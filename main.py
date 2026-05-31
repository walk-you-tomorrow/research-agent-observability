"""
main.py — Observable Research Agent 진입점

이 모듈은 에이전트의 메인 실행 파일로, 다음 역할을 수행한다:
1. YAML 설정 파일(config/agent_config.yaml)을 로드한다
2. 다중 턴 세션을 관리한다 (세션 ID 생성, 턴별 상태 초기화)
3. 각 턴마다 Langfuse CallbackHandler를 생성하여 LangGraph 그래프와 연결한다
4. 턴 간 상태(messages, turn_conclusions)를 전달하여 연속 대화를 지원한다

실행 방법:
    python main.py

사용 흐름:
    main() → load_config() → run_session(queries) → graph.invoke(turn_state)
"""
import time
import uuid

import truststore
import yaml
from dotenv import load_dotenv
from langfuse import get_client

from agent.graph import build_turn_graph
from agent.log_writer import setup_session_log, teardown_session_log
from agent.monitoring_schema import THRESHOLDS
from agent.turn_runner import (
    evaluate_and_diagnose,
    init_session_state,
    run_turn,
)
# 하위 호환 re-export: 기존 importer(tests/unit/test_cross_turn.py,
# scripts/test_phase37_*.py)가 `from main import _build_trace_data, _run_evaluation`로
# 참조한다. 엔진은 turn_runner로 이동했으나 진입점 호환을 위해 main에서 노출한다.
from agent.turn_runner import _build_trace_data, _execute_turn, _run_evaluation  # noqa: F401

# macOS 시스템 키체인의 인증서를 사용한다 (회사 프록시 SSL 인증서 호환)
truststore.inject_into_ssl()

# .env 파일에서 환경변수를 로드한다 (ANTHROPIC_API_KEY, LANGFUSE_* 등)
load_dotenv()


def load_config() -> dict:
    """YAML 설정 파일을 로드하여 딕셔너리로 반환한다.

    설정 파일 위치: config/agent_config.yaml
    포함 내용: LLM 모델, 재시도 횟수, 컨텍스트 윈도우 설정, knowledge_base 경로
    """
    with open("config/agent_config.yaml") as f:
        return yaml.safe_load(f)


# --- 분석 범위 상수 ---
# generate_analysis 노드가 이전 턴 결론을 최대 5개까지만 참조한다.
# Literature Review #21 (LLMs Get Lost In Multi-Turn Conversation): 다중 턴 시 -35% 성능 하락
MAX_TURNS_IN_SCOPE = 5


def _interpret_utilization(utilization: float) -> str:
    """윈도우 사용률에 대한 해석 텍스트를 반환한다."""
    if utilization < 0.1:
        return "여유 (10% 미만)"
    elif utilization < 0.3:
        return "적정"
    elif utilization < 0.7:
        return "주의 (목표 범위 내)"
    else:
        return "위험 (70%+ → 성능 저하 가능)"


def _interpret_noise(noise: float, utilization: float) -> str:
    """노이즈 비율에 대한 해석 텍스트를 반환한다."""
    if noise < 0.15:
        return "양호 — 대부분 새 데이터"
    elif noise < 0.35:
        return "보통 — 이전 턴 누적 있음"
    elif utilization < 0.1:
        return "높음 (단, 윈도우 여유 충분 → 실제 위험 낮음)"
    else:
        return "높음 — 이전 턴 데이터가 지배적"


def _interpret_rot_risk(rot_risk: float) -> str:
    """context rot risk에 대한 해석 텍스트를 반환한다."""
    if rot_risk < 0.05:
        return "안전"
    elif rot_risk < 0.2:
        return "경미"
    elif rot_risk < 0.5:
        return "주의"
    else:
        return "위험 — 컨텍스트 정리 필요"


def _print_context_summary(
    turn_number: int,
    result: dict,
    session_state: dict,
    wall_time_ms: int | None = None,
    trace_id: str | None = None,
    scores: dict[str, float] | None = None,
) -> None:
    """턴 완료 후 Context Monitoring 요약을 출력한다.

    4차원 품질 지표와 컨텍스트 전달 상태를 해석 텍스트와 함께
    터미널에 표시하여 컨텍스트가 어떻게 보존/변화하는지 보여준다.

    Args:
        turn_number: 현재 턴 번호.
        result: graph.invoke()의 반환값.
        session_state: 세션 상태 딕셔너리.
        wall_time_ms: 턴 실행 시간 (밀리초). None이면 표시하지 않는다.
        trace_id: Langfuse trace ID. None이면 표시하지 않는다.
        scores: 4D 평가 점수 딕셔너리. None이면 표시하지 않는다.
    """
    ctx_meta = result.get("context_metadata", {})
    ctx_eval = result.get("context_evaluation", {})
    verification = result.get("verification", {})
    conclusions = session_state.get("turn_conclusions", [])
    messages = session_state.get("messages", [])
    referenced = result.get("referenced_turns", [])

    print(f"\n{'═'*60}")
    print(f"  📊 Context Monitoring — Turn {turn_number}")
    if trace_id:
        print(f"  Langfuse trace: {trace_id}")
    print(f"{'═'*60}")

    # ── 컨텍스트 전달 상태 ──
    print(f"\n  [컨텍스트 전달]")
    print(f"    누적 메시지:       {len(messages)}개")
    print(f"    누적 턴 결론:      {len(conclusions)}개")

    # Literature Review #21: 분석 범위 턴 표시 — scope 밖 턴 정보 손실 감지
    prior_turns = turn_number - 1
    if prior_turns > 0:
        turns_in_scope = min(prior_turns, MAX_TURNS_IN_SCOPE)
        scope_warning = " ⚠" if turns_in_scope < prior_turns else ""
        print(f"    분석 범위 턴:      {turns_in_scope}/{prior_turns}{scope_warning}")
        if turns_in_scope < prior_turns:
            out = prior_turns - turns_in_scope
            print(f"      → Turn 1~{out} 은 scope 밖 (generate_analysis 참조 불가)")

    if referenced:
        print(f"    참조한 이전 턴:    {referenced}")

    # v3 통합: conflict_tracking dict 경유로 모순 정보 추출
    _ct = result.get("conflict_tracking") or {}
    if _ct.get("detected"):
        _resolution = _ct.get("resolution") or {}
        resolved = "해결됨" if _resolution.get("has_explanation") else "미해결"
        prev_conclusion = _resolution.get("conflict_summary", "")
        print(f"    ⚠ 이전 턴과 모순:  {resolved}")
        if prev_conclusion:
            # 이전 결론의 첫 80자만 표시
            print(f"      이전 결론: \"{prev_conclusion[:80]}...\"")

    # Literature Review #46 (Honeycomb): latency는 first-class 모니터링 신호
    if wall_time_ms is not None:
        print(f"    턴 실행 시간:      {wall_time_ms:,}ms")

    # ── 완전성 (Completeness) ──
    gathered = result.get("gathered_data", [])
    is_sufficient = ctx_eval.get("is_sufficient", "N/A")
    confidence = ctx_eval.get("confidence_score", "N/A")
    print(f"\n  [완전성 Completeness]")
    print(f"    수집 데이터:       {len(gathered)}건")
    # 수집된 소스명 표시 — 어디서 데이터가 왔는지 근거 제공
    if gathered:
        sources = [g.get("source", "?") for g in gathered[:5]]
        print(f"      소스: {', '.join(sources)}")
    print(f"    충분성 판단:       {is_sufficient} (confidence: {confidence})")
    if is_sufficient is True:
        print(f"      → evaluate_context가 데이터 충분으로 판단 → generate_analysis 진행")
    elif is_sufficient is False:
        gather_iter = result.get("gather_iteration", 0)
        print(f"      → 데이터 부족 (iteration {gather_iter}/3)")
    missing = ctx_eval.get("missing_info", "")
    if missing:
        print(f"    부족 정보:         {missing}")

    # ── 효율성 (Efficiency) ──
    total_tokens = ctx_meta.get("total_tokens", "N/A")
    utilization = ctx_meta.get("context_window_utilization", "N/A")
    print(f"\n  [효율성 Efficiency]")
    print(f"    총 토큰:           {total_tokens}")
    if isinstance(utilization, (int, float)):
        interp = _interpret_utilization(utilization)
        print(f"    윈도우 사용률:     {utilization:.1%} — {interp}")
    else:
        print(f"    윈도우 사용률:     {utilization}")

    # 토큰 소스별 분배 — source_breakdown 딕셔너리에서 추출
    source = ctx_meta.get("source_breakdown", {})
    src_gathered = source.get("gathered_data", 0)
    src_prev = source.get("previous_turns", 0)
    src_concl = source.get("turn_conclusions", 0)
    src_total = src_gathered + src_prev + src_concl
    if src_total > 0:
        pct_g = src_gathered / src_total * 100
        pct_p = src_prev / src_total * 100
        pct_c = src_concl / src_total * 100
        print(f"    ┌ 수집 데이터:     {src_gathered:>6} tokens ({pct_g:.0f}%)")
        print(f"    ├ 이전 턴 메시지:  {src_prev:>6} tokens ({pct_p:.0f}%)")
        print(f"    └ 턴 결론:         {src_concl:>6} tokens ({pct_c:.0f}%)")

    # ── 관련성 (Relevance) ──
    noise = ctx_meta.get("noise_ratio", "N/A")
    rot_risk = ctx_meta.get("rot_risk", None)
    print(f"\n  [관련성 Relevance]")
    if isinstance(noise, (int, float)):
        util_val = utilization if isinstance(utilization, (int, float)) else 0
        interp = _interpret_noise(noise, util_val)
        print(f"    노이즈 비율:       {noise:.1%} — {interp}")
    else:
        print(f"    노이즈 비율:       {noise}")

    # Literature Review #26 (Anthropic): context rot = utilization × noise_ratio
    if rot_risk is not None:
        interp = _interpret_rot_risk(rot_risk)
        print(f"    Context Rot Risk:  {rot_risk:.4f} — {interp}")
        if isinstance(noise, (int, float)) and isinstance(utilization, (int, float)):
            print(f"      (= 사용률 {utilization:.1%} × 노이즈 {noise:.1%})")

    # ── 일관성 (Consistency) ──
    conf_delta = ctx_meta.get("confidence_delta", None)
    missing_resolved = ctx_meta.get("missing_info_resolved", None)
    print(f"\n  [일관성 Consistency]")

    # 패턴 A: 같은 턴 내 iteration 간 변화
    if conf_delta is not None and conf_delta != 0:
        direction = "개선 ↑" if conf_delta > 0 else "하락 ↓"
        print(f"    신뢰도 변화:       {conf_delta:+.3f} ({direction})")
    if missing_resolved is True:
        print(f"    부족 정보:         해결됨 ✓ (이전 iteration에서 지적된 정보가 수집됨)")

    # 패턴 B + 검증 결과
    verdict = verification.get("overall_verdict", "N/A")
    print(f"    검증 결과:         {verdict}")

    # ── 컨텍스트 보존도 ──
    continuity = ctx_meta.get("continuity_score", None)
    if continuity is not None:
        cont_interp = "완전 보존" if continuity >= 0.9 else (
            "일부 손실" if continuity >= 0.7 else "상당 손실 ⚠"
        )
        print(f"\n  [컨텍스트 보존도]")
        print(f"    컨텍스트 보존도:   {continuity:.3f} — {cont_interp}")

    # ── 4D 평가 점수 (Langfuse Score) ──
    if scores:
        thresholds = {
            "completeness_score": THRESHOLDS["completeness"],
            "efficiency_score": THRESHOLDS["efficiency"],
            "relevance_score": THRESHOLDS["relevance"],
            "consistency_score": THRESHOLDS["consistency"],
        }
        print(f"\n  [4D 평가 Score — Langfuse 부착됨]")
        for name, value in scores.items():
            t = thresholds.get(name, 0.7)
            verdict_mark = "PASS ✓" if value >= t else "FAIL ✗"
            short_name = name.replace("_score", "").capitalize()
            print(f"    {short_name:<14} {value:.2f}  [{verdict_mark}, threshold={t}]")

    print(f"\n{'═'*60}")


def run_session(queries: list[str], config: dict | None = None) -> dict:
    """다중 턴 세션을 실행한다.

    Args:
        queries: 턴별 사용자 질문 리스트. 예: ["카페 창업 어디가 좋을까?", "강남 vs 마포?"]
        config: agent_config.yaml 설정. None이면 로그 저장 비활성.

    Returns:
        최종 세션 상태 딕셔너리. messages, turn_conclusions, current_turn 등 포함.

    동작 흐름:
        1. 고유 세션 ID를 생성한다 (sess_xxxxxxxx 형식)
        2. LangGraph 그래프를 빌드한다 (6노드 파이프라인)
        3. 각 질문에 대해 턴을 순차 실행한다:
           a. Langfuse CallbackHandler 생성 (트레이스 연결)
           b. 턴 상태 초기화 (이전 턴 결과 + 새 질문)
           c. 그래프 실행 (analyze → gather → evaluate → generate → verify → respond)
           d. 세션 상태 업데이트 (다음 턴으로 전달할 데이터)
    """
    # 세션 ID는 Langfuse에서 같은 세션의 모든 턴을 그룹화하는 데 사용된다
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # 로그 자동 저장: config가 있고 logging.enabled=true이면 TeeWriter 설치
    if config:
        setup_session_log(config, session_id)

    # 6노드 LangGraph 그래프를 컴파일한다
    graph = build_turn_graph()

    # 세션 전체에 걸쳐 유지되는 상태. 턴 간에 messages와 turn_conclusions가 전달된다.
    session_state = init_session_state(session_id)

    # --- 턴 간 대기 시간 (Rate Limit 대응) ---
    # 50K tokens/min 제한에서 7턴 시나리오가 안전하게 실행되려면 턴 사이에 대기가 필요하다.
    retry_cfg = config.get("retry", {}) if config else {}
    inter_turn_delay = retry_cfg.get("inter_turn_delay_seconds", 0)

    # 각 턴의 graph.invoke() 결과를 축적하여 테스트에서 per-turn 검증이 가능하게 한다.
    per_turn_results: list[dict] = []

    # --- 턴별 실행 루프 ---
    for i, query in enumerate(queries):
        turn_number = i + 1

        # 첫 턴이 아닌 경우, 턴 간 대기하여 Rate Limit 여유를 확보한다
        if i > 0 and inter_turn_delay > 0:
            print(f"\n  ⏳ Rate Limit 대응: {inter_turn_delay}초 대기 중...")
            time.sleep(inter_turn_delay)

        print(f"\n{'='*60}")
        print(f"Turn {turn_number}: {query}")
        print(f"{'='*60}")
        msg_count = len(session_state.get("messages", []))
        concl_count = len(session_state.get("turn_conclusions", []))
        print(f"  📎 컨텍스트: 이전 메시지 {msg_count}개, 턴 결론 {concl_count}개")

        # 턴 실행: turn_runner.run_turn이 턴 상태 구성 → _execute_turn(그래프 invoke +
        # Langfuse trace) → 다음 턴용 새 session_state(불변) 생성을 한 번에 처리한다.
        prev_diagnosis = session_state.get("previous_diagnosis", [])
        turn_out = run_turn(graph, session_state, query, turn_number, config)
        result = turn_out["result"]
        trace_id = turn_out["trace_id"]
        turn_wall_time_ms = turn_out["wall_time_ms"]
        session_state = turn_out["session_state"]

        # 턴 결과를 축적하여 테스트에서 per-turn 검증이 가능하게 한다.
        per_turn_results.append(result)

        print(f"\nAgent: {turn_out['response']}")

        # --- 4D 평가 + 자동 진단 + Langfuse 기록 (요약 출력 전에 실행) ---
        diag_out = (
            evaluate_and_diagnose(config, trace_id, result, prev_diagnosis)
            if trace_id else {"scores": None, "diagnosis": [], "improvement": False}
        )
        turn_scores = diag_out["scores"]
        # 다음 턴에 진단 결과 전달 (개선 여부 판정용)
        session_state["previous_diagnosis"] = diag_out["diagnosis"]

        # --- Context Monitoring 요약 (평가 점수 포함) ---
        _print_context_summary(
            turn_number, result, session_state,
            wall_time_ms=turn_wall_time_ms,
            trace_id=trace_id,
            scores=turn_scores,
        )

        # Langfuse에 버퍼링된 이벤트를 즉시 전송한다.
        get_client().flush()

    session_state["turn_results"] = per_turn_results
    teardown_session_log()
    return session_state


def run_interactive_session(config: dict | None = None) -> dict:
    """대화형 세션을 실행한다.

    사용자로부터 질문을 하나씩 입력받아 즉시 응답한 뒤,
    다음 질문을 기다린다. 'q' 또는 빈 입력으로 종료한다.

    Args:
        config: agent_config.yaml 설정. None이면 로그 저장 비활성.

    Returns:
        최종 세션 상태 딕셔너리.
    """
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # 로그 자동 저장: config가 있고 logging.enabled=true이면 TeeWriter 설치
    if config:
        setup_session_log(config, session_id)

    graph = build_turn_graph()

    session_state = init_session_state(session_id)

    turn_number = 0
    while True:
        query = input("\n질문> ").strip()
        if not query or query.lower() == "q":
            break

        turn_number += 1
        print(f"\n{'='*60}")
        print(f"Turn {turn_number}: {query}")
        print(f"{'='*60}")
        msg_count = len(session_state.get("messages", []))
        concl_count = len(session_state.get("turn_conclusions", []))
        print(f"  📎 컨텍스트: 이전 메시지 {msg_count}개, 턴 결론 {concl_count}개")

        # 턴 실행 (turn_runner 공용 엔진) — 응답을 먼저 보여준 뒤 평가/진단을 수행한다.
        prev_diagnosis = session_state.get("previous_diagnosis", [])
        turn_out = run_turn(graph, session_state, query, turn_number, config)
        result = turn_out["result"]
        trace_id = turn_out["trace_id"]
        turn_wall_time_ms = turn_out["wall_time_ms"]
        session_state = turn_out["session_state"]

        print(f"\nAgent: {turn_out['response']}")

        # --- 4D 평가 + 자동 진단 + Langfuse 기록 ---
        diag_out = (
            evaluate_and_diagnose(config, trace_id, result, prev_diagnosis)
            if trace_id else {"scores": None, "diagnosis": [], "improvement": False}
        )
        turn_scores = diag_out["scores"]
        session_state["previous_diagnosis"] = diag_out["diagnosis"]

        # --- Context Monitoring 요약 (평가 점수 포함) ---
        _print_context_summary(
            turn_number, result, session_state,
            wall_time_ms=turn_wall_time_ms,
            trace_id=trace_id,
            scores=turn_scores,
        )

        get_client().flush()

    teardown_session_log()
    return session_state


def main() -> None:
    """에이전트의 메인 실행 함수.

    사용자로부터 질문을 입력받아 즉시 응답하는 대화형 세션을 실행한다.
    'q' 또는 빈 입력으로 종료한다.
    """
    config = load_config()

    print("서울 상권 분석 Research Agent")
    print("종료: 'q' 입력 또는 빈 줄 엔터")
    print("-" * 40)

    run_interactive_session(config=config)


if __name__ == "__main__":
    main()
