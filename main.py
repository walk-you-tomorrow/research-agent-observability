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
from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

from agent.graph import build_turn_graph
from agent.log_writer import setup_session_log, teardown_session_log
from agent.monitoring_schema import ATTRS, THRESHOLDS
from evaluation.diagnosis import diagnose_quality
from evaluation.run_evaluation import evaluate_turn

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


def _build_trace_data(result: dict) -> dict:
    """AgentState 결과를 Judge가 기대하는 trace_data 형태로 매핑한다.

    각 judge_*.py의 build_*_input() 함수는 trace_data["metadata"]에서
    Layer 2 attribute를 추출한다. 이 함수는 graph.invoke() 결과의
    중첩 딕셔너리를 flat metadata로 변환한다.

    Args:
        result: graph.invoke()의 반환값 (AgentState 딕셔너리).

    Returns:
        {"metadata": {...}} 구조의 딕셔너리. evaluate_turn()에 전달 가능.
    """
    ctx_meta = result.get("context_metadata", {})
    ctx_eval = result.get("context_evaluation", {})
    verification = result.get("verification", {})
    query_analysis = result.get("query_analysis", {})
    source = ctx_meta.get("source_breakdown", {})
    prev_fidelity = result.get("previous_turn_fidelity", {})

    metadata = {
        # 완전성 (Completeness) 지표
        ATTRS["context.is_sufficient"]: ctx_eval.get("is_sufficient"),
        ATTRS["context.sufficiency_confidence"]: ctx_eval.get("confidence_score", 0),
        ATTRS["context.missing_info"]: ctx_eval.get("missing_info", ""),
        ATTRS["context.truncated_items"]: ctx_meta.get("truncated_items", []),
        ATTRS["verify.numeric_check_passed"]: (verification.get("numeric_check", {}) or {}).get("passed"),
        ATTRS["query.intent"]: query_analysis.get("intent", "unknown"),

        # 효율성 (Efficiency) 지표 — 자원 관점 (2026-04-27 analysis/31)
        ATTRS["context.total_tokens"]: ctx_meta.get("total_tokens", 0),
        ATTRS["context.window_utilization"]: ctx_meta.get("context_window_utilization", 0),
        ATTRS["context.source.system_prompt_tokens"]: source.get("system_prompt", 0),
        ATTRS["context.source.query_analysis_tokens"]: source.get("query_analysis", 0),
        ATTRS["context.source.gathered_data_tokens"]: source.get("gathered_data", 0),
        ATTRS["context.source.previous_turns_tokens"]: source.get("previous_turns", 0) + source.get("turn_conclusions", 0),
        # v3 폐기: turn_conclusions_tokens → previous_turns_tokens에 통합
        # 신규 효율성 자원 지표 (2026-04-27)
        ATTRS["turn.tool_call_count"]: len(result.get("tools_called", [])),
        ATTRS["turn.total_cost_usd"]: 0.0,  # placeholder (Langfuse usage 사후 계산 예정)
        ATTRS["turn.wall_time_ms"]: result.get("turn_wall_time_ms", 0),  # _execute_turn에서 전파 필요

        # 관련성 (Relevance) 지표
        ATTRS["gather.tools_called"]: result.get("tools_called", []),
        ATTRS["query.tool_plan"]: query_analysis.get("tool_plan", []),
        ATTRS["gather.items_collected"]: ctx_meta.get("gathered_count", 0),
        ATTRS["gather.items_excluded"]: result.get("excluded_items_count", 0),
        ATTRS["gather.excluded_items"]: result.get("excluded_items_sources", []),
        ATTRS["context.noise_ratio"]: ctx_meta.get("noise_ratio", 0),
        ATTRS["context.effective_noise_ratio"]: ctx_meta.get("effective_noise_ratio", 0),
        ATTRS["context.rot_risk"]: ctx_meta.get("rot_risk", 0),
        ATTRS["context.rot_velocity"]: ctx_meta.get("rot_velocity", 0),

        # A4+A5: Rot Gate
        ATTRS["context.dead_weight_tokens"]: ctx_meta.get("dead_weight_tokens", 0),
        ATTRS["context.rot_gate_triggered"]: ctx_meta.get("rot_gate_triggered", False),
        # v3 폐기: rot_gate_pruned_tokens (derived: dead_weight_tokens × rot_gate_triggered)

        # 일관성 (Consistency) 지표 — 패턴 A + 패턴 B + 패턴 C
        ATTRS["context.missing_info_resolved"]: ctx_meta.get("missing_info_resolved", False),
        ATTRS["context.confidence_delta"]: ctx_meta.get("confidence_delta", 0),
        # v3 통합: 3 attribute → conflict_tracking dict (analysis.conflict_tracking)
        ATTRS["analysis.conflict_tracking"]: result.get("conflict_tracking", {
            "detected": False,
            "resolution": {"has_explanation": False, "conflict_summary": "", "source_resolution": ""},
        }),
        ATTRS["analysis.referenced_turns"]: result.get("referenced_turns", []),

        # G1: 충실도 (일관성 패턴 C)
        ATTRS["context.fidelity_score"]: ctx_meta.get("fidelity_score", 1.0),
        ATTRS["context.fidelity_detail"]: ctx_meta.get("fidelity_detail", {}),
        # v3 폐기: continuity_score (fidelity_score와 의미 중복)

        # G5: 탈락 이유
        ATTRS["gather.exclusion_reasons"]: result.get("exclusion_reasons", []),

        # G3: 교차 턴 진화
        # v3 폐기: new_data_ratio / token_delta (derived from total_tokens 비교)
        ATTRS["context.contributing_turns"]: ctx_meta.get("contributing_turns", 0),
        # v3 폐기: sufficiency_by_source (Judge 활용 약함)

        # Post-1: 인과 전파
        ATTRS["context.causal_sources"]: ctx_meta.get("causal_sources", []),

        # Post-3: 의미적 정보 밀도
        ATTRS["context.information_density"]: ctx_meta.get("information_density", 0),
        ATTRS["context.redundancy_ratio"]: ctx_meta.get("redundancy_ratio", 0),

        # Phase 2 Step 2.1: respond_to_user 신규 속성 (previous_turn_fidelity 경유)
        ATTRS["response.key_claims_preserved"]: prev_fidelity.get("key_claims_preserved"),
        # v3 폐기: conditions_preserved / conditions_detail (Tier 4 결함)
        ATTRS["response.lost_claims"]: prev_fidelity.get("lost_claims"),
        # v3 폐기: compression_ratio → context.fidelity_detail에 흡수

        # H0 재설계: groundedness 평가를 위한 최종 응답 텍스트
        ATTRS["response.final_text"]: result.get("response", "")[:2000],

        # 이탈 감지 (Phase 3.7) — alignment judge 입력용 + diagnosis.py 규칙 발화용
        # user_query: AgentState 직접 참조. analysis.summary: analysis_result.summary[:3000].
        # session_continuity: AgentState 필드 (Turn 1 / Ollama 미가용 시 None).
        ATTRS["query.user_query"]: result.get("user_query", ""),
        ATTRS["analysis.summary"]: (result.get("analysis_result") or {}).get("summary", "")[:3000],
        ATTRS["query.session_continuity"]: result.get("session_continuity"),
    }
    return {"metadata": metadata}


def _run_evaluation(
    config: dict, trace_id: str, result: dict
) -> tuple[dict[str, float], dict[str, float]] | tuple[None, dict]:
    """설정에 따라 4D 평가를 실행한다.

    config에 evaluation.auto_evaluate=true가 설정되어 있으면 평가를 실행하고,
    아니면 건너뛴다. 평가 실패 시 에러를 출력하고 (None, {})을 반환한다.

    Args:
        config: agent_config.yaml 설정 딕셔너리.
        trace_id: Langfuse trace ID.
        result: graph.invoke()의 반환값.

    Returns:
        (scores_4d, alignment_scores) 튜플.
        scores_4d: 4D 점수 딕셔너리 또는 None (평가 비활성/실패 시).
        alignment_scores: 이탈 감지 점수 딕셔너리 (BL-008: diagnose_quality 주입용).
    """
    eval_config = config.get("evaluation", {}) if config else {}
    if not eval_config.get("auto_evaluate", False):
        return None, {}

    try:
        trace_data = _build_trace_data(result)
        return evaluate_turn(trace_id, trace_data)
    except Exception as e:
        print(f"  ⚠ 4D 평가 실패: {e}")
        return None, {}


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


@observe()
def _execute_turn(
    graph,
    turn_state: dict,
    session_id: str,
    turn_number: int,
) -> tuple[dict, str, int]:
    """턴을 실행하고 Langfuse trace에 세션/모니터링 메타데이터를 기록한다.

    @observe()가 trace를 생성하고, update_current_trace()로 session_id/name/tags를 설정한다.
    CallbackHandler는 trace_context로 이 trace에 연결되어 노드 스팬이 하위에 생성된다.
    턴 완료 후 모니터링 속성을 trace-level metadata로 기록하여 visualize_session.py가 조회할 수 있게 한다.

    Args:
        graph: 컴파일된 LangGraph 그래프.
        turn_state: 이번 턴의 초기 상태.
        session_id: Langfuse session ID (세션 그룹화용).
        turn_number: 현재 턴 번호.

    Returns:
        (result, trace_id, wall_time_ms) 튜플.
    """
    # Langfuse trace에 세션 정보 설정 (session_id/tags는 여기서, name은 invoke 이후에)
    # name을 먼저 설정하면 CallbackHandler(update_trace=True)가 "LangGraph"로 덮어쓴다
    get_client().update_current_trace(
        session_id=session_id,
        tags=["research_agent", "context_monitoring"],
    )
    trace_id = get_client().get_current_trace_id()

    # CallbackHandler를 이 trace에 연결하여 노드 스팬이 하위에 생성되게 한다
    langfuse_handler = CallbackHandler(
        update_trace=True,
        trace_context={"trace_id": trace_id},
    )

    # 그래프 실행 + wall_time 측정
    turn_start = time.monotonic()
    result = graph.invoke(turn_state, config={"callbacks": [langfuse_handler]})
    wall_time_ms = round((time.monotonic() - turn_start) * 1000)

    # span-level 모니터링 속성을 trace-level metadata로 승격
    # visualize_session.py가 trace.metadata에서 읽을 수 있게 한다
    # name을 여기서 설정해야 CallbackHandler의 "LangGraph" 덮어쓰기 이후에 반영된다
    ctx_meta = result.get("context_metadata", {})
    ctx_eval = result.get("context_evaluation", {})
    source = ctx_meta.get("source_breakdown", {})
    get_client().update_current_trace(
        name=f"turn_{turn_number}",
        metadata={
            # 턴 식별 + 효율성 자원 (2026-04-27 analysis/31 — 효율성 = 자원 관점)
            ATTRS["turn.number"]: turn_number,
            ATTRS["turn.wall_time_ms"]: wall_time_ms,
            ATTRS["turn.tool_call_count"]: len(result.get("tools_called", [])),
            ATTRS["turn.total_cost_usd"]: 0.0,  # TODO: Langfuse usage에서 사후 계산. 현재는 placeholder.

            # 효율성 (Efficiency)
            ATTRS["context.total_tokens"]: ctx_meta.get("total_tokens"),
            ATTRS["context.window_utilization"]: ctx_meta.get("context_window_utilization"),
            ATTRS["context.source.query_analysis_tokens"]: source.get("query_analysis", 0),
            ATTRS["context.source.gathered_data_tokens"]: source.get("gathered_data", 0),
            ATTRS["context.source.previous_turns_tokens"]: source.get("previous_turns", 0) + source.get("turn_conclusions", 0),
            # v3 폐기: turn_conclusions_tokens → previous_turns_tokens에 통합

            # 관련성 (Relevance)
            ATTRS["context.noise_ratio"]: ctx_meta.get("noise_ratio"),
            ATTRS["context.rot_risk"]: ctx_meta.get("rot_risk"),

            # 완전성 (Completeness)
            ATTRS["context.is_sufficient"]: ctx_eval.get("is_sufficient"),
            ATTRS["context.sufficiency_confidence"]: ctx_eval.get("confidence_score"),

            # 일관성 (Consistency)
            # v3 통합: 3 attribute → conflict_tracking dict
            ATTRS["analysis.conflict_tracking"]: result.get("conflict_tracking", {
                "detected": False,
                "resolution": {"has_explanation": False, "conflict_summary": "", "source_resolution": ""},
            }),
            ATTRS["context.confidence_delta"]: ctx_meta.get("confidence_delta"),

            # v3 폐기: continuity_score (fidelity_score와 의미 중복)

            # A4+A5: Rot Gate
            ATTRS["context.dead_weight_tokens"]: ctx_meta.get("dead_weight_tokens"),
            ATTRS["context.rot_gate_triggered"]: ctx_meta.get("rot_gate_triggered"),
            # v3 폐기: rot_gate_pruned_tokens (derived)

            # Phase 2 Step 2.1 신규 속성
            ATTRS["context.effective_noise_ratio"]: ctx_meta.get("effective_noise_ratio"),
            ATTRS["context.rot_velocity"]: ctx_meta.get("rot_velocity"),
            ATTRS["context.fidelity_score"]: ctx_meta.get("fidelity_score"),
            ATTRS["context.fidelity_detail"]: ctx_meta.get("fidelity_detail"),
            ATTRS["response.key_claims_preserved"]: result.get("previous_turn_fidelity", {}).get("key_claims_preserved"),
            # v3 폐기: conditions_preserved / conditions_detail (Tier 4 결함)
            ATTRS["response.lost_claims"]: result.get("previous_turn_fidelity", {}).get("lost_claims"),

            # H0 재설계: groundedness 평가용 최종 응답 텍스트
            ATTRS["response.final_text"]: result.get("response", "")[:2000],

            # 이탈 감지 (Phase 3.7) — trace-level 가시화 + alignment judge 입력 보조
            ATTRS["query.user_query"]: result.get("user_query", ""),
            ATTRS["analysis.summary"]: (result.get("analysis_result") or {}).get("summary", "")[:3000],
            ATTRS["query.session_continuity"]: result.get("session_continuity"),
        },
    )

    # @observe() 컨텍스트가 닫히기 전에 데이터를 Langfuse로 전송한다
    # 턴 6~7이 누락되는 문제 방지: @observe() 밖의 flush()만으로는 불충분할 수 있음
    get_client().flush()

    return result, trace_id, wall_time_ms


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
    session_state: dict = {
        "messages": [],           # LangGraph 메시지 히스토리 (누적)
        "session_id": session_id,  # Langfuse 세션 추적용
        "current_turn": 0,         # 현재 턴 번호
        "turn_conclusions": [],    # 각 턴의 결론 요약 리스트 (일관성 패턴 B에 사용)
        "session_intent_history": [],  # Phase 3.7: intent 이력 (Rot Gate 독립, 절대 prune 안 됨)
    }

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

        # 턴 상태 초기화: 세션 상태를 복사하고 이번 턴에 필요한 필드를 추가/초기화한다.
        # 반복 제어 변수(gather_retry_count, verify_retry_count 등)는 매 턴 0으로 리셋된다.
        turn_state = {
            **session_state,                     # 이전 턴에서 전달된 상태 (messages, turn_conclusions)
            "user_query": query,                 # 이번 턴의 사용자 질문
            "current_turn": turn_number,         # 현재 턴 번호
            "gather_iteration": 0,               # 데이터 수집 반복 횟수 (재수집 시 증가)
            "gather_retry_count": 0,             # gather 재시도 카운터
            "verify_retry_count": 0,             # verify 재시도 카운터
            "previous_missing_info": "",         # 일관성 패턴 A: 이전 iteration의 missing_info
            "previous_confidence": 0.0,          # 일관성 패턴 A: 이전 iteration의 confidence
            # v3 통합 (2026-04-29): 일관성 패턴 B 모순 추적을 단일 dict로
            "conflict_tracking": {
                "detected": False,
                "resolution": {"has_explanation": False, "conflict_summary": "", "source_resolution": ""},
            },
            "referenced_turns": [],              # 일관성 패턴 B: 참조한 이전 턴 번호
        }

        # _execute_turn이 @observe()로 trace를 생성하고, session_id/name/tags를 설정하며,
        # 모니터링 속성을 trace-level metadata에 기록한다.
        result, trace_id, turn_wall_time_ms = _execute_turn(
            graph, turn_state, session_id, turn_number,
        )

        # 턴 결과를 축적하여 테스트에서 per-turn 검증이 가능하게 한다.
        per_turn_results.append(result)

        # 세션 상태 업데이트: 이번 턴의 결과에서 다음 턴으로 전달할 데이터를 추출한다.
        session_state["messages"] = result.get("messages", [])
        session_state["turn_conclusions"] = result.get("turn_conclusions", [])
        session_state["current_turn"] = turn_number
        # Phase 3.7: session_intent_history는 Rot Gate와 독립. 누락 시 Turn 2+의 session_continuity가 항상 None.
        session_state["session_intent_history"] = result.get("session_intent_history", [])
        # P2 보강 (2026-05-08): previous_turn_fidelity를 다음 턴에 전달.
        # 이전엔 누락되어 evaluate_context가 항상 fallback path → fidelity_score=1.0 고정.
        session_state["previous_turn_fidelity"] = result.get("previous_turn_fidelity", {})
        # 패턴 A 데이터(missing_info, confidence) 다음 턴 전달
        session_state["previous_missing_info"] = result.get("previous_missing_info", "")
        session_state["previous_confidence"] = result.get("previous_confidence", 0.0)

        print(f"\nAgent: {result.get('response', '(응답 없음)')}")

        # --- 4D 평가 + Langfuse Score 부착 (요약 출력 전에 실행) ---
        turn_scores = None
        _alignment_scores: dict[str, float] = {}
        if trace_id:
            turn_scores, _alignment_scores = _run_evaluation(config, trace_id, result)

        # --- Post-5: 자동 진단 + Langfuse 기록 ---
        # 4D 점수가 있으면 diagnose_quality()로 개선 제안 생성하고 Langfuse에 기록
        diagnosis_results = []
        if turn_scores:
            try:
                trace_data = _build_trace_data(result)
                # BL-008: alignment 점수를 trace_data에 주입 → Pattern I/II/III 규칙 발화
                trace_data["metadata"].update(_alignment_scores)
                diagnosis_results = diagnose_quality(turn_scores, trace_data)
                # 이전 턴 진단과 현재 점수를 비교하여 개선 여부 판정
                prev_diag = session_state.get("previous_diagnosis", [])
                improvement_applied = False
                if prev_diag:
                    prev_dims = {d["dimension"] for d in prev_diag}
                    curr_fail_dims = {d["dimension"] for d in diagnosis_results}
                    # 이전에 진단된 차원이 현재 진단에서 사라졌으면 개선된 것
                    improvement_applied = bool(prev_dims - curr_fail_dims)
                # Langfuse trace에 진단 결과 기록 (v3: ingestion batch API)
                langfuse_client = Langfuse()
                langfuse_client.api.ingestion.batch(batch=[{
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "timestamp": None,
                    "body": {
                        "id": trace_id,
                        "metadata": {
                            ATTRS["eval.diagnosis"]: diagnosis_results,
                            ATTRS["eval.improvement_applied"]: improvement_applied,
                        },
                    },
                }])
            except Exception as e:
                print(f"  ⚠ 진단 기록 실패: {e}")

        # 다음 턴에 진단 결과 전달 (개선 여부 판정용)
        session_state["previous_diagnosis"] = diagnosis_results

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

    session_state: dict = {
        "messages": [],
        "session_id": session_id,
        "current_turn": 0,
        "turn_conclusions": [],
        "session_intent_history": [],  # Phase 3.7: intent 이력 (Rot Gate 독립, 절대 prune 안 됨)
    }

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

        turn_state = {
            **session_state,
            "user_query": query,
            "current_turn": turn_number,
            "gather_iteration": 0,
            "gather_retry_count": 0,
            "verify_retry_count": 0,
            "previous_missing_info": "",
            "previous_confidence": 0.0,
            # v3 통합 (2026-04-29): 모순 추적을 단일 dict로
            "conflict_tracking": {
                "detected": False,
                "resolution": {"has_explanation": False, "conflict_summary": "", "source_resolution": ""},
            },
            "referenced_turns": [],
        }

        result, trace_id, turn_wall_time_ms = _execute_turn(
            graph, turn_state, session_id, turn_number,
        )

        session_state["messages"] = result.get("messages", [])
        session_state["turn_conclusions"] = result.get("turn_conclusions", [])
        session_state["current_turn"] = turn_number
        # Phase 3.7: session_intent_history는 Rot Gate와 독립. 누락 시 Turn 2+의 session_continuity가 항상 None.
        session_state["session_intent_history"] = result.get("session_intent_history", [])
        # P2 보강 (2026-05-08): interactive 세션도 다음 턴에 fidelity/패턴A 데이터 전달
        session_state["previous_turn_fidelity"] = result.get("previous_turn_fidelity", {})
        session_state["previous_missing_info"] = result.get("previous_missing_info", "")
        session_state["previous_confidence"] = result.get("previous_confidence", 0.0)

        print(f"\nAgent: {result.get('response', '(응답 없음)')}")

        # --- 4D 평가 + Langfuse Score 부착 (요약 출력 전에 실행) ---
        turn_scores = None
        _alignment_scores: dict[str, float] = {}
        if trace_id:
            turn_scores, _alignment_scores = _run_evaluation(config, trace_id, result)

        # --- Post-5: 자동 진단 + Langfuse 기록 ---
        if turn_scores:
            try:
                trace_data = _build_trace_data(result)
                # BL-008: alignment 점수를 trace_data에 주입 → Pattern I/II/III 규칙 발화
                trace_data["metadata"].update(_alignment_scores)
                diag = diagnose_quality(turn_scores, trace_data)
                prev_diag = session_state.get("previous_diagnosis", [])
                improvement = bool(
                    {d["dimension"] for d in prev_diag} - {d["dimension"] for d in diag}
                ) if prev_diag else False
                langfuse_client = Langfuse()
                langfuse_client.api.ingestion.batch(batch=[{
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "timestamp": None,
                    "body": {
                        "id": trace_id,
                        "metadata": {
                            ATTRS["eval.diagnosis"]: diag,
                            ATTRS["eval.improvement_applied"]: improvement,
                        },
                    },
                }])
                session_state["previous_diagnosis"] = diag
            except Exception as e:
                print(f"  ⚠ 진단 기록 실패: {e}")

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
