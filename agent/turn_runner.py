"""
agent/turn_runner.py — 단일 턴 실행 엔진 (CLI · 채팅 공용)

역할:
    main.py의 CLI 루프와 chat_app.py의 Streamlit 채팅이 공유하는 턴 실행 로직을 모은다.
    이전에 main.py에 private로 묶여 있던 단일 턴 코드를 추출하여, 두 진입점이 동일한
    엔진을 재사용하도록(DRY) 한다.

제공 기능:
    - init_session_state(session_id): 세션 시작 시 빈 상태 dict 생성
    - run_turn(...):                  한 턴 실행 + 턴 간 상태 carry-over (불변 반환)
    - evaluate_and_diagnose(...):     4D 평가 + 진단 + Langfuse 기록 (백그라운드 스레드 호출 가능)
    - _execute_turn(...):             그래프 invoke + Langfuse trace 기록 (@observe)

데이터 흐름:
    init_session_state(session_id)
      → run_turn(graph, session_state, query, turn_number, config)
          → (호출자가 응답을 UI/터미널에 표시)
      → evaluate_and_diagnose(config, trace_id, result, previous_diagnosis)

Langfuse 기록 attribute:
    turn.*, context.*, analysis.*, response.*, query.*, eval.* (모두 ATTRS 경유)
"""
import time
import uuid

from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

from agent.monitoring_schema import ATTRS
from evaluation.diagnosis import diagnose_quality
from evaluation.run_evaluation import evaluate_turn


def init_session_state(session_id: str) -> dict:
    """세션 시작 시 빈 세션 상태 dict를 생성한다.

    턴 간에 유지되는 필드만 초기화한다. 턴별 입력/제어 필드는 run_turn이 채운다.

    Args:
        session_id: Langfuse 세션 그룹화용 ID (예: sess_xxxxxxxx).

    Returns:
        messages/turn_conclusions/session_intent_history 등을 비운 상태 dict.
    """
    return {
        "messages": [],                # LangGraph 메시지 히스토리 (누적)
        "session_id": session_id,      # Langfuse 세션 추적용
        "current_turn": 0,             # 현재 턴 번호
        "turn_conclusions": [],        # 각 턴의 결론 요약 (일관성 패턴 B)
        "session_intent_history": [],  # Phase 3.7: intent 이력 (Rot Gate 독립, 절대 prune 안 됨)
    }


def _build_turn_state(session_state: dict, query: str, turn_number: int) -> dict:
    """세션 상태 + 이번 턴 입력으로 graph.invoke()에 전달할 턴 상태를 구성한다.

    반복 제어 변수(gather_iteration 등)는 매 턴 0으로 리셋된다.

    Args:
        session_state: 이전 턴에서 전달된 상태 (messages, turn_conclusions 등).
        query: 이번 턴의 사용자 질문.
        turn_number: 현재 턴 번호.

    Returns:
        graph.invoke()에 넘길 턴 상태 dict.
    """
    return {
        **session_state,                     # 이전 턴 상태 (messages, turn_conclusions, ...)
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


def _carry_over(session_state: dict, result: dict, turn_number: int) -> dict:
    """이번 턴 결과에서 다음 턴으로 전달할 새 세션 상태를 만든다 (불변 패턴).

    기존 session_state를 변경하지 않고, 다음 턴에 필요한 필드만 갱신한 새 dict를 반환한다.

    Args:
        session_state: 현재 세션 상태.
        result: graph.invoke()의 반환값.
        turn_number: 방금 실행한 턴 번호.

    Returns:
        다음 턴 입력으로 쓸 새 세션 상태 dict.
    """
    return {
        **session_state,
        "messages": result.get("messages", []),
        "turn_conclusions": result.get("turn_conclusions", []),
        "current_turn": turn_number,
        # Phase 3.7: session_intent_history는 Rot Gate와 독립. 누락 시 Turn 2+의 session_continuity가 항상 None.
        "session_intent_history": result.get("session_intent_history", []),
        # P2 보강 (2026-05-08): previous_turn_fidelity를 다음 턴에 전달.
        "previous_turn_fidelity": result.get("previous_turn_fidelity", {}),
        # 패턴 A 데이터(missing_info, confidence) 다음 턴 전달
        "previous_missing_info": result.get("previous_missing_info", ""),
        "previous_confidence": result.get("previous_confidence", 0.0),
    }


def run_turn(
    graph,
    session_state: dict,
    query: str,
    turn_number: int,
    config: dict | None = None,
) -> dict:
    """한 턴을 실행하고 다음 턴으로 전달할 세션 상태까지 만들어 반환한다.

    턴 상태 구성 → _execute_turn(그래프 invoke + Langfuse trace) → carry-over를 한 번에 처리한다.
    4D 평가/진단은 포함하지 않는다 (호출자가 응답을 먼저 보여준 뒤 evaluate_and_diagnose 호출).

    Args:
        graph: 컴파일된 LangGraph 그래프 (build_turn_graph() 결과).
        session_state: 이전 턴에서 전달된 세션 상태.
        query: 이번 턴의 사용자 질문.
        turn_number: 현재 턴 번호 (1부터).
        config: agent_config.yaml 설정 (현재 turn 실행엔 미사용, 시그니처 일관성용).

    Returns:
        {
          "response": str,        # 사용자에게 보여줄 응답 텍스트
          "trace_id": str,        # Langfuse trace ID (평가 부착용)
          "result": dict,         # graph.invoke() 원본 반환값 (평가/요약 입력)
          "wall_time_ms": int,    # 턴 실행 시간
          "session_state": dict,  # 다음 턴 입력으로 쓸 새 세션 상태 (불변)
        }
    """
    turn_state = _build_turn_state(session_state, query, turn_number)
    result, trace_id, wall_time_ms = _execute_turn(
        graph, turn_state, session_state["session_id"], turn_number,
    )
    new_session_state = _carry_over(session_state, result, turn_number)
    return {
        "response": result.get("response", "(응답 없음)"),
        "trace_id": trace_id,
        "result": result,
        "wall_time_ms": wall_time_ms,
        "session_state": new_session_state,
    }


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
    config: dict | None, trace_id: str, result: dict
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


def evaluate_and_diagnose(
    config: dict | None,
    trace_id: str,
    result: dict,
    previous_diagnosis: list | None = None,
) -> dict:
    """4D 평가 + 자동 진단 + Langfuse 기록을 한 번에 수행한다.

    st.*/UI에 의존하지 않으므로 백그라운드 스레드(채팅 앱)에서 호출해도 안전하다.
    응답을 먼저 사용자에게 보여준 뒤 호출하여 고객 체감 지연을 최소화한다.

    Args:
        config: agent_config.yaml 설정. auto_evaluate=false면 평가를 건너뛴다.
        trace_id: Langfuse trace ID.
        result: graph.invoke()의 반환값.
        previous_diagnosis: 직전 턴의 진단 결과 (개선 여부 판정용).

    Returns:
        {
          "scores": dict | None,   # 4D 점수 (평가 비활성/실패 시 None)
          "diagnosis": list,       # 진단 결과 (개선 제안)
          "improvement": bool,     # 직전 턴 대비 개선 여부
        }
    """
    scores, alignment_scores = _run_evaluation(config, trace_id, result)
    if not scores:
        return {"scores": None, "diagnosis": [], "improvement": False}

    diagnosis: list = []
    improvement = False
    try:
        trace_data = _build_trace_data(result)
        # BL-008: alignment 점수를 trace_data에 주입 → Pattern I/II/III 규칙 발화
        trace_data["metadata"].update(alignment_scores)
        diagnosis = diagnose_quality(scores, trace_data)
        # 직전 턴 진단과 비교하여 개선 여부 판정 (이전 차원이 사라졌으면 개선)
        if previous_diagnosis:
            prev_dims = {d["dimension"] for d in previous_diagnosis}
            curr_fail_dims = {d["dimension"] for d in diagnosis}
            improvement = bool(prev_dims - curr_fail_dims)
        # Langfuse trace에 진단 결과 기록 (v3: ingestion batch API)
        Langfuse().api.ingestion.batch(batch=[{
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": None,
            "body": {
                "id": trace_id,
                "metadata": {
                    ATTRS["eval.diagnosis"]: diagnosis,
                    ATTRS["eval.improvement_applied"]: improvement,
                },
            },
        }])
    except Exception as e:
        print(f"  ⚠ 진단 기록 실패: {e}")

    return {"scores": scores, "diagnosis": diagnosis, "improvement": improvement}


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
