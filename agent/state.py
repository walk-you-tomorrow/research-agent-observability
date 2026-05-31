"""
agent/state.py — 에이전트 공유 상태 타입 정의 (AgentState)

이 모듈은 LangGraph 그래프의 모든 노드가 공유하는 상태 스키마를 정의한다.
AgentState는 TypedDict로, 각 노드가 읽고 쓰는 필드를 명시한다.

데이터 흐름:
    main.py에서 초기 상태를 생성하고 → 각 노드가 자신의 출력 필드를 업데이트 →
    다음 노드가 이전 노드의 출력을 읽어 사용한다.

필드 표기 규칙:
    [생성자 → 소비자] 형식으로, 어떤 노드/모듈이 값을 생성하고
    어떤 노드가 그 값을 소비하는지 나타낸다.

참고:
    messages 필드의 Annotated[list, add_messages]는 LangGraph의 리듀서 패턴으로,
    노드가 반환하는 메시지가 기존 리스트에 append된다 (덮어쓰기가 아님).
"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """모든 노드가 공유하는 상태. [생성자 → 소비자] 표기.

    이 TypedDict는 LangGraph StateGraph의 상태 스키마로 사용된다.
    각 노드는 이 상태에서 필요한 값을 읽고, 자신의 출력 필드를 업데이트한다.
    """

    # ── 세션 레벨 (main.py에서 초기화, 턴 간 유지) ──

    # LangGraph 메시지 히스토리. add_messages 리듀서로 새 메시지가 자동 append된다.
    messages: Annotated[list, add_messages]

    # Langfuse 세션 ID. 한 세션의 모든 턴을 그룹화하는 데 사용된다.
    session_id: str

    # 현재 턴 번호 (1부터 시작). 각 턴이 끝날 때 main.py에서 업데이트된다.
    current_turn: int

    # ── 현재 턴 입력 (main.py에서 설정) ──

    # 이번 턴의 사용자 질문 원문. analyze_query 노드의 입력이 된다.
    user_query: str  # [main.py → analyze_query]

    # ── analyze_query 노드의 출력 ──

    # 질의 분석 결과. 사용자 의도(intent), 필요 데이터, 도구 호출 계획 등을 포함한다.
    # 구조: {intent, required_data, required_docs, tool_plan, references_previous, referenced_turns}
    query_analysis: dict  # [analyze_query → gather_data, evaluate_context]

    # ── gather_data 노드의 출력 ──

    # 수집된 데이터 항목 리스트. 각 항목은 출처, 도구명, 요약, 토큰 수, 관련성 사유를 포함한다.
    # 구조: [{source, tool_used, data_summary, token_count, relevance_reason}]
    gathered_data: list[dict]  # [gather_data → evaluate_context, generate_analysis]

    # 수집 전략 설명. 초기 수집인지, 재수집인지, 어떤 도구를 호출했는지 기록한다.
    gather_strategy: str  # [gather_data → evaluate_context]

    # 수집 반복 횟수. 1이면 초기 수집, 2 이상이면 재수집을 의미한다.
    gather_iteration: int  # [gather_data → evaluate_context]

    # 실제 호출된 도구 목록. _build_trace_data에서 judge 평가 입력으로 사용된다.
    tools_called: list[str]  # [gather_data → main.py _build_trace_data]

    # 관련성 필터로 제외된 항목 수. judge 평가 입력으로 사용된다.
    excluded_items_count: int  # [gather_data → main.py _build_trace_data]

    # 제외된 항목의 소스명 목록. judge 평가 입력으로 사용된다.
    excluded_items_sources: list[str]  # [gather_data → main.py _build_trace_data]

    # ── evaluate_context 노드의 출력 (★ Context Monitoring 핵심) ──

    # 컨텍스트 메타데이터. 토큰 수, 소스별 분해, 윈도우 사용률 등 관측 전용 데이터.
    # 구조: {total_tokens, source_breakdown, truncated_items, ...}
    context_metadata: dict  # [evaluate_context → Langfuse 메타데이터 기록 전용]

    # 컨텍스트 충분성 평가 결과. 분기 ①(should_continue_gather)에서 사용된다.
    # 구조: {is_sufficient, missing_info, confidence_score}
    context_evaluation: dict  # [evaluate_context → should_continue_gather 분기]

    # ── 일관성 패턴 A (같은 턴 내 iteration 간 비교) ──

    # 이전 iteration의 부족한 정보 항목 목록. 재수집 후 해결 여부(missing_info_resolved)를 판단하는 데 사용된다.
    previous_missing_info: list[str]  # [evaluate_context → 다음 evaluate_context iteration]

    # 이전 iteration의 신뢰도 점수. 재수집 후 변화량(confidence_delta)을 계산하는 데 사용된다.
    previous_confidence: float  # [evaluate_context → 다음 evaluate_context iteration]

    # ── 일관성 패턴 B (턴 간 비교) ──

    # 모든 턴의 결론 요약 리스트. 턴이 끝날 때마다 respond_to_user가 append한다.
    # 다음 턴의 analyze_query, generate_analysis에서 이전 결론을 참조하는 데 사용된다.
    # 구조: [{turn_number, conclusion_summary, key_claims, data_sources_used}]
    turn_conclusions: list[dict]  # [respond_to_user → 다음 턴 전체]

    # 턴별 사용자 intent 히스토리. analyze_query가 각 턴 시작 시 append한다.
    # Rot Gate와 완전히 독립 — turn_conclusions와 달리 절대 prune되지 않는다.
    # query.session_continuity 계산(Phase 3.7 이탈 감지)에 사용된다.
    # 구조: [{turn_number: int, intent: str}]
    session_intent_history: list[dict]  # [analyze_query → analyze_query (session_continuity 계산)]

    # 이번 턴의 session continuity 점수. analyze_query가 계산한다.
    # Turn 1 또는 Ollama 미가용 시 None. _build_trace_data()가 diagnosis.py에 전달.
    session_continuity: float | None  # [analyze_query → main._build_trace_data (diagnosis 주입)]

    # analyze_query가 감지한, 이번 질문이 참조하는 이전 턴 번호 리스트.
    referenced_turns: list[int]  # [analyze_query → generate_analysis]

    # v3 통합 (2026-04-29): 일관성 패턴 B의 모순 추적을 단일 dict로 통합.
    # 구조:
    #   detected: bool                            — 모순 감지 (구 contradicts_previous)
    #   resolution.has_explanation: bool          — 해결 설명 (구 contradiction_resolved)
    #   resolution.conflict_summary: str          — 이전 결론 요약 (구 previous_conclusion)
    #   resolution.source_resolution: str         — 소스 충돌 해결 (구 source_conflict_resolution)
    conflict_tracking: dict  # [generate_analysis → respond_to_user, main.py trace]

    # ── 일관성 패턴 C (Memory → Organize 충실도) ── (G1)

    # respond_to_user가 기록한 G2 metrics를 다음 턴의 evaluate_context에 전달한다.
    # 구조: {compression_ratio, conditions_preserved, key_claims_preserved, total_claims}
    previous_turn_fidelity: dict  # [respond_to_user → 다음 턴 evaluate_context]

    # ── 교차 턴 진화 (Cross-Turn Evolution) ── (G3)

    # 이전 턴의 총 토큰 수. token_delta 계산에 사용된다.
    previous_total_tokens: int  # [evaluate_context → 다음 턴 evaluate_context]

    # A1: 이전 턴의 rot_risk. rot_velocity(턴 간 rot_risk 변화율) 계산에 사용된다.
    previous_rot_risk: float  # [evaluate_context → 다음 턴 evaluate_context]

    # ── generate_analysis 노드의 출력 ──

    # 데이터 기반 분석 결과. 요약, 주장(claims), 데이터 참조, 주의사항을 포함한다.
    # 구조: {summary, claims, data_references, caveats}
    analysis_result: dict  # [generate_analysis → verify_result]

    # ── verify_result 노드의 출력 ──

    # 검증 결과. 수치 검증, 해석 검증, 최종 판정(pass/fail_numeric/fail_interpretation)을 포함한다.
    # 구조: {numeric_check, interpretation_check, overall_verdict, issues}
    verification: dict  # [verify_result → route_after_verify 분기]

    # ── 반복 제어 카운터 ──

    # 데이터 재수집 시도 횟수. MAX_GATHER_RETRIES(3)에 도달하면 강제 진행한다.
    gather_retry_count: int  # [graph.py 분기 로직에서 사용]

    # 검증 재시도 횟수. MAX_VERIFY_RETRIES(2)에 도달하면 응답으로 진행한다.
    verify_retry_count: int  # [graph.py 분기 로직에서 사용]

    # ── 최종 응답 ──

    # respond_to_user가 생성한 사용자 대상 최종 응답 텍스트.
    response: str  # [respond_to_user → main.py 출력]
