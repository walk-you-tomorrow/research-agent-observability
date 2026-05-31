"""
agent/graph.py — LangGraph 그래프 구성 및 분기 로직

이 모듈은 6노드 LangGraph 파이프라인을 구성한다:
    analyze_query → gather_data → evaluate_context → generate_analysis → verify_result → respond_to_user

핵심 기능:
    1. safe_node() 래퍼: 노드 에러 시 그래프가 중단되지 않도록 보호
    2. should_continue_gather(): 컨텍스트 충분성 기반 분기 (분기 ①)
    3. route_after_verify(): 검증 결과 기반 분기 (분기 ②)
    4. build_turn_graph(): 위 요소들을 조합하여 한 턴의 그래프를 빌드

그래프 흐름:
    START → analyze_query → gather_data → evaluate_context ─┐
                   ↑                                         │
                   │  (부족)                                  │ (충분)
                   └─────────────────────────────────────────┘
                                                              ↓
           ┌── gather_data ← (수치 오류) ── verify_result ← generate_analysis
           │                                     │
           │   generate_analysis ← (해석 오류) ──┘
           │                                     │ (통과)
           └─────────────────────────────────────→ respond_to_user → END
"""
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.analyze_query import analyze_query
from agent.nodes.gather_data import gather_data
from agent.nodes.evaluate_context import evaluate_context
from agent.nodes.generate_analysis import generate_analysis
from agent.nodes.verify_result import verify_result
from agent.nodes.respond_to_user import respond_to_user

# --- 반복 제한 상수 ---
# 데이터 수집 최대 재시도 횟수. 이 횟수를 초과하면 부족한 상태로 강제 진행한다.
MAX_GATHER_RETRIES = 3

# 검증 실패 시 최대 재시도 횟수. 이 횟수를 초과하면 현재 결과로 응답한다.
MAX_VERIFY_RETRIES = 2

# 충분성 판단 임계값. confidence_score가 이 값 이상이면 "충분"으로 판정한다.
SUFFICIENCY_THRESHOLD = 0.7


def safe_node(fn):
    """노드 에러 시 그래프가 중단되지 않게 감싸는 래퍼 (데코레이터).

    Args:
        fn: 원래 노드 함수. state: dict를 받아 dict를 반환해야 한다.

    Returns:
        래핑된 노드 함수. 에러 발생 시 에러 정보를 담은 기본 상태를 반환한다.

    동작:
        - 정상: fn(state) 그대로 반환
        - 예외 발생: verification.overall_verdict="error"와 에러 메시지를 포함한 dict 반환
        - 그래프는 중단되지 않고 다음 노드로 계속 진행한다

    사용 이유:
        LangGraph에서 한 노드가 예외를 던지면 전체 그래프가 중단된다.
        safe_node로 감싸면 에러가 발생해도 나머지 파이프라인이 계속 실행되어
        부분적인 결과라도 사용자에게 전달할 수 있다.
    """

    def wrapper(state: dict) -> dict:
        print(f"  ▶ [{fn.__name__}] 시작")
        try:
            result = fn(state)
            print(f"  ✔ [{fn.__name__}] 완료")
            return result
        except Exception as e:
            # 에러 발생 시: verification에 에러 정보를 담아 반환한다.
            # analysis_result는 기존 값이 있으면 유지하고, 없으면 에러 메시지로 대체한다.
            print(f"  ✘ [{fn.__name__}] 에러: {type(e).__name__}: {str(e)[:200]}")

            error_state = {
                "verification": {
                    "overall_verdict": "error",
                    "issues": [f"{type(e).__name__}: {str(e)[:500]}"],
                },
                "analysis_result": state.get(
                    "analysis_result",
                    {
                        "summary": f"처리 중 오류: {str(e)[:200]}",
                        "claims": [],
                        "data_references": [],
                        "caveats": [],
                    },
                ),
            }

            # respond_to_user 실패 시: 폴백 turn_conclusion을 저장한다.
            # 이렇게 하지 않으면 Rate Limit 등으로 respond_to_user가 실패할 때
            # turn_conclusions에 이번 턴이 누락되어 후속 검증(assert len==7)이 깨진다.
            if fn.__name__ == "respond_to_user":
                turn_number = state.get("current_turn", 0)
                analysis = error_state["analysis_result"]
                fallback_conclusion = {
                    "turn_number": turn_number,
                    "conclusion_summary": analysis.get("summary", f"턴 {turn_number} 처리 중 오류 발생"),
                    "key_claims": [],
                    "data_sources_used": [],
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                }
                existing = list(state.get("turn_conclusions", []))
                error_state["turn_conclusions"] = existing + [fallback_conclusion]
                error_state["response"] = ""
                error_state["current_turn"] = turn_number
                print(f"    ↳ 폴백 turn_conclusion 저장 (turn {turn_number})")

            return error_state

    # 래퍼 함수의 __name__을 원래 함수와 동일하게 설정한다.
    # LangGraph가 노드 이름을 함수명에서 가져오기 때문에 필요하다.
    wrapper.__name__ = fn.__name__
    return wrapper


def should_continue_gather(state: dict) -> str:
    """분기 ① — evaluate_context 이후: 컨텍스트가 충분한가?

    이 함수는 LangGraph의 conditional_edges에서 호출되어
    다음 노드를 결정한다.

    판단 로직:
        1. is_sufficient=true AND confidence >= 0.7 → generate_analysis (분석 진행)
        2. 위 조건 불만족 AND iteration < 3 → gather_data (재수집)
        3. 재시도 소진 (iteration >= 3) → generate_analysis (부족한 상태로 강제 진행)

    Args:
        state: 현재 AgentState. context_evaluation과 gather_iteration을 참조한다.

    Returns:
        다음 노드 이름 문자열: "generate_analysis" 또는 "gather_data"
    """
    eval_result = state.get("context_evaluation", {})
    iteration = state.get("gather_iteration", 1)
    is_sufficient = eval_result.get("is_sufficient", False)
    confidence = eval_result.get("confidence_score", 0.0)

    if is_sufficient and confidence >= SUFFICIENCY_THRESHOLD:
        # 컨텍스트가 충분하다 → 분석 생성 단계로 진행
        print(f"  ↳ 분기①: 충분 (confidence={confidence:.2f}) → generate_analysis")
        return "generate_analysis"
    elif iteration < MAX_GATHER_RETRIES:
        # 컨텍스트가 부족하고 재시도 여유가 있다 → 데이터 재수집
        print(f"  ↳ 분기①: 부족 (confidence={confidence:.2f}, iteration={iteration}/{MAX_GATHER_RETRIES}) → gather_data 재수집")
        return "gather_data"
    else:
        # 재시도 횟수 소진 → 부족하지만 강제로 분석 진행 (무한 루프 방지)
        print(f"  ↳ 분기①: 재시도 소진 (iteration={iteration}) → generate_analysis 강제 진행")
        return "generate_analysis"


def route_after_verify(state: dict) -> str:
    """분기 ② — verify_result 이후: 검증을 통과했는가?

    이 함수는 검증 결과(verdict)에 따라 다음 동작을 결정한다.

    판단 로직:
        1. verdict="pass" 또는 재시도 소진 → respond_to_user (응답 생성)
        2. verdict="fail_numeric" → gather_data (수치 오류 → 데이터 재수집부터)
        3. verdict="fail_interpretation" → generate_analysis (해석 오류 → 분석 재생성)
        4. 기타 (error 등) → respond_to_user (에러 메시지라도 응답)

    Args:
        state: 현재 AgentState. verification과 verify_retry_count를 참조한다.

    Returns:
        다음 노드 이름 문자열: "respond_to_user", "gather_data", 또는 "generate_analysis"
    """
    verification = state.get("verification", {})
    verdict = verification.get("overall_verdict", "pass")
    retries = state.get("verify_retry_count", 0)

    if verdict == "pass" or retries >= MAX_VERIFY_RETRIES:
        # 검증 통과 또는 재시도 소진 → 응답 생성으로 진행
        print(f"  ↳ 분기②: verdict={verdict}, retries={retries} → respond_to_user")
        return "respond_to_user"
    elif verdict == "fail_numeric":
        # 수치 검증 실패 → 데이터 재수집부터 다시 시작
        print(f"  ↳ 분기②: 수치 오류 (retries={retries}/{MAX_VERIFY_RETRIES}) → gather_data")
        return "gather_data"
    elif verdict == "fail_interpretation":
        # 해석 검증 실패 → 분석 재생성 (데이터는 유지)
        print(f"  ↳ 분기②: 해석 오류 (retries={retries}/{MAX_VERIFY_RETRIES}) → generate_analysis")
        return "generate_analysis"
    else:
        # 기타 (error 등) → 에러 상태라도 응답을 생성
        print(f"  ↳ 분기②: {verdict} → respond_to_user")
        return "respond_to_user"


def build_turn_graph():
    """한 턴을 처리하는 LangGraph 그래프를 빌드하고 컴파일한다.

    Returns:
        컴파일된 StateGraph. main.py에서 graph.invoke(state)로 실행된다.

    그래프 구조:
        6개 노드를 safe_node()로 감싸서 등록하고,
        2개의 조건부 분기(conditional_edges)로 연결한다.

        선형 경로: analyze_query → gather_data → evaluate_context
        분기 ①: evaluate_context → (충분?) → generate_analysis / gather_data(재수집)
        선형 경로: generate_analysis → verify_result
        분기 ②: verify_result → (통과?) → respond_to_user / gather_data / generate_analysis
        종료: respond_to_user → END
    """
    # AgentState를 상태 스키마로 사용하는 StateGraph 생성
    g = StateGraph(AgentState)

    # --- 6개 노드 등록 ---
    # 각 노드는 safe_node()로 감싸져 있어 에러 시에도 그래프가 중단되지 않는다.
    g.add_node("analyze_query", safe_node(analyze_query))       # 노드 1: 질의 분석
    g.add_node("gather_data", safe_node(gather_data))           # 노드 2: 데이터 수집
    g.add_node("evaluate_context", safe_node(evaluate_context)) # 노드 3: 컨텍스트 평가
    g.add_node("generate_analysis", safe_node(generate_analysis)) # 노드 4: 분석 생성
    g.add_node("verify_result", safe_node(verify_result))       # 노드 5: 결과 검증
    g.add_node("respond_to_user", safe_node(respond_to_user))   # 노드 6: 응답 생성

    # --- 엣지 연결 ---

    # 시작점: 그래프 실행 시 analyze_query부터 시작
    g.set_entry_point("analyze_query")

    # 순방향 엣지: 질의 분석 → 데이터 수집 → 컨텍스트 평가
    g.add_edge("analyze_query", "gather_data")
    g.add_edge("gather_data", "evaluate_context")

    # 분기 ①: evaluate_context 이후 — 컨텍스트가 충분한가?
    # should_continue_gather 함수가 반환하는 문자열에 따라 다음 노드가 결정된다.
    g.add_conditional_edges(
        "evaluate_context",
        should_continue_gather,
        {
            "generate_analysis": "generate_analysis",  # 충분 → 분석 진행
            "gather_data": "gather_data",              # 부족 → 재수집 (루프)
        },
    )

    # 순방향 엣지: 분석 생성 → 결과 검증
    g.add_edge("generate_analysis", "verify_result")

    # 분기 ②: verify_result 이후 — 검증을 통과했는가?
    # route_after_verify 함수가 반환하는 문자열에 따라 다음 노드가 결정된다.
    g.add_conditional_edges(
        "verify_result",
        route_after_verify,
        {
            "respond_to_user": "respond_to_user",      # 통과 → 응답
            "gather_data": "gather_data",              # 수치 오류 → 재수집
            "generate_analysis": "generate_analysis",  # 해석 오류 → 분석 재생성
        },
    )

    # 종료: 응답 생성 후 그래프 종료
    g.add_edge("respond_to_user", END)

    # 그래프를 컴파일하여 실행 가능한 형태로 반환
    return g.compile()
