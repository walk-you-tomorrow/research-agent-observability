"""
agent/nodes/gather_data.py — 노드 2: 도구 호출로 데이터/문서 수집

역할:
    analyze_query가 생성한 도구 계획(tool_plan)에 따라 9개 도구 중 필요한 것을 호출하여
    4개 소스(CSV, LightRAG, 웹 검색, 서울시 API)에서 데이터를 수집한다.
    수집된 항목은 관련성 필터링을 거친다.

프로세스 단계: ② Collect
품질 차원: 완전성(수집 항목 수), 관련성(제외 항목 수)

데이터 흐름:
    입력: state["query_analysis"] (도구 계획), state["context_evaluation"] (재수집 시 missing_info)
    출력: state["gathered_data"], state["gather_strategy"], state["gather_iteration"]

Langfuse 기록:
    gather.strategy: 수집 전략 설명
    gather.tools_called: 실제 호출된 도구 목록
    gather.items_collected: 수집 항목 수
    gather.items_excluded: 제외 항목 수
    gather.iteration: 수집 반복 횟수
    gather.api_called: 호출된 API 키 (api_query 사용 시)
    gather.api_response_count: API 응답 데이터 건수 (api_query 사용 시)
    web.search_count: 웹 검색 실행 횟수 (web_search 사용 시)
    web.result_count: 웹 검색 결과 수 (web_search 사용 시)
    web.source_domains: 검색 결과 도메인 목록 (web_search 사용 시)
    web.freshness: 검색 결과 시점 (web_search 사용 시)

9개 도구 매핑:
    rag_search, rag_deep_read, rag_global_summary, rag_compare → rag_tools.py (RAG 도구)
    pandas_query                                                → data_tools.py (데이터 도구)
    calculate, lookup_previous                                  → result_tools.py (결과 도구)
    api_query                                                   → api_tools.py (서울시 상권분석서비스 API)
    web_search                                                  → web_tools.py (Claude 웹 검색)
"""
import json
import re

from langfuse import get_client, observe

from agent.monitoring_schema import ATTRS
from agent.token_counter import count_tokens
from agent.tools.api_tools import api_query
from agent.tools.data_tools import pandas_query
from agent.tools.rag_tools import rag_compare, rag_deep_read, rag_global_summary, rag_search
from agent.tools.result_tools import calculate, lookup_previous
from agent.tools.web_tools import web_search

# --- 도구 이름 → 함수 매핑 ---
# analyze_query가 생성한 tool_plan의 문자열 이름을 실제 함수로 매핑한다.
# 예: tool_plan=["rag_search", "pandas_query"] → [rag_search(), pandas_query()] 호출
TOOL_MAP = {
    "rag_search": rag_search,                 # LightRAG mix 모드 시맨틱 검색
    "rag_deep_read": rag_deep_read,           # LightRAG local 모드 엔티티 상세
    "rag_global_summary": rag_global_summary, # LightRAG global 모드 패턴/트렌드
    "rag_compare": rag_compare,               # LightRAG hybrid 모드 비교
    "pandas_query": pandas_query,             # CSV pandas 쿼리 (정확한 수치)
    "calculate": calculate,                   # 수치 계산/검증
    "lookup_previous": lookup_previous,       # 이전 턴 결론 조회
    "api_query": api_query,                   # 서울시 상권분석서비스 API 호출
    "web_search": web_search,                 # Claude 웹 검색 (실시간 정보)
}


@observe(name="gather_data")
def gather_data(state: dict) -> dict:
    """도구를 호출하여 데이터/문서를 수집한다.

    Args:
        state: 현재 AgentState. query_analysis, gather_iteration, context_evaluation을 참조.

    Returns:
        {
            "gathered_data": list[dict],     # 수집된 항목 리스트
            "gather_strategy": str,          # 수집 전략 설명
            "gather_iteration": int,         # 현재 반복 횟수
        }

    동작 과정:
        1. 수집 전략 결정 (초기 수집 vs 재수집)
        2. tool_plan의 각 도구를 순차 호출
        3. 관련성 필터링 (irrelevant 결과는 제외)
        4. 각 수집 항목의 토큰 수 계산
        5. Langfuse에 수집 통계 기록
    """
    query_analysis = state.get("query_analysis", {})
    tool_plan = query_analysis.get("tool_plan", [])

    # 수집 반복 횟수 증가. 1이면 초기 수집, 2 이상이면 재수집이다.
    iteration = state.get("gather_iteration", 0) + 1

    # --- 수집 전략 결정 ---
    # 재수집 시: evaluate_context의 missing_info를 참고하여 어떤 데이터가 추가로 필요한지 기록.
    # 초기 수집 시: tool_plan 그대로 사용.
    if iteration > 1:
        missing = state.get("context_evaluation", {}).get("missing_info", "")
        strategy = f"재수집 #{iteration}: missing_info='{missing}'에 따라 추가 tool 호출"
    else:
        strategy = f"초기 수집: tool_plan={tool_plan}"

    gathered = []           # 수집 성공한 항목 리스트
    tools_called = []       # 실제 호출된 도구 이름 리스트
    excluded_items = []     # 관련성 부족으로 제외된 항목 리스트

    # --- 도구 순차 호출 ---
    for tool_name in tool_plan:
        # TOOL_MAP에 없는 도구 이름은 건너뛴다 (오타나 미구현 도구 방어)
        if tool_name not in TOOL_MAP:
            continue

        tool_fn = TOOL_MAP[tool_name]
        try:
            # 모든 도구는 동일한 인터페이스를 가진다:
            # tool_fn(query_analysis=dict, state=dict) → dict
            result = tool_fn(query_analysis=query_analysis, state=state)

            # --- 관련성 필터링 ---
            # 도구가 relevance="irrelevant"을 반환하면 해당 결과를 수집에서 제외한다.
            # 예: data_compare인데 비교 대상이 1개뿐인 경우
            if result.get("relevance") == "irrelevant":
                # source가 빈 문자열이면 tool_name으로 대체
                source = result.get("source", "") or tool_name
                excluded_items.append({
                    "source": source,
                    "reason": result.get("exclusion_reason", "intent와 무관"),
                })
                continue

            # 수집 성공: 항목을 gathered 리스트에 추가한다.
            # token_count는 Context Monitoring에서 효율성 측정에 사용된다.
            item = {
                "source": result.get("source", tool_name),       # 데이터 출처 (파일명 등)
                "tool_used": tool_name,                          # 사용된 도구 이름
                "data_summary": result.get("summary", ""),       # 데이터 요약 설명
                "token_count": count_tokens(json.dumps(result.get("data", ""))),  # 토큰 수
                "relevance_reason": result.get("relevance_reason", ""),  # 관련성 사유
            }

            # web_search 도구는 모니터링용 web_meta를 반환한다
            if tool_name == "web_search" and "web_meta" in result:
                item["web_meta"] = result["web_meta"]

            gathered.append(item)
            tools_called.append(tool_name)

        except Exception as e:
            # 도구 실행 에러: 에러 정보를 gathered에 포함하여 evaluate_context에서
            # "이 도구는 실패했다"는 정보를 참조할 수 있게 한다.
            gathered.append({
                "source": tool_name,
                "tool_used": tool_name,
                "data_summary": f"Error: {str(e)[:200]}",
                "token_count": 0,
                "relevance_reason": "tool_error",
            })

    # --- API 호출 통계 집계 ---
    # api_query 도구가 호출된 경우, 어떤 API가 호출되었고 응답 건수가 얼마인지 기록한다.
    api_called = ""
    api_response_count = 0
    for item in gathered:
        if item.get("tool_used") == "api_query":
            source = item.get("source", "")
            if ":" in source:
                api_called = source.split(":")[1].split()[0]  # "api_query:commercial_change" → "commercial_change"
            # data_summary에서 "N건 조회" 패턴으로 응답 건수 추출
            summary = item.get("data_summary", "")
            count_match = re.search(r"(\d+)건 조회", summary)
            if count_match:
                api_response_count = int(count_match.group(1))

    # --- 웹 검색 통계 집계 ---
    web_meta = {}
    for item in gathered:
        if item.get("tool_used") == "web_search":
            web_meta = item.get("web_meta", {})
            break

    # --- Langfuse 메타데이터 기록 (프로세스 단계 ② Collect) ---
    # 수집 통계를 Layer 2 attribute로 기록한다.
    # - 완전성: items_collected로 "충분히 수집했는가" 측정
    # - 관련성: items_excluded로 "불필요한 데이터를 얼마나 걸렀는가" 측정
    metadata = {
        # v3 폐기: gather.strategy (Judge 입력 X, gather.iteration으로 충분)
        ATTRS["gather.tools_called"]: tools_called,                      # 실제 호출 도구
        ATTRS["gather.items_collected"]: len(gathered),                   # 수집 항목 수
        ATTRS["gather.items_excluded"]: len(excluded_items),              # 제외 항목 수
        ATTRS["gather.excluded_items"]: [e["source"] for e in excluded_items],  # 제외된 소스
        ATTRS["gather.exclusion_reasons"]: excluded_items,                # G5: 전체 사유 포함
        ATTRS["gather.iteration"]: iteration,                            # 반복 횟수
    }

    # API 호출이 있는 경우에만 API 관련 속성을 추가한다
    if api_called:
        metadata[ATTRS["gather.api_called"]] = api_called
        # v3 폐기: gather.api_response_count → gather.items_collected에 흡수

    # 웹 검색이 있는 경우에만 웹 관련 속성을 추가한다
    if web_meta:
        # v3 폐기: web.search_count (gather.iteration이 더 정확) / web.result_count (items_collected에 흡수)
        metadata[ATTRS["web.source_domains"]] = web_meta.get("source_domains", [])
        metadata[ATTRS["web.freshness"]] = web_meta.get("freshness", [])

    get_client().update_current_span(metadata=metadata)

    return {
        "gathered_data": gathered,
        "gather_strategy": strategy,
        "gather_iteration": iteration,
        "tools_called": tools_called,                                      # main.py _build_trace_data용
        "excluded_items_count": len(excluded_items),                        # judge 평가용
        "excluded_items_sources": [e["source"] for e in excluded_items],    # judge 평가용
        "exclusion_reasons": excluded_items,                                # G5: 전체 사유 포함 (judge용)
    }
