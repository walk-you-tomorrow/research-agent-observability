"""
agent/tools/rag_tools.py — RAG 도구: rag_search, rag_deep_read, rag_global_summary, rag_compare

LightRAG Knowledge Graph를 검색하는 4개 도구를 제공한다.
모든 도구는 동일한 인터페이스를 따른다:
    tool_fn(query_analysis: dict, state: dict) → dict

반환값 구조:
    {
        "source": str,              # 데이터 출처 ("lightrag:{mode}")
        "summary": str,             # 검색 결과 요약
        "data": any,                # 원본 검색 결과
        "relevance": str,           # "relevant" 또는 "irrelevant"
        "relevance_reason": str,    # 관련성 판단 사유
    }

4개 도구와 LightRAG 검색 모드 매핑:
    - rag_search: mix 모드 (시맨틱 + KG 통합 검색)
    - rag_deep_read: local 모드 (엔티티 중심 상세 검색)
    - rag_global_summary: global 모드 (전역 패턴/트렌드 요약)
    - rag_compare: hybrid 모드 (비교 분석)
"""

from agent.lightrag_adapter import query_knowledge

# lightrag_adapter는 relevance를 float (0.0~1.0)로 반환하지만,
# gather_data.py는 "irrelevant" 문자열로 필터링한다.
# 이 임계값 이하이면 "irrelevant"로 변환한다.
RELEVANCE_THRESHOLD = 0.3


def _convert_relevance(score: float | int) -> str:
    """float 관련성 점수를 문자열로 변환한다.

    gather_data.py의 `result.get("relevance") == "irrelevant"` 체크와 호환.

    Args:
        score: 관련성 점수 (0.0~1.0).

    Returns:
        "relevant" 또는 "irrelevant".
    """
    if isinstance(score, str):
        return score  # 이미 문자열이면 그대로 반환
    return "irrelevant" if score <= RELEVANCE_THRESHOLD else "relevant"


def _build_query(query_analysis: dict, extra_context: str = "") -> str:
    """query_analysis에서 LightRAG 쿼리 문자열을 조합한다.

    intent + keywords를 결합하여 검색 쿼리를 생성한다.

    Args:
        query_analysis: 질의 분석 결과. intent, keywords 키를 참조.
        extra_context: 추가 컨텍스트 문자열 (도구별 특화 정보).

    Returns:
        LightRAG에 전달할 쿼리 문자열.
    """
    intent = query_analysis.get("intent", "")
    keywords = query_analysis.get("keywords", [])

    query = intent
    if keywords:
        query += " " + " ".join(keywords)
    if extra_context:
        query += " " + extra_context
    return query.strip()


def rag_search(query_analysis: dict, state: dict) -> dict:
    """LightRAG mix 모드로 시맨틱 검색을 수행한다.

    시맨틱 벡터 검색과 Knowledge Graph 관계 탐색을 결합한 통합 검색.
    가장 범용적인 검색 도구로, 일반적인 정보 탐색에 사용된다.

    Args:
        query_analysis: 질의 분석 결과. intent, keywords를 쿼리 조합에 사용.
        state: 현재 AgentState (이 도구에서는 사용하지 않음).

    Returns:
        표준 도구 반환 형식. relevance는 문자열 ("relevant"/"irrelevant").
    """
    query = _build_query(query_analysis)
    raw = query_knowledge(query, mode="mix", top_k=10)

    return {
        "source": raw["source"],
        "summary": raw["summary"],
        "data": raw["data"],
        "relevance": _convert_relevance(raw["relevance"]),
        "relevance_reason": raw["relevance_reason"],
    }


def rag_deep_read(query_analysis: dict, state: dict) -> dict:
    """LightRAG local 모드로 엔티티 중심 상세 검색을 수행한다.

    특정 동(洞), 구(區), 업종 등 개별 엔티티에 대한 상세 정보를 검색한다.
    required_docs에 명시된 문서 키워드를 쿼리에 추가하여 검색 정밀도를 높인다.

    Args:
        query_analysis: 질의 분석 결과. required_docs를 extra_context로 활용.
        state: 현재 AgentState (이 도구에서는 사용하지 않음).

    Returns:
        표준 도구 반환 형식.
    """
    # required_docs의 키워드를 추가 컨텍스트로 활용
    docs = query_analysis.get("required_docs", [])
    extra = " ".join(docs) if docs else ""

    query = _build_query(query_analysis, extra_context=extra)
    raw = query_knowledge(query, mode="local", top_k=10)

    return {
        "source": raw["source"],
        "summary": raw["summary"],
        "data": raw["data"],
        "relevance": _convert_relevance(raw["relevance"]),
        "relevance_reason": raw["relevance_reason"],
    }


def rag_global_summary(query_analysis: dict, state: dict) -> dict:
    """LightRAG global 모드로 전역 패턴과 트렌드를 요약한다.

    Knowledge Graph 전체에서 테마별 패턴, 트렌드, 공통점을 추출한다.
    교차 분석이나 컨텍스트 요약이 필요할 때 사용된다.

    Args:
        query_analysis: 질의 분석 결과.
        state: 현재 AgentState (이 도구에서는 사용하지 않음).

    Returns:
        표준 도구 반환 형식.
    """
    query = _build_query(query_analysis, extra_context="전반적 트렌드와 패턴")
    raw = query_knowledge(query, mode="global", top_k=10)

    return {
        "source": raw["source"],
        "summary": raw["summary"],
        "data": raw["data"],
        "relevance": _convert_relevance(raw["relevance"]),
        "relevance_reason": raw["relevance_reason"],
    }


def rag_compare(query_analysis: dict, state: dict) -> dict:
    """LightRAG hybrid 모드로 비교 분석을 수행한다.

    두 개 이상의 엔티티(구 vs 구, 동 vs 동, 업종 vs 업종)를 비교할 때 사용된다.
    벡터 유사성과 KG 관계를 결합하여 비교 관점의 정보를 추출한다.

    Args:
        query_analysis: 질의 분석 결과.
        state: 현재 AgentState (이 도구에서는 사용하지 않음).

    Returns:
        표준 도구 반환 형식.
    """
    query = _build_query(query_analysis, extra_context="비교 분석")
    raw = query_knowledge(query, mode="hybrid", top_k=10)

    return {
        "source": raw["source"],
        "summary": raw["summary"],
        "data": raw["data"],
        "relevance": _convert_relevance(raw["relevance"]),
        "relevance_reason": raw["relevance_reason"],
    }
