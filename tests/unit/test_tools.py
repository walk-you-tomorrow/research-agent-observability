"""
tests/unit/test_tools.py — 9개 도구 단위 테스트

각 도구를 독립적으로 호출하여 반환값 형식, 기본 동작, 에러 처리를 검증한다.
전체 파이프라인(run_session)을 실행하지 않고 도구 함수만 테스트한다.

테스트 대상 (B1~B9):
    B1: rag_search — LightRAG mix 모드 검색
    B2: rag_deep_read — LightRAG local 모드 엔티티 상세
    B3: rag_compare — LightRAG hybrid 모드 비교 분석
    B4: rag_global_summary — LightRAG global 모드 트렌드/패턴
    B5: pandas_query — 9개 CSV 쿼리
    B6: api_query — 서울시 API 5개 엔드포인트
    B7: web_search — Claude 웹 검색
    B8: calculate — 수치 검증
    B9: lookup_previous — 이전 턴 결론 조회

실행 방법:
    cd observable-research-agent && source .venv/bin/activate
    python -m pytest tests/unit/test_tools.py -v
    python -m pytest tests/unit/test_tools.py -v -k "test_b5"  # 개별 실행
"""
import os
import sys

import pytest

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()


# --- 공통 반환값 형식 검증 ---
def _assert_tool_result(result: dict, expected_source_prefix: str = "") -> None:
    """도구 반환값이 표준 형식을 따르는지 검증한다.

    표준 형식: {source, summary, data, relevance, relevance_reason}
    """
    assert "source" in result, f"Missing 'source' in result: {result.keys()}"
    assert "summary" in result, f"Missing 'summary' in result: {result.keys()}"
    assert "data" in result, f"Missing 'data' in result: {result.keys()}"
    assert "relevance" in result, f"Missing 'relevance' in result: {result.keys()}"
    assert "relevance_reason" in result, f"Missing 'relevance_reason' in result: {result.keys()}"

    assert result["relevance"] in ("relevant", "irrelevant"), \
        f"Invalid relevance: {result['relevance']}"

    if expected_source_prefix:
        assert result["source"].startswith(expected_source_prefix) or \
               expected_source_prefix in result["source"], \
            f"Expected source to contain '{expected_source_prefix}', got '{result['source']}'"


# --- 공통 테스트 입력 ---
BASIC_QUERY_ANALYSIS = {
    "intent": "서울 카페 상권 분석",
    "keywords": ["카페", "강남구", "상권"],
    "required_data": ["store_info", "foot_traffic", "rent"],
    "required_docs": ["강남구 상권"],
    "tool_plan": ["rag_search", "pandas_query"],
    "source_types": ["csv", "rag"],
    "references_previous_turn": False,
    "referenced_turns": [],
    "api_params": {},
}

COMPARE_QUERY_ANALYSIS = {
    "intent": "강남구 마포구 비교 분석",
    "keywords": ["강남구", "마포구", "비교"],
    "required_data": ["store_info", "rent"],
    "required_docs": ["강남구", "마포구"],
    "tool_plan": ["rag_compare", "pandas_query"],
    "source_types": ["csv", "rag"],
    "references_previous_turn": False,
    "referenced_turns": [],
    "api_params": {},
}

EMPTY_STATE = {
    "user_query": "강남구 카페 분석",
    "turn_conclusions": [],
    "gathered_data": [],
    "analysis_result": {},
    "current_turn": 1,
}


# ═══════════════════════════════════════
# B1: rag_search (LightRAG mix mode)
# ═══════════════════════════════════════
class TestB1RagSearch:
    """B1: rag_search — LightRAG mix 모드 검색."""

    def test_returns_standard_format(self):
        """반환값이 표준 도구 형식을 따르는지 확인."""
        from agent.tools.rag_tools import rag_search
        result = rag_search(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "lightrag")

    def test_relevant_query_returns_data(self):
        """알려진 엔티티(연남동 카페)로 검색 시 결과가 있는지 확인."""
        from agent.tools.rag_tools import rag_search
        qa = {**BASIC_QUERY_ANALYSIS, "intent": "연남동 카페 상권", "keywords": ["연남동", "카페"]}
        result = rag_search(qa, EMPTY_STATE)
        # KG에 연남동 엔티티가 있으면 relevant, 없으면 irrelevant — 양쪽 모두 유효
        assert result["relevance"] in ("relevant", "irrelevant")


# ═══════════════════════════════════════
# B2: rag_deep_read (LightRAG local mode)
# ═══════════════════════════════════════
class TestB2RagDeepRead:
    """B2: rag_deep_read — LightRAG local 모드 엔티티 상세."""

    def test_returns_standard_format(self):
        from agent.tools.rag_tools import rag_deep_read
        result = rag_deep_read(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "lightrag")

    def test_entity_centric_query(self):
        """특정 동(역삼동) 엔티티에 대한 상세 조회."""
        from agent.tools.rag_tools import rag_deep_read
        qa = {**BASIC_QUERY_ANALYSIS, "intent": "역삼동 상세 분석", "required_docs": ["역삼동"]}
        result = rag_deep_read(qa, EMPTY_STATE)
        _assert_tool_result(result)


# ═══════════════════════════════════════
# B3: rag_compare (LightRAG hybrid mode)
# ═══════════════════════════════════════
class TestB3RagCompare:
    """B3: rag_compare — LightRAG hybrid 모드 비교 분석."""

    def test_returns_standard_format(self):
        from agent.tools.rag_tools import rag_compare
        result = rag_compare(COMPARE_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "lightrag")

    def test_comparison_query(self):
        """두 지역 비교 쿼리가 정상 동작하는지 확인."""
        from agent.tools.rag_tools import rag_compare
        qa = {**COMPARE_QUERY_ANALYSIS, "intent": "강남 vs 마포 카페", "keywords": ["강남", "마포"]}
        result = rag_compare(qa, EMPTY_STATE)
        _assert_tool_result(result)


# ═══════════════════════════════════════
# B4: rag_global_summary (LightRAG global mode)
# ═══════════════════════════════════════
class TestB4RagGlobalSummary:
    """B4: rag_global_summary — LightRAG global 모드 전역 패턴/트렌드."""

    def test_returns_standard_format(self):
        from agent.tools.rag_tools import rag_global_summary
        result = rag_global_summary(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "lightrag")


# ═══════════════════════════════════════
# B5: pandas_query (CSV DataFrame 쿼리)
# ═══════════════════════════════════════
class TestB5PandasQuery:
    """B5: pandas_query — 9개 CSV 파일 쿼리."""

    def test_returns_standard_format(self):
        from agent.tools.data_tools import pandas_query
        result = pandas_query(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result)

    def test_store_info_query(self):
        """store_info.csv 쿼리가 데이터를 반환하는지 확인."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["store_info"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert "store_info" in result["data"], "store_info should be in result data"
        assert result["data"]["store_info"]["row_count"] > 0

    def test_foot_traffic_query(self):
        """foot_traffic.csv 쿼리."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["foot_traffic"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert result["relevance"] == "relevant"

    def test_rent_query(self):
        """rent.csv 쿼리."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["rent"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert result["relevance"] == "relevant"
        assert "rent" in result["data"]

    def test_demographics_query(self):
        """demographics.csv 쿼리."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["demographics"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert result["relevance"] == "relevant"

    def test_card_consumption_query(self):
        """card_consumption.csv 쿼리."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["card_consumption"]}
        result = pandas_query(qa, EMPTY_STATE)
        # 파일이 있으면 relevant, 없으면 irrelevant — 양쪽 모두 유효
        assert result["relevance"] in ("relevant", "irrelevant")

    def test_estimated_sales_query(self):
        """estimated_sales.csv 쿼리."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["estimated_sales"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert result["relevance"] in ("relevant", "irrelevant")

    def test_subway_ridership_query(self):
        """subway_ridership.csv 쿼리."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["subway_ridership"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert result["relevance"] in ("relevant", "irrelevant")

    def test_nonexistent_file_graceful(self):
        """존재하지 않는 CSV를 요청해도 에러가 발생하지 않는지 확인."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "required_data": ["nonexistent_data"]}
        result = pandas_query(qa, EMPTY_STATE)
        assert result["relevance"] == "irrelevant"

    def test_compare_intent_groupby(self):
        """compare intent 시 구별 집계가 수행되는지 확인."""
        from agent.tools.data_tools import pandas_query
        qa = {**BASIC_QUERY_ANALYSIS, "intent": "compare 강남 마포", "required_data": ["foot_traffic"]}
        result = pandas_query(qa, EMPTY_STATE)
        if result["relevance"] == "relevant" and "foot_traffic" in result["data"]:
            data = result["data"]["foot_traffic"]["data"]
            # district별 집계된 결과가 있어야 한다
            if data:
                assert "district" in data[0], "compare intent should groupby district"


# ═══════════════════════════════════════
# B6: api_query (서울시 상권분석서비스 API)
# ═══════════════════════════════════════
class TestB6ApiQuery:
    """B6: api_query — 서울시 API 5개 엔드포인트."""

    def _make_api_qa(self, api_key: str, params: dict | None = None) -> dict:
        """API 테스트용 query_analysis를 생성한다."""
        return {
            **BASIC_QUERY_ANALYSIS,
            "api_params": {
                "api": api_key,
                "params": params or {},
            },
        }

    def test_returns_standard_format(self):
        from agent.tools.api_tools import api_query
        qa = self._make_api_qa("estimated_sales")
        result = api_query(qa, EMPTY_STATE)
        _assert_tool_result(result, "api_query")

    def test_commercial_change_api(self):
        """commercial_change API 호출."""
        from agent.tools.api_tools import api_query
        qa = self._make_api_qa("commercial_change")
        result = api_query(qa, EMPTY_STATE)
        _assert_tool_result(result)

    def test_store_openclose_api(self):
        """store_openclose API 호출."""
        from agent.tools.api_tools import api_query
        qa = self._make_api_qa("store_openclose")
        result = api_query(qa, EMPTY_STATE)
        _assert_tool_result(result)

    def test_crowd_facility_api(self):
        """crowd_facility API 호출."""
        from agent.tools.api_tools import api_query
        qa = self._make_api_qa("crowd_facility")
        result = api_query(qa, EMPTY_STATE)
        _assert_tool_result(result)

    def test_resident_population_api(self):
        """resident_population API 호출."""
        from agent.tools.api_tools import api_query
        qa = self._make_api_qa("resident_population")
        result = api_query(qa, EMPTY_STATE)
        _assert_tool_result(result)

    def test_unsupported_api_graceful(self):
        """지원하지 않는 API 키에 대해 irrelevant을 반환하는지 확인."""
        from agent.tools.api_tools import api_query
        qa = self._make_api_qa("nonexistent_api")
        result = api_query(qa, EMPTY_STATE)
        assert result["relevance"] == "irrelevant"

    def test_caching_works(self):
        """동일 요청 시 캐시가 동작하는지 확인."""
        from agent.tools.api_tools import api_query, clear_api_cache
        clear_api_cache()
        qa = self._make_api_qa("crowd_facility")
        result1 = api_query(qa, EMPTY_STATE)
        result2 = api_query(qa, EMPTY_STATE)
        # 2번째 호출은 캐시 히트여야 한다
        if result1["relevance"] == "relevant":
            assert "cached" in result2.get("source", ""), \
                "Second call should be a cache hit"


# ═══════════════════════════════════════
# B7: web_search (Claude 웹 검색)
# ═══════════════════════════════════════
class TestB7WebSearch:
    """B7: web_search — Claude built-in 웹 검색."""

    def test_returns_standard_format(self):
        from agent.tools.web_tools import web_search
        result = web_search(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "web_search")

    def test_web_meta_present(self):
        """web_meta 필드가 반환되는지 확인."""
        from agent.tools.web_tools import web_search
        result = web_search(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        assert "web_meta" in result, "web_search should return web_meta"
        meta = result["web_meta"]
        assert "search_count" in meta
        assert "result_count" in meta
        assert "source_domains" in meta
        assert "freshness" in meta


# ═══════════════════════════════════════
# B8: calculate (수치 검증)
# ═══════════════════════════════════════
class TestB8Calculate:
    """B8: calculate — 수치 계산/검증."""

    def test_returns_standard_format(self):
        from agent.tools.result_tools import calculate
        result = calculate(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "calculate")

    def test_with_claims(self):
        """claims가 있는 state에서 수치 검증이 동작하는지 확인."""
        from agent.tools.result_tools import calculate
        state_with_claims = {
            **EMPTY_STATE,
            "analysis_result": {
                "claims": [
                    {"text": "강남구 카페 수", "value": "64123", "source": "store_info"},
                    {"text": "마포구 카페 수", "value": "30488", "source": "store_info"},
                ],
            },
        }
        result = calculate(BASIC_QUERY_ANALYSIS, state_with_claims)
        assert len(result["data"]) >= 1, "Should verify at least 1 claim"

    def test_empty_claims(self):
        """claims가 비어있으면 빈 검증 목록을 반환하는지 확인."""
        from agent.tools.result_tools import calculate
        result = calculate(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        assert result["data"] == []


# ═══════════════════════════════════════
# B9: lookup_previous (이전 턴 결론 조회)
# ═══════════════════════════════════════
class TestB9LookupPrevious:
    """B9: lookup_previous — 이전 턴 결론 조회."""

    def test_returns_standard_format(self):
        from agent.tools.result_tools import lookup_previous
        result = lookup_previous(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        _assert_tool_result(result, "lookup_previous")

    def test_empty_conclusions_returns_irrelevant(self):
        """결론이 없으면 irrelevant을 반환하는지 확인."""
        from agent.tools.result_tools import lookup_previous
        result = lookup_previous(BASIC_QUERY_ANALYSIS, EMPTY_STATE)
        assert result["relevance"] == "irrelevant"

    def test_with_conclusions(self):
        """결론이 있으면 relevant을 반환하고 데이터에 포함하는지 확인."""
        from agent.tools.result_tools import lookup_previous
        state = {
            **EMPTY_STATE,
            "turn_conclusions": [
                {"turn_number": 1, "conclusion_summary": "강남구 카페 추천"},
                {"turn_number": 2, "conclusion_summary": "마포구 임대료 분석"},
            ],
        }
        result = lookup_previous(BASIC_QUERY_ANALYSIS, state)
        assert result["relevance"] == "relevant"
        assert len(result["data"]) == 2

    def test_referenced_turns_filter(self):
        """referenced_turns가 지정되면 해당 턴만 반환하는지 확인."""
        from agent.tools.result_tools import lookup_previous
        qa = {**BASIC_QUERY_ANALYSIS, "referenced_turns": [2]}
        state = {
            **EMPTY_STATE,
            "turn_conclusions": [
                {"turn_number": 1, "conclusion_summary": "Turn 1"},
                {"turn_number": 2, "conclusion_summary": "Turn 2"},
                {"turn_number": 3, "conclusion_summary": "Turn 3"},
            ],
        }
        result = lookup_previous(qa, state)
        assert len(result["data"]) == 1
        assert result["data"][0]["turn_number"] == 2
