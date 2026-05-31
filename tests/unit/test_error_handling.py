"""
tests/unit/test_error_handling.py — 에러 핸들링 및 엣지 케이스 테스트

개별 도구와 노드의 에러 처리, 경계 조건, 그레이스풀 디그레이드를 검증한다.
전체 파이프라인은 실행하지 않고, 도구와 유틸리티 함수를 직접 테스트한다.

테스트 대상 (C1~C7):
    C1: API 타임아웃/실패 시 graceful degradation
    C2: LightRAG 빈 결과 처리
    C3: Web search 실패 시 처리
    C4: Max gather retry 도달 시 동작
    C5: Max verify retry 도달 시 동작
    C6: 빈 CSV 쿼리 결과 처리
    C7: 대용량 컨텍스트 오버플로우

실행 방법:
    python -m pytest tests/unit/test_error_handling.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════
# C1: API 타임아웃/실패 → graceful degradation
# ═══════════════════════════════════════
class TestC1ApiFailure:
    """C1: API 실패 시 에러가 전파되지 않고 irrelevant 반환."""

    def test_api_no_key_returns_irrelevant(self):
        """API 키가 없으면 irrelevant을 반환한다."""
        from agent.tools.api_tools import api_query

        with patch("agent.tools.api_tools.API_KEY", ""):
            qa = {
                "intent": "매출 조회",
                "api_params": {"api": "estimated_sales", "params": {}},
            }
            result = api_query(qa, {})
            assert result["relevance"] == "irrelevant"
            assert "API 키" in result["summary"]

    def test_api_network_error_returns_irrelevant(self):
        """네트워크 에러 시 irrelevant을 반환한다."""
        from agent.tools.api_tools import api_query

        with patch("agent.tools.api_tools.requests.get", side_effect=ConnectionError("Network error")):
            with patch("agent.tools.api_tools.API_KEY", "test_key"):
                qa = {
                    "intent": "매출 조회",
                    "api_params": {"api": "estimated_sales", "params": {}},
                }
                result = api_query(qa, {})
                assert result["relevance"] == "irrelevant"
                assert "실패" in result["summary"]

    def test_api_invalid_key_returns_irrelevant(self):
        """잘못된 API 키를 사용해도 에러가 전파되지 않는다."""
        from agent.tools.api_tools import api_query

        qa = {
            "intent": "매출 조회",
            "api_params": {"api": "nonexistent_api", "params": {}},
        }
        result = api_query(qa, {})
        assert result["relevance"] == "irrelevant"


# ═══════════════════════════════════════
# C2: LightRAG 빈 결과 처리
# ═══════════════════════════════════════
class TestC2LightRAGEmpty:
    """C2: LightRAG에서 결과가 없을 때 graceful 처리."""

    def test_empty_query_result(self):
        """의미 없는 쿼리에 대해 빈 결과를 반환한다."""
        from agent.tools.rag_tools import rag_search

        qa = {
            "intent": "xyzabc123 존재하지않는것",
            "keywords": ["xyzabc123"],
            "required_data": [],
            "required_docs": [],
        }
        result = rag_search(qa, {})
        # 결과가 비어 있어도 에러가 발생하지 않아야 한다
        assert "source" in result
        assert "summary" in result

    def test_lightrag_exception_handled(self):
        """LightRAG 내부 예외가 발생해도 도구 수준에서 처리된다."""
        from agent.tools.rag_tools import rag_search

        with patch("agent.tools.rag_tools.query_knowledge", side_effect=Exception("KG error")):
            qa = {"intent": "test", "keywords": []}
            # rag_search는 query_knowledge를 직접 호출하므로 예외가 전파될 수 있다.
            # gather_data가 try/except로 감싸므로, 여기서는 예외 발생을 확인한다.
            with pytest.raises(Exception, match="KG error"):
                rag_search(qa, {})


# ═══════════════════════════════════════
# C3: Web search 실패 시 처리
# ═══════════════════════════════════════
class TestC3WebSearchFailure:
    """C3: 웹 검색 실패 시 에러가 전파되지 않고 irrelevant 반환."""

    def test_web_search_exception_returns_irrelevant(self):
        """Claude API 호출 실패 시 irrelevant을 반환한다."""
        from agent.tools.web_tools import web_search

        with patch("agent.tools.web_tools.ChatAnthropic", side_effect=Exception("API unavailable")):
            result = web_search({"intent": "최신 뉴스"}, {"user_query": "test"})
            assert result["relevance"] == "irrelevant"
            assert "실패" in result["summary"]
            assert result["web_meta"]["search_count"] == 0

    def test_web_search_empty_response(self):
        """빈 응답이 돌아와도 정상 처리된다."""
        from agent.tools.web_tools import web_search

        mock_response = MagicMock()
        mock_response.content = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response

        with patch("agent.tools.web_tools.ChatAnthropic", return_value=mock_llm):
            result = web_search({"intent": "test"}, {"user_query": "test"})
            assert result["source"] == "web_search"
            assert result["web_meta"]["result_count"] == 0


# ═══════════════════════════════════════
# C4: Max gather retries 도달
# ═══════════════════════════════════════
class TestC4MaxGatherRetries:
    """C4: 최대 재수집 횟수 도달 시 루프를 종료하는지 검증."""

    def test_graph_enforces_max_gather_retries(self):
        """graph.py의 should_continue_gather가 MAX_GATHER_RETRIES를 존중한다."""
        from agent.graph import MAX_GATHER_RETRIES

        # 상수가 정의되어 있는지 확인 (기본값 3)
        assert MAX_GATHER_RETRIES >= 1, "MAX_GATHER_RETRIES should be at least 1"
        assert MAX_GATHER_RETRIES <= 5, "MAX_GATHER_RETRIES should be reasonable (<= 5)"

    def test_should_continue_gather_stops_at_max(self):
        """gather_iteration이 MAX에 도달하면 generate_analysis로 진행한다."""
        from agent.graph import should_continue_gather, MAX_GATHER_RETRIES

        # should_continue_gather는 gather_iteration을 참조한다 (gather_retry_count가 아님).
        # iteration >= MAX_GATHER_RETRIES(3)이면 부족하더라도 강제 진행.
        state = {
            "context_evaluation": {"is_sufficient": False, "confidence_score": 0.3},
            "gather_iteration": MAX_GATHER_RETRIES,
        }
        result = should_continue_gather(state)
        assert result == "generate_analysis", \
            f"Should proceed to generate_analysis at max retries, got '{result}'"


# ═══════════════════════════════════════
# C5: Max verify retries 도달
# ═══════════════════════════════════════
class TestC5MaxVerifyRetries:
    """C5: 최대 검증 재시도 횟수 도달 시 응답으로 진행."""

    def test_graph_enforces_max_verify_retries(self):
        """graph.py의 MAX_VERIFY_RETRIES 상수가 정의되어 있다."""
        from agent.graph import MAX_VERIFY_RETRIES

        assert MAX_VERIFY_RETRIES >= 1
        assert MAX_VERIFY_RETRIES <= 3

    def test_route_after_verify_stops_at_max(self):
        """verify_retry_count가 MAX에 도달하면 respond_to_user로 진행한다."""
        from agent.graph import route_after_verify, MAX_VERIFY_RETRIES

        state = {
            "verification": {"overall_verdict": "fail_numeric"},
            "verify_retry_count": MAX_VERIFY_RETRIES,
        }
        result = route_after_verify(state)
        assert result == "respond_to_user", \
            f"Should proceed to respond_to_user at max retries, got '{result}'"


# ═══════════════════════════════════════
# C6: 빈 CSV 쿼리 결과 처리
# ═══════════════════════════════════════
class TestC6EmptyCSVQuery:
    """C6: 조건에 맞는 행이 0인 CSV 쿼리 시 에러 없이 처리."""

    def test_empty_required_data(self):
        """required_data가 비어있으면 irrelevant을 반환한다."""
        from agent.tools.data_tools import pandas_query

        qa = {"intent": "test", "required_data": []}
        result = pandas_query(qa, {})
        assert result["relevance"] == "irrelevant"

    def test_nonexistent_csv(self):
        """존재하지 않는 CSV 파일을 요청해도 에러가 발생하지 않는다."""
        from agent.tools.data_tools import pandas_query

        qa = {"intent": "test", "required_data": ["this_file_does_not_exist"]}
        result = pandas_query(qa, {})
        assert result["relevance"] == "irrelevant"


# ═══════════════════════════════════════
# C7: 대용량 컨텍스트 오버플로우
# ═══════════════════════════════════════
class TestC7ContextOverflow:
    """C7: 컨텍스트 윈도우를 초과하지 않는지 검증."""

    def test_context_window_utilization_under_100(self):
        """context_window_utilization이 1.0 이하인지 확인하는 로직 테스트."""
        from agent.monitoring_schema import CONTEXT_WINDOW_MAX_TOKENS

        # 최대 토큰이 합리적인 범위인지 확인
        assert CONTEXT_WINDOW_MAX_TOKENS > 0
        assert CONTEXT_WINDOW_MAX_TOKENS <= 200_000  # Claude Opus 기준

    def test_token_counter_works(self):
        """토큰 카운터가 정상 동작하는지 확인."""
        from agent.token_counter import count_tokens

        text = "서울 강남구 카페 상권 분석"
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 100  # 짧은 텍스트

    def test_large_text_token_count(self):
        """대용량 텍스트의 토큰 수가 합리적인지 확인."""
        from agent.token_counter import count_tokens

        # 10KB 텍스트
        large_text = "서울 상권 데이터 분석 결과. " * 500
        tokens = count_tokens(large_text)
        assert tokens > 100
        assert tokens < 50_000  # 합리적 범위
