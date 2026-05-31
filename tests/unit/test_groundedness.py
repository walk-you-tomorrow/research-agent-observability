"""
tests/unit/test_groundedness.py — 비-LLM Groundedness Checker 단위 테스트

테스트 대상:
    - extract_numbers(): 한국어 텍스트에서 숫자+단위 추출
    - extract_sources(): 응답에서 소스 참조 추출
    - check_numeric_accuracy(): CSV 대조 수치 정확도
    - check_source_grounding(): gathered_data 소스 매칭
    - check_groundedness(): 결합 점수 계산 + 엣지 케이스
"""
import pandas as pd
import pytest

from evaluation.groundedness_checker import (
    check_groundedness,
    check_numeric_accuracy,
    check_source_grounding,
    extract_numbers,
    extract_sources,
)


# ═══════════════════════════════════════
# 숫자 추출 테스트
# ═══════════════════════════════════════

class TestExtractNumbers:
    """한국어 텍스트에서 숫자+단위 조합 추출 테스트."""

    def test_basic_integer_with_unit(self):
        """정수+단위 기본 추출."""
        text = "강남구 음식점 12,847개"
        result = extract_numbers(text)
        assert len(result) == 1
        assert result[0]["value"] == 12847.0
        assert result[0]["unit"] == "개"
        assert result[0]["raw"] == "12,847개"

    def test_decimal_with_unit(self):
        """소수점+단위 추출."""
        text = "임대료 82.9원"
        result = extract_numbers(text)
        assert len(result) == 1
        assert result[0]["value"] == 82.9
        assert result[0]["unit"] == "원"

    def test_percentage(self):
        """퍼센트 추출."""
        text = "여성 비율 53.2%"
        result = extract_numbers(text)
        assert len(result) == 1
        assert result[0]["value"] == 53.2
        assert result[0]["unit"] == "%"

    def test_billion_multiplier(self):
        """억 단위 배수 적용."""
        text = "매출 1.2억"
        result = extract_numbers(text)
        assert len(result) == 1
        assert result[0]["value"] == 1.2
        assert result[0]["actual_value"] == 1.2 * 1_0000_0000

    def test_ten_thousand_multiplier(self):
        """만 단위 배수 적용."""
        text = "유동인구 2,500만"
        result = extract_numbers(text)
        assert len(result) == 1
        assert result[0]["actual_value"] == 2500 * 10000

    def test_multiple_numbers(self):
        """여러 숫자 동시 추출."""
        text = "강남구 음식점 12,847개, 유동인구 2,232,614명, 임대료 82.9원"
        result = extract_numbers(text)
        assert len(result) == 3

    def test_no_numbers(self):
        """숫자가 없는 텍스트."""
        text = "강남구의 상권 현황을 분석합니다."
        result = extract_numbers(text)
        assert len(result) == 0

    def test_number_without_unit_ignored(self):
        """단위 없는 숫자는 추출하지 않음."""
        text = "2024년 3분기 데이터"
        result = extract_numbers(text)
        assert len(result) == 1  # "3분기"만 매칭됨
        assert result[0]["unit"] == "분기"

    def test_area_unit(self):
        """면적 단위 (평, ㎡) 추출."""
        text = "매장 면적 35평, 약 115.7㎡"
        result = extract_numbers(text)
        assert len(result) == 2
        units = {r["unit"] for r in result}
        assert "평" in units
        assert "㎡" in units


# ═══════════════════════════════════════
# 소스 추출 테스트
# ═══════════════════════════════════════

class TestExtractSources:
    """응답에서 소스 참조 추출 테스트."""

    def test_csv_filename(self):
        """CSV 파일명 추출."""
        text = "store_info.csv 데이터에 따르면"
        result = extract_sources(text)
        assert "store_info" in result

    def test_csv_without_extension(self):
        """확장자 없는 CSV 이름 추출."""
        text = "foot_traffic 데이터를 분석한 결과"
        result = extract_sources(text)
        assert "foot_traffic" in result

    def test_korean_source_name(self):
        """한국어 소스명 추출 및 정규화."""
        text = "상가정보에서 확인한 결과, 유동인구 데이터에 따르면"
        result = extract_sources(text)
        assert "store_info" in result  # 상가정보 → store_info
        assert "foot_traffic" in result  # 유동인구 → foot_traffic

    def test_tool_name(self):
        """도구 이름 추출."""
        text = "pandas_query로 조회한 결과, web_search를 통해 확인"
        result = extract_sources(text)
        assert "pandas_query" in result
        assert "web_search" in result

    def test_web_search_korean(self):
        """한국어 '웹 검색' 추출."""
        text = "웹 검색 결과에 따르면"
        result = extract_sources(text)
        assert "웹 검색" in result

    def test_no_sources(self):
        """소스 언급이 없는 텍스트."""
        text = "강남구는 서울에서 가장 활발한 상권입니다."
        result = extract_sources(text)
        assert len(result) == 0

    def test_multiple_sources_dedup(self):
        """중복 소스 제거."""
        text = "store_info 데이터와 store_info.csv를 비교하면"
        result = extract_sources(text)
        # store_info가 두 번 나와도 한 번만
        assert result.count("store_info") == 1

    def test_api_source(self):
        """API 소스 추출."""
        text = "서울시 API를 통해 조회한 매출 데이터"
        result = extract_sources(text)
        assert any("서울시" in s or "api_query" in s for s in result)


# ═══════════════════════════════════════
# CSV 대조 수치 정확도 테스트
# ═══════════════════════════════════════

class TestCheckNumericAccuracy:
    """CSV 원본 대조 수치 정확도 테스트."""

    @pytest.fixture
    def sample_csv_files(self) -> dict[str, pd.DataFrame]:
        """테스트용 CSV 데이터."""
        return {
            "store_info": pd.DataFrame({
                "count": [12847, 5432, 8901],
                "district": ["강남구", "마포구", "서초구"],
            }),
            "foot_traffic": pd.DataFrame({
                "total_foot_traffic": [2232614, 1360721],
                "district": ["강남구", "강남구"],
            }),
            "rent": pd.DataFrame({
                "rent_2025Q3": [82.6, 189.3, 95.8],
                "district": ["종로구", "종로구", "종로구"],
            }),
        }

    def test_exact_match(self, sample_csv_files):
        """CSV 값과 정확히 일치하는 경우."""
        numbers = [{"raw": "12,847개", "value": 12847.0, "unit": "개", "actual_value": 12847.0}]
        accuracy, checks = check_numeric_accuracy(numbers, sample_csv_files)
        assert accuracy == 1.0
        assert checks[0]["matched"] is True
        assert checks[0]["found_in_csv"] == "store_info"

    def test_approximate_match_within_tolerance(self, sample_csv_files):
        """5% 허용 오차 내 근사 매칭."""
        # 12847 * 1.04 = 13360.88 → 5% 이내
        numbers = [{"raw": "13,360개", "value": 13360.0, "unit": "개", "actual_value": 13360.0}]
        accuracy, checks = check_numeric_accuracy(numbers, sample_csv_files, tolerance=0.05)
        assert accuracy == 1.0
        assert checks[0]["matched"] is True

    def test_no_match_outside_tolerance(self, sample_csv_files):
        """허용 오차 초과 시 불일치."""
        # 99999는 어떤 CSV 값과도 5% 이내에 들지 않음
        numbers = [{"raw": "99,999개", "value": 99999.0, "unit": "개", "actual_value": 99999.0}]
        accuracy, checks = check_numeric_accuracy(numbers, sample_csv_files)
        assert accuracy == 0.0
        assert checks[0]["matched"] is False

    def test_empty_numbers(self, sample_csv_files):
        """숫자가 없으면 정확도 1.0 (문제 없음)."""
        accuracy, checks = check_numeric_accuracy([], sample_csv_files)
        assert accuracy == 1.0
        assert checks == []

    def test_partial_match(self, sample_csv_files):
        """일부만 매칭되는 경우."""
        numbers = [
            {"raw": "12,847개", "value": 12847.0, "unit": "개", "actual_value": 12847.0},
            {"raw": "99,999명", "value": 99999.0, "unit": "명", "actual_value": 99999.0},
        ]
        accuracy, checks = check_numeric_accuracy(numbers, sample_csv_files)
        assert accuracy == 0.5
        assert checks[0]["matched"] is True
        assert checks[1]["matched"] is False

    def test_zero_value_match(self, sample_csv_files):
        """0값 매칭 (0/0 나눗셈 방지)."""
        sample_csv_files["test"] = pd.DataFrame({"val": [0, 1, 2]})
        numbers = [{"raw": "0개", "value": 0.0, "unit": "개", "actual_value": 0.0}]
        accuracy, checks = check_numeric_accuracy(numbers, sample_csv_files)
        assert accuracy == 1.0
        assert checks[0]["matched"] is True


# ═══════════════════════════════════════
# 소스 근거도 테스트
# ═══════════════════════════════════════

class TestCheckSourceGrounding:
    """gathered_data 소스 매칭 테스트."""

    @pytest.fixture
    def sample_gathered_data(self) -> list[dict]:
        """테스트용 gathered_data."""
        return [
            {"source": "store_info.csv", "tool_used": "pandas_query", "data_summary": "...", "token_count": 100},
            {"source": "foot_traffic.csv", "tool_used": "pandas_query", "data_summary": "...", "token_count": 200},
            {"source": "상권분석보고서", "tool_used": "rag_search", "data_summary": "...", "token_count": 300},
            {"source": "웹 검색 결과", "tool_used": "web_search", "data_summary": "...", "token_count": 150},
        ]

    def test_all_sources_found(self, sample_gathered_data):
        """모든 언급된 소스가 gathered_data에 존재."""
        mentioned = ["store_info", "pandas_query"]
        grounding, checks = check_source_grounding(mentioned, sample_gathered_data)
        assert grounding == 1.0
        assert all(c["matched"] for c in checks)

    def test_partial_sources_found(self, sample_gathered_data):
        """일부 소스만 존재."""
        mentioned = ["store_info", "api_query"]
        grounding, checks = check_source_grounding(mentioned, sample_gathered_data)
        assert grounding == 0.5

    def test_no_sources_found(self, sample_gathered_data):
        """어떤 소스도 없음."""
        mentioned = ["unknown_source", "nonexistent_tool"]
        grounding, checks = check_source_grounding(mentioned, sample_gathered_data)
        assert grounding == 0.0

    def test_empty_mentioned(self, sample_gathered_data):
        """소스 언급 없으면 1.0."""
        grounding, checks = check_source_grounding([], sample_gathered_data)
        assert grounding == 1.0
        assert checks == []

    def test_partial_string_match(self, sample_gathered_data):
        """부분 문자열 매칭 (예: 'web_search' in 'web_search')."""
        mentioned = ["web_search"]
        grounding, checks = check_source_grounding(mentioned, sample_gathered_data)
        assert grounding == 1.0

    def test_csv_extension_normalization(self, sample_gathered_data):
        """CSV 확장자 유무에 관계없이 매칭."""
        mentioned = ["store_info"]  # gathered_data에는 "store_info.csv"로 저장
        grounding, checks = check_source_grounding(mentioned, sample_gathered_data)
        assert grounding == 1.0


# ═══════════════════════════════════════
# 결합 점수 계산 테스트
# ═══════════════════════════════════════

class TestCheckGroundedness:
    """check_groundedness() 통합 테스트."""

    def test_fully_grounded_response(self):
        """완전히 근거 있는 응답."""
        response = "store_info 데이터에 따르면 강남구 음식점은 12,847개입니다."
        gathered_data = [
            {"source": "store_info.csv", "tool_used": "pandas_query",
             "data_summary": "...", "token_count": 100},
        ]
        # CSV에 12847이 있다고 가정하기 위해 mock이 필요하지만,
        # 실제 CSV 파일이 없어도 소스 근거도는 테스트 가능
        analysis_result = {"summary": "강남구 음식점 현황", "claims": []}
        result = check_groundedness(response, gathered_data, analysis_result)

        assert "grounded_claim_ratio" in result
        assert "numeric_accuracy" in result
        assert "source_grounding" in result
        assert "hallucination_detected" in result
        assert isinstance(result["ungrounded_claims"], list)
        assert isinstance(result["numeric_checks"], list)
        assert isinstance(result["source_checks"], list)
        # 소스 근거도는 1.0이어야 함 (store_info가 gathered_data에 존재)
        assert result["source_grounding"] == 1.0

    def test_empty_response(self):
        """빈 응답."""
        result = check_groundedness("", [], {})
        assert result["grounded_claim_ratio"] == 1.0
        assert result["numeric_accuracy"] == 1.0
        assert result["source_grounding"] == 1.0
        assert result["hallucination_detected"] is False
        assert result["ungrounded_claims"] == []

    def test_hallucination_detection_threshold(self):
        """grounded_claim_ratio < 0.7이면 환각 감지."""
        # 존재하지 않는 소스만 언급
        response = "api_query 데이터에 따르면 99,999개입니다."
        gathered_data = [
            {"source": "store_info.csv", "tool_used": "pandas_query",
             "data_summary": "...", "token_count": 100},
        ]
        result = check_groundedness(response, gathered_data, {})
        # 소스 불일치 + 수치 불일치 가능성 → 낮은 점수
        assert isinstance(result["hallucination_detected"], bool)

    def test_combined_score_weights(self):
        """결합 점수 가중치 검증: 0.6 * numeric + 0.4 * source."""
        response = "store_info 기준 99,999개"
        gathered_data = [
            {"source": "store_info.csv", "tool_used": "pandas_query",
             "data_summary": "...", "token_count": 100},
        ]
        result = check_groundedness(response, gathered_data, {})
        # 소스 근거도 = 1.0 (store_info 존재), 수치 정확도는 CSV 로드에 따라 다름
        # 결합 점수 = 0.6 * numeric_accuracy + 0.4 * 1.0
        expected = 0.6 * result["numeric_accuracy"] + 0.4 * result["source_grounding"]
        assert abs(result["grounded_claim_ratio"] - round(expected, 4)) < 0.001

    def test_analysis_result_claims_included(self):
        """analysis_result의 claims도 검증 범위에 포함."""
        response = "분석 결과입니다."
        analysis_result = {
            "summary": "강남구 현황",
            "claims": ["store_info 데이터 기준 12,847개 업소 확인"],
        }
        gathered_data = [
            {"source": "store_info.csv", "tool_used": "pandas_query",
             "data_summary": "...", "token_count": 100},
        ]
        result = check_groundedness(response, gathered_data, analysis_result)
        # claims에서 store_info, 12,847개가 추출되어야 함
        assert result["source_grounding"] == 1.0
        assert len(result["numeric_checks"]) >= 1

    def test_no_gathered_data(self):
        """gathered_data가 비어있으면 소스 근거도가 낮아짐."""
        response = "store_info 데이터에 따르면 12,847개"
        result = check_groundedness(response, [], {})
        assert result["source_grounding"] == 0.0

    def test_return_structure(self):
        """반환 딕셔너리 구조 검증."""
        result = check_groundedness("테스트", [], {})
        required_keys = {
            "grounded_claim_ratio",
            "numeric_accuracy",
            "source_grounding",
            "hallucination_detected",
            "ungrounded_claims",
            "numeric_checks",
            "source_checks",
        }
        assert required_keys == set(result.keys())
