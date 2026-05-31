"""
evaluation/groundedness_checker.py — 비-LLM 기반 답변 근거성(Groundedness) 검증기

역할:
    에이전트의 최종 응답이 실제 수집된 데이터에 근거하는지를 LLM 호출 없이
    결정론적으로 검증한다. 두 가지 방법을 조합한다:
    1) 수치 정확도 (Numeric Accuracy): 응답 내 숫자를 CSV 원본과 대조
    2) 소스 근거도 (Source Grounding): 응답에서 언급한 소스가 gathered_data에 존재하는지 확인

데이터 흐름:
    입력: response_text (응답 텍스트), gathered_data (수집 데이터 리스트), analysis_result (분석 결과)
    출력: 근거성 검증 결과 딕셔너리 (grounded_claim_ratio, numeric_accuracy, source_grounding 등)
"""

import os
import re
from pathlib import Path

import pandas as pd

# --- CSV 데이터 디렉토리 ---
_BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DATA_DIR = _BASE_DIR / "knowledge_base" / "data"

# --- 수치 추출 정규식 패턴 ---
# 한국어 텍스트에서 숫자+단위 조합을 추출한다.
# 예: "12,847개", "1.2억 원", "35.7%", "2,500만 원"
_NUMBER_UNIT_PATTERN = re.compile(
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(개|건|원|명|%|평|㎡|억|만|호|곳|위|분기)"
)

# 한국어 배수 단위를 실제 숫자로 변환하는 매핑
_KOREAN_MULTIPLIERS = {
    "억": 1_0000_0000,  # 1억 = 100,000,000
    "만": 1_0000,        # 1만 = 10,000
}

# --- 소스 이름 패턴 ---
# 응답 텍스트에서 소스 참조를 추출하기 위한 패턴
_SOURCE_PATTERNS = [
    # CSV 파일명 패턴
    re.compile(r"(store_info|foot_traffic|rent|demographics|dong_summary|"
               r"estimated_sales|card_consumption|subway_ridership|business_codes)"
               r"(?:\.csv)?"),
    # 한국어 소스명 패턴
    re.compile(r"(상권분석보고서|상가정보|유동인구|임대료|인구통계|매출|카드소비|지하철|업종코드)"),
    # 도구 이름 패턴
    re.compile(r"(pandas_query|rag_search|rag_deep_read|rag_global_summary|"
               r"rag_compare|web_search|api_query|calculate|lookup_previous)"),
    # 일반 소스 유형 패턴
    re.compile(r"(웹\s*검색|API\s*(?:조회|데이터)|서울시\s*API|공공\s*데이터)"),
]

# 한국어 소스명 → CSV 파일 매핑 (소스 매칭 정규화용)
_KOREAN_TO_CSV = {
    "상가정보": "store_info",
    "유동인구": "foot_traffic",
    "임대료": "rent",
    "인구통계": "demographics",
    "매출": "estimated_sales",
    "카드소비": "card_consumption",
    "지하철": "subway_ridership",
    "업종코드": "business_codes",
}


def _parse_number(raw: str) -> float:
    """콤마가 포함된 문자열을 float로 변환한다.

    Args:
        raw: 콤마 포함 숫자 문자열 (예: "12,847")

    Returns:
        파싱된 float 값
    """
    return float(raw.replace(",", ""))


def _apply_multiplier(value: float, unit: str) -> float:
    """한국어 배수 단위(억, 만)를 적용하여 실제 값을 반환한다.

    Args:
        value: 원본 숫자 값
        unit: 단위 문자열

    Returns:
        배수가 적용된 실제 값. 배수 단위가 아니면 원본 그대로 반환.
    """
    if unit in _KOREAN_MULTIPLIERS:
        return value * _KOREAN_MULTIPLIERS[unit]
    return value


def extract_numbers(text: str) -> list[dict]:
    """응답 텍스트에서 숫자+단위 조합을 추출한다.

    Args:
        text: 에이전트 응답 텍스트

    Returns:
        추출된 숫자 정보 리스트. 각 항목은 {raw, value, unit, actual_value} 딕셔너리.
        - raw: 원본 매칭 텍스트 (예: "12,847개")
        - value: 파싱된 숫자 (예: 12847.0)
        - unit: 단위 (예: "개")
        - actual_value: 배수 적용된 실제 값 (예: 억/만 적용 후)
    """
    results = []
    for match in _NUMBER_UNIT_PATTERN.finditer(text):
        raw_num = match.group(1)
        unit = match.group(2)
        value = _parse_number(raw_num)
        actual_value = _apply_multiplier(value, unit)
        results.append({
            "raw": match.group(0),
            "value": value,
            "unit": unit,
            "actual_value": actual_value,
        })
    return results


def extract_sources(text: str) -> list[str]:
    """응답 텍스트에서 소스 참조를 추출한다.

    여러 패턴을 순회하며 매칭되는 소스명을 수집한다.
    중복을 제거하고 정규화된 소스명 리스트를 반환한다.

    Args:
        text: 에이전트 응답 텍스트

    Returns:
        정규화된 소스명 리스트 (중복 제거됨)
    """
    found: set[str] = set()
    for pattern in _SOURCE_PATTERNS:
        for match in pattern.finditer(text):
            source = match.group(1).strip()
            # 한국어 소스명을 영문 키로 정규화
            normalized = _KOREAN_TO_CSV.get(source, source)
            found.add(normalized)
    return sorted(found)


def _load_csv_files() -> dict[str, pd.DataFrame]:
    """knowledge_base/data/ 디렉토리의 모든 CSV 파일을 로드한다.

    Returns:
        파일명(확장자 제외) → DataFrame 매핑 딕셔너리
    """
    csv_files: dict[str, pd.DataFrame] = {}
    if not CSV_DATA_DIR.exists():
        return csv_files

    for csv_path in CSV_DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            csv_files[csv_path.stem] = df
        except Exception:
            # 파싱 실패한 CSV는 건너뛴다
            continue
    return csv_files


def _collect_csv_numeric_values(csv_files: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
    """모든 CSV에서 숫자형 값을 추출한다.

    Args:
        csv_files: 파일명 → DataFrame 매핑

    Returns:
        (파일명, 숫자값) 튜플 리스트
    """
    values: list[tuple[str, float]] = []
    for name, df in csv_files.items():
        for col in df.select_dtypes(include=["number"]).columns:
            for val in df[col].dropna().unique():
                values.append((name, float(val)))
    return values


def check_numeric_accuracy(
    numbers: list[dict],
    csv_files: dict[str, pd.DataFrame] | None = None,
    tolerance: float = 0.05,
) -> tuple[float, list[dict]]:
    """추출된 숫자를 CSV 원본과 대조하여 정확도를 계산한다.

    5% 허용 오차 내에서 매칭되면 "일치"로 판정한다.
    퍼센트(%) 단위는 CSV의 비율 컬럼과 직접 비교한다.

    Args:
        numbers: extract_numbers()의 반환값
        csv_files: 파일명 → DataFrame 매핑. None이면 자동 로드.
        tolerance: 허용 오차 비율 (기본 5%)

    Returns:
        (정확도 점수 0.0~1.0, 개별 검사 결과 리스트)
    """
    if not numbers:
        return 1.0, []  # 숫자가 없으면 정확도 문제 없음

    if csv_files is None:
        csv_files = _load_csv_files()

    csv_values = _collect_csv_numeric_values(csv_files)
    checks: list[dict] = []

    for num_info in numbers:
        actual = num_info["actual_value"]
        matched = False
        found_in = None

        for csv_name, csv_val in csv_values:
            if csv_val == 0 and actual == 0:
                matched = True
                found_in = csv_name
                break
            if csv_val != 0 and abs(actual - csv_val) / abs(csv_val) <= tolerance:
                matched = True
                found_in = csv_name
                break

        checks.append({
            "value": num_info["raw"],
            "actual_value": actual,
            "found_in_csv": found_in,
            "matched": matched,
        })

    matched_count = sum(1 for c in checks if c["matched"])
    accuracy = matched_count / len(checks)
    return accuracy, checks


def check_source_grounding(
    mentioned_sources: list[str],
    gathered_data: list[dict],
) -> tuple[float, list[dict]]:
    """응답에서 언급된 소스가 gathered_data에 실제 존재하는지 검증한다.

    gathered_data의 source 필드와 tool_used 필드를 모두 확인한다.

    Args:
        mentioned_sources: extract_sources()의 반환값
        gathered_data: 수집된 데이터 항목 리스트

    Returns:
        (근거도 점수 0.0~1.0, 개별 검사 결과 리스트)
    """
    if not mentioned_sources:
        return 1.0, []  # 소스 언급이 없으면 근거 문제 없음

    # gathered_data에서 소스명과 도구명을 수집
    gathered_sources: set[str] = set()
    for item in gathered_data:
        if "source" in item and item["source"]:
            # CSV 파일명에서 확장자 제거하여 정규화
            src = item["source"]
            gathered_sources.add(src)
            if src.endswith(".csv"):
                gathered_sources.add(src.replace(".csv", ""))
        if "tool_used" in item and item["tool_used"]:
            gathered_sources.add(item["tool_used"])

    checks: list[dict] = []
    for source in mentioned_sources:
        # 정확한 매칭 또는 부분 매칭 시도
        matched = False
        if source in gathered_sources:
            matched = True
        else:
            # 부분 매칭: gathered_sources 내 항목에 source가 포함되거나 그 반대
            for gs in gathered_sources:
                if source in gs or gs in source:
                    matched = True
                    break

        checks.append({
            "source_mentioned": source,
            "found_in_gathered": matched,
            "matched": matched,
        })

    matched_count = sum(1 for c in checks if c["matched"])
    grounding = matched_count / len(checks)
    return grounding, checks


def check_groundedness(
    response_text: str,
    gathered_data: list[dict],
    analysis_result: dict,
) -> dict:
    """비-LLM 기반 답변 근거성 검증.

    두 가지 검증 방법을 조합하여 최종 근거성 점수를 산출한다:
    - 수치 정확도 (가중치 0.6): 응답 내 숫자가 CSV 원본과 일치하는지
    - 소스 근거도 (가중치 0.4): 언급된 소스가 실제 수집 데이터에 있는지

    Args:
        response_text: 에이전트의 최종 응답 텍스트
        gathered_data: 수집된 데이터 항목 리스트 [{source, tool_used, data_summary, token_count}]
        analysis_result: generate_analysis 결과 {summary, claims, data_references, caveats}

    Returns:
        근거성 검증 결과 딕셔너리:
        - grounded_claim_ratio: 최종 근거성 점수 (0.0~1.0)
        - numeric_accuracy: 수치 정확도 (0.0~1.0)
        - source_grounding: 소스 근거도 (0.0~1.0)
        - hallucination_detected: 환각 감지 여부 (grounded_claim_ratio < 0.7)
        - ungrounded_claims: 근거 없는 주장 리스트
        - numeric_checks: 수치 검사 상세 결과
        - source_checks: 소스 검사 상세 결과
    """
    # analysis_result에서 추가 텍스트를 합산하여 검증 범위 확대
    full_text = response_text
    if analysis_result:
        if "summary" in analysis_result:
            full_text += "\n" + str(analysis_result["summary"])
        if "claims" in analysis_result:
            for claim in analysis_result.get("claims", []):
                full_text += "\n" + str(claim)

    # ═══════════════════════════════════════
    # STEP 1: 수치 정확도 검증
    # ═══════════════════════════════════════
    numbers = extract_numbers(full_text)
    csv_files = _load_csv_files()
    numeric_accuracy, numeric_checks = check_numeric_accuracy(numbers, csv_files)

    # ═══════════════════════════════════════
    # STEP 2: 소스 근거도 검증
    # ═══════════════════════════════════════
    mentioned_sources = extract_sources(full_text)
    source_grounding, source_checks = check_source_grounding(
        mentioned_sources, gathered_data
    )

    # ═══════════════════════════════════════
    # STEP 3: 결합 점수 계산
    # ═══════════════════════════════════════
    # 가중 평균: 수치 정확도 60% + 소스 근거도 40%
    grounded_claim_ratio = 0.6 * numeric_accuracy + 0.4 * source_grounding

    # 근거 없는 주장 수집
    ungrounded_claims: list[str] = []
    for check in numeric_checks:
        if not check["matched"]:
            ungrounded_claims.append(f"수치 불일치: {check['value']}")
    for check in source_checks:
        if not check["matched"]:
            ungrounded_claims.append(f"소스 미확인: {check['source_mentioned']}")

    return {
        "grounded_claim_ratio": round(grounded_claim_ratio, 4),
        "numeric_accuracy": round(numeric_accuracy, 4),
        "source_grounding": round(source_grounding, 4),
        "hallucination_detected": grounded_claim_ratio < 0.7,
        "ungrounded_claims": ungrounded_claims,
        "numeric_checks": numeric_checks,
        "source_checks": source_checks,
    }
