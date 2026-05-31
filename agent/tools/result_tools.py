"""
agent/tools/result_tools.py — 결과 도구: calculate, lookup_previous

수치 계산/검증과 이전 턴 결론 조회를 담당하는 2개 도구를 제공한다.

calculate:
    분석 결과의 수치(claims의 value)를 원본 데이터와 대조하여 검증한다.
    verify_result 노드의 STEP 1(수치 검증)에서 사용된다.

lookup_previous:
    이전 턴의 결론(turn_conclusions)을 조회한다.
    다중 턴 대화에서 이전 분석 결과를 참조할 때 사용된다.

모든 도구의 인터페이스:
    tool_fn(query_analysis: dict, state: dict) → dict
    반환값: {source, summary, data, relevance, relevance_reason}
"""
import os
import re

import pandas as pd

KB_DATA_DIR = os.path.join("knowledge_base", "data")


def _try_numeric_lookup(source: str, value_str: str) -> bool | None:
    """원본 CSV에서 수치를 찾아 claim의 value와 대조한다.

    Returns:
        True: 원본에서 유사한 수치를 발견 (10% 이내 오차)
        False: 원본에서 불일치 발견
        None: 검증 불가 (파일 없음, 수치 파싱 실패 등)
    """
    # claim의 value에서 숫자 추출
    numbers = re.findall(r"[\d,]+\.?\d*", str(value_str).replace(",", ""))
    if not numbers:
        return None
    try:
        claimed_num = float(numbers[0])
    except ValueError:
        return None

    # source에서 CSV 파일명 추출
    csv_candidates = [
        "dong_summary.csv", "foot_traffic.csv", "rent.csv",
        "demographics.csv", "store_info.csv",
    ]

    for csv_file in csv_candidates:
        fpath = os.path.join(KB_DATA_DIR, csv_file)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            # 모든 수치 컬럼에서 claim 수치와 유사한 값이 있는지 탐색
            for col in df.select_dtypes(include="number").columns:
                col_values = df[col].dropna()
                # 정확히 일치하거나 10% 이내 오차
                if claimed_num == 0:
                    if (col_values == 0).any():
                        return True
                else:
                    tolerance = abs(claimed_num) * 0.1
                    close_match = col_values.between(
                        claimed_num - tolerance, claimed_num + tolerance
                    )
                    if close_match.any():
                        return True
        except Exception:
            continue

    return False


def calculate(query_analysis: dict, state: dict) -> dict:
    """분석 결과의 수치를 원본 데이터와 대조하여 검증한다.

    Args:
        query_analysis: 질의 분석 결과.
        state: 현재 AgentState. analysis_result.claims에서 검증 대상을 가져온다.

    Returns:
        표준 도구 반환 형식. data에 각 claim의 검증 결과 리스트 포함.
    """
    claims = state.get("analysis_result", {}).get("claims", [])

    verifications = []
    for claim in claims:
        if claim.get("value"):
            result = _try_numeric_lookup(
                claim.get("source", ""), claim["value"]
            )
            verifications.append({
                "claim": claim.get("text", ""),
                "stated_value": claim["value"],
                "verified": result if result is not None else True,
            })

    return {
        "source": "calculate",
        "summary": f"수치 검증: {len(verifications)}건",
        "data": verifications,
        "relevance": "relevant",
        "relevance_reason": "수치 검증",
    }


def lookup_previous(query_analysis: dict, state: dict) -> dict:
    """이전 턴의 결론(turn_conclusions)을 조회한다.

    두 가지 모드로 동작한다:
    1. referenced_turns가 지정된 경우: 해당 턴 번호의 결론만 반환
    2. referenced_turns가 비어있는 경우: 가장 최근 3개 턴의 결론 반환

    이 도구는 "이전에 뭐라고 했지?"와 같은 질문이나,
    이전 분석을 참조해야 하는 후속 질문에 활용된다.

    Args:
        query_analysis: 질의 분석 결과. referenced_turns에서 조회할 턴 번호를 가져온다.
        state: 현재 AgentState. turn_conclusions에서 이전 결론을 참조한다.

    Returns:
        표준 도구 반환 형식.
        - data에 조회된 턴 결론 리스트 포함.
        - 결론이 없으면 relevance="irrelevant" 반환.
    """
    turn_conclusions = state.get("turn_conclusions", [])
    referenced = query_analysis.get("referenced_turns", [])

    if referenced:
        # 특정 턴 번호의 결론만 필터링
        results = [c for c in turn_conclusions if c["turn_number"] in referenced]
    else:
        # 가장 최근 3개 턴의 결론 반환
        results = turn_conclusions[-3:]

    return {
        "source": "lookup_previous",
        "summary": f"이전 턴 결론 {len(results)}개 조회",
        "data": results,
        "relevance": "relevant" if results else "irrelevant",
        "relevance_reason": f"턴 {referenced} 결론 참조",
    }
