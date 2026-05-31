"""
agent/tools/data_tools.py — 데이터 도구: pandas_query

CSV 기반 서울 상권 데이터를 pandas로 쿼리하는 도구를 제공한다.
정확한 수치 집계가 필요할 때 사용된다 (LightRAG 비경유).

인터페이스:
    tool_fn(query_analysis: dict, state: dict) → dict

반환값 구조:
    {
        "source": str,              # 데이터 출처 (파일명 등)
        "summary": str,             # 수행 결과 요약
        "data": any,                # 실제 데이터 (pandas DataFrame의 dict 변환 등)
        "relevance": str,           # "relevant" 또는 "irrelevant"
        "relevance_reason": str,    # 관련성 판단 사유
    }

지원하는 CSV 파일:
    - store_info.csv: 상가 정보 (5개 구: 강남, 마포, 서초, 종로, 영등포)
    - foot_traffic.csv: 유동인구 데이터 (시간대·연령대별)
    - rent.csv: 임대료 데이터 (구 수준)
    - demographics.csv: 인구통계 데이터
    - business_codes.csv: 업종코드 데이터
    - dong_summary.csv: 동(洞)별 집계 데이터
"""
import os

import pandas as pd

from agent.config_loader import get_token_budget

# knowledge_base/data/ 디렉토리 경로. 모든 CSV 파일이 이 디렉토리에 위치한다.
KB_DATA_DIR = os.path.join("knowledge_base", "data")

# CSV 파일 키 → 파일명 매핑.
# analyze_query가 required_data에 키(예: "foot_traffic")를 넣으면,
# 이 매핑으로 실제 파일 경로를 찾는다.
FILE_MAP = {
    "store_info": "store_info.csv",             # 상가 정보
    "foot_traffic": "foot_traffic.csv",         # 유동인구
    "rent": "rent.csv",                         # 임대료
    "demographics": "demographics.csv",         # 인구통계
    "business_codes": "business_codes.csv",     # 업종코드
    "dong_summary": "dong_summary.csv",         # 동별 집계
    "card_consumption": "card_consumption.csv", # 카드 소비
    "estimated_sales": "estimated_sales.csv",   # 추정 매출
    "subway_ridership": "subway_ridership.csv", # 지하철 승하차
}


def pandas_query(query_analysis: dict, state: dict) -> dict:
    """pandas 표현식으로 CSV를 쿼리한다.

    사용자 의도(intent)에 따라 적절한 pandas 연산을 수행한다:
    - "compare" 의도 + district 컬럼 존재 → 구(district)별 평균 집계
    - 그 외 → 상위 20행 반환 (탐색용)

    Args:
        query_analysis: 질의 분석 결과. intent와 required_data를 참조.
        state: 현재 AgentState (이 도구에서는 사용하지 않음).

    Returns:
        표준 도구 반환 형식. data에 각 데이터셋의 쿼리 결과 포함.
    """
    intent = query_analysis.get("intent", "")
    required = query_analysis.get("required_data", [])

    results = {}
    for key in required:
        fname = FILE_MAP.get(key, f"{key}.csv")
        fpath = os.path.join(KB_DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue

        # store_info는 182K행이므로 집계 쿼리만 필요한 경우 최적화
        if key == "store_info":
            df = pd.read_csv(fpath, usecols=["district", "dong", "biz_category", "biz_detail"])
        else:
            df = pd.read_csv(fpath)
        has_district = "district" in df.columns

        # 의도 기반 쿼리 분기
        if "compare" in intent and has_district:
            # 구 비교: 구별 수치 집계
            result_df = df.groupby("district").mean(numeric_only=True).reset_index()
        elif "dong" in intent and has_district and "dong" in df.columns:
            # 동 비교: 동별 데이터 반환
            result_df = df.sort_values("dong").head(30)
        elif key == "store_info" and has_district:
            # 상가정보: 구별 업종 분포 집계
            result_df = df.groupby(["district", "biz_category"]).size().reset_index(name="count")
        elif key == "rent":
            # 임대료: 전체 반환 (22행으로 소량)
            result_df = df
        elif key == "foot_traffic" and has_district:
            # 유동인구: 구별 합산
            result_df = df.groupby("district").sum(numeric_only=True).reset_index()
        else:
            result_df = df.head(20)

        # 결과 행수 제한: 토큰 초과 방지. 잘린 경우 총 행수를 기록한다.
        max_rows = get_token_budget()["pandas_max_rows"]
        total_rows = len(result_df)
        if total_rows > max_rows:
            result_df = result_df.head(max_rows)

        results[key] = {
            "data": result_df.to_dict(orient="records"),
            "row_count": len(result_df),
            "total_rows": total_rows,          # 잘림 전 전체 행수
            "columns_used": list(result_df.columns),
        }

    return {
        "source": ", ".join(f"{k}.csv" for k in results),
        "summary": f"쿼리 완료: {len(results)}개 데이터셋",
        "data": results,
        "relevance": "relevant" if results else "irrelevant",
        "relevance_reason": f"intent '{intent}'에 필요한 데이터 쿼리",
    }
