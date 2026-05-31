"""
tests/scenarios/multi_source/test_a1_csv_only.py — 시나리오 A1: CSV 전용 쿼리

CSV 데이터만으로 충분히 답할 수 있는 단순 수치 질문을 던져
소스 선택이 CSV로 한정되는지 검증한다.

검증 포인트:
    - source.types_selected에 "csv"만 포함 (또는 csv가 주 소스)
    - pandas_query 도구가 호출됨
    - rag_search, web_search, api_query는 호출되지 않을 것을 기대
    - 정확한 수치가 응답에 포함됨

Trace 패턴 (Langfuse에서 확인):
    analyze_query(csv 전용) → gather_data(pandas_query만) → evaluate_context
    → generate_analysis → verify_result → respond_to_user

실행 방법:
    python tests/scenarios/multi_source/test_a1_csv_only.py
"""
import sys
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from main import run_session, load_config
from tests.scenarios.assert_helpers import (
    get_turn_result,
    assert_first_turn_defaults,
    assert_evolution_metrics_present,
    assert_exclusion_reasons_structure,
    assert_information_density_positive,
    assert_sufficiency_by_source_present,
    ScenarioTestLog,
)

# --- 시나리오 질문 ---
# 단순 수치 질문: CSV(store_info.csv)에서 바로 답할 수 있다.
# LightRAG, 웹 검색, API가 필요하지 않은 질문.
QUERIES = [
    "강남구에 카페 매장이 몇 개 있어?",
]


def run():
    """CSV 전용 쿼리 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론, is_sufficient
        - Soft: pandas_query 호출, source_types에 csv, rag/web 미호출

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="A1",
        name="CSV-Only Query",
        description="CSV 데이터만으로 충분히 답할 수 있는 단순 수치 질문을 던져\n"
                    "소스 선택이 CSV로 한정되는지 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재 + is_sufficient",
            "[SOFT] pandas_query 호출",
            "[SOFT] source_types에 csv",
            "[SOFT] rag_search / web_search 미호출",
            "[SOFT] G1 첫 턴 기본값 (fidelity=1.0, inherited=0.0)",
            "[SOFT] G3 진화 지표 존재",
            "[SOFT] G5 exclusion_reasons 구조",
            "[SOFT] Post-3 information_density > 0",
            "[SOFT] G3 sufficiency_by_source 존재",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(len(conclusions) >= 1, "결론 존재", f"{len(conclusions)}개")

    r = get_turn_result(session_state, 0)
    ctx_eval = r.get("context_evaluation", {})
    log.check_hard(
        ctx_eval.get("is_sufficient", False),
        "is_sufficient",
        f"confidence={ctx_eval.get('confidence_score')}",
    )

    tools = r.get("tools_called", [])
    source_types = r.get("query_analysis", {}).get("source_types", [])
    log.check_soft("pandas_query" in tools, "pandas_query 호출", f"tools={tools}")
    log.check_soft("csv" in source_types, "source_types에 csv", f"types={source_types}")
    log.check_soft("rag_search" not in tools, "rag_search 미호출", f"tools={tools}")
    log.check_soft("web_search" not in tools, "web_search 미호출", f"tools={tools}")

    # G1~G4 + Post-G5 관측 체계 검증
    log.check_soft(assert_first_turn_defaults(r), "G1 첫 턴 기본값")
    log.check_soft(assert_evolution_metrics_present(r), "G3 진화 지표 존재")
    log.check_soft(assert_exclusion_reasons_structure(r), "G5 exclusion_reasons 구조")
    log.check_soft(assert_information_density_positive(r), "Post-3 정보 밀도 > 0")
    log.check_soft(assert_sufficiency_by_source_present(r), "G3 소스별 충분성")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
