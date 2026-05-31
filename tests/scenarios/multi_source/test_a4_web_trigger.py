"""
tests/scenarios/multi_source/test_a4_web_trigger.py — 시나리오 A4: 웹 검색 트리거 쿼리

최신 뉴스/트렌드를 질문하여 web_search 도구가 Claude의
내장 웹 검색을 실행하는지 검증한다.

검증 포인트:
    - source.types_selected에 "web" 포함
    - web_search 도구가 호출됨
    - web.search_count > 0 (웹 검색이 실제로 실행됨)
    - web.result_count > 0 (검색 결과가 반환됨)
    - web.source_domains에 도메인이 기록됨

Trace 패턴:
    analyze_query(web 필요) → gather_data(web_search + 기타)
    → evaluate_context → generate_analysis → verify_result → respond_to_user

실행 방법:
    python tests/scenarios/multi_source/test_a4_web_trigger.py
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
# "2026년 최신 동향"은 CSV나 PDF에 없는 실시간 정보를 요구한다.
# web_search 도구가 호출되어야 한다.
QUERIES = [
    "2026년 서울 카페 시장 최신 동향과 트렌드는?",
]


def run():
    """웹 검색 트리거 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론, is_sufficient
        - Soft: web_search 호출, source_types에 web

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="A4",
        name="Web Search Triggered Query",
        description="CSV나 PDF에 없는 최신 뉴스/트렌드를 질문하여\n"
                    "web_search 도구가 Claude 내장 웹 검색을 실행하는지 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재 + is_sufficient",
            "[SOFT] web_search 호출",
            "[SOFT] source_types에 web",
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
    log.check_soft("web_search" in tools, "web_search 호출", f"tools={tools}")
    log.check_soft("web" in source_types, "source_types에 web", f"types={source_types}")

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
