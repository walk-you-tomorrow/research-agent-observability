"""
tests/scenarios/multi_source/test_a3_api_trigger.py — 시나리오 A3: API 트리거 쿼리

CSV에 없는 데이터(상권변화지표)를 질문하여 api_query 도구가
서울시 상권분석서비스 API를 호출하는지 검증한다.

검증 포인트:
    - source.types_selected에 "api" 포함
    - api_query 도구가 호출됨
    - gather.api_called에 API 키(예: commercial_change)가 기록됨
    - gather.api_response_count > 0 (API가 실제 데이터를 반환)
    - API 데이터가 gathered_data에 포함됨

Trace 패턴:
    analyze_query(api 필요) → gather_data(api_query + 기타)
    → evaluate_context → generate_analysis → verify_result → respond_to_user

실행 방법:
    python tests/scenarios/multi_source/test_a3_api_trigger.py
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
# "상권이 성장하고 있나?"는 상권변화지표(OA-15576)가 필요한 질문이다.
# CSV에 없으므로 api_query(commercial_change)가 호출되어야 한다.
QUERIES = [
    "마포구 상권이 성장하고 있나? 최근 상권변화지표를 보여줘",
]


def run():
    """API 트리거 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론, is_sufficient
        - Soft: api_query 호출, source_types에 api, gathered_data에 api 항목

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="A3",
        name="API-Triggered Query",
        description="CSV에 없는 상권변화지표를 질문하여\n"
                    "서울시 상권분석서비스 API(api_query)가 호출되는지 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재 + is_sufficient",
            "[SOFT] api_query 호출",
            "[SOFT] source_types에 api",
            "[SOFT] gathered_data에 api 항목",
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
    log.check_soft("api_query" in tools, "api_query 호출", f"tools={tools}")
    log.check_soft("api" in source_types, "source_types에 api", f"types={source_types}")

    gathered = r.get("gathered_data", [])
    has_api_data = any(g.get("tool_used") == "api_query" for g in gathered)
    log.check_soft(has_api_data, "gathered_data에 api 항목", f"gathered={len(gathered)}개")

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
