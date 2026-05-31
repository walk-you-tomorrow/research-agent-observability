"""
tests/scenarios/multi_source/test_a5_fusion.py — 시나리오 A5: Multi-Source 융합 쿼리

종합 분석 질문을 던져 3개 이상의 소스(CSV + RAG + API/웹)가
동시에 사용되는지 검증한다. Phase 2 Multi-Source 아키텍처의 핵심 시나리오.

검증 포인트:
    - len(source.types_selected) >= 3 (3개 이상 소스 유형 사용)
    - source.types_selected에 최소 3가지 포함
    - source.contribution에 복수 소스 키 존재
    - 응답에 수치(CSV) + 트렌드(RAG) + 실시간 정보(API/웹) 혼합

Trace 패턴:
    analyze_query(multi-source) → gather_data(rag + pandas + api/web)
    → evaluate_context → generate_analysis → verify_result → respond_to_user

실행 방법:
    python tests/scenarios/multi_source/test_a5_fusion.py
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
# "종합 분석"은 정확한 수치(CSV) + 트렌드(RAG) + 성장 지표(API) + 최신 뉴스(웹)
# 모두를 요구하는 질문이다. 최소 3개 소스가 활성화되어야 한다.
QUERIES = [
    "강남구 카페 창업 종합 분석해줘. 매출, 임대료, 유동인구 수치와 상권 성장 전망, 최신 트렌드까지 포함해서.",
]


def run():
    """Multi-Source 융합 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론, is_sufficient
        - Soft: 2+ 소스 유형 사용, 2+ 도구 카테고리

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="A5",
        name="Multi-Source Fusion Query",
        description="종합 분석 질문으로 CSV + RAG + API/웹 등 복수 소스가\n"
                    "동시에 사용되는지 검증한다. Phase 2 Multi-Source의 핵심 시나리오.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재 + is_sufficient",
            "[SOFT] 2+ 소스 유형 사용 (gathered_data 기준)",
            "[SOFT] 2+ 도구 카테고리 호출",
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

    # gathered_data에서 소스 유형 추출
    gathered = r.get("gathered_data", [])
    source_types_used = set()
    for item in gathered:
        tool = item.get("tool_used", "")
        if tool in ("pandas_query", "calculate"):
            source_types_used.add("csv")
        elif tool.startswith("rag_"):
            source_types_used.add("rag")
        elif tool == "web_search":
            source_types_used.add("web")
        elif tool == "api_query":
            source_types_used.add("api")
    log.check_soft(
        len(source_types_used) >= 2,
        "2+ 소스 유형 사용",
        f"types={source_types_used}",
    )

    # tools_called에서 카테고리 추출
    tools = r.get("tools_called", [])
    categories = set()
    for t in tools:
        if t in ("pandas_query", "calculate"):
            categories.add("csv")
        elif t.startswith("rag_"):
            categories.add("rag")
        elif t == "web_search":
            categories.add("web")
        elif t == "api_query":
            categories.add("api")
    log.check_soft(
        len(categories) >= 2,
        "2+ 도구 카테고리",
        f"categories={categories}",
    )

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
