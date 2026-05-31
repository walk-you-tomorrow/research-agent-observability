"""
tests/scenarios/pipeline/insufficient_regather.py — 시나리오 P1: 컨텍스트 부족 → 재수집

이 시나리오는 컨텍스트 충분성 분기(should_continue_gather)의 "부족" 경로를 검증한다.
복잡한 비교 질문을 던져 evaluate_context가 is_sufficient=false를 반환하도록 유도하고,
재수집(gather_data 루프) 후 충분해지는 과정을 확인한다.

검증 포인트:
    - evaluate_context에서 is_sufficient=false 판정 발생
    - should_continue_gather 분기 ①에서 gather_data로 복귀
    - gather_iteration > 1 기록 (재수집이 실제로 발생)
    - 재수집 후 missing_info_resolved=true (일관성 패턴 A: 이전 부족 정보 해결)
    - 재수집 후 모든 Layer 2 attribute가 재계산됨

Trace 패턴 (Langfuse에서 확인):
    analyze_query → gather_data(iteration=1) → evaluate_context(insufficient)
    → gather_data(iteration=2) → evaluate_context(sufficient)
    → generate_analysis → verify_result → respond_to_user

이 시나리오가 중요한 이유:
    에이전트가 "부족하다"고 판단했을 때 적절히 재수집할 수 있는지,
    그리고 재수집 후 품질이 실제로 개선되는지 확인한다.
    이는 Context Monitoring의 핵심 가치인 "데이터 기반 신뢰성 개선"을 검증하는 것이다.

실행 방법:
    python tests/scenarios/pipeline/insufficient_regather.py
"""
import sys
import os

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, PROJECT_ROOT)

# CWD를 프로젝트 루트로 변경 (config/, knowledge_base/ 등 상대 경로 해결)
os.chdir(PROJECT_ROOT)

from main import run_session, load_config
from tests.scenarios.assert_helpers import (
    get_turn_result,
    assert_first_turn_defaults,
    assert_evolution_metrics_present,
    assert_exclusion_reasons_structure,
    assert_sufficiency_by_source_present,
    ScenarioTestLog,
)


# --- 시나리오 질문 ---
# 의도적으로 여러 데이터 소스의 교차 분석이 필요한 복잡한 질문.
# "합정동과 상수동의 임대료 차이를 유동인구 대비로 비교" →
# rent.csv + foot_traffic.csv + dong_summary.csv 모두 필요.
# 초기 수집에서 일부 데이터가 부족할 가능성이 높다.
QUERIES = [
    "합정동과 상수동의 임대료 차이를 유동인구 대비로 비교해줘",
]


def run():
    """재수집 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론
        - Soft: gather_iteration > 1, missing_info_resolved=True,
                confidence_delta > 0

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="P1",
        name="Insufficient → Re-gather",
        description="복잡한 교차 분석 질문으로 초기 수집이 부족하도록 유도하고,\n"
                    "재수집(gather_iteration > 1) 후 충분해지는 과정을 검증한다.\n"
                    "Context Monitoring의 핵심인 '데이터 기반 신뢰성 개선'을 확인.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재",
            "[SOFT] gather_iteration > 1 (재수집 발생)",
            "[SOFT] missing_info_resolved = True",
            "[SOFT] confidence_delta > 0 (신뢰도 개선)",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(len(conclusions) >= 1, "결론 존재", f"{len(conclusions)}개")

    r = get_turn_result(session_state, 0)

    iteration = r.get("gather_iteration", 0)
    log.check_soft(iteration > 1, "gather_iteration > 1", f"iteration={iteration}")

    ctx_meta = r.get("context_metadata", {})
    log.check_soft(
        ctx_meta.get("missing_info_resolved", False),
        "missing_info_resolved",
        f"value={ctx_meta.get('missing_info_resolved', False)}",
    )

    confidence_delta = ctx_meta.get("confidence_delta", 0)
    log.check_soft(
        confidence_delta > 0,
        "confidence_delta > 0",
        f"delta={confidence_delta}",
    )

    # G1~G4 + Post-G5 관측 체계 검증
    log.check_soft(assert_first_turn_defaults(r), "G1 첫 턴 기본값")
    log.check_soft(assert_evolution_metrics_present(r), "G3 진화 지표 존재")
    log.check_soft(assert_exclusion_reasons_structure(r), "G5 exclusion_reasons 구조")
    log.check_soft(assert_sufficiency_by_source_present(r), "G3 소스별 충분성")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
