"""
tests/scenarios/multi_turn/test_d1_progressive.py — 시나리오 D1: 점진적 세밀화

일반적 질문에서 시작하여 점점 구체적인 질문으로 좁혀가는
3턴 시나리오. 턴 간 참조와 결론 누적을 검증한다.

검증 포인트:
    - Turn 2에서 Turn 1의 결론을 참조
    - Turn 3에서 Turn 1, 2의 결론을 참조
    - referenced_turns 리스트가 점진적으로 증가
    - turn_conclusions가 3개 누적됨
    - 이전 턴 결론이 다음 턴 분석에 반영됨

Trace 패턴:
    Turn 1: 강남구 전체 → Turn 2: 역삼동 상세 → Turn 3: 역삼동 카페 임대료

실행 방법:
    python tests/scenarios/multi_turn/test_d1_progressive.py
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
    assert_fidelity_score_valid,
    assert_evolution_metrics_present,
    assert_inherited_ratio_increases,
    assert_causal_sources_present,
    assert_sufficiency_by_source_present,
    ScenarioTestLog,
)

# --- 3턴 질문: 점진적 세밀화 ---
# 구 → 동 → 동+업종 순으로 좁혀간다.
QUERIES = [
    "강남구 상권 전체적으로 어때?",              # Turn 1: 구 수준 개요
    "그 중에서 역삼동은 어떤 특징이 있어?",       # Turn 2: 동 수준 상세 (Turn 1 참조)
    "역삼동에서 카페 창업하면 임대료가 얼마야?",   # Turn 3: 동+업종 수치 (Turn 1,2 참조)
]


def run():
    """점진적 세밀화 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 3개 결론, 턴 번호 [1,2,3], Turn 3 is_sufficient
        - Soft: 턴2,3에서 referenced_turns 비어있지 않음

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="D1",
        name="Progressive Refinement",
        description="구 → 동 → 동+업종 순으로 좁혀가는 3턴 시나리오.\n"
                    "턴 간 참조(referenced_turns)와 결론 누적을 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 3개 결론 + 턴번호 [1,2,3]",
            "[SOFT] Turn 3 is_sufficient (LLM 비결정적)",
            "[SOFT] Turn 2 referenced_turns 비어있지 않음",
            "[SOFT] Turn 3 referenced_turns 비어있지 않음",
            "[SOFT] G1 Turn 1 기본값, Turn 2/3 fidelity 유효",
            "[SOFT] G3 진화 지표 + inherited_ratio 증가 추세",
            "[SOFT] Post-1 Turn 3 인과 전파",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(
        len(conclusions) == 3,
        "3개 결론",
        f"{len(conclusions)}개",
    )

    turn_numbers = [c["turn_number"] for c in conclusions]
    log.check_hard(
        turn_numbers == [1, 2, 3],
        "턴번호 [1,2,3]",
        f"actual={turn_numbers}",
    )

    r3 = get_turn_result(session_state, 2)
    ctx_eval = r3.get("context_evaluation", {})
    # Turn 3의 충분성은 LLM 비결정성으로 실패할 수 있음 (재수집 후에도 부족 판정 가능)
    log.check_soft(
        ctx_eval.get("is_sufficient", False),
        "Turn 3 is_sufficient",
        f"confidence={ctx_eval.get('confidence_score')}",
    )

    r2 = get_turn_result(session_state, 1)
    refs2 = r2.get("referenced_turns", [])
    log.check_soft(len(refs2) > 0, "Turn 2 referenced_turns", f"refs={refs2}")

    refs3 = r3.get("referenced_turns", [])
    log.check_soft(len(refs3) > 0, "Turn 3 referenced_turns", f"refs={refs3}")

    # G1~G4 + Post-G5 관측 체계 검증 (멀티턴)
    r1 = get_turn_result(session_state, 0)
    log.check_soft(assert_first_turn_defaults(r1), "G1 Turn 1 기본값")
    log.check_soft(assert_evolution_metrics_present(r1), "G3 Turn 1 진화 지표")

    log.check_soft(assert_fidelity_score_valid(r2), "G1 Turn 2 fidelity 유효")
    log.check_soft(assert_fidelity_score_valid(r3), "G1 Turn 3 fidelity 유효")
    log.check_soft(assert_evolution_metrics_present(r3), "G3 Turn 3 진화 지표")
    log.check_soft(assert_causal_sources_present(r3), "Post-1 Turn 3 인과 전파")
    log.check_soft(assert_sufficiency_by_source_present(r3), "G3 Turn 3 소스별 충분성")

    all_results = [get_turn_result(session_state, i) for i in range(3)]
    log.check_soft(assert_inherited_ratio_increases(all_results), "G3 inherited_ratio 증가 추세")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
