"""
tests/scenarios/multi_turn/test_d2_contradiction.py — 시나리오 D2: 교차 턴 모순 + 해결

Turn 1에서 특정 지역을 추천한 후 Turn 2에서 반대 논점을 제시하여
모순 감지(contradicts_previous=true) 및 해결을 검증한다.

검증 포인트:
    - Turn 2에서 analysis.contradicts_previous=true 기록
    - analysis.contradiction_resolved=true (모순이 명시적으로 해결됨)
    - analysis.previous_conclusion에 Turn 1 결론이 기록됨
    - analysis.referenced_turns에 Turn 1 번호 포함

Trace 패턴:
    Turn 1: 마포구 추천 → Turn 2: 강남구가 더 나은 이유 (모순 발생 → 해결)

실행 방법:
    python tests/scenarios/multi_turn/test_d2_contradiction.py
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
    ScenarioTestLog,
)

# --- 2턴 질문: 모순 트리거 ---
# Turn 1에서 마포구를 추천하고, Turn 2에서 강남구가 낫다는 반론을 제기한다.
QUERIES = [
    "카페 창업 하려는데 마포구가 좋을까?",
    "근데 강남구가 매출이 훨씬 높잖아. 강남이 더 나은 거 아니야?",
]


def run():
    """모순 감지 + 해결 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 2개 결론
        - Soft: Turn 2에서 contradicts_previous=True, contradicts_turn 존재

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="D2",
        name="Contradiction & Resolution",
        description="Turn 1에서 마포구를 추천한 후 Turn 2에서 반대 논점을 제시하여\n"
                    "모순 감지(contradicts_previous) 및 해결을 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 2개 결론",
            "[SOFT] Turn 2 contradicts_previous=True",
            "[SOFT] Turn 2 결론에 contradicts_turn 존재",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(len(conclusions) == 2, "2개 결론", f"{len(conclusions)}개")

    r2 = get_turn_result(session_state, 1)
    contradicts = r2.get("contradicts_previous", False)
    log.check_soft(
        contradicts is True,
        "contradicts_previous=True",
        f"actual={contradicts}",
    )

    turn2_conclusion = conclusions[1] if len(conclusions) > 1 else {}
    has_contradiction = turn2_conclusion.get("contradicts_turn") is not None
    log.check_soft(
        has_contradiction,
        "contradicts_turn 존재",
        f"contradicts_turn={turn2_conclusion.get('contradicts_turn')}",
    )

    # G1~G4 + Post-G5 관측 체계 검증
    r1 = get_turn_result(session_state, 0)
    log.check_soft(assert_first_turn_defaults(r1), "G1 Turn 1 기본값")
    log.check_soft(assert_fidelity_score_valid(r2), "G1 Turn 2 fidelity 유효")
    log.check_soft(assert_evolution_metrics_present(r2), "G3 Turn 2 진화 지표")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
