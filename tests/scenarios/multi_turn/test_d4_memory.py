"""
tests/scenarios/multi_turn/test_d4_memory.py — 시나리오 D4: 세션 메모리 지속성

3턴 세션에서 turn_conclusions이 정확히 턴 수만큼 누적되는지,
그리고 이전 턴 상태(messages)가 올바르게 전달되는지 검증한다.

검증 포인트:
    - 3턴 완료 후 turn_conclusions 리스트 길이 == 3
    - 각 결론의 turn_number가 1~3으로 연속
    - messages가 턴 간 누적됨 (6개 이상, AI+Human 쌍)
    - current_turn이 최종적으로 3
    - turn_results가 3개 축적됨

Trace 패턴:
    Turn 1~3: 순차 질문, 상태 누적

실행 방법:
    python tests/scenarios/multi_turn/test_d4_memory.py
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

# --- 3턴 질문: 독립적이지만 세션 컨텍스트 누적 ---
# 기존 5턴에서 3턴으로 축소. 누적 검증에는 3턴이면 충분하다.
QUERIES = [
    "서울에서 카페 창업 어디가 좋아?",
    "강남구 상가 임대료 알려줘",
    "마포구 유동인구는?",
]


def run():
    """세션 메모리 지속성 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 3개 결론, 턴번호 [1,2,3], messages >= 6,
                current_turn=3, turn_results=3개

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="D4",
        name="Session Memory Persistence",
        description="3턴 세션에서 turn_conclusions 누적, messages 누적,\n"
                    "current_turn 갱신, turn_results 축적을 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 3개 결론 + 턴번호 [1,2,3]",
            "[HARD] messages >= 6 (3턴 × Human+AI 쌍)",
            "[HARD] current_turn = 3",
            "[HARD] turn_results = 3개",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(len(conclusions) == 3, "3개 결론", f"{len(conclusions)}개")

    turn_numbers = [c["turn_number"] for c in conclusions]
    log.check_hard(
        turn_numbers == [1, 2, 3],
        "턴번호 [1,2,3]",
        f"actual={turn_numbers}",
    )

    messages = session_state.get("messages", [])
    log.check_hard(len(messages) >= 6, "messages >= 6", f"{len(messages)}개")

    log.check_hard(
        session_state.get("current_turn") == 3,
        "current_turn = 3",
        f"actual={session_state.get('current_turn')}",
    )

    turn_results = session_state.get("turn_results", [])
    log.check_hard(
        len(turn_results) == 3,
        "turn_results = 3개",
        f"{len(turn_results)}개",
    )

    # G1~G4 + Post-G5 관측 체계 검증
    r1 = get_turn_result(session_state, 0)
    log.check_soft(assert_first_turn_defaults(r1), "G1 Turn 1 기본값")
    log.check_soft(assert_evolution_metrics_present(r1), "G3 Turn 1 진화 지표")

    r3 = get_turn_result(session_state, 2)
    log.check_soft(assert_fidelity_score_valid(r3), "G1 Turn 3 fidelity 유효")

    all_results = [get_turn_result(session_state, i) for i in range(3)]
    log.check_soft(assert_inherited_ratio_increases(all_results), "G3 inherited_ratio 증가 추세")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
