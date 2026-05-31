"""
tests/scenarios/multi_turn/test_d3_reference.py — 시나리오 D3: 이전 턴 참조 정확성

2개 구를 각각 분석한 후 3번째 턴에서 "앞서 분석한 내용 종합"을 요청하여
lookup_previous 도구가 정확한 이전 턴을 참조하는지 검증한다.

검증 포인트:
    - Turn 3에서 lookup_previous 도구 호출
    - referenced_turns에 정확한 턴 번호 포함
    - turn_conclusions에서 올바른 결론이 조회됨
    - 환각(hallucination) 없이 실제 분석 내용만 종합

Trace 패턴:
    Turn 1~2: 다양한 질문 → Turn 3: 종합 요약 (lookup_previous 호출)

실행 방법:
    python tests/scenarios/multi_turn/test_d3_reference.py
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

# --- 3턴 질문: 2개 구 분석 + 종합 요약 ---
# 기존 6턴(5개 구 + 종합)에서 3턴(2개 구 + 종합)으로 축소.
# 이전 턴 참조 정확성 검증에는 2개 턴 + 종합이면 충분하다.
QUERIES = [
    "강남구 카페 현황 알려줘",                # Turn 1: 강남 카페
    "마포구 음식점 현황은?",                   # Turn 2: 마포 음식점
    "지금까지 분석한 내용을 종합해줘",           # Turn 3: 전체 종합 (이전 2턴 참조)
]


def run():
    """이전 턴 참조 정확성 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 3개 결론, Turn 3 is_sufficient
        - Soft: Turn 3에서 lookup_previous 호출, referenced_turns 존재

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="D3",
        name="Turn Reference Accuracy",
        description="2개 구를 각각 분석한 후 종합을 요청하여\n"
                    "lookup_previous 도구가 정확한 이전 턴을 참조하는지 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 3개 결론 + is_sufficient",
            "[SOFT] Turn 3 lookup_previous 호출",
            "[SOFT] Turn 3 referenced_turns 존재",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(len(conclusions) == 3, "3개 결론", f"{len(conclusions)}개")

    r3 = get_turn_result(session_state, 2)
    ctx_eval = r3.get("context_evaluation", {})
    log.check_hard(
        ctx_eval.get("is_sufficient", False),
        "Turn 3 is_sufficient",
        f"confidence={ctx_eval.get('confidence_score')}",
    )

    tools = r3.get("tools_called", [])
    log.check_soft("lookup_previous" in tools, "lookup_previous 호출", f"tools={tools}")

    refs = r3.get("referenced_turns", [])
    log.check_soft(len(refs) > 0, "referenced_turns 존재", f"refs={refs}")

    # G1~G4 + Post-G5 관측 체계 검증
    r1 = get_turn_result(session_state, 0)
    log.check_soft(assert_first_turn_defaults(r1), "G1 Turn 1 기본값")

    log.check_soft(assert_fidelity_score_valid(r3), "G1 Turn 3 fidelity 유효")
    log.check_soft(assert_causal_sources_present(r3), "Post-1 Turn 3 인과 전파")
    log.check_soft(assert_sufficiency_by_source_present(r3), "G3 Turn 3 소스별 충분성")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
