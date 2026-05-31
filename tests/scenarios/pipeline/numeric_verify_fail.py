"""
tests/scenarios/pipeline/numeric_verify_fail.py — 시나리오 P2: 수치 검증 실패 → 재생성

이 시나리오는 검증 분기(route_after_verify)의 "fail_numeric" 경로를 검증한다.
수치 비교가 핵심인 질문을 던져 verify_result에서 수치 검증 실패가 발생하도록 유도하고,
데이터 재수집 + 분석 재생성 후 검증을 통과하는 과정을 확인한다.

검증 포인트:
    - verify.numeric_check_passed=false 기록
    - verify.overall_verdict="fail_numeric" 기록
    - route_after_verify 분기 ②에서 gather_data로 복귀
    - 재수집 후 generate_analysis 재실행
    - 2차 verify_result에서 통과 (overall_verdict="pass")
    - verify_retry_count 증가 기록

Trace 패턴 (Langfuse에서 확인):
    1차: analyze_query → gather_data → evaluate_context(sufficient)
         → generate_analysis → verify_result(fail_numeric)
    2차: → gather_data(재수집) → evaluate_context
         → generate_analysis(재생성) → verify_result(pass)
         → respond_to_user

이 시나리오가 중요한 이유:
    에이전트가 생성한 분석에서 수치적 오류를 스스로 감지하고 교정할 수 있는지 확인한다.
    이는 "이중 검증" 메커니즘의 실효성을 검증하는 것으로,
    AI Agent의 정확성을 자동으로 보장하는 핵심 장치이다.

실행 방법:
    python tests/scenarios/pipeline/numeric_verify_fail.py
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
    ScenarioTestLog,
)


# --- 시나리오 질문 ---
# "정확한 수치를 비교"라는 키워드로 수치 정확성을 강조하는 질문.
# verify_result의 수치 검증(STEP 1)에서 불일치가 발생할 가능성이 높다.
QUERIES = [
    "강남구와 마포구의 카페 매출 정확한 수치를 비교해줘",
]


def run():
    """수치 검증 실패 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론
        - Soft: verify_retry_count >= 1, 최종 verdict=pass

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="P2",
        name="Numeric Verify Fail → Re-generate",
        description="수치 비교가 핵심인 질문으로 verify_result 수치 검증 실패를 유도하고,\n"
                    "데이터 재수집 + 분석 재생성 후 검증 통과 과정을 검증한다.\n"
                    "'이중 검증' 메커니즘의 실효성을 확인하는 핵심 시나리오.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재",
            "[SOFT] verify_retry_count >= 1",
            "[SOFT] 최종 verdict = pass",
        ],
    )
    log.print_header()

    config = load_config()
    session_state = run_session(QUERIES, config=config)

    # --- Assertions ---
    conclusions = session_state.get("turn_conclusions", [])
    log.check_hard(len(conclusions) >= 1, "결론 존재", f"{len(conclusions)}개")

    r = get_turn_result(session_state, 0)

    retry_count = r.get("verify_retry_count", 0)
    log.check_soft(retry_count >= 1, "verify_retry >= 1", f"count={retry_count}")

    verification = r.get("verification", {})
    verdict = verification.get("overall_verdict", "N/A")
    log.check_soft(verdict == "pass", "verdict = pass", f"actual={verdict}")

    # G1~G4 + Post-G5 관측 체계 검증
    log.check_soft(assert_first_turn_defaults(r), "G1 첫 턴 기본값")
    log.check_soft(assert_evolution_metrics_present(r), "G3 진화 지표 존재")

    log.print_turn_details(session_state)
    log.print_summary()
    return session_state


if __name__ == "__main__":
    run()
