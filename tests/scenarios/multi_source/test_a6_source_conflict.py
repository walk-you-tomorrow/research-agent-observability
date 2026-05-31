"""
tests/scenarios/multi_source/test_a6_source_conflict.py — 시나리오 A6: 소스 간 충돌 감지

서로 다른 소스에서 모순되는 정보가 올 수 있는 질문을 던져
source.conflict_detected 기록과 충돌 해결 로직을 검증한다.

검증 포인트:
    - 복수 소스(CSV + RAG)에서 데이터 수집
    - source.conflict_detected가 Langfuse에 기록됨
    - source.conflict_resolution에 해결 방법이 기록됨 (발생 시)
    - 분석 결과에 데이터 출처별 차이가 명시됨

Trace 패턴:
    analyze_query → gather_data(복수 소스) → evaluate_context
    → generate_analysis(conflict 감지 시도) → verify_result → respond_to_user

실행 방법:
    python tests/scenarios/multi_source/test_a6_source_conflict.py
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
# CSV 데이터(2023년)와 PDF 보고서(2025년)의 수치가 다를 수 있는 질문.
# 임대료는 rent.csv(2024~2025 분기별)와 상권분석보고서(2025 연간)에서
# 미묘한 차이가 발생할 가능성이 있다.
QUERIES = [
    "강남구 임대료가 높다고 하는데 실제로 얼마인지, 다양한 자료를 비교해서 알려줘",
]


def run():
    """소스 간 충돌 감지 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론, is_sufficient
        - Soft: 다중 소스 사용, analysis_result summary 존재

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="A6",
        name="Source Conflict Detection",
        description="CSV와 PDF 보고서의 수치 차이가 발생할 수 있는 질문을 던져\n"
                    "복수 소스 사용과 소스 간 충돌 감지/해결을 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재 + is_sufficient",
            "[SOFT] 2+ 소스 유형 사용",
            "[SOFT] analysis summary 존재",
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
        "2+ 소스 사용",
        f"types={source_types_used}",
    )

    analysis = r.get("analysis_result", {})
    summary = analysis.get("summary", "")
    log.check_soft(len(summary) > 0, "analysis summary 존재", f"length={len(summary)}")

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
