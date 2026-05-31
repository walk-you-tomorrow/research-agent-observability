"""
tests/scenarios/multi_source/test_a2_rag_only.py — 시나리오 A2: RAG 전용 쿼리

상권 트렌드/정책 분석 등 LightRAG Knowledge Graph에서
시맨틱 검색으로 답할 수 있는 질문을 던져
소스 선택이 RAG 위주인지 검증한다.

검증 포인트:
    - source.types_selected에 "rag" 포함
    - rag_search 또는 rag_deep_read 등 RAG 도구가 호출됨
    - KG 기반 엔티티/관계 정보가 응답에 포함됨

Trace 패턴:
    analyze_query(rag 위주) → gather_data(rag_search, rag_deep_read)
    → evaluate_context → generate_analysis → verify_result → respond_to_user

실행 방법:
    python tests/scenarios/multi_source/test_a2_rag_only.py
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
# 상권분석보고서(PDF)에서 인덱싱된 정보를 검색하는 질문.
# CSV 수치보다 정성적 분석이 필요한 질문으로, RAG가 주 소스가 된다.
QUERIES = [
    "마포구 상권의 최근 트렌드와 성장 동력은 무엇인가?",
]


def run():
    """RAG 전용 쿼리 시나리오를 실행하고 검증한다.

    검증 내용:
        - Hard: 최소 1개 결론, is_sufficient
        - Soft: RAG 도구 호출, source_types에 rag

    Returns:
        최종 세션 상태 딕셔너리.
    """
    log = ScenarioTestLog(
        scenario_id="A2",
        name="RAG-Only Query",
        description="상권분석보고서(PDF) 기반 정성적 분석 질문을 던져\n"
                    "LightRAG Knowledge Graph가 주 소스로 선택되는지 검증한다.",
        queries=QUERIES,
        key_checks=[
            "[HARD] 결론 존재 + is_sufficient",
            "[SOFT] RAG 도구 호출 (rag_search/rag_deep_read/rag_global_summary/rag_compare)",
            "[SOFT] source_types에 rag",
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
    rag_tools = ["rag_search", "rag_deep_read", "rag_global_summary", "rag_compare"]
    log.check_soft(
        any(t in tools for t in rag_tools),
        "RAG 도구 호출",
        f"tools={tools}",
    )
    source_types = r.get("query_analysis", {}).get("source_types", [])
    log.check_soft("rag" in source_types, "source_types에 rag", f"types={source_types}")

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
