"""
scripts/phase_c_step4_verify.py — Phase C Step 4: 5턴 trace 검증 (1세션)

목적:
    v3 yaml swap 후 5턴 시나리오 1세션을 실행하여 v3 attribute가 trace에 정상 기록되는지 검증.
    옵션 A 단계적 실측 1단계 (~$0.5 비용 추정).

검증 항목:
    1. 5턴 모두 정상 완료 (오류 없이)
    2. v3 신규/REDEFINE attribute가 metadata에 기록되는지:
       - analysis.conflict_tracking (dict 구조)
       - context.redundancy_ratio (임베딩 cosine 결과)
       - source.conflict_detected (정량 측정)
       - context.contributing_turns (방향 정정 후 referenced_turns 기반)
       - analysis.conclusion_utilization (조건부 None/float)
       - context.truncated_items (list[dict] 구조)
    3. v3 폐기 attribute가 metadata에 없는지:
       - analysis.contradicts_previous, contradiction_resolved, previous_conclusion
       - context.continuity_score, new_data_ratio, token_delta
       - response.compression_ratio, conditions_preserved 등

세션 ID는 출력 후 Langfuse에서 확인 가능.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_session, load_config

# 5턴 강남구 카페 창업 시나리오 (Phase 4 reference 시나리오와 유사)
QUERIES_5TURN = [
    "강남구에서 카페 창업하기 좋은 동은 어디인가?",
    "마포구 합정동과 비교하면 어떤 차이가 있나?",
    "두 지역의 임대료 차이는 얼마나 되나?",
    "유동인구 추세는 어떻게 다른가?",
    "지금까지의 분석을 종합하면 어디를 추천하는가?",
]


def main() -> None:
    print("=" * 70)
    print("Phase C Step 4 — 5턴 trace 검증 (1세션)")
    print("=" * 70)
    print(f"Queries: {len(QUERIES_5TURN)} turns")
    for i, q in enumerate(QUERIES_5TURN, 1):
        print(f"  T{i}: {q}")
    print("-" * 70)

    config = load_config()
    final_state = run_session(QUERIES_5TURN, config=config)

    print("\n" + "=" * 70)
    print("✓ 5턴 시나리오 완료")
    print("=" * 70)
    print(f"current_turn: {final_state.get('current_turn')}")
    print(f"turn_conclusions: {len(final_state.get('turn_conclusions', []))}")
    print("\nLangfuse에서 trace를 확인하여 v3 attribute 정상 기록 여부를 검증하시오.")


if __name__ == "__main__":
    main()
