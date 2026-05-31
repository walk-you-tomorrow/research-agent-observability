"""
scripts/phase_c_step4_validate_trace.py — Phase C Step 4: trace 검증

Langfuse에서 sess_f55d9481의 5턴 trace를 가져와
v3 신규/REDEFINE attribute가 정상 기록되었는지 + 폐기 attribute가 부재한지 검증.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.data_loader import load_enriched_session_data

SESSION_ID = "sess_f55d9481"

# v3 신규 + REDEFINE — trace에 기록되어야 함
EXPECTED_V3_ATTRS = [
    "analysis.conflict_tracking",       # REDEFINE: dict 통합
    "context.redundancy_ratio",          # REDEFINE: 임베딩 cosine
    "source.conflict_detected",          # REDEFINE: 정량 + LLM
    "context.contributing_turns",        # REDEFINE: referenced_turns 기반
    "analysis.conclusion_utilization",   # REDEFINE: 조건부
    "context.truncated_items",           # REDEFINE: list[dict]
    "context.fidelity_score",            # 명세 강화 (Pattern C)
    "response.hallucination_detected",   # 명세 강화 (Pattern D)
    "context.total_tokens",              # 효율성 자원
    "turn.tool_call_count",
    "turn.wall_time_ms",
]

# v3 폐기 — trace에 없어야 함 (코드에서 기록 제거됨)
DEPRECATED_ATTRS = [
    "analysis.contradicts_previous",
    "analysis.contradiction_resolved",
    "analysis.previous_conclusion",
    "context.continuity_score",
    "context.new_data_ratio",
    "context.token_delta",
    "context.rot_gate_pruned_tokens",
    "context.source.turn_conclusions_tokens",
    "context.messages_tokens",
    "context.sufficiency_by_source",
    "context.truncation_reasons",
    "response.compression_ratio",
    "response.conditions_preserved",
    "response.conditions_detail",
    "response.conclusion_token_count",
    "analysis.claims_count",
    "analysis.data_references_count",
    "gather.api_response_count",
    "gather.strategy",
    "query.referenced_turn_numbers",
    "source.types_selected",
    "source.conflict_resolution",
    "web.result_count",
    "web.search_count",
]


def main() -> None:
    print("=" * 70)
    print(f"Phase C Step 4 — Trace 검증 ({SESSION_ID})")
    print("=" * 70)

    turns = load_enriched_session_data(SESSION_ID)
    if not turns:
        print(f"❌ 세션 {SESSION_ID}에 trace 없음 (Langfuse 동기화 지연일 수 있음)")
        sys.exit(1)

    print(f"\n[1] Trace 발견: {len(turns)} turns")
    for t in turns:
        meta_keys = len(t.get("metadata", {}))
        print(f"  Turn {t.get('turn_number')}: {meta_keys} metadata keys, "
              f"trace_id={t.get('trace_id', 'N/A')[:16]}...")

    # 모든 turn의 metadata를 합쳐 attribute 존재 검증
    all_keys: set[str] = set()
    for t in turns:
        all_keys.update(t.get("metadata", {}).keys())

    print(f"\n[2] 누적 metadata keys: {len(all_keys)}")

    # ── v3 신규/REDEFINE 검증 ──
    print(f"\n[3] v3 신규/REDEFINE attribute 기록 검증 (총 {len(EXPECTED_V3_ATTRS)}개)")
    missing_v3 = []
    for attr in EXPECTED_V3_ATTRS:
        if attr in all_keys:
            print(f"  ✓ {attr}")
        else:
            print(f"  ❌ {attr} — 누락")
            missing_v3.append(attr)

    # ── 폐기 attribute 부재 검증 ──
    print(f"\n[4] 폐기 attribute 부재 검증 (총 {len(DEPRECATED_ATTRS)}개)")
    leaked_deprecated = []
    for attr in DEPRECATED_ATTRS:
        if attr in all_keys:
            print(f"  ⚠ {attr} — 여전히 trace에 기록됨 (코드 정리 누락)")
            leaked_deprecated.append(attr)
    if not leaked_deprecated:
        print("  ✓ 24개 폐기 attribute 모두 trace에서 제거됨")

    # ── conflict_tracking 구조 검증 ──
    print(f"\n[5] analysis.conflict_tracking dict 구조 검증")
    for t in turns:
        ct = t.get("metadata", {}).get("analysis.conflict_tracking")
        tn = t.get("turn_number")
        if ct is None:
            print(f"  Turn {tn}: 키 없음")
            continue
        if not isinstance(ct, dict):
            print(f"  ⚠ Turn {tn}: dict 아님 (type={type(ct).__name__})")
            continue
        detected = ct.get("detected")
        resolution = ct.get("resolution") or {}
        has_expl = resolution.get("has_explanation")
        summary = (resolution.get("conflict_summary") or "")[:60]
        print(f"  Turn {tn}: detected={detected}, has_explanation={has_expl}, summary='{summary}...'")

    # ── redundancy_ratio 검증 (임베딩 작동) ──
    print(f"\n[6] context.redundancy_ratio 임베딩 cosine 작동 검증")
    for t in turns:
        r = t.get("metadata", {}).get("context.redundancy_ratio")
        tn = t.get("turn_number")
        print(f"  Turn {tn}: redundancy_ratio={r}")

    # ── source.conflict_detected ──
    print(f"\n[7] source.conflict_detected (v3 정량 측정)")
    for t in turns:
        c = t.get("metadata", {}).get("source.conflict_detected")
        tn = t.get("turn_number")
        print(f"  Turn {tn}: conflict_detected={c}")

    # ── 결과 ──
    print("\n" + "=" * 70)
    if not missing_v3 and not leaked_deprecated:
        print("✅ Trace 검증 통과 — Phase C Step 4 trace 단계 OK")
    else:
        print(f"⚠ 검증 이슈: missing v3={len(missing_v3)}, leaked deprecated={len(leaked_deprecated)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
