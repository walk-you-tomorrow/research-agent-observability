"""
scripts/a12_dead_weight_analysis.py — A-12: dead_weight > 0 검증 (n=24 trace 재분석)

목표:
    1. dead_weight_tokens 분포 (5세션 25 turns)
    2. Rot Gate 트리거 빈도 + 정합성 검증 (rot_gate_triggered 조건)
    3. dead_weight > 0 케이스의 efficiency score 패턴
    4. rot_risk vs dead_weight 상관 (선후관계)
    5. v3 변경 정합성: rot_gate_pruned_tokens 폐기 후 dead_weight_tokens가 신호 역할
       — derived 관계 (`pruned ≈ dead × triggered`) 검증

비용: 0 (이미 측정된 trace 재사용)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.data_loader import load_enriched_session_data

SESSIONS = [
    ("gangnam", "sess_f55d9481"),
    ("mapo", "sess_680b7c59"),
    ("jongno", "sess_3130be49"),
    ("seocho", "sess_387e8ece"),
    ("ydp", "sess_c2a652b2"),
]

ROT_GATE_THRESHOLD = 0.3  # constants.context_window.rot_gate_threshold


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for domain, sid in SESSIONS:
        turns = load_enriched_session_data(sid)
        if not turns:
            print(f"⚠ {domain} ({sid}): trace 없음")
            continue
        for t in turns:
            meta = t.get("metadata", {})
            scores = t.get("scores", {})
            rows.append({
                "domain": domain,
                "session": sid,
                "turn": t.get("turn_number"),
                "dead_weight_tokens": meta.get("context.dead_weight_tokens"),
                "rot_gate_triggered": meta.get("context.rot_gate_triggered"),
                "rot_risk": meta.get("context.rot_risk"),
                "rot_velocity": meta.get("context.rot_velocity"),
                "previous_turns_tokens": meta.get("context.source.previous_turns_tokens"),
                "total_tokens": meta.get("context.total_tokens"),
                "noise_ratio": meta.get("context.noise_ratio"),
                "efficiency_score": scores.get("efficiency_score") or scores.get("efficiency"),
                "completeness_score": scores.get("completeness_score") or scores.get("completeness"),
            })
    return rows


def main() -> None:
    print("=" * 78)
    print("A-12 — dead_weight > 0 검증 (n=24 trace 재분석)")
    print("=" * 78)

    rows = collect_rows()
    print(f"\n[1] 데이터 수집: {len(rows)} turns")

    # ── 1. dead_weight_tokens 분포 ──
    print(f"\n[2] dead_weight_tokens 분포")
    dw_values = [r["dead_weight_tokens"] for r in rows if r["dead_weight_tokens"] is not None]
    dw_zero = sum(1 for v in dw_values if v == 0)
    dw_pos = sum(1 for v in dw_values if v > 0)
    print(f"  유효 측정: {len(dw_values)}/{len(rows)}")
    print(f"  = 0:  {dw_zero} turns")
    print(f"  > 0:  {dw_pos} turns")
    if dw_pos > 0:
        pos_vals = sorted([v for v in dw_values if v > 0])
        print(f"  > 0 통계: min={pos_vals[0]}, median={pos_vals[len(pos_vals)//2]}, max={pos_vals[-1]}")
        print(f"  > 0 분포: {pos_vals}")

    # ── 2. Rot Gate 트리거 빈도 ──
    print(f"\n[3] Rot Gate 트리거 빈도 + 조건 검증")
    triggered_count = sum(1 for r in rows if r["rot_gate_triggered"] is True)
    not_triggered = sum(1 for r in rows if r["rot_gate_triggered"] is False)
    none_count = sum(1 for r in rows if r["rot_gate_triggered"] is None)
    print(f"  triggered=True:  {triggered_count} turns")
    print(f"  triggered=False: {not_triggered} turns")
    print(f"  None (미측정):    {none_count} turns")

    # 조건 정합성 검증: triggered = (rot_risk > 0.3 AND dead_weight > 0)
    print(f"\n[4] Trigger 조건 정합성 (rot_risk > {ROT_GATE_THRESHOLD} AND dead_weight > 0)")
    consistent = 0
    inconsistent = []
    for r in rows:
        triggered = r["rot_gate_triggered"]
        rr = r["rot_risk"] or 0
        dw = r["dead_weight_tokens"] or 0
        if triggered is None:
            continue
        expected = (rr > ROT_GATE_THRESHOLD and dw > 0)
        if triggered == expected:
            consistent += 1
        else:
            inconsistent.append({
                "domain": r["domain"], "turn": r["turn"],
                "triggered": triggered, "rot_risk": rr, "dead_weight": dw,
                "expected": expected,
            })
    print(f"  정합 turns: {consistent}")
    if inconsistent:
        print(f"  ⚠ 불일치 {len(inconsistent)} turns:")
        for i in inconsistent[:5]:
            print(f"    {i['domain']} T{i['turn']}: triggered={i['triggered']}, "
                  f"rot_risk={i['rot_risk']}, dead_weight={i['dead_weight']}, "
                  f"expected={i['expected']}")
    else:
        print("  ✓ 모든 turn에서 조건 정합 (rot_gate_triggered = rot_risk>0.3 AND dead_weight>0)")

    # ── 3. dead_weight > 0 케이스의 efficiency 패턴 ──
    print(f"\n[5] dead_weight > 0 케이스의 efficiency score 패턴")
    pos_eff = [r["efficiency_score"] for r in rows if (r["dead_weight_tokens"] or 0) > 0 and r["efficiency_score"] is not None]
    zero_eff = [r["efficiency_score"] for r in rows if r["dead_weight_tokens"] == 0 and r["efficiency_score"] is not None]
    if pos_eff:
        print(f"  dead_weight > 0 efficiency 평균: {sum(pos_eff)/len(pos_eff):.3f} (n={len(pos_eff)})")
    if zero_eff:
        print(f"  dead_weight = 0 efficiency 평균: {sum(zero_eff)/len(zero_eff):.3f} (n={len(zero_eff)})")
    if pos_eff and zero_eff:
        diff = sum(pos_eff)/len(pos_eff) - sum(zero_eff)/len(zero_eff)
        print(f"  차이: {diff:+.3f}  ({'dead_weight 있을 때 efficiency↓' if diff < 0 else '의외로 ≥'})")

    # ── 4. rot_risk vs dead_weight 상관 ──
    print(f"\n[6] rot_risk와 dead_weight_tokens의 관계")
    pairs = [(r["rot_risk"], r["dead_weight_tokens"]) for r in rows
             if r["rot_risk"] is not None and r["dead_weight_tokens"] is not None]
    if len(pairs) >= 3:
        rr_pos = [(rr, dw) for rr, dw in pairs if dw > 0]
        rr_zero = [(rr, dw) for rr, dw in pairs if dw == 0]
        if rr_pos:
            print(f"  dead_weight > 0 케이스의 rot_risk: "
                  f"mean={sum(r for r,_ in rr_pos)/len(rr_pos):.3f}, "
                  f"max={max(r for r,_ in rr_pos):.3f}, n={len(rr_pos)}")
        if rr_zero:
            print(f"  dead_weight = 0 케이스의 rot_risk: "
                  f"mean={sum(r for r,_ in rr_zero)/len(rr_zero):.3f}, "
                  f"max={max(r for r,_ in rr_zero):.3f}, n={len(rr_zero)}")

    # ── 5. 도메인별 dead_weight 패턴 ──
    print(f"\n[7] 도메인별 dead_weight_tokens 평균")
    print(f"  {'domain':<10} | {'n':>2} | {'dw_mean':>8} | {'dw_max':>7} | {'gate_count':>10}")
    print(f"  {'-'*10} | {'-'*2} | {'-'*8} | {'-'*7} | {'-'*10}")
    for d, _ in SESSIONS:
        domain_rows = [r for r in rows if r["domain"] == d]
        dws = [r["dead_weight_tokens"] for r in domain_rows if r["dead_weight_tokens"] is not None]
        gates = sum(1 for r in domain_rows if r["rot_gate_triggered"] is True)
        if dws:
            print(f"  {d:<10} | {len(domain_rows):>2} | {sum(dws)/len(dws):>8.0f} | {max(dws):>7} | {gates:>10}")

    # ── 6. 턴별 dead_weight 진화 ──
    print(f"\n[8] 턴별 dead_weight_tokens 평균 (turn 진행에 따른 누적 패턴)")
    print(f"  {'turn':>4} | {'n':>2} | {'dw_mean':>8} | {'dw_max':>7} | {'rot_risk_mean':>14} | {'gate_count':>10}")
    print(f"  {'-'*4} | {'-'*2} | {'-'*8} | {'-'*7} | {'-'*14} | {'-'*10}")
    for t in range(1, 6):
        turn_rows = [r for r in rows if r["turn"] == t]
        dws = [r["dead_weight_tokens"] for r in turn_rows if r["dead_weight_tokens"] is not None]
        rrs = [r["rot_risk"] for r in turn_rows if r["rot_risk"] is not None]
        gates = sum(1 for r in turn_rows if r["rot_gate_triggered"] is True)
        if dws:
            print(f"  T{t:<3} | {len(turn_rows):>2} | {sum(dws)/len(dws):>8.0f} | {max(dws):>7} | "
                  f"{sum(rrs)/len(rrs) if rrs else 0:>14.3f} | {gates:>10}")

    # ── 7. v3 derived 관계 검증 ──
    print(f"\n[9] v3 derived 관계 검증 (rot_gate_pruned_tokens 폐기 → dead_weight × triggered)")
    print(f"  v2 공식: rot_gate_pruned_tokens = dead_weight_tokens (when triggered=True)")
    print(f"  v3에서는 pruned가 trace에 없음. 진단 메시지에서 dead_weight 직접 사용.")
    if triggered_count > 0:
        triggered_rows = [r for r in rows if r["rot_gate_triggered"] is True]
        print(f"  triggered=True 케이스의 dead_weight: "
              f"{[r['dead_weight_tokens'] for r in triggered_rows]}")
        print(f"  → derived (pruned == dead_weight when triggered) 관계 검증 가능")
    else:
        print(f"  ⚠ triggered=True 케이스 0건 (H4b와 동일 — 실측 개입 없음)")
        print(f"  → A-12의 진정한 검증은 dead_weight > 0 시나리오 추가 측정 필요")

    # ── 8. 결론 ──
    print(f"\n" + "=" * 78)
    print("A-12 결론")
    print("=" * 78)
    if dw_pos == 0:
        print("⚠ dead_weight_tokens가 0인 케이스만 발견 → Rot Gate 작동 시나리오 부재")
        print("  필요 조건: 5턴 이상 + 누적 토큰 큰 시나리오 + key_claims 활용 패턴")
    else:
        print(f"✓ dead_weight_tokens > 0 발견: {dw_pos}/{len(dw_values)} turns")
        if triggered_count > 0:
            print(f"✓ Rot Gate 트리거: {triggered_count} turns — v3 로직 작동 확인")
        else:
            print(f"⚠ dead_weight > 0이지만 triggered=False — rot_risk가 임계 미만 가능")
    if not inconsistent:
        print(f"✓ Trigger 조건 정합성: {consistent}/{consistent} (rot_risk × dead_weight × enabled)")


if __name__ == "__main__":
    main()
