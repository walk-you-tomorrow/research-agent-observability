"""
scripts/phase_c_r_compute.py — Phase C 옵션 A 2단계 r 측정 분석

5세션 × 5턴 = 25 turns의 4D Score를 Langfuse에서 끌어와
efficiency ↔ relevance Spearman rho 및 4D 6쌍 직교성 검증.

목표: r < 0.5 (구공식 0.92 대비)
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

DIMENSIONS = ["completeness", "efficiency", "relevance", "consistency"]


def collect_scores() -> list[dict]:
    """모든 세션의 모든 턴에서 4D score 수집."""
    rows: list[dict] = []
    for domain, sid in SESSIONS:
        turns = load_enriched_session_data(sid)
        if not turns:
            print(f"⚠ {domain} ({sid}): trace 없음")
            continue
        for t in turns:
            scores = t.get("scores", {})
            row = {
                "domain": domain,
                "session": sid,
                "turn": t.get("turn_number"),
            }
            for d in DIMENSIONS:
                # Langfuse score 키 형식: "{dimension}_score" (예: efficiency_score)
                row[d] = scores.get(f"{d}_score") or scores.get(d)
            rows.append(row)
    return rows


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. n < 3이면 NaN."""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        # 동순위는 평균 rank
        idx = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and values[idx[j + 1]] == values[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based 평균 rank
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    dx2 = sum((r - mean_x) ** 2 for r in rx)
    dy2 = sum((r - mean_y) ** 2 for r in ry)
    if dx2 == 0 or dy2 == 0:
        return float("nan")
    return num / (dx2 * dy2) ** 0.5


def main() -> None:
    print("=" * 70)
    print("Phase C 옵션 A 2단계 — r 측정 (n=25 목표)")
    print("=" * 70)

    rows = collect_scores()
    print(f"\n[1] 데이터 수집: {len(rows)} turns")

    # 도메인/세션별 카운트
    by_domain: dict[str, int] = {}
    for r in rows:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    for d, c in by_domain.items():
        print(f"  {d}: {c} turns")

    # 4D score 누락 검사
    print(f"\n[2] 4D score 가용성")
    for dim in DIMENSIONS:
        valid = [r[dim] for r in rows if isinstance(r.get(dim), (int, float))]
        print(f"  {dim}: {len(valid)}/{len(rows)} valid")

    # 6쌍 Spearman rho
    print(f"\n[3] Spearman rho — 4D 6쌍 (n={len(rows)})")
    pairs = [
        ("completeness", "efficiency"),
        ("completeness", "relevance"),
        ("completeness", "consistency"),
        ("efficiency", "relevance"),  # ★ 핵심
        ("efficiency", "consistency"),
        ("relevance", "consistency"),
    ]
    for d1, d2 in pairs:
        xs, ys = [], []
        for r in rows:
            v1, v2 = r.get(d1), r.get(d2)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                xs.append(float(v1))
                ys.append(float(v2))
        if len(xs) < 3:
            print(f"  {d1} ↔ {d2}: n={len(xs)} 부족")
            continue
        rho = spearman_rho(xs, ys)
        marker = ""
        if d1 == "efficiency" and d2 == "relevance":
            if rho < 0.5:
                marker = " ★ 목표 < 0.5 달성 ✓"
            else:
                marker = f" (목표 < 0.5 미달, 구공식 0.92 / 1세션 0.60 비교)"
        print(f"  {d1} ↔ {d2}: rho = {rho:+.3f} (n={len(xs)}){marker}")

    # 세션별 평균
    print(f"\n[4] 세션별 4D 평균")
    print(f"  {'domain':<10} | {'n':>2} | {'compl':>6} | {'effic':>6} | {'relev':>6} | {'const':>6}")
    print(f"  {'-'*10} | {'-'*2} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    for d, _ in SESSIONS:
        domain_rows = [r for r in rows if r["domain"] == d]
        if not domain_rows:
            continue
        avgs = {}
        for dim in DIMENSIONS:
            vs = [r[dim] for r in domain_rows if isinstance(r.get(dim), (int, float))]
            avgs[dim] = (sum(vs) / len(vs)) if vs else None
        line = f"  {d:<10} | {len(domain_rows):>2} | "
        line += " | ".join(
            f"{avgs[dim]:>6.3f}" if avgs[dim] is not None else "  N/A "
            for dim in DIMENSIONS
        )
        print(line)

    # 턴별 추이
    print(f"\n[5] 턴별 4D 평균 (학습 곡선 — 턴 진행에 따른 score 변화)")
    print(f"  {'turn':>4} | {'n':>2} | {'compl':>6} | {'effic':>6} | {'relev':>6} | {'const':>6}")
    print(f"  {'-'*4} | {'-'*2} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    for t in range(1, 6):
        turn_rows = [r for r in rows if r["turn"] == t]
        if not turn_rows:
            continue
        avgs = {}
        for dim in DIMENSIONS:
            vs = [r[dim] for r in turn_rows if isinstance(r.get(dim), (int, float))]
            avgs[dim] = (sum(vs) / len(vs)) if vs else None
        line = f"  T{t:<3} | {len(turn_rows):>2} | "
        line += " | ".join(
            f"{avgs[dim]:>6.3f}" if avgs[dim] is not None else "  N/A "
            for dim in DIMENSIONS
        )
        print(line)


if __name__ == "__main__":
    main()
