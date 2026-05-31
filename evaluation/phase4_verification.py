"""
evaluation/phase4_verification.py — Phase 4: 가정 종합 검증

역할:
    1. H1 완전 검증: 50턴+ 데이터로 소스 구성 ↔ 4D Spearman 상관
    2. H2 독립성 재분석: 4D 차원 쌍 6쌍 상관 (|r| < 0.5 기준)
    3. FAIL 턴 추출: H5 블라인드 테스트용
    4. H0/H3 사람 평가 자료 준비

사용 방법:
    python -m evaluation.phase4_verification --sessions sess_a sess_b ...

데이터 흐름:
    입력: Langfuse session IDs (50턴 이상)
    출력: docs/analysis/NN-PHASE4_VERIFICATION.md + 사람 평가 시트
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.correlation_analysis import (
    collect_turn_data,
    spearman_correlation,
)
from agent.monitoring_schema import ATTRS


SCORE_NAMES = {
    "completeness": "completeness_score",
    "efficiency": "efficiency_score",
    "relevance": "relevance_score",
    "consistency": "consistency_score",
}


# --- H1 완전 검증 ---
def verify_h1(turns: list[dict]) -> dict:
    """H1: 소스 구성 ↔ 4D 상관 (50턴+). 성공 기준: noise_ratio ↔ 4D avg r < -0.4."""
    context_attrs = [
        ("noise_ratio", ATTRS["context.noise_ratio"]),
        ("effective_noise_ratio", ATTRS["context.effective_noise_ratio"]),
        ("window_utilization", ATTRS["context.window_utilization"]),
        ("new_data_ratio", ATTRS.get("context.new_data_ratio", "context.new_data_ratio")),
        ("rot_risk", ATTRS["context.rot_risk"]),
        ("information_density", ATTRS["context.information_density"]),
        ("fidelity_score", ATTRS["context.fidelity_score"]),
    ]

    results = {}
    for attr_label, attr_key in context_attrs:
        dim_corrs = {}
        for dim_label, score_name in SCORE_NAMES.items():
            pairs = [
                (float(t["metadata"].get(attr_key)), float(t["scores"].get(score_name)))
                for t in turns
                if t["metadata"].get(attr_key) is not None and t["scores"].get(score_name) is not None
            ]
            if len(pairs) >= 5:
                rho = spearman_correlation([p[0] for p in pairs], [p[1] for p in pairs])
                dim_corrs[dim_label] = {"rho": rho, "n": len(pairs)}
        if dim_corrs:
            results[attr_label] = dim_corrs

    # noise_ratio ↔ 4D 평균 (핵심 지표)
    avg_pairs = []
    for t in turns:
        nr = t["metadata"].get(ATTRS["context.noise_ratio"])
        scores = [t["scores"].get(sn) for sn in SCORE_NAMES.values()]
        if nr is not None and all(s is not None for s in scores):
            avg_pairs.append((float(nr), sum(float(s) for s in scores) / 4))

    key_rho = spearman_correlation(
        [p[0] for p in avg_pairs], [p[1] for p in avg_pairs]
    ) if len(avg_pairs) >= 5 else None

    verdict = "PASS" if key_rho is not None and key_rho < -0.4 else "FAIL"
    return {
        "correlations": results,
        "key_rho": key_rho,
        "key_n": len(avg_pairs),
        "verdict": verdict,
    }


# --- H2 독립성 재분석 ---
def verify_h2(turns: list[dict]) -> dict:
    """H2: 4D 차원 쌍 독립성. 성공 기준: 6쌍 모두 |r| < 0.5."""
    dims = list(SCORE_NAMES.keys())
    pairs_result = []

    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            d1, d2 = dims[i], dims[j]
            sn1, sn2 = SCORE_NAMES[d1], SCORE_NAMES[d2]
            vals = [
                (float(t["scores"][sn1]), float(t["scores"][sn2]))
                for t in turns
                if t["scores"].get(sn1) is not None and t["scores"].get(sn2) is not None
            ]
            if len(vals) >= 5:
                rho = spearman_correlation([v[0] for v in vals], [v[1] for v in vals])
                passed = abs(rho) < 0.5 if rho is not None else None
                pairs_result.append({
                    "pair": f"{d1} ↔ {d2}",
                    "rho": rho,
                    "n": len(vals),
                    "passed": passed,
                })

    all_pass = all(p["passed"] for p in pairs_result if p["passed"] is not None)
    # F-001: efficiency ↔ relevance 예외 (이미 알려진 고상관)
    known_exception = next(
        (p for p in pairs_result if "efficiency" in p["pair"] and "relevance" in p["pair"]),
        None
    )

    return {
        "pairs": pairs_result,
        "all_pass": all_pass,
        "known_exception": known_exception,
        "verdict": "PASS" if all_pass else f"FAIL (known: eff↔rel={known_exception['rho'] if known_exception else 'N/A'})",
    }


# --- FAIL 턴 추출 (H5 블라인드 테스트용) ---
def extract_fail_turns(turns: list[dict], max_count: int = 10) -> list[dict]:
    """4D 점수 중 하나라도 FAIL(< 0.7)인 턴을 추출한다."""
    fail_turns = []
    for t in turns:
        scores = {dim: t["scores"].get(sn) for dim, sn in SCORE_NAMES.items()}
        fail_dims = [
            dim for dim, score in scores.items()
            if score is not None and score < 0.7
        ]
        if fail_dims:
            fail_turns.append({
                "session_id": t["session_id"],
                "turn_number": t["turn_number"],
                "trace_id": t["trace_id"],
                "scores": scores,
                "fail_dims": fail_dims,
                # H5 블라인드용: 진단 결과는 제외하고 메타데이터 요약만
                "context_summary": {
                    "total_tokens": t["metadata"].get(ATTRS["context.total_tokens"]),
                    "noise_ratio": t["metadata"].get(ATTRS["context.noise_ratio"]),
                    "is_sufficient": t["metadata"].get(ATTRS["context.is_sufficient"]),
                    "rot_risk": t["metadata"].get(ATTRS["context.rot_risk"]),
                },
            })
    # 점수가 가장 낮은 순으로 정렬
    fail_turns.sort(key=lambda t: min(
        (s for s in t["scores"].values() if s is not None), default=1.0
    ))
    return fail_turns[:max_count]


# --- H0 평가 자료 준비 ---
def prepare_h0_sheet(turns: list[dict], max_count: int = 20) -> list[dict]:
    """H0 사람 평가용 턴 목록 (trace_id + 4D 점수 요약)."""
    # 4D 점수가 모두 있는 턴에서 균등 샘플
    scored = [
        t for t in turns
        if all(t["scores"].get(sn) is not None for sn in SCORE_NAMES.values())
    ]
    # 다양한 점수 범위를 포함하도록 4D 평균으로 정렬 후 균등 선택
    scored.sort(key=lambda t: sum(
        float(t["scores"][sn]) for sn in SCORE_NAMES.values()
    ) / 4)

    step = max(1, len(scored) // max_count)
    selected = scored[::step][:max_count]

    return [
        {
            "trace_id": t["trace_id"],
            "session_id": t["session_id"],
            "turn_number": t["turn_number"],
            "scores_4d_avg": round(sum(
                float(t["scores"][sn]) for sn in SCORE_NAMES.values()
            ) / 4, 3),
        }
        for t in selected
    ]


# --- 리포트 생성 ---
def generate_report(
    h1: dict, h2: dict, fail_turns: list, h0_sheet: list,
    n_turns: int, session_ids: list[str],
) -> str:
    """Phase 4 종합 검증 리포트."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Phase 4: 가정 종합 검증 결과",
        "",
        f"> **Date:** {now_str}",
        f"> **데이터:** {n_turns}턴 ({len(session_ids)}개 세션)",
        "",
        "---",
        "",
        "## 1. H1 완전 검증: 소스 구성 ↔ 4D 상관",
        "",
        f"### 판정: **{h1['verdict']}**",
        f"- noise_ratio ↔ 4D 평균: rho = {h1['key_rho']} (n={h1['key_n']})",
        f"- 기준: rho < -0.4",
        "",
        "### 상관 매트릭스",
        "",
        "| 속성 | completeness | efficiency | relevance | consistency |",
        "|------|:-----------:|:----------:|:---------:|:-----------:|",
    ]

    for attr_label, dim_corrs in h1["correlations"].items():
        row = [f"`{attr_label}`"]
        for dim in ["completeness", "efficiency", "relevance", "consistency"]:
            c = dim_corrs.get(dim)
            if c:
                mark = "**" if abs(c["rho"]) > 0.3 else ""
                row.append(f"{mark}{c['rho']:.3f}{mark} (n={c['n']})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. H2 독립성 재분석: 4D 차원 쌍 상관",
        "",
        f"### 판정: **{h2['verdict']}**",
        "",
        "| 쌍 | rho | n | |r|<0.5 |",
        "|----|:---:|:-:|:------:|",
    ])
    for p in h2["pairs"]:
        status = "✅" if p["passed"] else "❌"
        rho_str = f"{p['rho']:.3f}" if p['rho'] is not None else "N/A"
        lines.append(f"| {p['pair']} | {rho_str} | {p['n']} | {status} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. H5 블라인드 테스트용 FAIL 턴 (진단 결과 비공개)",
        "",
        f"FAIL 턴 {len(fail_turns)}개 추출됨.",
        "",
        "| # | session | turn | 가장 낮은 차원 | 최저 점수 | trace_id |",
        "|:-:|---------|:----:|:------------:|:---------:|----------|",
    ])
    for i, ft in enumerate(fail_turns, 1):
        worst_dim = min(
            ((d, s) for d, s in ft["scores"].items() if s is not None),
            key=lambda x: x[1],
        )
        lines.append(
            f"| {i} | {ft['session_id'][:12]}... | {ft['turn_number']} | "
            f"{worst_dim[0]} | {worst_dim[1]:.2f} | `{ft['trace_id'][:16]}...` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. H0 사람 평가 시트 (답변 품질 1~5점)",
        "",
        f"{len(h0_sheet)}턴 선정됨 (4D 평균 기준 균등 분포).",
        "",
        "| # | trace_id | session | turn | 4D avg | 사람 평가 |",
        "|:-:|----------|---------|:----:|:------:|:--------:|",
    ])
    for i, item in enumerate(h0_sheet, 1):
        lines.append(
            f"| {i} | `{item['trace_id'][:16]}...` | {item['session_id'][:12]}... | "
            f"{item['turn_number']} | {item['scores_4d_avg']} | ___/5 |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. 다음 단계",
        "",
        "1. **H0**: 위 시트의 각 턴 응답을 Langfuse UI에서 확인 후 1~5점 채점",
        "2. **H5**: FAIL 턴 목록의 각 턴에 대해 '왜 나쁜가?' 블라인드 판단",
        "3. **H3**: conditions_preserved=False 턴에서 품질 저하 수동 확인",
        "4. 채점 결과를 `evaluation/phase4_verification.py --human-scores` 로 전달하면 상관분석 자동 실행",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 종합 검증")
    parser.add_argument("--sessions", nargs="+", required=True, help="세션 ID 목록")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Phase 4 검증 — {len(args.sessions)}개 세션")
    turns = collect_turn_data(args.sessions)
    scored = [
        t for t in turns
        if any(t["scores"].get(sn) is not None for sn in SCORE_NAMES.values())
    ]
    print(f"  총 {len(turns)}턴, 점수 있는 턴 {len(scored)}개")

    # H1
    print("\n─── H1 완전 검증 ───")
    h1 = verify_h1(scored)
    print(f"  noise_ratio ↔ 4D avg: rho={h1['key_rho']} (n={h1['key_n']})")
    print(f"  판정: {h1['verdict']}")

    # H2
    print("\n─── H2 독립성 재분석 ───")
    h2 = verify_h2(scored)
    for p in h2["pairs"]:
        status = "✅" if p["passed"] else "❌"
        print(f"  {p['pair']}: rho={p['rho']:.3f} {status}")
    print(f"  판정: {h2['verdict']}")

    # FAIL 턴
    print("\n─── FAIL 턴 추출 ───")
    fail_turns = extract_fail_turns(scored)
    print(f"  {len(fail_turns)}개 FAIL 턴 추출")

    # H0 시트
    h0_sheet = prepare_h0_sheet(scored)
    print(f"  H0 평가 시트: {len(h0_sheet)}턴 선정")

    # 리포트
    report = generate_report(h1, h2, fail_turns, h0_sheet, len(scored), args.sessions)
    if args.output:
        output_path = args.output
    else:
        analysis_dir = "docs/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        existing = [f for f in os.listdir(analysis_dir) if f.endswith(".md")]
        next_num = max((int(f.split("-")[0]) for f in existing if f[0].isdigit()), default=0) + 1
        output_path = f"{analysis_dir}/{next_num:02d}-PHASE4_VERIFICATION.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 리포트: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
