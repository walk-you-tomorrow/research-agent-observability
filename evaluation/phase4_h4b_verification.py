"""
evaluation/phase4_h4b_verification.py — H4b 검증: Rot Gate 개입 효과

역할:
    1. 동일한 장기 세션 질문(24턴)을 Rot Gate ON / OFF 두 그룹으로 실행
    2. 각 그룹의 4D 평균(완전성/효율성/관련성/일관성)을 비교
    3. Gate 판정: ON 그룹의 4D 평균이 OFF 그룹보다 유의미하게 높으면 H4b PASS

비교 단위:
    전체 8턴 × 3세션 = 24턴의 4D 평균 (그룹별)
    사용자 결정(2026-04-23): "전체 8턴 × 3세션의 4D 평균 비교로 진행"

사용 방법:
    # 두 그룹 실행 + 검증
    python -m evaluation.phase4_h4b_verification

    # 기존 세션으로 검증만 (ON/OFF 세션 ID 직접 전달)
    python -m evaluation.phase4_h4b_verification \\
        --on-sessions sess_on_a sess_on_b sess_on_c \\
        --off-sessions sess_off_a sess_off_b sess_off_c

데이터 흐름:
    입력: 8턴 × 3세션 × 2그룹 = 48턴
    출력: H4b Gate 판정 (PASS/FAIL) + docs/analysis/NN-H4B_VERIFICATION.md

환경변수:
    ROT_GATE_ENABLED: "1"(기본, ON) / "0"(OFF). 세션 실행 전에 설정됨.
"""
import argparse
import math
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

from evaluation.correlation_analysis import collect_turn_data
from evaluation.phase3_h4a_verification import LONG_SESSION_QUERIES


# --- 4D 점수 키 (Langfuse에 기록된 이름) ---
SCORE_NAMES = [
    "completeness_score",
    "efficiency_score",
    "relevance_score",
    "consistency_score",
]


def run_group_sessions(group_label: str, rot_gate_enabled: bool) -> list[str]:
    """특정 그룹(ON/OFF)으로 3개 장기 세션을 실행한다.

    Args:
        group_label: "ON" 또는 "OFF" — 터미널 출력용 라벨.
        rot_gate_enabled: True면 Rot Gate 활성화, False면 비활성화.

    Returns:
        실행된 session_id 리스트.
    """
    # 런타임 평가: is_rot_gate_enabled()가 os.environ을 호출 시점에 읽으므로
    # main.py 임포트 전에 환경변수를 설정하지 않아도 된다.
    os.environ["ROT_GATE_ENABLED"] = "1" if rot_gate_enabled else "0"

    from main import load_config, run_session

    config = load_config()
    session_ids = []

    print(f"\n{'#'*70}")
    print(f"#  그룹 {group_label} (ROT_GATE_ENABLED={os.environ['ROT_GATE_ENABLED']})")
    print(f"{'#'*70}")

    for i, queries in enumerate(LONG_SESSION_QUERIES):
        label = chr(ord("A") + i)
        print(f"\n{'='*60}")
        print(f"  [{group_label}] 장기 세션 {label}: {len(queries)}턴")
        print(f"{'='*60}\n")

        try:
            state = run_session(queries, config=config)
            sid = state.get("session_id", "")
            session_ids.append(sid)
            print(f"\n  → [{group_label}] Session {label}: {sid}")
        except Exception as e:
            print(f"\n  ✗ [{group_label}] Session {label} 실패: {e}")

    return session_ids


def compute_group_stats(turns: list[dict]) -> dict:
    """그룹의 4D 점수 평균/표준편차/n을 계산한다.

    Args:
        turns: collect_turn_data()의 반환값.

    Returns:
        {
            "n_turns": int,
            "n_with_scores": int,  # 4D 점수가 모두 있는 턴 수
            "per_dimension": {dim: {mean, std, n, values}},
            "avg_4d": {mean, std, n, values},  # 턴별 4D 평균의 분포
            "rot_gate_triggered_ratio": float,  # rot_gate_triggered=True 턴 비율
            "rot_gate_pruned_tokens_total": int,
            "rot_risk_mean": float,
        }
    """
    from agent.monitoring_schema import ATTRS

    per_dim = {s: [] for s in SCORE_NAMES}
    avg_4d_values = []
    rot_gate_flags = []
    rot_pruned_tokens = []
    rot_risks = []

    for t in turns:
        # 4D 점수 수집
        scores = t["scores"]
        dim_scores = [scores.get(s) for s in SCORE_NAMES]
        if all(s is not None for s in dim_scores):
            try:
                floats = [float(s) for s in dim_scores]
                for s_name, s_val in zip(SCORE_NAMES, floats):
                    per_dim[s_name].append(s_val)
                avg_4d_values.append(sum(floats) / 4)
            except (TypeError, ValueError):
                pass

        # Rot Gate 통계
        meta = t["metadata"]
        rgt = meta.get(ATTRS["context.rot_gate_triggered"])
        if rgt is not None:
            rot_gate_flags.append(bool(rgt))
        rpt = meta.get(ATTRS.get("context.rot_gate_pruned_tokens", "context.rot_gate_pruned_tokens"))
        if rpt is not None:
            try:
                rot_pruned_tokens.append(int(rpt))
            except (TypeError, ValueError):
                pass
        rr = meta.get(ATTRS["context.rot_risk"])
        if rr is not None:
            try:
                rot_risks.append(float(rr))
            except (TypeError, ValueError):
                pass

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std": None, "n": 0, "values": []}
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        return {"mean": round(mean, 4), "std": round(std, 4), "n": n, "values": values}

    return {
        "n_turns": len(turns),
        "n_with_scores": len(avg_4d_values),
        "per_dimension": {dim: _stats(per_dim[dim]) for dim in SCORE_NAMES},
        "avg_4d": _stats(avg_4d_values),
        "rot_gate_triggered_ratio": (
            round(sum(rot_gate_flags) / len(rot_gate_flags), 3)
            if rot_gate_flags else None
        ),
        "rot_gate_pruned_tokens_total": sum(rot_pruned_tokens),
        "rot_risk_mean": (
            round(sum(rot_risks) / len(rot_risks), 4) if rot_risks else None
        ),
    }


def mann_whitney_u(x: list[float], y: list[float]) -> dict:
    """Mann-Whitney U 검정 (정규성 가정 없음, 순위 기반).

    H0: 두 그룹이 동일 분포에서 추출됨.
    H1: ON 그룹의 점수가 OFF 그룹보다 큼 (단측 검정).

    Args:
        x: ON 그룹 값.
        y: OFF 그룹 값.

    Returns:
        {"u_statistic": float, "z_score": float, "p_one_sided": float}
        n이 너무 작으면 값에 None.
    """
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return {"u_statistic": None, "z_score": None, "p_one_sided": None}

    # 순위 계산 (동순위는 평균 순위)
    combined = [(v, "x") for v in x] + [(v, "y") for v in y]
    combined.sort(key=lambda p: p[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) - 1 and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    # x 그룹 순위 합
    r1 = sum(r for r, (_, tag) in zip(ranks, combined) if tag == "x")
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1

    # 큰 샘플 근사 (z-score)
    mean_u = n1 * n2 / 2
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if std_u == 0:
        return {"u_statistic": u1, "z_score": None, "p_one_sided": None}
    z = (u1 - mean_u) / std_u

    # 단측 p (정규 근사, ON > OFF 방향)
    p_one_sided = 0.5 * (1 - math.erf(z / math.sqrt(2)))

    return {
        "u_statistic": round(u1, 2),
        "z_score": round(z, 4),
        "p_one_sided": round(p_one_sided, 4),
    }


def verify_h4b(on_stats: dict, off_stats: dict) -> dict:
    """H4b 판정: ON 그룹의 4D 평균 > OFF 그룹 + 통계적 유의성.

    판정 기준:
        1. mean_diff = ON.avg_4d.mean - OFF.avg_4d.mean > 0
        2. Mann-Whitney U 단측 p < 0.1 (n이 작아 관대하게)
        3. 차원별로도 ON이 대체로 우세 (최소 2/4 차원에서 ON > OFF)

    Returns:
        판정 결과 딕셔너리.
    """
    on_avg = on_stats["avg_4d"]["mean"]
    off_avg = off_stats["avg_4d"]["mean"]
    if on_avg is None or off_avg is None:
        return {"verdict": "INCONCLUSIVE", "reason": "둘 중 한 그룹에 4D 점수 없음"}

    mean_diff = round(on_avg - off_avg, 4)

    # Mann-Whitney U (턴별 4D 평균 분포)
    mw = mann_whitney_u(
        on_stats["avg_4d"]["values"],
        off_stats["avg_4d"]["values"],
    )

    # 차원별 비교
    dim_wins = 0
    dim_compare = {}
    for dim in SCORE_NAMES:
        on_m = on_stats["per_dimension"][dim]["mean"]
        off_m = off_stats["per_dimension"][dim]["mean"]
        if on_m is not None and off_m is not None:
            diff = round(on_m - off_m, 4)
            dim_compare[dim] = {"on": on_m, "off": off_m, "diff": diff}
            if diff > 0:
                dim_wins += 1

    # 최종 판정
    passes_mean = mean_diff > 0
    passes_p = mw["p_one_sided"] is not None and mw["p_one_sided"] < 0.1
    passes_dims = dim_wins >= 2

    if passes_mean and passes_p and passes_dims:
        verdict = "PASS"
    elif passes_mean and passes_dims:
        verdict = "WEAK_PASS"  # 방향은 맞지만 통계적 유의성 미확보
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "mean_diff": mean_diff,
        "on_avg_4d": on_avg,
        "off_avg_4d": off_avg,
        "mann_whitney": mw,
        "dim_wins_for_on": dim_wins,
        "dim_compare": dim_compare,
        "criteria": {
            "mean_positive": passes_mean,
            "p_under_0.1": passes_p,
            "dim_majority_on": passes_dims,
        },
    }


def generate_h4b_report(
    verdict: dict,
    on_stats: dict,
    off_stats: dict,
    on_sessions: list[str],
    off_sessions: list[str],
) -> str:
    """H4b 검증 리포트를 Markdown으로 생성한다."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _fmt(v):
        return "N/A" if v is None else f"{v}"

    lines = [
        "# H4b 검증 결과: Rot Gate 개입 효과 (ON/OFF 비교)",
        "",
        f"> **Date:** {now_str}",
        "> **Context:** Phase 4 Step 4.5 — 동일 질문에 대해 Rot Gate ON / OFF 실행 후 4D 평균 비교",
        f"> **비교 단위:** 전체 8턴 × 3세션 ({on_stats['n_turns']}턴 ON vs {off_stats['n_turns']}턴 OFF)",
        "",
        "---",
        "",
        f"## Gate 판정: **{verdict['verdict']}**",
        "",
        f"- **ON 그룹 4D 평균:** {_fmt(verdict['on_avg_4d'])}  (n={on_stats['avg_4d']['n']})",
        f"- **OFF 그룹 4D 평균:** {_fmt(verdict['off_avg_4d'])}  (n={off_stats['avg_4d']['n']})",
        f"- **차이 (ON − OFF):** {_fmt(verdict['mean_diff'])}",
        "",
        "**Mann-Whitney U (단측, ON > OFF):**",
        f"- U = {_fmt(verdict['mann_whitney']['u_statistic'])}",
        f"- z = {_fmt(verdict['mann_whitney']['z_score'])}",
        f"- p = {_fmt(verdict['mann_whitney']['p_one_sided'])}",
        "",
        "**판정 기준:**",
        f"- 평균 차이 > 0: {'✅' if verdict['criteria']['mean_positive'] else '❌'}",
        f"- 단측 p < 0.1: {'✅' if verdict['criteria']['p_under_0.1'] else '❌'}",
        f"- 4차원 중 2+에서 ON 우세: {'✅' if verdict['criteria']['dim_majority_on'] else '❌'} ({verdict['dim_wins_for_on']}/4)",
        "",
        "---",
        "",
        "## 차원별 비교",
        "",
        "| 차원 | ON 평균 | OFF 평균 | 차이 (ON-OFF) |",
        "|------|:---:|:---:|:---:|",
    ]
    for dim, cmp in verdict["dim_compare"].items():
        arrow = "🔼" if cmp["diff"] > 0 else ("🔽" if cmp["diff"] < 0 else "—")
        lines.append(f"| {dim.replace('_score','')} | {cmp['on']} | {cmp['off']} | {arrow} {cmp['diff']} |")

    lines.extend([
        "",
        "---",
        "",
        "## Rot Gate 관측 지표",
        "",
        "| 지표 | ON 그룹 | OFF 그룹 |",
        "|------|:---:|:---:|",
        f"| rot_gate_triggered 비율 | {_fmt(on_stats['rot_gate_triggered_ratio'])} | {_fmt(off_stats['rot_gate_triggered_ratio'])} |",
        f"| 총 pruned_tokens | {on_stats['rot_gate_pruned_tokens_total']} | {off_stats['rot_gate_pruned_tokens_total']} |",
        f"| rot_risk 평균 | {_fmt(on_stats['rot_risk_mean'])} | {_fmt(off_stats['rot_risk_mean'])} |",
        "",
        "**해석:**",
        "- OFF 그룹에서 rot_gate_triggered 비율이 0이어야 정상 (토글이 제대로 작동).",
        "- pruned_tokens는 OFF 그룹에서 0이어야 정상 (pruning 없음).",
        "",
        "---",
        "",
        "## 세션 목록",
        "",
        "### ON 그룹 (Rot Gate 활성화)",
        "",
        "```",
    ])
    for sid in on_sessions:
        lines.append(sid)
    lines.extend(["```", "", "### OFF 그룹 (Rot Gate 비활성화)", "", "```"])
    for sid in off_sessions:
        lines.append(sid)
    lines.extend(["```", ""])

    lines.extend([
        "---",
        "",
        "## 해석 가이드",
        "",
        "- **PASS**: ON 그룹의 4D 평균이 통계적으로 유의미하게 높음. Rot Gate 개입이 품질 저하를 방지.",
        "- **WEAK_PASS**: 방향은 맞지만 n이 작아 통계적 유의성 미확보. 더 많은 세션 필요.",
        "- **FAIL**: ON 그룹이 OFF보다 낫다는 증거 없음. Rot Gate 효과 미확인 또는 역효과.",
        "",
        "**한계:**",
        "- LLM 응답의 비결정성으로 동일 질문이라도 턴별로 변동 존재.",
        "- n=24 vs 24는 통계적 검정력이 제한적 (효과 크기 중간 이상에서만 검출).",
        "- 차원별 Judge의 독립성 문제(F-001, H2 부분 FAIL) 잔존.",
        "",
    ])

    return "\n".join(lines)


def main():
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="H4b 검증: Rot Gate ON/OFF 비교")
    parser.add_argument("--on-sessions", nargs="+", help="ON 그룹 기존 세션 ID")
    parser.add_argument("--off-sessions", nargs="+", help="OFF 그룹 기존 세션 ID")
    parser.add_argument("--output", default=None, help="리포트 저장 경로")
    parser.add_argument(
        "--order",
        choices=["off-first", "on-first"],
        default="off-first",
        help="두 그룹 실행 순서 (기본: OFF→ON, 컨텍스트 누적 효과 중립화)",
    )
    args = parser.parse_args()

    # --- 세션 실행 또는 기존 세션 사용 ---
    if args.on_sessions and args.off_sessions:
        on_sessions = args.on_sessions
        off_sessions = args.off_sessions
        print(f"기존 세션 사용: ON={len(on_sessions)}개, OFF={len(off_sessions)}개")
    else:
        # 실행 순서: 기본은 OFF 먼저 (Rot Gate 없이 baseline 확보)
        if args.order == "off-first":
            off_sessions = run_group_sessions("OFF", rot_gate_enabled=False)
            on_sessions = run_group_sessions("ON", rot_gate_enabled=True)
        else:
            on_sessions = run_group_sessions("ON", rot_gate_enabled=True)
            off_sessions = run_group_sessions("OFF", rot_gate_enabled=False)

    if not on_sessions or not off_sessions:
        print("세션이 부족합니다.")
        sys.exit(1)

    # --- Langfuse ingestion 대기 ---
    print(f"\n⏳ Langfuse ingestion 대기 (10초)...")
    time.sleep(10)

    # --- 데이터 수집 ---
    print("\nON 그룹 데이터 수집 중...")
    on_turns = collect_turn_data(on_sessions)
    print(f"  {len(on_turns)}턴 수집 완료")

    print("\nOFF 그룹 데이터 수집 중...")
    off_turns = collect_turn_data(off_sessions)
    print(f"  {len(off_turns)}턴 수집 완료")

    # --- 통계 계산 ---
    on_stats = compute_group_stats(on_turns)
    off_stats = compute_group_stats(off_turns)

    # --- H4b 판정 ---
    verdict = verify_h4b(on_stats, off_stats)

    print(f"\n{'='*60}")
    print(f"  H4b Gate 판정: {verdict['verdict']}")
    print(f"  ON 4D avg:  {verdict['on_avg_4d']}")
    print(f"  OFF 4D avg: {verdict['off_avg_4d']}")
    print(f"  차이: {verdict['mean_diff']}")
    print(f"  단측 p: {verdict['mann_whitney']['p_one_sided']}")
    print(f"{'='*60}")

    # --- 리포트 생성 ---
    report = generate_h4b_report(verdict, on_stats, off_stats, on_sessions, off_sessions)

    if args.output:
        output_path = args.output
    else:
        analysis_dir = "docs/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        existing = [f for f in os.listdir(analysis_dir) if f.endswith(".md")]
        next_num = max(
            (int(f.split("-")[0]) for f in existing if f[0].isdigit()),
            default=0,
        ) + 1
        output_path = f"{analysis_dir}/{next_num:02d}-H4B_VERIFICATION.md"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 리포트: {os.path.abspath(output_path)}")
    return verdict


if __name__ == "__main__":
    main()
