"""
evaluation/phase3_h4a_verification.py — H4a 검증: rot_risk ↔ 4D 점수 상관분석

역할:
    1. 8턴 장기 세션 3개를 실행하여 context rot 누적 데이터를 수집
    2. rot_risk/rot_velocity ↔ 4D 점수 간 Spearman 상관 계산
    3. Gate 판단: r < -0.4이면 H4a PASS → Phase 3 Step 3.2 진행

사용 방법:
    # 3개 장기 세션 실행 + 검증
    python -m evaluation.phase3_h4a_verification

    # 기존 세션으로 검증만
    python -m evaluation.phase3_h4a_verification --sessions sess_a sess_b sess_c

데이터 흐름:
    입력: 8턴 × 3세션 = 24턴 데이터
    출력: H4a Gate 판정 (PASS/FAIL) + docs/analysis/NN-H4A_VERIFICATION.md
"""
import argparse
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

# --- 장기 세션 질문 세트 (8턴) ---
# 동일 도메인(서울 상권)에서 점진적으로 질문하여
# 이전 턴 결론이 누적되면서 context rot이 자연스럽게 발생하도록 유도
LONG_SESSION_QUERIES = [
    [
        # 세션 A: 강남구 중심 깊이 탐색
        "강남구 전체 상권 현황을 알려줘",
        "강남구에서 가장 활발한 동은 어디야?",
        "역삼동의 업종별 매출 비교해줘",
        "역삼동 카페 매출 추세가 어때?",
        "역삼동 카페와 음식점 임대료를 비교해줘",
        "강남구 전체에서 임대료 대비 매출이 좋은 동은?",
        "그 동에서 창업하기 좋은 업종은?",
        "지금까지 분석한 내용을 종합하면 강남구에서 가장 유망한 상권은?",
    ],
    [
        # 세션 B: 마포구 중심 + 비교 분석
        "마포구 상권 전체 현황을 알려줘",
        "마포구에서 유동인구가 가장 많은 동은?",
        "합정동과 연남동 상권을 비교해줘",
        "합정동 카페 매출은 얼마야?",
        "마포구와 강남구 카페 매출을 비교해줘",
        "마포구에서 임대료가 낮은 동은?",
        "마포구에서 소규모 창업에 적합한 동은?",
        "마포구 상권의 강점과 약점을 종합해줘",
    ],
    [
        # 세션 C: 교차 구 비교
        "서울 5개 구(강남, 마포, 서초, 종로, 영등포) 상권 비교해줘",
        "5개 구 중 유동인구가 가장 많은 곳은?",
        "종로구의 전통 상권 특징을 알려줘",
        "종로구와 영등포구의 매출 구조를 비교해줘",
        "영등포구에서 성장 중인 동은 어디야?",
        "서초구와 강남구 비교 — 어디가 더 나아?",
        "5개 구에서 임대료 대비 매출이 좋은 동 3곳을 추천해줘",
        "지금까지 분석을 종합하면 어느 구가 가장 유망해?",
    ],
]


def run_long_sessions() -> list[str]:
    """3개 장기 세션(8턴)을 실행하고 session_id를 반환한다."""
    from main import run_session, load_config

    config = load_config()
    session_ids = []

    for i, queries in enumerate(LONG_SESSION_QUERIES):
        label = chr(ord("A") + i)
        print(f"\n{'='*60}")
        print(f"  장기 세션 {label}: {len(queries)}턴")
        print(f"{'='*60}\n")

        try:
            state = run_session(queries, config=config)
            sid = state.get("session_id", "")
            session_ids.append(sid)
            print(f"\n  → Session {label}: {sid}")
        except Exception as e:
            print(f"\n  ✗ Session {label} 실패: {e}")

    return session_ids


def analyze_h4a(turns: list[dict]) -> dict:
    """H4a 검증: rot_risk/rot_velocity ↔ 4D 점수 상관분석.

    Args:
        turns: collect_turn_data()의 반환값.

    Returns:
        {
            "n_turns": int,
            "rot_risk_correlations": {dimension: rho},
            "rot_velocity_correlations": {dimension: rho},
            "gate_rho": float (rot_risk ↔ 4D 평균의 상관),
            "gate_verdict": "PASS" | "FAIL",
            "rot_velocity_discriminative": bool,
        }
    """
    score_names = {
        "completeness": "completeness_score",
        "efficiency": "efficiency_score",
        "relevance": "relevance_score",
        "consistency": "consistency_score",
    }

    # rot_risk ↔ 각 4D 차원
    rot_risk_corrs = {}
    rot_vel_corrs = {}
    for dim_label, score_name in score_names.items():
        # rot_risk pairs
        pairs = []
        for t in turns:
            rr = t["metadata"].get(ATTRS["context.rot_risk"])
            sc = t["scores"].get(score_name)
            if rr is not None and sc is not None:
                try:
                    pairs.append((float(rr), float(sc)))
                except (TypeError, ValueError):
                    pass
        if len(pairs) >= 3:
            rho = spearman_correlation([p[0] for p in pairs], [p[1] for p in pairs])
            rot_risk_corrs[dim_label] = rho

        # rot_velocity pairs
        pairs_v = []
        for t in turns:
            rv = t["metadata"].get(ATTRS["context.rot_velocity"])
            sc = t["scores"].get(score_name)
            if rv is not None and sc is not None:
                try:
                    pairs_v.append((float(rv), float(sc)))
                except (TypeError, ValueError):
                    pass
        if len(pairs_v) >= 3:
            rho_v = spearman_correlation([p[0] for p in pairs_v], [p[1] for p in pairs_v])
            rot_vel_corrs[dim_label] = rho_v

    # Gate 판단: rot_risk ↔ 4D 평균
    avg_pairs = []
    for t in turns:
        rr = t["metadata"].get(ATTRS["context.rot_risk"])
        scores_4d = [t["scores"].get(sn) for sn in score_names.values()]
        if rr is not None and all(s is not None for s in scores_4d):
            try:
                avg_4d = sum(float(s) for s in scores_4d) / 4
                avg_pairs.append((float(rr), avg_4d))
            except (TypeError, ValueError):
                pass

    gate_rho = spearman_correlation(
        [p[0] for p in avg_pairs], [p[1] for p in avg_pairs]
    ) if len(avg_pairs) >= 3 else None

    gate_verdict = "PASS" if gate_rho is not None and gate_rho < -0.4 else "FAIL"

    # rot_velocity 구분력: 양수(악화)/음수(개선) 둘 다 존재하는가?
    velocities = []
    for t in turns:
        rv = t["metadata"].get(ATTRS["context.rot_velocity"])
        if rv is not None:
            try:
                velocities.append(float(rv))
            except (TypeError, ValueError):
                pass
    has_positive = any(v > 0.001 for v in velocities)
    has_negative = any(v < -0.001 for v in velocities)
    discriminative = has_positive and has_negative

    return {
        "n_turns": len(turns),
        "n_with_rot": len(avg_pairs),
        "rot_risk_correlations": rot_risk_corrs,
        "rot_velocity_correlations": rot_vel_corrs,
        "gate_rho": gate_rho,
        "gate_verdict": gate_verdict,
        "rot_velocity_discriminative": discriminative,
        "velocities_summary": {
            "count": len(velocities),
            "positive": sum(1 for v in velocities if v > 0.001),
            "negative": sum(1 for v in velocities if v < -0.001),
            "zero": sum(1 for v in velocities if abs(v) <= 0.001),
        },
    }


def generate_h4a_report(result: dict, session_ids: list[str]) -> str:
    """H4a 검증 리포트를 Markdown으로 생성한다."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# H4a 검증 결과: Context Rot ↔ 4D 품질 상관",
        "",
        f"> **Date:** {now_str}",
        f"> **Context:** Phase 3 Step 3.1 — 장기 세션 데이터 기반 rot_risk ↔ 4D 상관분석",
        f"> **데이터:** {result['n_turns']}턴 ({len(session_ids)}개 세션), rot 데이터 포함 {result['n_with_rot']}턴",
        "",
        "---",
        "",
        f"## Gate 판정: **{result['gate_verdict']}**",
        "",
        f"- **rot_risk ↔ 4D 평균 상관:** rho = {result['gate_rho']}",
        f"- **기준:** rho < -0.4이면 PASS",
        f"- **판정:** {'rot_risk 증가 시 4D 품질 저하 경향 확인' if result['gate_verdict'] == 'PASS' else 'rot_risk와 4D 품질 간 유의미한 음의 상관 미확인'}",
        "",
        "---",
        "",
        "## rot_risk ↔ 4D 차원별 상관",
        "",
        "| 차원 | rho | 해석 |",
        "|------|:---:|------|",
    ]
    for dim, rho in result["rot_risk_correlations"].items():
        direction = "음의 상관 (rot↑ → 품질↓)" if rho and rho < -0.3 else "유의미하지 않음"
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"
        lines.append(f"| {dim} | {rho_str} | {direction} |")

    lines.extend([
        "",
        "## rot_velocity ↔ 4D 차원별 상관",
        "",
        "| 차원 | rho | 해석 |",
        "|------|:---:|------|",
    ])
    for dim, rho in result["rot_velocity_correlations"].items():
        direction = "음의 상관" if rho and rho < -0.3 else "유의미하지 않음"
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"
        lines.append(f"| {dim} | {rho_str} | {direction} |")

    vs = result["velocities_summary"]
    lines.extend([
        "",
        "## rot_velocity 구분력",
        "",
        f"- **구분력:** {'✅ 양방향 변화 존재' if result['rot_velocity_discriminative'] else '❌ 단방향만 존재'}",
        f"- **분포:** 양수(악화) {vs['positive']}개, 음수(개선) {vs['negative']}개, 0(무변화) {vs['zero']}개",
        "",
        "---",
        "",
        "## 세션 목록",
        "",
        "```",
    ])
    for sid in session_ids:
        lines.append(sid)
    lines.extend(["```", ""])

    return "\n".join(lines)


def main():
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="H4a 검증: rot_risk ↔ 4D 상관분석")
    parser.add_argument("--sessions", nargs="+", help="기존 세션 ID (실행 건너뜀)")
    parser.add_argument("--output", default=None, help="리포트 저장 경로")
    args = parser.parse_args()

    if args.sessions:
        session_ids = args.sessions
    else:
        session_ids = run_long_sessions()

    if not session_ids:
        print("세션이 없습니다.")
        sys.exit(1)

    # Langfuse 데이터 수집
    print(f"\n⏳ Langfuse ingestion 대기 (5초)...")
    time.sleep(5)

    print(f"\n데이터 수집 중...")
    turns = collect_turn_data(session_ids)
    print(f"  {len(turns)}턴 수집 완료")

    # H4a 분석
    result = analyze_h4a(turns)
    print(f"\n{'='*60}")
    print(f"  H4a Gate 판정: {result['gate_verdict']}")
    print(f"  rot_risk ↔ 4D 평균: rho = {result['gate_rho']}")
    print(f"  rot_velocity 구분력: {result['rot_velocity_discriminative']}")
    print(f"{'='*60}")

    # 리포트 생성
    report = generate_h4a_report(result, session_ids)
    if args.output:
        output_path = args.output
    else:
        analysis_dir = "docs/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        existing = [f for f in os.listdir(analysis_dir) if f.endswith(".md")]
        next_num = max((int(f.split("-")[0]) for f in existing if f[0].isdigit()), default=0) + 1
        output_path = f"{analysis_dir}/{next_num:02d}-H4A_VERIFICATION.md"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 리포트: {os.path.abspath(output_path)}")
    return result


if __name__ == "__main__":
    main()
