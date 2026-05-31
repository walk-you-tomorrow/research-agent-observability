"""
evaluation/correlation_analysis.py — H1/H3 예비 검증: 상관분석 + 패턴 C 동작 확인

역할:
    1. H1 예비 검증: 소스 구성 비율(noise_ratio 등) ↔ 4D 점수 간 Spearman 상관 계산
    2. H3 예비 검증: 패턴 C 속성(conditions_preserved, lost_claims, fidelity_detail) 기록 확인
    3. Phase 2 신규 속성(rot_velocity, fidelity_detail, conditions_detail, lost_claims) 기록 검증

사용 방법:
    # 세션 ID 목록으로 분석
    python -m evaluation.correlation_analysis --sessions sess_abc sess_def

    # 최근 N일 내 전체 세션 분석
    python -m evaluation.correlation_analysis --days 7

데이터 흐름:
    입력: Langfuse session IDs
    출력: 터미널 리포트 + docs/analysis/NN-H1_H3_PRELIMINARY.md
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

from agent.monitoring_schema import ATTRS


# --- Langfuse API 재시도 ---
def _api_retry(fn, *args, max_retries: int = 8, **kwargs):
    """429/502 대응 지수 백오프 재시도."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ApiError as e:
            if e.status_code in (429, 502, 503, 504) and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⏳ Rate limit, {wait}s 후 재시도...")
                time.sleep(wait)
            else:
                raise


# --- 세션 데이터 수집 ---
def collect_turn_data(session_ids: list[str]) -> list[dict]:
    """Langfuse에서 세션별 턴 데이터를 수집한다.

    Args:
        session_ids: Langfuse session ID 리스트.

    Returns:
        턴별 데이터 리스트. 각 항목은 session_id, turn_number, scores, metadata를 포함.
    """
    langfuse = Langfuse(timeout=60)
    all_turns = []

    for sid in session_ids:
        print(f"  Fetching session: {sid}")
        # trace 목록 조회
        traces = []
        page = 1
        while True:
            resp = _api_retry(langfuse.api.trace.list, session_id=sid, limit=50, page=page)
            traces.extend(resp.data)
            if len(resp.data) < 50:
                break
            page += 1

        for trace_summary in traces:
            trace = _api_retry(langfuse.api.trace.get, trace_id=trace_summary.id)
            metadata = trace.metadata or {}
            scores = {}
            if trace.scores:
                for score in trace.scores:
                    scores[score.name] = score.value

            turn_number = metadata.get(ATTRS["turn.number"], 0)
            if not turn_number and trace.name:
                try:
                    parts = trace.name.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        turn_number = int(parts[-1])
                except (ValueError, IndexError):
                    pass

            all_turns.append({
                "session_id": sid,
                "turn_number": turn_number,
                "trace_id": trace.id,
                "scores": scores,
                "metadata": metadata,
            })

    all_turns.sort(key=lambda t: (t["session_id"], t["turn_number"]))
    return all_turns


# --- Spearman 상관계수 (scipy 불필요 — 수동 구현) ---
def _rank(values: list[float]) -> list[float]:
    """값 리스트의 순위를 반환한다 (동순위는 평균)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """Spearman 순위 상관계수를 계산한다.

    Args:
        x, y: 동일 길이 수치 리스트.

    Returns:
        -1.0 ~ 1.0 상관계수. 데이터 부족 시 None.
    """
    if len(x) != len(y) or len(x) < 3:
        return None
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    d_sq_sum = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return round(rho, 4)


# --- H1 예비 검증: 소스 구성 ↔ 4D 상관 ---
def analyze_h1(turns: list[dict]) -> dict:
    """H1 예비 검증: 소스 구성 비율과 4D 점수의 상관을 분석한다.

    분석 대상 context 속성:
        - noise_ratio: 이전 턴 토큰 비율
        - effective_noise_ratio: 인과 기반 실효 노이즈
        - window_utilization: 컨텍스트 윈도우 사용률
        - new_data_ratio: 새 데이터 비율
        - rot_risk: context rot 복합 지표
        - rot_velocity: rot 변화율 (A1)
        - information_density: 정보 밀도

    4D 점수:
        - completeness_score, efficiency_score, relevance_score, consistency_score

    Returns:
        {
            "n_turns": int,
            "correlations": [{context_attr, dimension, rho, n}],
            "summary": str,
        }
    """
    # 4D 점수명 매핑
    score_names = {
        "completeness": "completeness_score",
        "efficiency": "efficiency_score",
        "relevance": "relevance_score",
        "consistency": "consistency_score",
    }

    # 분석 대상 context 속성
    context_attrs = [
        ("noise_ratio", ATTRS["context.noise_ratio"]),
        ("effective_noise_ratio", ATTRS["context.effective_noise_ratio"]),
        ("window_utilization", ATTRS["context.window_utilization"]),
        ("new_data_ratio", ATTRS.get("context.new_data_ratio", "context.new_data_ratio")),
        ("rot_risk", ATTRS["context.rot_risk"]),
        ("rot_velocity", ATTRS["context.rot_velocity"]),
        ("information_density", ATTRS["context.information_density"]),
        ("fidelity_score", ATTRS["context.fidelity_score"]),
    ]

    correlations = []
    for attr_label, attr_key in context_attrs:
        for dim_label, score_name in score_names.items():
            # (x, y) 쌍 수집: 둘 다 값이 있는 턴만
            pairs = []
            for t in turns:
                x_val = t["metadata"].get(attr_key)
                y_val = t["scores"].get(score_name)
                if x_val is not None and y_val is not None:
                    try:
                        pairs.append((float(x_val), float(y_val)))
                    except (TypeError, ValueError):
                        pass

            if len(pairs) >= 3:
                x_list = [p[0] for p in pairs]
                y_list = [p[1] for p in pairs]
                rho = spearman_correlation(x_list, y_list)
                correlations.append({
                    "context_attr": attr_label,
                    "dimension": dim_label,
                    "rho": rho,
                    "n": len(pairs),
                })

    # 유의미한 상관 요약 (|rho| > 0.3)
    significant = [c for c in correlations if c["rho"] is not None and abs(c["rho"]) > 0.3]
    if significant:
        top = max(significant, key=lambda c: abs(c["rho"]))
        summary = f"가장 강한 상관: {top['context_attr']} ↔ {top['dimension']} (rho={top['rho']}, n={top['n']})"
    else:
        summary = "유의미한 상관(|rho| > 0.3)이 발견되지 않음 (데이터 부족 가능성)"

    return {
        "n_turns": len(turns),
        "correlations": correlations,
        "significant": significant,
        "summary": summary,
    }


# --- H3 예비 검증: 패턴 C 동작 확인 ---
def analyze_h3(turns: list[dict]) -> dict:
    """H3 예비 검증: 패턴 C 속성이 정상 기록되는지 확인한다.

    확인 대상 (Phase 2 Step 2.1 신규):
        - conditions_preserved (bool): F2 마커 확장 후 기록 여부
        - conditions_detail (dict): F2 마커별 보존 상세
        - key_claims_preserved (float): F3 비율 기반 전환
        - lost_claims (list): A3 손실 주장 추적
        - fidelity_detail (dict): F4+A7 구성 요소 분리
        - rot_velocity (float): A1 rot 진행 속도

    Returns:
        {
            "n_turns": int,
            "attr_coverage": {attr: {recorded, total, sample}},
            "h3_verdict": str,
        }
    """
    attrs_to_check = {
        "conditions_preserved": ATTRS.get("response.conditions_preserved", "response.conditions_preserved"),
        "conditions_detail": ATTRS.get("response.conditions_detail", "response.conditions_detail"),
        "key_claims_preserved": ATTRS["response.key_claims_preserved"],
        "lost_claims": ATTRS["response.lost_claims"],
        "fidelity_score": ATTRS["context.fidelity_score"],
        "fidelity_detail": ATTRS["context.fidelity_detail"],
        "rot_velocity": ATTRS["context.rot_velocity"],
    }

    coverage = {}
    for label, attr_key in attrs_to_check.items():
        recorded = 0
        sample = None
        for t in turns:
            val = t["metadata"].get(attr_key)
            if val is not None:
                recorded += 1
                if sample is None:
                    sample = val
        coverage[label] = {
            "recorded": recorded,
            "total": len(turns),
            "pct": round(recorded / len(turns) * 100, 1) if turns else 0,
            "sample": sample,
        }

    # 핵심 속성 기록률 판정: 최소 50% 이상 기록되면 PASS
    core_attrs = ["conditions_preserved", "key_claims_preserved", "fidelity_score", "rot_velocity"]
    core_pass = all(coverage[a]["pct"] >= 50 for a in core_attrs if a in coverage)

    # F2/F3 정확성 확인
    f2_ok = coverage.get("conditions_detail", {}).get("recorded", 0) > 0
    f3_ok = False
    for t in turns:
        val = t["metadata"].get(ATTRS["response.key_claims_preserved"])
        if val is not None and isinstance(val, float) and 0.0 <= val <= 1.0:
            f3_ok = True
            break

    verdict_parts = []
    if core_pass:
        verdict_parts.append("핵심 속성 기록률 PASS (≥50%)")
    else:
        verdict_parts.append("핵심 속성 기록률 FAIL (<50%)")
    if f2_ok:
        verdict_parts.append("F2(conditions_detail) 기록 확인")
    else:
        verdict_parts.append("F2(conditions_detail) 미기록")
    if f3_ok:
        verdict_parts.append("F3(claims 비율) 정상 (0.0~1.0 float)")
    else:
        verdict_parts.append("F3(claims 비율) 미확인")

    return {
        "n_turns": len(turns),
        "attr_coverage": coverage,
        "h3_verdict": " | ".join(verdict_parts),
    }


# --- 리포트 생성 ---
def generate_report(h1_result: dict, h3_result: dict, session_ids: list[str]) -> str:
    """H1/H3 예비 검증 결과를 Markdown 리포트로 생성한다.

    Returns:
        리포트 텍스트 (Markdown).
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# H1/H3 예비 검증 결과",
        f"",
        f"> **Date:** {now_str}",
        f"> **Context:** Phase 2 Step 2.3 — 시나리오 실행 데이터 기반 예비 상관분석 및 패턴 C 동작 확인",
        f"> **데이터:** {h1_result['n_turns']}턴 ({len(session_ids)}개 세션)",
        f"> **주의:** 50턴 미만이므로 예비적 결과. Phase 4에서 50턴 이상으로 완전 검증 예정.",
        f"",
        f"---",
        f"",
        f"## 1. H1 예비 검증: 소스 구성 ↔ 4D 점수 상관",
        f"",
        f"### 요약",
        f"",
        f"- **데이터 수:** {h1_result['n_turns']}턴",
        f"- **결과:** {h1_result['summary']}",
        f"",
    ]

    # 상관 테이블
    if h1_result["correlations"]:
        lines.append("### 상관계수 매트릭스")
        lines.append("")
        lines.append("| Context 속성 | Completeness | Efficiency | Relevance | Consistency |")
        lines.append("|-------------|:------------:|:----------:|:---------:|:-----------:|")

        # context_attr별 그룹화
        from collections import defaultdict
        by_attr = defaultdict(dict)
        for c in h1_result["correlations"]:
            by_attr[c["context_attr"]][c["dimension"]] = c

        for attr_label in by_attr:
            row = [f"`{attr_label}`"]
            for dim in ["completeness", "efficiency", "relevance", "consistency"]:
                c = by_attr[attr_label].get(dim)
                if c and c["rho"] is not None:
                    rho = c["rho"]
                    mark = "**" if abs(rho) > 0.3 else ""
                    row.append(f"{mark}{rho:.3f}{mark} (n={c['n']})")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # 유의미한 상관
    if h1_result["significant"]:
        lines.append("### 유의미한 상관 (|rho| > 0.3)")
        lines.append("")
        for c in sorted(h1_result["significant"], key=lambda x: -abs(x["rho"])):
            direction = "양의" if c["rho"] > 0 else "음의"
            lines.append(f"- `{c['context_attr']}` ↔ `{c['dimension']}`: rho={c['rho']} ({direction} 상관, n={c['n']})")
        lines.append("")
    else:
        lines.append("유의미한 상관(|rho| > 0.3) 없음. 데이터 부족 가능성.")
        lines.append("")

    # H3 검증
    lines.extend([
        "---",
        "",
        "## 2. H3 예비 검증: 패턴 C 동작 확인",
        "",
        f"### 판정: {h3_result['h3_verdict']}",
        "",
        "### 속성별 기록률",
        "",
        "| 속성 | 기록 턴/전체 턴 | 기록률 | 샘플 값 |",
        "|------|:--------------:|:------:|---------|",
    ])

    for label, info in h3_result["attr_coverage"].items():
        sample = str(info["sample"])[:60] if info["sample"] is not None else "—"
        lines.append(f"| `{label}` | {info['recorded']}/{info['total']} | {info['pct']}% | {sample} |")
    lines.append("")

    # 해석
    lines.extend([
        "### 해석",
        "",
        "- **conditions_detail**: F2 마커 확장이 정상 기록되면, 마커별 보존/손실 추적이 가능해진다.",
        "- **key_claims_preserved**: float(0.0~1.0)으로 기록되면 F3 비율 전환 성공.",
        "- **lost_claims**: 비어있어도 정상 (모든 claims가 보존된 경우). 값이 있으면 A3 동작 확인.",
        "- **fidelity_detail**: 3요소(cond_score, claims_ratio, compression_penalty) 분리 확인.",
        "- **rot_velocity**: 첫 턴 이후부터 0이 아닌 값이 나오면 A1 정상.",
        "",
    ])

    # 세션 ID 목록
    lines.extend([
        "---",
        "",
        "## 3. 데이터 출처",
        "",
        "```",
    ])
    for sid in session_ids:
        lines.append(sid)
    lines.extend([
        "```",
        "",
    ])

    return "\n".join(lines)


def main():
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="H1/H3 예비 검증: 상관분석 + 패턴 C 동작 확인")
    parser.add_argument("--sessions", nargs="+", help="세션 ID 목록")
    parser.add_argument("--days", type=int, default=None, help="최근 N일 내 세션 자동 검색")
    parser.add_argument("--output", default=None, help="리포트 저장 경로")
    args = parser.parse_args()

    session_ids = args.sessions or []

    if args.days and not session_ids:
        # 최근 N일 내 세션 자동 검색
        langfuse = Langfuse(timeout=60)
        since = datetime.now() - timedelta(days=args.days)
        print(f"최근 {args.days}일 내 세션 검색 중...")
        response = _api_retry(langfuse.api.sessions.list, limit=100)
        for sess in response.data:
            if hasattr(sess, "created_at") and sess.created_at and sess.created_at >= since:
                session_ids.append(sess.id)
        print(f"  {len(session_ids)}개 세션 발견")

    if not session_ids:
        print("세션 ID를 지정하거나 --days를 사용하세요.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  H1/H3 예비 검증 — {len(session_ids)}개 세션")
    print(f"{'='*60}\n")

    # 데이터 수집
    turns = collect_turn_data(session_ids)
    print(f"\n  총 {len(turns)}턴 수집 완료\n")

    if not turns:
        print("수집된 턴이 없습니다.")
        sys.exit(1)

    # H1 분석
    print("─── H1 예비 검증 ───")
    h1_result = analyze_h1(turns)
    print(f"  상관 분석 완료: {len(h1_result['correlations'])}개 쌍")
    print(f"  유의미한 상관: {len(h1_result['significant'])}개")
    print(f"  {h1_result['summary']}")

    # H3 분석
    print("\n─── H3 예비 검증 ───")
    h3_result = analyze_h3(turns)
    print(f"  판정: {h3_result['h3_verdict']}")
    for label, info in h3_result["attr_coverage"].items():
        print(f"    {label}: {info['recorded']}/{info['total']} ({info['pct']}%)")

    # 리포트 생성
    report = generate_report(h1_result, h3_result, session_ids)
    if args.output:
        output_path = args.output
    else:
        # 기존 분석 파일 번호 자동 결정
        analysis_dir = "docs/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        existing = [f for f in os.listdir(analysis_dir) if f.endswith(".md")]
        next_num = max((int(f.split("-")[0]) for f in existing if f[0].isdigit()), default=0) + 1
        output_path = f"{analysis_dir}/{next_num:02d}-H1_H3_PRELIMINARY.md"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 리포트 저장: {os.path.abspath(output_path)}")
    print(f"\n✅ H1/H3 예비 검증 완료")


if __name__ == "__main__":
    main()
