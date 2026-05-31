"""
evaluation/h0_verification.py — H0 가설 이중 검증: 관측 + 실험

역할:
    H0 가설("Context Monitoring은 응답이 수집 컨텍스트에서 벗어날 때 감지할 수 있다")을
    2가지 독립 경로로 검증한다.

    1. 관측 경로: 기존 세션의 4D 점수 ↔ groundedness 상관분석
    2. 실험 경로: 오류 주입 시뮬레이션으로 groundedness 하락 확인

PASS 기준:
    1. 관측: Spearman(4D_avg, groundedness) > 0.4
    2. 실험: treatment hallucination_rate > 2× control

데이터 흐름:
    입력: Langfuse session IDs
    출력: H0 검증 결과 dict + 콘솔 리포트

사용 방법:
    python -m evaluation.h0_verification --sessions sess_a sess_b
    python -m evaluation.h0_verification --days 7
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
from evaluation.correlation_analysis import (
    collect_turn_data,
    spearman_correlation,
)
from evaluation.error_injection import (
    create_poisoned_gathered_data,
    run_injection_experiment,
)


# --- Langfuse API 재시도 ---
def _api_retry(fn, *args, max_retries: int = 8, **kwargs):
    """429 rate limit 대응 지수 백오프 재시도."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ApiError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry] Rate limit, {wait}s 후 재시도...")
                time.sleep(wait)
            else:
                raise


# --- 4D 점수명 ---
SCORE_NAMES = {
    "completeness": "completeness_score",
    "efficiency": "efficiency_score",
    "relevance": "relevance_score",
    "consistency": "consistency_score",
}


# ═══════════════════════════════════════
# STEP 1: 관측 경로 — 4D ↔ groundedness 상관분석
# ═══════════════════════════════════════

def verify_h0_observational(session_ids: list[str]) -> dict:
    """관측 경로: 기존 세션에서 4D 점수 ↔ groundedness 상관을 분석한다.

    Langfuse에 저장된 세션 데이터를 수집하고, 각 턴의:
    - 4D 평균 점수 (completeness + efficiency + relevance + consistency) / 4
    - groundedness 점수 (groundedness_checker로 실시간 계산)
    사이의 Spearman 상관을 계산한다.

    Args:
        session_ids: Langfuse session ID 리스트.

    Returns:
        {
            "n_turns": int,
            "n_with_both": int,        — 4D + groundedness 둘 다 있는 턴 수
            "spearman_rho": float | None,
            "individual_correlations": {dim: rho},  — 개별 차원별 상관
            "verdict": "PASS" | "FAIL" | "INSUFFICIENT_DATA",
            "details": str,
        }
    """
    from evaluation.groundedness_checker import check_groundedness

    turns = collect_turn_data(session_ids)

    # 4D 평균 + groundedness 쌍 추출
    pairs_avg = []
    individual_pairs: dict[str, list[tuple[float, float]]] = {
        dim: [] for dim in SCORE_NAMES
    }

    for t in turns:
        scores = t.get("scores", {})
        metadata = t.get("metadata", {})

        # response.final_text로 groundedness 실시간 계산
        final_text = metadata.get(ATTRS.get("response.final_text", "response.final_text"), "")
        if not final_text:
            continue

        # gathered_data 재구성 (trace metadata에서)
        gathered_proxy = []
        key_claims = metadata.get(ATTRS.get("response.key_claims", "response.key_claims"), [])
        conclusion = metadata.get(ATTRS.get("response.conclusion_summary", "response.conclusion_summary"), "")
        if key_claims or conclusion:
            gathered_proxy.append({
                "source": "trace_proxy",
                "tool_used": "proxy",
                "data_summary": conclusion + " " + " ".join(key_claims) if isinstance(key_claims, list) else str(key_claims),
                "token_count": 0,
            })

        # groundedness 계산
        grounded_result = check_groundedness(final_text, gathered_proxy, {})
        grounded_val = grounded_result.get("grounded_claim_ratio", 0.0)

        # 4D 점수 수집
        dim_scores = {}
        for dim_label, score_name in SCORE_NAMES.items():
            val = scores.get(score_name)
            if val is not None:
                try:
                    dim_scores[dim_label] = float(val)
                except (TypeError, ValueError):
                    pass

        if not dim_scores:
            continue

        # 4D 평균 계산
        avg_4d = sum(dim_scores.values()) / len(dim_scores)

        pairs_avg.append((avg_4d, grounded_val))

        # 개별 차원별 쌍 추가
        for dim_label, dim_val in dim_scores.items():
            individual_pairs[dim_label].append((dim_val, grounded_val))

    # Spearman 상관 계산
    n_with_both = len(pairs_avg)

    if n_with_both < 5:
        return {
            "n_turns": len(turns),
            "n_with_both": n_with_both,
            "spearman_rho": None,
            "individual_correlations": {},
            "verdict": "INSUFFICIENT_DATA",
            "details": f"4D + groundedness 쌍이 {n_with_both}개뿐 (최소 5개 필요)",
        }

    rho = spearman_correlation(
        [p[0] for p in pairs_avg],
        [p[1] for p in pairs_avg],
    )

    # 개별 차원별 상관
    ind_corrs = {}
    for dim_label, pairs in individual_pairs.items():
        if len(pairs) >= 5:
            r = spearman_correlation([p[0] for p in pairs], [p[1] for p in pairs])
            ind_corrs[dim_label] = r

    # 판정: rho > 0.4이면 PASS
    if rho is not None and rho > 0.4:
        verdict = "PASS"
        details = f"Spearman(4D_avg, groundedness) = {rho:.4f} > 0.4"
    elif rho is not None:
        verdict = "FAIL"
        details = f"Spearman(4D_avg, groundedness) = {rho:.4f} <= 0.4"
    else:
        verdict = "FAIL"
        details = "상관계수 계산 불가"

    return {
        "n_turns": len(turns),
        "n_with_both": n_with_both,
        "spearman_rho": rho,
        "individual_correlations": ind_corrs,
        "verdict": verdict,
        "details": details,
    }


# ═══════════════════════════════════════
# STEP 2: 실험 경로 — 오류 주입 시뮬레이션
# ═══════════════════════════════════════

def _extract_turn_data_for_experiment(session_ids: list[str]) -> list[dict]:
    """Langfuse 세션에서 실험용 턴 데이터를 추출한다.

    gathered_data 원본은 Langfuse에 저장되지 않으므로 (너무 큼),
    metadata에서 복원 가능한 정보를 활용한다:
    - response.final_text 또는 response.conclusion_summary → 응답 텍스트
    - source.contribution → 소스별 기여도 (gathered_data 대리)
    - response.key_claims → 주장 목록

    실제 gathered_data가 없으므로, source contribution 정보로
    가상의 gathered_data 항목을 구성한다.

    Args:
        session_ids: Langfuse session ID 리스트.

    Returns:
        실험용 턴 데이터 리스트. 각 항목:
        {
            "turn_id": str,
            "response_text": str,
            "gathered_data": list[dict],  — 메타데이터에서 복원한 근사치
        }
    """
    langfuse = Langfuse(timeout=60)
    experiment_data = []

    for sid in session_ids:
        print(f"  세션 {sid} 데이터 추출 중...")
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

            # 응답 텍스트 추출 (여러 가능한 필드에서 시도)
            response_text = (
                metadata.get("response.final_text", "")
                or metadata.get("response.conclusion_summary", "")
                or ""
            )

            if not response_text:
                # span metadata에서도 시도
                try:
                    obs_resp = _api_retry(
                        langfuse.api.observations.get_many,
                        trace_id=trace.id, limit=100, page=1,
                    )
                    for obs in obs_resp.data:
                        obs_meta = getattr(obs, "metadata", None) or {}
                        rt = (
                            obs_meta.get("response.final_text", "")
                            or obs_meta.get("response.conclusion_summary", "")
                        )
                        if rt:
                            response_text = rt
                            break
                except Exception:
                    pass

            if not response_text:
                continue

            # 가상 gathered_data 구성: source contribution에서 역추출
            gathered_data = []
            source_contrib = metadata.get("source.contribution", {})
            if isinstance(source_contrib, dict):
                for source_type, contrib_info in source_contrib.items():
                    gathered_data.append({
                        "source": source_type,
                        "tool_used": source_type,
                        "data_summary": str(contrib_info) if contrib_info else "",
                    })

            # key_claims가 있으면 가상 gathered_data에 포함
            key_claims = metadata.get("response.key_claims", [])
            if isinstance(key_claims, list) and key_claims:
                gathered_data.append({
                    "source": "analysis_claims",
                    "tool_used": "generate_analysis",
                    "data_summary": " / ".join(str(c) for c in key_claims),
                })

            # conclusion_summary를 가상 data_summary로 활용
            conclusion = metadata.get("response.conclusion_summary", "")
            if conclusion:
                gathered_data.append({
                    "source": "conclusion",
                    "tool_used": "respond_to_user",
                    "data_summary": conclusion,
                })

            if not gathered_data:
                continue

            experiment_data.append({
                "turn_id": f"{sid}/{trace.id}",
                "response_text": response_text,
                "gathered_data": gathered_data,
            })

    return experiment_data


def verify_h0_experimental(session_ids: list[str]) -> dict:
    """실험 경로: 오류 주입 시뮬레이션으로 H0를 검증한다.

    기존 세션의 response와 gathered_data(메타데이터에서 복원)를 사용하여:
    1. Control: groundedness(response, real_data)
    2. Treatment: groundedness(response, poisoned_data)
    3. 비교: treatment에서 grounded_claim_ratio가 유의미하게 하락하는지 확인

    API 호출 없이 오프라인으로 실행한다.

    Args:
        session_ids: Langfuse session ID 리스트.

    Returns:
        {
            "n_turns": int,
            "experiment_result": dict (run_injection_experiment 결과),
            "verdict": "PASS" | "WEAK_PASS" | "FAIL" | "INSUFFICIENT_DATA",
            "details": str,
        }
    """
    # groundedness_checker 임포트 (동시 개발 중이므로 지연 임포트)
    try:
        from evaluation.groundedness_checker import check_groundedness
    except ImportError:
        return {
            "n_turns": 0,
            "experiment_result": {},
            "verdict": "SKIP",
            "details": "groundedness_checker 모듈이 아직 없음 (동시 개발 중)",
        }

    # 실험 데이터 추출
    print("\n실험 데이터 추출 중...")
    turn_data_list = _extract_turn_data_for_experiment(session_ids)

    if len(turn_data_list) < 3:
        return {
            "n_turns": len(turn_data_list),
            "experiment_result": {},
            "verdict": "INSUFFICIENT_DATA",
            "details": f"실험 가능한 턴이 {len(turn_data_list)}개뿐 (최소 3개 필요)",
        }

    print(f"  {len(turn_data_list)}개 턴 추출 완료")

    # 오류 주입 실험 실행
    print("\n오류 주입 실험 실행 중...")
    experiment_result = run_injection_experiment(
        turn_data_list, check_groundedness,
    )

    # 판정: treatment hallucination이 control의 2배 이상이면 PASS
    verdict = experiment_result.get("overall_verdict", "FAIL")
    reason = experiment_result.get("verdict_reason", "")

    return {
        "n_turns": len(turn_data_list),
        "experiment_result": experiment_result,
        "verdict": verdict,
        "details": reason,
    }


# ═══════════════════════════════════════
# STEP 3: 이중 검증 통합
# ═══════════════════════════════════════

def run_full_h0_verification(session_ids: list[str]) -> dict:
    """관측 + 실험 이중 검증을 실행한다.

    PASS 기준 (양쪽 모두 충족):
    1. 관측: Spearman(4D_avg, groundedness) > 0.4
    2. 실험: treatment에서 groundedness 하락률 > 50% (null_injection)

    Args:
        session_ids: Langfuse session ID 리스트.

    Returns:
        {
            "observational": dict,
            "experimental": dict,
            "combined_verdict": "PASS" | "PARTIAL_PASS" | "FAIL",
            "summary": str,
        }
    """
    print("=" * 60)
    print("  H0 이중 검증: Context Monitoring → 이탈 감지")
    print("=" * 60)

    # 1. 관측 경로
    print("\n[1/2] 관측 경로: 4D ↔ groundedness 상관분석")
    obs_result = verify_h0_observational(session_ids)
    print(f"  결과: {obs_result['verdict']} — {obs_result['details']}")

    # 2. 실험 경로
    print("\n[2/2] 실험 경로: 오류 주입 시뮬레이션")
    exp_result = verify_h0_experimental(session_ids)
    print(f"  결과: {exp_result['verdict']} — {exp_result['details']}")

    # 종합 판정
    obs_pass = obs_result["verdict"] == "PASS"
    exp_pass = exp_result["verdict"] in ("PASS", "WEAK_PASS")
    exp_skip = exp_result["verdict"] == "SKIP"

    if obs_pass and exp_pass:
        combined = "PASS"
        summary = "양쪽 경로 모두 PASS — H0 지지"
    elif obs_pass or exp_pass:
        combined = "PARTIAL_PASS"
        parts = []
        if obs_pass:
            parts.append("관측 PASS")
        else:
            parts.append(f"관측 {obs_result['verdict']}")
        if exp_pass:
            parts.append("실험 PASS")
        elif exp_skip:
            parts.append("실험 SKIP (groundedness_checker 미완)")
        else:
            parts.append(f"실험 {exp_result['verdict']}")
        summary = f"한쪽만 PASS — {', '.join(parts)}"
    else:
        combined = "FAIL"
        summary = (
            f"양쪽 모두 미충족 — "
            f"관측: {obs_result['verdict']}, 실험: {exp_result['verdict']}"
        )

    result = {
        "observational": obs_result,
        "experimental": exp_result,
        "combined_verdict": combined,
        "summary": summary,
    }

    # 콘솔 요약
    print(f"\n{'=' * 60}")
    print(f"  H0 이중 검증 결과: [{combined}]")
    print(f"  {summary}")
    print(f"{'=' * 60}")

    return result


# ═══════════════════════════════════════
# CLI 진입점
# ═══════════════════════════════════════

def main():
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="H0 이중 검증: 관측 + 실험")
    parser.add_argument("--sessions", nargs="+", help="세션 ID 목록")
    parser.add_argument("--days", type=int, default=None, help="최근 N일 내 세션 자동 검색")
    parser.add_argument(
        "--observational-only", action="store_true",
        help="관측 경로만 실행 (실험 생략)",
    )
    parser.add_argument(
        "--experimental-only", action="store_true",
        help="실험 경로만 실행 (관측 생략)",
    )
    args = parser.parse_args()

    session_ids = args.sessions or []

    if args.days and not session_ids:
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

    if args.observational_only:
        result = verify_h0_observational(session_ids)
        print(f"\n관측 경로 결과: [{result['verdict']}] {result['details']}")
    elif args.experimental_only:
        result = verify_h0_experimental(session_ids)
        print(f"\n실험 경로 결과: [{result['verdict']}] {result['details']}")
    else:
        result = run_full_h0_verification(session_ids)

    # exit code: 0=PASS, 1=FAIL, 2=PARTIAL
    verdict = result.get("combined_verdict", result.get("verdict", "FAIL"))
    if verdict == "PASS":
        sys.exit(0)
    elif verdict in ("PARTIAL_PASS", "WEAK_PASS"):
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
