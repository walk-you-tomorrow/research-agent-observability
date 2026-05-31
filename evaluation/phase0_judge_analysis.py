"""
evaluation/phase0_judge_analysis.py --- Phase 0.1: 4D Judge 점수 분포 분석

역할:
    Langfuse에 저장된 기존 4D 평가 점수(completeness, efficiency, relevance, consistency)를
    추출하여 분포 통계와 차원 간 상관관계를 분석한다.

목적:
    - 판별력(Discriminability): 각 Judge의 표준편차 > 0.1 인지 확인
    - 독립성(Independence): 6개 차원 쌍의 |r| < 0.5 인지 확인

데이터 흐름:
    입력: Langfuse Cloud의 score_v_2 API (NUMERIC scores)
    출력: 콘솔 리포트 (통계 요약 + Pass/Fail 판정)

사용 방법:
    cd observable-research-agent
    source .venv/bin/activate
    python -m evaluation.phase0_judge_analysis
"""
import math
import time
from collections import defaultdict
from itertools import combinations

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

# --- 4D Score 이름 ---
SCORE_NAMES = [
    "completeness_score",
    "efficiency_score",
    "relevance_score",
    "consistency_score",
]

# --- 판정 기준 ---
DISCRIMINABILITY_THRESHOLD = 0.1   # 표준편차 최소 기준
INDEPENDENCE_THRESHOLD = 0.5       # |상관계수| 최대 기준


# ═══════════════════════════════════════
# STEP 1: Langfuse API 유틸리티
# ═══════════════════════════════════════

def _api_call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Langfuse API 호출을 rate limit 대응 재시도와 함께 실행한다.

    Args:
        fn: 호출할 API 함수.
        max_retries: 최대 재시도 횟수.

    Returns:
        API 응답 결과.
    """
    retryable_codes = {429, 502, 503, 504}
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ApiError as e:
            if e.status_code in retryable_codes and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [재시도] API 오류 {e.status_code}, {wait}초 후 재시도... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except (TimeoutError, OSError):
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [재시도] 네트워크 오류, {wait}초 후 재시도... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════
# STEP 2: Score 수집
# ═══════════════════════════════════════

def fetch_all_scores(client: Langfuse) -> dict[str, list[dict]]:
    """Langfuse에서 4D score를 모두 가져온다.

    score_v_2 API로 score_name별로 페이지네이션하여 전체 수집한다.

    Args:
        client: Langfuse 클라이언트.

    Returns:
        {score_name: [{trace_id, value, comment}, ...]} 딕셔너리.
    """
    all_scores: dict[str, list[dict]] = {name: [] for name in SCORE_NAMES}

    for score_name in SCORE_NAMES:
        print(f"  [{score_name}] 수집 중...")
        page = 1
        while True:
            resp = _api_call_with_retry(
                client.api.score_v_2.get,
                name=score_name,
                data_type="NUMERIC",
                limit=100,
                page=page,
            )
            for s in resp.data:
                all_scores[score_name].append({
                    "trace_id": s.trace_id,
                    "value": s.value,
                    "comment": getattr(s, "comment", None) or "",
                })
            fetched = len(resp.data)
            if fetched < 100:
                break
            page += 1
        print(f"    -> {len(all_scores[score_name])}건 수집 완료")

    return all_scores


# ═══════════════════════════════════════
# STEP 3: 기술 통계 계산
# ═══════════════════════════════════════

def compute_stats(values: list[float]) -> dict:
    """값 리스트의 기술 통계를 계산한다.

    Args:
        values: float 값 리스트.

    Returns:
        {count, mean, median, min, max, std, histogram} 딕셔너리.
    """
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None, "histogram": {}}

    sorted_vals = sorted(values)
    mean = sum(values) / n
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # 표준편차 (모집단)
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    # 히스토그램: 10개 구간 [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bins = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    for v in values:
        bin_idx = min(int(v * 10), 9)  # 1.0은 마지막 구간에 포함
        bin_key = f"{bin_idx/10:.1f}-{(bin_idx+1)/10:.1f}"
        bins[bin_key] += 1

    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "std": round(std, 4),
        "histogram": bins,
    }


# ═══════════════════════════════════════
# STEP 4: 상관계수 계산
# ═══════════════════════════════════════

def compute_correlation(x: list[float], y: list[float]) -> float | None:
    """Pearson 상관계수를 계산한다.

    Args:
        x: 첫 번째 차원의 값 리스트.
        y: 두 번째 차원의 값 리스트.

    Returns:
        상관계수 (-1.0 ~ 1.0). 계산 불가 시 None.
    """
    n = len(x)
    if n < 3:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / (n - 1))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / (n - 1))

    if std_x == 0 or std_y == 0:
        return None

    return round(cov / (std_x * std_y), 4)


def compute_correlation_matrix(
    all_scores: dict[str, list[dict]],
) -> dict[tuple[str, str], float | None]:
    """6개 차원 쌍의 상관계수를 계산한다.

    같은 trace_id를 가진 score끼리만 페어링한다.

    Args:
        all_scores: fetch_all_scores()의 반환값.

    Returns:
        {(dim_a, dim_b): correlation} 딕셔너리. 6쌍.
    """
    # trace_id → {score_name: value} 매핑 구성
    trace_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for score_name, scores in all_scores.items():
        for s in scores:
            trace_scores[s["trace_id"]][score_name] = s["value"]

    correlations = {}
    for dim_a, dim_b in combinations(SCORE_NAMES, 2):
        # 두 차원 모두 존재하는 trace만 추출
        paired_x = []
        paired_y = []
        for tid, scores_map in trace_scores.items():
            if dim_a in scores_map and dim_b in scores_map:
                paired_x.append(scores_map[dim_a])
                paired_y.append(scores_map[dim_b])

        label_a = dim_a.replace("_score", "")
        label_b = dim_b.replace("_score", "")
        correlations[(label_a, label_b)] = compute_correlation(paired_x, paired_y)

    return correlations


# ═══════════════════════════════════════
# STEP 5: 리포트 출력
# ═══════════════════════════════════════

def print_report(
    all_scores: dict[str, list[dict]],
    stats: dict[str, dict],
    correlations: dict[tuple[str, str], float | None],
) -> None:
    """분석 결과를 콘솔에 출력한다.

    Args:
        all_scores: 원본 score 데이터.
        stats: 차원별 기술 통계.
        correlations: 6개 차원 쌍의 상관계수.
    """
    print("\n" + "=" * 70)
    print("  Phase 0.1: 4D Judge 점수 분포 분석 리포트")
    print("=" * 70)

    # --- 총 데이터 요약 ---
    total_scores = sum(len(v) for v in all_scores.values())
    unique_traces = set()
    for scores in all_scores.values():
        for s in scores:
            unique_traces.add(s["trace_id"])
    print(f"\n총 score 수: {total_scores}건 (고유 trace: {len(unique_traces)}개)")

    # --- 차원별 기술 통계 ---
    print("\n" + "-" * 70)
    print("  [1] 차원별 기술 통계")
    print("-" * 70)

    discriminability_results = {}
    for score_name in SCORE_NAMES:
        dim = score_name.replace("_score", "")
        s = stats[score_name]
        print(f"\n  {dim.upper()}")
        if s["count"] == 0:
            print("    (데이터 없음)")
            discriminability_results[dim] = None
            continue

        print(f"    건수: {s['count']}")
        print(f"    평균: {s['mean']:.4f}")
        print(f"    중앙값: {s['median']:.4f}")
        print(f"    최솟값: {s['min']:.4f}")
        print(f"    최댓값: {s['max']:.4f}")
        print(f"    표준편차: {s['std']:.4f}")

        # 판별력 판정
        passed = s["std"] > DISCRIMINABILITY_THRESHOLD
        discriminability_results[dim] = passed
        verdict = "PASS" if passed else "FAIL"
        print(f"    판별력: {verdict} (std={s['std']:.4f}, 기준>{DISCRIMINABILITY_THRESHOLD})")

        # 히스토그램
        print(f"    분포:")
        max_count = max(s["histogram"].values()) if s["histogram"] else 1
        for bin_range, count in s["histogram"].items():
            bar_len = int(count / max(max_count, 1) * 30) if max_count > 0 else 0
            bar = "#" * bar_len
            print(f"      [{bin_range}] {count:3d} {bar}")

    # --- 상관계수 행렬 ---
    print("\n" + "-" * 70)
    print("  [2] 차원 간 상관계수 (Pearson r)")
    print("-" * 70)

    independence_results = {}
    for (dim_a, dim_b), r in correlations.items():
        if r is None:
            print(f"  {dim_a} vs {dim_b}: 계산 불가 (데이터 부족)")
            independence_results[(dim_a, dim_b)] = None
        else:
            passed = abs(r) < INDEPENDENCE_THRESHOLD
            verdict = "PASS" if passed else "FAIL"
            independence_results[(dim_a, dim_b)] = passed
            print(f"  {dim_a} vs {dim_b}: r={r:+.4f}  {verdict} (|r|<{INDEPENDENCE_THRESHOLD})")

    # --- 종합 판정 ---
    print("\n" + "=" * 70)
    print("  [3] 종합 판정")
    print("=" * 70)

    # 판별력 종합
    disc_pass = [k for k, v in discriminability_results.items() if v is True]
    disc_fail = [k for k, v in discriminability_results.items() if v is False]
    disc_na = [k for k, v in discriminability_results.items() if v is None]

    print(f"\n  판별력 (Discriminability): std > {DISCRIMINABILITY_THRESHOLD}")
    if disc_pass:
        print(f"    PASS: {', '.join(disc_pass)}")
    if disc_fail:
        print(f"    FAIL: {', '.join(disc_fail)}")
    if disc_na:
        print(f"    N/A:  {', '.join(disc_na)}")

    # 독립성 종합
    indep_pass = [f"{a}-{b}" for (a, b), v in independence_results.items() if v is True]
    indep_fail = [f"{a}-{b}" for (a, b), v in independence_results.items() if v is False]
    indep_na = [f"{a}-{b}" for (a, b), v in independence_results.items() if v is None]

    print(f"\n  독립성 (Independence): |r| < {INDEPENDENCE_THRESHOLD}")
    if indep_pass:
        print(f"    PASS: {', '.join(indep_pass)}")
    if indep_fail:
        print(f"    FAIL: {', '.join(indep_fail)}")
    if indep_na:
        print(f"    N/A:  {', '.join(indep_na)}")

    # 전체 결론
    all_disc_pass = len(disc_fail) == 0 and len(disc_na) == 0
    all_indep_pass = len(indep_fail) == 0 and len(indep_na) == 0

    print("\n" + "-" * 70)
    if total_scores == 0:
        print("  결론: 데이터 없음. 평가 실행 후 재분석 필요.")
    elif total_scores < 10:
        print(f"  결론: 데이터 부족 ({total_scores}건). 최소 10건 이상 필요. 참고 수치만 제공.")
    elif all_disc_pass and all_indep_pass:
        print("  결론: 모든 기준 충족. 4D Judge가 충분한 판별력과 독립성을 보유.")
    else:
        issues = []
        if disc_fail:
            issues.append(f"판별력 미달: {', '.join(disc_fail)}")
        if indep_fail:
            issues.append(f"독립성 미달: {', '.join(indep_fail)}")
        print(f"  결론: 일부 기준 미충족. {'; '.join(issues)}")

    print("=" * 70 + "\n")


# ═══════════════════════════════════════
# STEP 6: 메인 실행
# ═══════════════════════════════════════

def main():
    """Phase 0.1 분석을 실행한다."""
    print("Phase 0.1: 4D Judge 점수 분포 분석 시작")
    print("-" * 40)

    # Langfuse 클라이언트 생성
    print("Langfuse 연결 중...")
    try:
        client = Langfuse(timeout=120)
    except Exception as e:
        print(f"Langfuse 연결 실패: {e}")
        return

    # Score 수집
    print("\nScore 수집 시작:")
    all_scores = fetch_all_scores(client)

    # 기술 통계 계산
    stats = {}
    for score_name in SCORE_NAMES:
        values = [s["value"] for s in all_scores[score_name]]
        stats[score_name] = compute_stats(values)

    # 상관계수 계산
    correlations = compute_correlation_matrix(all_scores)

    # 리포트 출력
    print_report(all_scores, stats, correlations)


if __name__ == "__main__":
    main()
