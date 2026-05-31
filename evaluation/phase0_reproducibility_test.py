"""
evaluation/phase0_reproducibility_test.py --- Phase 0.2: 4D Judge 재현성(Reproducibility) 테스트

역할:
    동일한 입력 데이터에 대해 4개 Judge를 각각 3회 실행하여
    점수의 분산(variance)이 허용 범위 내인지 검증한다.

목적:
    - 재현성(Reproducibility): 동일 입력 → 동일 점수 (variance < 0.1)
    - temperature=0.0 설정 하에서도 LLM 출력의 비결정성 정도를 측정

데이터 흐름:
    입력: Langfuse Cloud의 trace + observation metadata
    출력: 콘솔 리포트 (Judge별 variance + Pass/Fail 판정)

사용 방법:
    cd observable-research-agent
    source .venv/bin/activate
    python -m evaluation.phase0_reproducibility_test
"""
import math
import time

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError
from langchain_core.messages import HumanMessage

from agent.llm import create_llm, invoke_with_retry
from agent.parser import parse_llm_json
from agent.monitoring_schema import ATTRS
from evaluation.judge_completeness import build_completeness_input
from evaluation.judge_efficiency import build_efficiency_input
from evaluation.judge_relevance import build_relevance_input
from evaluation.judge_consistency import build_consistency_input
from evaluation.run_evaluation import JudgeScore

# --- 설정 ---
NUM_RUNS = 3                    # 동일 입력에 대한 반복 실행 횟수
MAX_TRACES = 5                  # 테스트할 최대 trace 수
VARIANCE_THRESHOLD = 0.1        # 허용 분산 임계값
LLM_CALL_DELAY_SEC = 2          # LLM 호출 간 대기 시간 (Rate Limit 대응)

# --- 4개 Judge 매핑 ---
JUDGES = {
    "completeness": build_completeness_input,
    "efficiency": build_efficiency_input,
    "relevance": build_relevance_input,
    "consistency": build_consistency_input,
}

# --- Langfuse 내부 metadata 키 (병합 시 제외) ---
_SKIP_METADATA_KEYS = {"resourceAttributes", "scope", "tags"}


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
# STEP 2: Trace 메타데이터 수집
# ═══════════════════════════════════════

def _extract_span_metadata(client: Langfuse, trace_id: str) -> dict:
    """trace의 observation들에서 span metadata를 병합하여 반환한다.

    dashboard/data_loader.py의 _extract_span_metadata()와 동일한 로직.
    SPAN observation의 metadata를 시간순으로 병합한다.

    Args:
        client: Langfuse 클라이언트.
        trace_id: Langfuse trace ID.

    Returns:
        병합된 metadata 딕셔너리.
    """
    all_observations = []
    page = 1
    while True:
        resp = _api_call_with_retry(
            client.api.observations.get_many,
            trace_id=trace_id,
            limit=100,
            page=page,
        )
        for obs in resp.data:
            all_observations.append({
                "name": obs.name,
                "type": obs.type,
                "metadata": getattr(obs, "metadata", None) or {},
                "start_time": str(obs.start_time) if obs.start_time else None,
            })
        if len(resp.data) < 100:
            break
        page += 1

    # SPAN만 추출, 시간순 정렬, 내부 노드 제외
    spans = [o for o in all_observations if o["type"] == "SPAN"]
    spans = [s for s in spans if s.get("name") not in ("_execute_turn", None, "")]
    spans.sort(key=lambda s: s.get("start_time") or "")

    merged: dict = {}
    for span in spans:
        metadata = span.get("metadata", {})
        filtered = {k: v for k, v in metadata.items() if k not in _SKIP_METADATA_KEYS}
        if filtered:
            merged.update(filtered)

    return merged


def fetch_traces_with_metadata(client: Langfuse, max_traces: int = MAX_TRACES) -> list[dict]:
    """Langfuse에서 4D 평가가 완료된 trace를 찾아 메타데이터를 수집한다.

    전략: 최근 세션을 조회하여 trace를 가져온 뒤,
    각 trace의 observation span metadata를 병합한다.
    4D score가 모두 있는 trace만 선별한다.

    Args:
        client: Langfuse 클라이언트.
        max_traces: 수집할 최대 trace 수.

    Returns:
        [{trace_id, metadata}] 리스트. metadata는 span 병합 결과.
    """
    print("최근 세션에서 trace 수집 중...")

    # 최근 세션 목록 조회
    sessions_resp = _api_call_with_retry(
        client.api.sessions.list, page=1, limit=10
    )

    candidates = []
    for session in sessions_resp.data:
        session_id = session.id
        print(f"  세션 {session_id} 조회 중...")

        try:
            session_detail = _api_call_with_retry(
                client.api.sessions.get, session_id
            )
        except Exception as e:
            print(f"    세션 조회 실패: {e}")
            continue

        if not session_detail.traces:
            continue

        for trace in session_detail.traces:
            trace_id = trace.id
            trace_metadata = trace.metadata or {}

            # 4D score 존재 여부 확인 (score_v_2 API)
            try:
                score_resp = _api_call_with_retry(
                    client.api.score_v_2.get,
                    trace_id=trace_id,
                    data_type="NUMERIC",
                    limit=10,
                    page=1,
                )
                score_names = {s.name for s in score_resp.data}
                has_all_4d = all(
                    f"{dim}_score" in score_names
                    for dim in ["completeness", "efficiency", "relevance", "consistency"]
                )
                if not has_all_4d:
                    continue
            except Exception:
                continue

            # Span metadata 병합
            print(f"    trace {trace_id[:12]}... span metadata 수집 중")
            span_metadata = _extract_span_metadata(client, trace_id)

            # trace-level metadata + span metadata 병합
            merged = {}
            merged.update(span_metadata)
            for k, v in trace_metadata.items():
                if k not in merged and k not in _SKIP_METADATA_KEYS:
                    merged[k] = v

            # 최소한의 메타데이터가 있는지 확인
            if len(merged) < 5:
                print(f"    -> metadata 부족 ({len(merged)}개), 건너뜀")
                continue

            candidates.append({
                "trace_id": trace_id,
                "metadata": merged,
            })
            print(f"    -> 수집 완료 (metadata {len(merged)}개 속성)")

            if len(candidates) >= max_traces:
                break

        if len(candidates) >= max_traces:
            break

    print(f"\n총 {len(candidates)}개 trace 수집 완료")
    return candidates


# ═══════════════════════════════════════
# STEP 3: Judge 재현성 테스트 실행
# ═══════════════════════════════════════

def run_reproducibility_test(
    traces: list[dict],
    num_runs: int = NUM_RUNS,
) -> dict[str, list[dict]]:
    """각 judge를 동일 입력으로 num_runs회 반복 실행한다.

    Args:
        traces: fetch_traces_with_metadata()의 반환값.
        num_runs: 반복 실행 횟수.

    Returns:
        {judge_name: [{trace_id, scores: [float, ...], variance: float}, ...]}
    """
    llm = create_llm(purpose="evaluation")
    results: dict[str, list[dict]] = {name: [] for name in JUDGES}

    total_calls = len(traces) * len(JUDGES) * num_runs
    call_count = 0

    for i, trace in enumerate(traces):
        trace_id = trace["trace_id"]
        trace_data = {"metadata": trace["metadata"]}

        print(f"\n--- Trace {i+1}/{len(traces)}: {trace_id[:16]}... ---")

        for judge_name, build_input_fn in JUDGES.items():
            # 프롬프트 구성 (동일한 입력 → 동일한 프롬프트)
            prompt = build_input_fn(trace_data)

            scores = []
            for run in range(num_runs):
                call_count += 1
                print(f"  [{judge_name}] run {run+1}/{num_runs} ({call_count}/{total_calls})")

                try:
                    response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
                    result = parse_llm_json(response.content, JudgeScore)
                    scores.append(result.score)
                    print(f"    score={result.score:.4f}  reasoning={result.reasoning[:60]}")
                except Exception as e:
                    print(f"    오류: {e}")
                    scores.append(None)

                # Rate Limit 대응 대기
                if call_count < total_calls:
                    time.sleep(LLM_CALL_DELAY_SEC)

            # None 제거 후 분산 계산
            valid_scores = [s for s in scores if s is not None]
            if len(valid_scores) >= 2:
                mean = sum(valid_scores) / len(valid_scores)
                variance = sum((s - mean) ** 2 for s in valid_scores) / (len(valid_scores) - 1)
            else:
                variance = None

            results[judge_name].append({
                "trace_id": trace_id,
                "scores": scores,
                "valid_scores": valid_scores,
                "variance": variance,
            })

    return results


# ═══════════════════════════════════════
# STEP 4: 리포트 출력
# ═══════════════════════════════════════

def print_report(results: dict[str, list[dict]]) -> None:
    """재현성 테스트 결과를 콘솔에 출력한다.

    Args:
        results: run_reproducibility_test()의 반환값.
    """
    print("\n" + "=" * 70)
    print("  Phase 0.2: 4D Judge 재현성(Reproducibility) 테스트 리포트")
    print("=" * 70)

    overall_pass = True

    for judge_name in JUDGES:
        trace_results = results[judge_name]
        print(f"\n  [{judge_name.upper()}]")
        print(f"  {'─' * 60}")

        variances = []
        for tr in trace_results:
            trace_short = tr["trace_id"][:16]
            scores_str = ", ".join(
                f"{s:.4f}" if s is not None else "N/A"
                for s in tr["scores"]
            )
            var_str = f"{tr['variance']:.6f}" if tr["variance"] is not None else "N/A"
            print(f"    trace {trace_short}...  scores=[{scores_str}]  var={var_str}")

            if tr["variance"] is not None:
                variances.append(tr["variance"])

        if variances:
            mean_var = sum(variances) / len(variances)
            max_var = max(variances)
            judge_pass = max_var < VARIANCE_THRESHOLD

            print(f"  평균 분산: {mean_var:.6f}")
            print(f"  최대 분산: {max_var:.6f}")
            print(f"  판정: {'PASS' if judge_pass else 'FAIL'} (max_var < {VARIANCE_THRESHOLD})")

            if not judge_pass:
                overall_pass = False
        else:
            print(f"  (유효한 분산 데이터 없음)")
            overall_pass = False

    # --- 종합 판정 ---
    print("\n" + "=" * 70)
    print("  [종합 판정]")
    print("=" * 70)
    if overall_pass:
        print(f"  결과: PASS — 모든 Judge의 점수 분산이 {VARIANCE_THRESHOLD} 미만")
    else:
        print(f"  결과: FAIL — 일부 Judge의 점수 분산이 {VARIANCE_THRESHOLD} 이상")
    print(f"  설정: {NUM_RUNS}회 반복, 분산 임계값={VARIANCE_THRESHOLD}")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════
# STEP 5: 메인 실행
# ═══════════════════════════════════════

def main():
    """Phase 0.2 재현성 테스트를 실행한다."""
    print("Phase 0.2: 4D Judge 재현성(Reproducibility) 테스트 시작")
    print("-" * 50)

    # Langfuse 클라이언트 생성
    print("Langfuse 연결 중...")
    try:
        client = Langfuse(timeout=120)
    except Exception as e:
        print(f"Langfuse 연결 실패: {e}")
        return

    # Trace 수집 (4D score가 있는 trace만)
    traces = fetch_traces_with_metadata(client, max_traces=MAX_TRACES)
    if not traces:
        print("테스트 가능한 trace가 없습니다. 4D 평가가 실행된 trace가 필요합니다.")
        return

    if len(traces) < 3:
        print(f"경고: {len(traces)}개 trace만 발견. 최소 3개 권장.")

    # 재현성 테스트 실행
    print(f"\n{len(traces)}개 trace x 4 judges x {NUM_RUNS}회 = "
          f"총 {len(traces) * 4 * NUM_RUNS}회 LLM 호출 예정")
    print(f"예상 소요 시간: ~{len(traces) * 4 * NUM_RUNS * (LLM_CALL_DELAY_SEC + 3) // 60}분")

    results = run_reproducibility_test(traces, num_runs=NUM_RUNS)

    # 리포트 출력
    print_report(results)


if __name__ == "__main__":
    main()
