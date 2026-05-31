"""
evaluation/phase1_causal_verification.py — N2: causal_sources impact 정확성 검증

역할:
    evaluate_context.py의 causal_sources impact 계산이 정확한지 검증한다.
    Langfuse에서 다중 턴 세션을 조회하고, impact=0인 턴의 결론이
    실제로 현재 턴의 분석에 참조되지 않았는지 확인한다.

검증 방법:
    1. Langfuse에서 8+ 턴 세션을 조회
    2. 각 턴의 span metadata에서 context.causal_sources 추출
    3. impact=0인 소스 턴에 대해:
       - 해당 소스 턴의 key_claims를 가져옴
       - 현재 턴의 generate_analysis 출력에 key_claims 키워드가 존재하는지 확인
       - 키워드가 없으면 impact=0 판정이 정확 (True Negative)
       - 키워드가 있으면 impact=0 판정이 부정확 (False Negative)
    4. 정확도 = True Negative / (True Negative + False Negative)

성공 기준: 정확도 > 0.8

데이터 흐름:
    입력: Langfuse 세션 데이터
    출력: 터미널 검증 리포트

사용 방법:
    python -m evaluation.phase1_causal_verification [--session-id SESSION_ID]
"""
import argparse
import sys
import time

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

from agent.monitoring_schema import ATTRS

load_dotenv()


# ═══════════════════════════════════════
# Langfuse API 유틸리티
# ═══════════════════════════════════════

def _api_call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Langfuse API 호출을 rate limit + 타임아웃 대응 재시도와 함께 실행한다.

    Args:
        fn: 호출할 API 함수.
        max_retries: 최대 재시도 횟수.

    Returns:
        API 응답 결과. 실패 시 None.
    """
    retryable_codes = {429, 502, 503, 504}
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ApiError as e:
            if e.status_code in retryable_codes and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry] API 오류 {e.status_code}, {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                raise
        except (TimeoutError, OSError) as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry] 네트워크 오류, {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════
# 한국어 조사 제거 (evaluate_context.py와 동일 로직)
# ═══════════════════════════════════════

_KOREAN_PARTICLES = [
    "에서는", "으로는", "이랑은",
    "에서", "으로", "이랑", "하고", "보다", "까지",
    "부터", "처럼", "마저", "조차", "에게",
    "은", "는", "이", "가", "을", "를",
    "에", "와", "과", "의", "도", "로", "만",
]


def _strip_particles(word: str) -> str:
    """한국어 단어에서 trailing 조사를 제거하여 어근을 반환한다."""
    for p in _KOREAN_PARTICLES:
        if word.endswith(p) and len(word) - len(p) >= 2:
            return word[:-len(p)]
    return word


def _claim_retained(claim: str, context_text: str) -> bool:
    """하나의 claim 문장이 컨텍스트 텍스트에 보존되어 있는지 확인한다.

    evaluate_context.py의 _claim_retained()과 동일한 로직을 사용하여
    검증 결과의 일관성을 보장한다.

    Args:
        claim: key_claim 텍스트.
        context_text: 검색 대상 텍스트 (소문자 변환 완료).

    Returns:
        claim의 핵심 키워드 중 하나 이상이 context_text에 존재하면 True.
    """
    for word in claim.lower().split():
        stem = _strip_particles(word)
        if len(stem) > 1 and stem in context_text:
            return True
    return False


# ═══════════════════════════════════════
# 세션 데이터 조회
# ═══════════════════════════════════════

def _fetch_session_turns(client: Langfuse, session_id: str) -> list[dict]:
    """세션의 모든 턴 데이터를 조회한다. (trace + observations)

    Args:
        client: Langfuse 클라이언트.
        session_id: Langfuse session ID.

    Returns:
        턴별 데이터 리스트. 각 항목:
        {turn_number, trace_id, metadata, observations}
    """
    # sessions.get()으로 trace 목록 조회
    session = _api_call_with_retry(client.api.sessions.get, session_id)
    if not session.traces:
        return []

    turns = []
    for trace in session.traces:
        metadata = trace.metadata or {}
        turn_number = metadata.get(ATTRS["turn.number"], 0)
        if not turn_number and trace.name:
            try:
                parts = trace.name.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    turn_number = int(parts[-1])
            except (ValueError, IndexError):
                pass

        turns.append({
            "turn_number": turn_number,
            "trace_id": trace.id,
            "metadata": metadata,
        })

    # 턴 번호 순 정렬
    turns.sort(key=lambda t: (t["turn_number"] or 999, t["trace_id"]))
    for i, turn in enumerate(turns):
        if not turn["turn_number"]:
            turn["turn_number"] = i + 1

    return turns


def _fetch_span_metadata(client: Langfuse, trace_id: str) -> dict:
    """trace의 모든 SPAN metadata를 병합하여 반환한다.

    Args:
        client: Langfuse 클라이언트.
        trace_id: Langfuse trace ID.

    Returns:
        병합된 metadata 딕셔너리.
    """
    _SKIP_KEYS = {"resourceAttributes", "scope", "tags"}
    merged = {}

    try:
        page = 1
        while True:
            response = _api_call_with_retry(
                client.api.observations.get_many,
                trace_id=trace_id,
                limit=100,
                page=page,
            )
            for obs in response.data:
                if obs.type != "SPAN":
                    continue
                obs_meta = getattr(obs, "metadata", None) or {}
                filtered = {k: v for k, v in obs_meta.items() if k not in _SKIP_KEYS}
                merged.update(filtered)
            if len(response.data) < 100:
                break
            page += 1
    except Exception as e:
        print(f"  [warn] span metadata 조회 실패: {e}")

    return merged


# ═══════════════════════════════════════
# 검증 로직
# ═══════════════════════════════════════

def _verify_causal_sources(
    turns: list[dict],
    turn_span_metadata: dict[int, dict],
) -> dict:
    """causal_sources의 impact=0 판정 정확성을 검증한다.

    검증 방법:
    - impact=0인 소스 턴의 key_claims를 추출
    - 현재 턴의 generate_analysis 출력(analysis.claims)에
      해당 키워드가 존재하는지 독립적으로 확인
    - 키워드가 없으면 → impact=0 판정이 정확 (true_negative)
    - 키워드가 있으면 → impact=0 판정이 부정확 (false_negative)

    Args:
        turns: 턴별 데이터 리스트.
        turn_span_metadata: 턴 번호 → span metadata 매핑.

    Returns:
        검증 결과 딕셔너리.
    """
    # 턴 번호 → key_claims 매핑 구축
    turn_claims: dict[int, list[str]] = {}
    for turn in turns:
        tn = turn["turn_number"]
        meta = turn_span_metadata.get(tn, {})
        # key_claims는 response.key_claims에 기록됨
        claims = meta.get("response.key_claims", [])
        if isinstance(claims, list) and claims:
            turn_claims[tn] = claims

    # 검증 결과 수집
    true_negatives = []   # impact=0이고 실제로도 참조 없음 (정확)
    false_negatives = []  # impact=0이지만 실제로는 참조 있음 (부정확)
    total_impact_zero = 0
    total_impact_positive = 0
    all_judgments = []

    for turn in turns:
        tn = turn["turn_number"]
        meta = turn_span_metadata.get(tn, {})

        # causal_sources 추출
        causal_sources = meta.get("context.causal_sources", [])
        if not isinstance(causal_sources, list) or not causal_sources:
            continue

        # 현재 턴의 분석 내용 구성 (검증 대상 텍스트)
        # generate_analysis의 출력을 사용하여 독립적으로 검증
        analysis_text_parts = []
        # analysis.claims_count 같은 요약 대신 실제 metadata에서 텍스트 추출
        # conclusion_summary가 가장 신뢰할 수 있는 분석 결과 텍스트
        conclusion = meta.get("response.conclusion_summary", "")
        if conclusion:
            analysis_text_parts.append(conclusion)
        # key_claims도 분석 결과의 핵심 주장
        current_claims = meta.get("response.key_claims", [])
        if isinstance(current_claims, list):
            analysis_text_parts.extend(current_claims)
        # 사용자 질문도 포함 (질문에 이전 턴 키워드가 언급될 수 있음)
        user_query = turn["metadata"].get(ATTRS.get("turn.user_query", "turn.user_query"), "")
        if user_query:
            analysis_text_parts.append(user_query)

        analysis_text = " ".join(analysis_text_parts).lower()

        if not analysis_text.strip():
            # 분석 텍스트가 없으면 검증 불가
            continue

        for cs in causal_sources:
            if not isinstance(cs, dict):
                continue
            source_turn = cs.get("turn", 0)
            impact = cs.get("impact", 0)
            claims_total = cs.get("claims_total", 0)
            claims_retained = cs.get("claims_retained", 0)

            if impact == 0:
                total_impact_zero += 1
                # 소스 턴의 key_claims 가져오기
                source_claims = turn_claims.get(source_turn, [])
                if not source_claims:
                    # key_claims가 없으면 검증 불가 — 건너뜀
                    continue

                # 독립 검증: 소스 턴의 key_claims가 현재 턴 분석에 나타나는지
                independently_found = any(
                    _claim_retained(claim, analysis_text)
                    for claim in source_claims
                )

                judgment = {
                    "current_turn": tn,
                    "source_turn": source_turn,
                    "impact": impact,
                    "claims_total": claims_total,
                    "source_claims_count": len(source_claims),
                }

                if not independently_found:
                    # impact=0이고 독립 검증에서도 참조 없음 → 정확
                    true_negatives.append(judgment)
                    judgment["verdict"] = "CORRECT (true negative)"
                else:
                    # impact=0이지만 독립 검증에서 참조 발견 → 부정확
                    false_negatives.append(judgment)
                    judgment["verdict"] = "INCORRECT (false negative)"

                all_judgments.append(judgment)
            else:
                total_impact_positive += 1

    total_verified = len(true_negatives) + len(false_negatives)
    accuracy = len(true_negatives) / total_verified if total_verified > 0 else None

    return {
        "total_impact_zero": total_impact_zero,
        "total_impact_positive": total_impact_positive,
        "total_verified": total_verified,
        "true_negatives": len(true_negatives),
        "false_negatives": len(false_negatives),
        "accuracy": accuracy,
        "judgments": all_judgments,
    }


# ═══════════════════════════════════════
# 세션 검색
# ═══════════════════════════════════════

def _find_multi_turn_sessions(client: Langfuse, min_turns: int = 8, limit: int = 20) -> list[str]:
    """다중 턴 세션을 검색한다.

    Args:
        client: Langfuse 클라이언트.
        min_turns: 최소 턴 수.
        limit: 검색할 최대 세션 수.

    Returns:
        min_turns 이상의 턴을 가진 세션 ID 리스트.
    """
    try:
        response = _api_call_with_retry(
            client.api.sessions.list, page=1, limit=limit
        )
    except Exception as e:
        print(f"  [error] 세션 목록 조회 실패: {e}")
        return []

    multi_turn_sessions = []
    for s in response.data:
        try:
            session_detail = _api_call_with_retry(client.api.sessions.get, s.id)
            trace_count = len(session_detail.traces) if session_detail.traces else 0
            if trace_count >= min_turns:
                multi_turn_sessions.append(s.id)
                print(f"  ✓ {s.id}: {trace_count}턴")
        except Exception:
            continue

    return multi_turn_sessions


# ═══════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="N2: causal_sources impact=0 정확성 검증"
    )
    parser.add_argument(
        "--session-id",
        help="검증할 특정 세션 ID. 미지정 시 8+ 턴 세션을 자동 검색.",
    )
    parser.add_argument(
        "--min-turns",
        type=int, default=4,
        help="최소 턴 수 (기본: 4). 8+ 턴 세션이 없으면 낮출 수 있음.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("N2: causal_sources impact=0 정확성 검증")
    print("=" * 60)

    client = Langfuse(timeout=120)

    # 세션 결정
    if args.session_id:
        session_ids = [args.session_id]
        print(f"\n지정 세션: {args.session_id}")
    else:
        print(f"\n{args.min_turns}+ 턴 세션 검색 중...")
        session_ids = _find_multi_turn_sessions(client, min_turns=args.min_turns)
        if not session_ids:
            print(f"\n[결과] {args.min_turns}+ 턴 세션을 찾을 수 없습니다.")
            print("--min-turns 값을 낮추거나 --session-id를 지정하세요.")
            sys.exit(1)
        print(f"\n{len(session_ids)}개 세션 발견")

    # 전체 검증 결과 집계
    global_true_neg = 0
    global_false_neg = 0
    global_total_zero = 0
    global_total_positive = 0
    all_session_results = []

    for session_id in session_ids:
        print(f"\n{'─' * 50}")
        print(f"세션: {session_id}")
        print(f"{'─' * 50}")

        # 턴 데이터 조회
        turns = _fetch_session_turns(client, session_id)
        print(f"  턴 수: {len(turns)}")

        if len(turns) < 2:
            print("  [skip] 2턴 미만 — causal_sources가 없음")
            continue

        # 각 턴의 span metadata 조회
        print("  span metadata 조회 중...")
        turn_span_metadata: dict[int, dict] = {}
        for turn in turns:
            tn = turn["turn_number"]
            span_meta = _fetch_span_metadata(client, turn["trace_id"])
            # trace metadata와 병합 (span이 우선)
            merged = {**turn["metadata"], **span_meta}
            turn_span_metadata[tn] = merged

        # causal_sources 유무 확인
        has_causal = False
        for tn, meta in turn_span_metadata.items():
            causal = meta.get("context.causal_sources", [])
            if isinstance(causal, list) and causal:
                has_causal = True
                break

        if not has_causal:
            print("  [skip] causal_sources 데이터 없음 (첫 턴만 있거나 미기록)")
            continue

        # 검증 실행
        result = _verify_causal_sources(turns, turn_span_metadata)
        all_session_results.append({"session_id": session_id, **result})

        # 세션별 결과 출력
        print(f"\n  --- 검증 결과 ---")
        print(f"  impact=0 총 건수: {result['total_impact_zero']}")
        print(f"  impact>0 총 건수: {result['total_impact_positive']}")
        print(f"  검증 가능 건수: {result['total_verified']}")
        print(f"  True Negative (정확): {result['true_negatives']}")
        print(f"  False Negative (부정확): {result['false_negatives']}")
        if result['accuracy'] is not None:
            status = "PASS" if result['accuracy'] >= 0.8 else "FAIL"
            print(f"  정확도: {result['accuracy']:.1%} [{status}]")
        else:
            print("  정확도: N/A (검증 가능 건수 없음)")

        # 개별 판정 상세
        if result['judgments']:
            print(f"\n  --- 개별 판정 ---")
            for j in result['judgments']:
                print(
                    f"  T{j['current_turn']} ← T{j['source_turn']}: "
                    f"impact={j['impact']}, claims={j['source_claims_count']} "
                    f"→ {j['verdict']}"
                )

        global_true_neg += result['true_negatives']
        global_false_neg += result['false_negatives']
        global_total_zero += result['total_impact_zero']
        global_total_positive += result['total_impact_positive']

    # 전체 요약
    print(f"\n{'=' * 60}")
    print("전체 검증 요약")
    print(f"{'=' * 60}")
    print(f"검증 세션 수: {len(all_session_results)}")
    print(f"impact=0 총 건수: {global_total_zero}")
    print(f"impact>0 총 건수: {global_total_positive}")

    total_verified = global_true_neg + global_false_neg
    print(f"검증 가능 건수: {total_verified}")
    print(f"True Negative (정확): {global_true_neg}")
    print(f"False Negative (부정확): {global_false_neg}")

    if total_verified > 0:
        accuracy = global_true_neg / total_verified
        status = "PASS" if accuracy >= 0.8 else "FAIL"
        print(f"\n최종 정확도: {accuracy:.1%}")
        print(f"판정: [{status}] (기준: > 80%)")
        sys.exit(0 if status == "PASS" else 1)
    else:
        print("\n[결과] 검증 가능한 impact=0 판정을 찾지 못했습니다.")
        print("원인: causal_sources가 없거나, impact=0인 소스의 key_claims가 없음")
        print("더 많은 턴을 가진 세션으로 재시도하거나 --min-turns를 낮추세요.")
        sys.exit(2)


if __name__ == "__main__":
    main()
