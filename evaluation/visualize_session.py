"""
evaluation/visualize_session.py — 세션 Context Preservation 시각화

이 모듈은 Langfuse에 기록된 세션 데이터를 조회하여,
턴별 4D 품질 점수와 컨텍스트 진화 추이를 터미널 테이블로 출력한다.

역할:
    1. Langfuse Python SDK로 session_id 기반 traces/scores 조회
    2. 턴별 4D 점수 테이블 출력 (터미널)
    3. 컨텍스트 진화 추이 출력 (tokens, noise, continuity, rot_risk)
    4. 구조적 무결성 요약 출력

사용 방법:
    python -m evaluation.visualize_session --session-id sess_a1b2c3d4

데이터 흐름:
    입력: session_id (Langfuse session identifier)
    출력: 터미널 테이블 (stdout)

의존:
    - Langfuse SDK v3 (환경변수: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
    - PASS_THRESHOLD 값은 각 judge 모듈에서 import
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
from agent.monitoring_schema import THRESHOLDS as _SCHEMA_THRESHOLDS

load_dotenv()

# --- Pass/Fail 임계값 매핑 ---
# YAML 스키마에서 중앙 관리. score 이름 → 임계값 매핑.
THRESHOLDS = {
    "completeness_score": _SCHEMA_THRESHOLDS["completeness"],
    "efficiency_score": _SCHEMA_THRESHOLDS["efficiency"],
    "relevance_score": _SCHEMA_THRESHOLDS["relevance"],
    "consistency_score": _SCHEMA_THRESHOLDS["consistency"],
}

# --- 분석 범위 상수 (main.py의 MAX_TURNS_IN_SCOPE와 동일) ---
MAX_TURNS_IN_SCOPE = 5


def _format_score(value: float | None, threshold: float) -> str:
    """점수를 pass/fail 마크와 함께 포맷한다.

    Args:
        value: 0.0~1.0 점수. None이면 "N/A" 반환.
        threshold: pass/fail 임계값.

    Returns:
        포맷된 문자열. 예: "0.85 ✓" 또는 "0.60 ✗"
    """
    if value is None:
        return " N/A  "
    mark = "✓" if value >= threshold else "✗"
    return f"{value:.2f} {mark}"


def _format_tokens(tokens: int | None) -> str:
    """토큰 수를 읽기 좋은 형식으로 포맷한다.

    Args:
        tokens: 토큰 수. None이면 "?" 반환.

    Returns:
        포맷된 문자열. 예: "1.2K", "450"
    """
    if tokens is None:
        return "?"
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}K"
    return str(tokens)


def _api_call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Langfuse API 호출을 429 rate limit 대응 재시도와 함께 실행한다.

    Args:
        fn: 호출할 API 함수.
        max_retries: 최대 재시도 횟수 (기본 5회).

    Returns:
        API 응답 결과.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ApiError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16초 지수 백오프
                print(f"  ⏳ Rate limit (429), {wait}s 후 재시도 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def fetch_session_data(session_id: str) -> list[dict]:
    """Langfuse에서 세션의 모든 턴 데이터를 조회한다.

    Args:
        session_id: Langfuse session ID (예: "sess_a1b2c3d4").

    Returns:
        턴별 데이터 리스트. 각 항목은 turn_number, scores, metadata, events를 포함.
        turn_number 기준 오름차순 정렬.
    """
    langfuse = Langfuse(timeout=60)  # 회사 네트워크 SSL 핸드셰이크 지연 대응

    # 세션의 모든 trace 조회 (페이지네이션 처리)
    all_traces = []
    page = 1
    while True:
        response = _api_call_with_retry(
            langfuse.api.trace.list, session_id=session_id, limit=50, page=page
        )
        all_traces.extend(response.data)
        if len(response.data) < 50:
            break
        page += 1

    if not all_traces:
        print(f"세션 '{session_id}'에 trace가 없습니다.")
        sys.exit(1)

    # 각 trace에서 상세 정보 조회 (scores, metadata 포함)
    turns = []
    for trace_summary in all_traces:
        trace = _api_call_with_retry(
            langfuse.api.trace.get, trace_id=trace_summary.id
        )
        metadata = trace.metadata or {}

        # scores를 딕셔너리로 변환
        scores = {}
        if trace.scores:
            for score in trace.scores:
                scores[score.name] = score.value

        # 턴 번호 추출: metadata 또는 trace name에서
        turn_number = metadata.get(ATTRS["turn.number"], 0)
        if not turn_number and trace.name:
            # trace name에서 턴 번호 추출 시도 (예: "turn_3")
            try:
                parts = trace.name.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    turn_number = int(parts[-1])
            except (ValueError, IndexError):
                pass

        # 이벤트 수집: 모순, scope 제한 등 (v2/v3 trace 양립)
        from agent.monitoring_schema import (
            get_contradicts_from_metadata,
            get_contradiction_resolved_from_metadata,
        )
        events = []
        if get_contradicts_from_metadata(metadata):
            resolved = "해결됨" if get_contradiction_resolved_from_metadata(metadata) else "미해결"
            events.append(f"이전 턴과 모순 발생 → {resolved}")

        turns.append({
            "turn_number": turn_number,
            "trace_id": trace.id,
            "scores": scores,
            "metadata": metadata,
            "events": events,
            "wall_time_ms": metadata.get(ATTRS["turn.wall_time_ms"]),
        })

    # 턴 번호 기준 정렬 (0인 경우 timestamp 순서 유지)
    turns.sort(key=lambda t: (t["turn_number"] or 999, t["trace_id"]))

    # 턴 번호가 0인 항목에 순서대로 번호 부여
    for i, turn in enumerate(turns):
        if not turn["turn_number"]:
            turn["turn_number"] = i + 1

    return turns


def print_score_table(session_id: str, turns: list[dict]) -> None:
    """4D 점수 테이블을 터미널에 출력한다.

    Args:
        session_id: 세션 ID (헤더 표시용).
        turns: fetch_session_data()의 반환값.
    """
    total_turns = len(turns)
    print(f"\nContext Preservation Report — {session_id} ({total_turns} turns)")
    print("═" * 78)
    print(f" Turn │ Complete │ Efficient │ Relevant │ Consist. │ Verdict │ Time(ms)")
    print("──────┼──────────┼───────────┼──────────┼──────────┼─────────┼─────────")

    for turn in turns:
        s = turn["scores"]
        comp = _format_score(s.get("completeness_score"), THRESHOLDS["completeness_score"])
        eff = _format_score(s.get("efficiency_score"), THRESHOLDS["efficiency_score"])
        rel = _format_score(s.get("relevance_score"), THRESHOLDS["relevance_score"])
        cons = _format_score(s.get("consistency_score"), THRESHOLDS["consistency_score"])

        # overall verdict: 모든 점수가 threshold 이상이면 pass
        score_names = ["completeness_score", "efficiency_score",
                       "relevance_score", "consistency_score"]
        all_scores = [s.get(name) for name in score_names]
        all_thresholds = [THRESHOLDS[name] for name in score_names]
        if all(v is not None for v in all_scores):
            verdict = "pass" if all(
                v >= t for v, t in zip(all_scores, all_thresholds)
            ) else "FAIL"
        else:
            verdict = " N/A "

        wall_time = turn.get("wall_time_ms")
        time_str = f"{wall_time:>7,}" if wall_time else "    N/A"

        tn = turn["turn_number"]
        print(f"  {tn:>3} │ {comp} │  {eff}  │ {rel} │ {cons} │ {verdict:>7} │{time_str}")

    print("═" * 78)


def print_context_evolution(turns: list[dict]) -> None:
    """컨텍스트 진화 추이를 출력한다.

    Args:
        turns: fetch_session_data()의 반환값.
    """
    print("\nContext Evolution:")

    # 토큰 추이
    tokens_line = "  Tokens:       "
    tokens_line += " → ".join(
        _format_tokens(t["metadata"].get(ATTRS["context.total_tokens"]))
        for t in turns
    )
    print(tokens_line)

    # 노이즈 비율 추이
    noise_line = "  Noise ratio:  "
    noise_line += " → ".join(
        f"{t['metadata'].get(ATTRS['context.noise_ratio'], 0):.0%}"
        if t["metadata"].get(ATTRS["context.noise_ratio"]) is not None else "?"
        for t in turns
    )
    print(noise_line)

    # continuity_score 추이
    cont_line = "  Continuity:   "
    cont_line += " → ".join(
        f"{t['metadata'].get(ATTRS['gen_ai.context.continuity_score'], 1.0):.2f}"
        if t["metadata"].get(ATTRS.get("context.continuity_score", "context.continuity_score")) is not None else "1.0"
        for t in turns
    )
    print(cont_line)

    # rot_risk 추이
    rot_line = "  Rot risk:     "
    rot_line += " → ".join(
        f"{t['metadata'].get(ATTRS['context.rot_risk'], 0.0):.2f}"
        if t["metadata"].get(ATTRS["context.rot_risk"]) is not None else "0.0"
        for t in turns
    )
    print(rot_line)

    # turns_in_scope 추이
    scope_line = "  Turns in scope: "
    scope_parts = []
    for t in turns:
        tn = t["turn_number"]
        prior = tn - 1
        if prior <= 0:
            scope_parts.append("0/0")
        else:
            in_scope = min(prior, MAX_TURNS_IN_SCOPE)
            warning = " ⚠" if in_scope < prior else ""
            scope_parts.append(f"{in_scope}/{prior}{warning}")
    scope_line += " → ".join(scope_parts)
    print(scope_line)


def print_events(turns: list[dict]) -> None:
    """턴별 이벤트를 출력한다.

    Args:
        turns: fetch_session_data()의 반환값.
    """
    has_events = any(t["events"] for t in turns)
    if not has_events:
        return

    print("\nEvents:")
    for turn in turns:
        for event in turn["events"]:
            print(f"  Turn {turn['turn_number']}: ⚠ {event}")

    # scope 제한 이벤트
    for turn in turns:
        tn = turn["turn_number"]
        prior = tn - 1
        if prior > MAX_TURNS_IN_SCOPE:
            out_of_scope = prior - MAX_TURNS_IN_SCOPE
            print(f"  Turn {tn}: ⚠ 이전 턴 {prior}개 중 {MAX_TURNS_IN_SCOPE}개만 scope 내 "
                  f"(turn 1~{out_of_scope} out of scope)")


def print_structural_summary(turns: list[dict]) -> None:
    """구조적 무결성 요약을 출력한다.

    Args:
        turns: fetch_session_data()의 반환값.
    """
    total = len(turns)
    # continuity_score 기반 구조적 무결성 요약
    cont_values = [
        t["metadata"].get(ATTRS.get("context.continuity_score", "context.continuity_score"))
        for t in turns
        if t["metadata"].get(ATTRS.get("context.continuity_score", "context.continuity_score")) is not None
    ]
    if cont_values:
        avg_cont = sum(cont_values) / len(cont_values)
        status = "healthy" if avg_cont >= 0.9 else ("warning" if avg_cont >= 0.7 else "degraded ⚠")
        print(f"\nStructural Integrity: {total} turns, avg continuity={avg_cont:.3f} ({status})")
    else:
        print(f"\nStructural Integrity: {total} turns (no continuity data)")


def visualize_session(session_id: str) -> None:
    """세션 전체의 Context Preservation 리포트를 출력한다.

    Args:
        session_id: Langfuse session ID.
    """
    turns = fetch_session_data(session_id)
    print_score_table(session_id, turns)
    print_context_evolution(turns)
    print_events(turns)
    print_structural_summary(turns)
    print()


def main() -> None:
    """CLI 진입점. --session-id 인자로 세션 ID를 받아 시각화한다."""
    parser = argparse.ArgumentParser(
        description="Context Preservation 세션 시각화",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Langfuse session ID (예: sess_a1b2c3d4)",
    )
    args = parser.parse_args()
    visualize_session(args.session_id)


if __name__ == "__main__":
    main()
