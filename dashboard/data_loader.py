"""
dashboard/data_loader.py — Langfuse API 데이터 조회 + Streamlit 캐싱

역할:
    Langfuse Cloud에서 세션/트레이스/관측 데이터를 조회하고,
    Streamlit 캐시로 API 호출을 최소화한다.
    evaluation/visualize_session.py의 fetch_session_data() 로직을 포팅하되,
    sys.exit() 대신 None 반환 + 에러 메시지로 대체한다.

데이터 흐름:
    입력: session_id, trace_id (Langfuse 식별자)
    출력: list[dict] (턴별 데이터, observation 데이터)

의존:
    - Langfuse SDK v3 (환경변수: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
    - agent.monitoring_schema — ATTRS
"""
import time

import streamlit as st
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

from agent.monitoring_schema import ATTR_META, ATTRS

load_dotenv()

# --- Langfuse 내부 metadata 키 (병합 시 제외) ---
_SKIP_METADATA_KEYS = {"resourceAttributes", "scope", "tags"}


# ═══════════════════════════════════════
# API 호출 유틸리티
# ═══════════════════════════════════════

def _api_call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Langfuse API 호출을 rate limit + 타임아웃 대응 재시도와 함께 실행한다.

    Args:
        fn: 호출할 API 함수.
        max_retries: 최대 재시도 횟수 (기본 5회).

    Returns:
        API 응답 결과. 실패 시 None.
    """
    # 429 (rate limit) + 502/503/504 (서버 일시 오류) 모두 재시도
    retryable_codes = {429, 502, 503, 504}
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ApiError as e:
            if e.status_code in retryable_codes and attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise
        except (TimeoutError, OSError) as e:
            # ReadTimeout, ConnectTimeout 등 네트워크 타임아웃 재시도
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise


def _get_langfuse_client() -> Langfuse | None:
    """Langfuse 클라이언트를 생성한다. 연결 실패 시 None 반환."""
    try:
        return Langfuse(timeout=120)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def check_connection() -> bool:
    """Langfuse 연결 상태를 확인한다. 60초 캐싱으로 rate limit 보호.

    Returns:
        연결 성공 시 True, 실패 시 False.
    """
    client = _get_langfuse_client()
    if client is None:
        return False
    try:
        _api_call_with_retry(client.api.sessions.list, page=1, limit=1)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════
# 세션 목록 조회
# ═══════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def list_recent_sessions(limit: int = 20) -> list[dict]:
    """최근 세션 목록을 조회한다.

    Args:
        limit: 조회할 최대 세션 수.

    Returns:
        세션 목록. 각 항목: {id, created_at, trace_count}.
        실패 시 빈 리스트.
    """
    client = _get_langfuse_client()
    if client is None:
        return []

    try:
        response = _api_call_with_retry(
            client.api.sessions.list, page=1, limit=limit
        )
        sessions = []
        for s in response.data:
            sessions.append({
                "id": s.id,
                "created_at": str(s.created_at) if hasattr(s, "created_at") else "",
            })
        return sessions
    except Exception:
        return []


# ═══════════════════════════════════════
# 세션 데이터 조회 (trace-level)
# ═══════════════════════════════════════

def _build_turn(trace) -> dict:
    """trace 객체에서 턴 데이터를 추출한다.

    Args:
        trace: Langfuse Trace 객체 (sessions.get 또는 trace.get 반환값).

    Returns:
        턴 딕셔너리: {turn_number, trace_id, scores, metadata, events, wall_time_ms}.
    """
    metadata = trace.metadata or {}

    # scores를 딕셔너리로 변환 (sessions.get 반환 trace에는 scores 없음)
    scores = {}
    if getattr(trace, "scores", None):
        for score in trace.scores:
            scores[score.name] = score.value

    # 턴 번호 추출: metadata 또는 trace name에서
    turn_number = metadata.get(ATTRS["turn.number"], 0)
    if not turn_number and trace.name:
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

    # alignment score → metadata mirror (_build_turn에서 일관 적용)
    for _attr in ("analysis.query_alignment", "response.query_alignment"):
        if _attr in scores and _attr not in metadata:
            metadata[_attr] = scores[_attr]

    return {
        "turn_number": turn_number,
        "trace_id": trace.id,
        "scores": scores,
        "score_comments": {},   # score_v_2 API로 별도 조회 후 병합
        "score_verdicts": {},   # score_v_2 API로 별도 조회 후 병합
        "metadata": metadata,
        "events": events,
        "wall_time_ms": metadata.get(ATTRS["turn.wall_time_ms"]),
    }


def _sort_and_number_turns(turns: list[dict]) -> list[dict]:
    """턴 리스트를 정렬하고 번호가 없는 항목에 순서 번호를 부여한다."""
    turns.sort(key=lambda t: (t["turn_number"] or 999, t["trace_id"]))
    for i, turn in enumerate(turns):
        if not turn["turn_number"]:
            turn["turn_number"] = i + 1
    return turns


def _mirror_alignment_scores_to_metadata(turn: dict) -> None:
    """Langfuse Score로 부착된 alignment 점수를 turn['metadata']에 mirror 주입한다.

    배경: `run_evaluation.py`는 `analysis.query_alignment` / `response.query_alignment` 를
    `langfuse.create_score(...)` 로 부착한다. 그러나 대시보드 차트
    (`query_alignment_trend`, `build_drift_stats` 등)는 `turn['metadata']` 에서 attribute를
    찾는다. 두 namespace를 연결하는 mirror 단계.

    Pattern I/II/III 진단을 위한 `query.session_continuity` 는 evaluate_context span의
    metadata에 이미 부착되어 있으므로 별도 처리 불필요 (enriched data가 자동 병합).
    """
    scores = turn.get("scores") or {}
    meta = turn.setdefault("metadata", {})
    for attr_key in ("analysis.query_alignment", "response.query_alignment"):
        if attr_key in scores and attr_key not in meta:
            meta[attr_key] = scores[attr_key]


def _fetch_trace_scores(client: Langfuse, trace_id: str) -> dict:
    """score_v_2 API로 단일 trace의 score를 조회한다.

    NUMERIC 점수(값 + comment)와 CATEGORICAL verdict를 모두 가져온다.

    Args:
        client: Langfuse 클라이언트.
        trace_id: Langfuse trace ID.

    Returns:
        {"values": {name: float}, "comments": {name: str}, "verdicts": {dim: str}}.
        실패 시 빈 구조.
    """
    values: dict[str, float] = {}
    comments: dict[str, str] = {}
    verdicts: dict[str, str] = {}
    try:
        resp = _api_call_with_retry(
            client.api.score_v_2.get,
            trace_id=trace_id,
            limit=50,
            page=1,
        )
        for s in resp.data:
            dt = getattr(s, "data_type", None)
            if dt == "CATEGORICAL":
                # "completeness_verdict" → "completeness"
                dim = s.name.replace("_verdict", "")
                verdicts[dim] = getattr(s, "string_value", "") or ""
            else:
                values[s.name] = s.value
                comment = getattr(s, "comment", None)
                if comment:
                    comments[s.name] = comment
    except Exception:
        pass
    return {"values": values, "comments": comments, "verdicts": verdicts}


@st.cache_data(ttl=300, show_spinner=False)
def load_session_data(session_id: str) -> list[dict] | None:
    """Langfuse에서 세션의 모든 턴 데이터를 조회한다.

    sessions.get()으로 세션 내 trace를 가져온다 (대용량 세션에서 trace.list가 504 타임아웃되는 문제 회피).
    sessions.get()의 trace에는 scores가 없으므로 score_v_2 API로 별도 조회하여 병합한다.
    sessions.get() 실패 시 trace.list + trace.get 방식으로 폴백한다.

    Args:
        session_id: Langfuse session ID (예: "sess_a1b2c3d4").

    Returns:
        턴별 데이터 리스트. 각 항목: {turn_number, trace_id, scores, metadata, events, wall_time_ms}.
        실패 시 None.
    """
    client = _get_langfuse_client()
    if client is None:
        return None

    # --- 1차: sessions.get()으로 trace 목록 + metadata 직접 추출 ---
    # trace.list()는 대용량 세션에서 504 타임아웃 발생하므로 sessions.get()을 우선 사용
    try:
        session = _api_call_with_retry(client.api.sessions.get, session_id)
        if not session.traces:
            return []
        turns = [_build_turn(t) for t in session.traces]

        # sessions.get() trace에는 scores가 없으므로 score_v_2 API로 trace별 조회
        for turn in turns:
            score_data = _fetch_trace_scores(client, turn["trace_id"])
            turn["scores"] = score_data["values"]
            turn["score_comments"] = score_data["comments"]
            turn["score_verdicts"] = score_data["verdicts"]
            _mirror_alignment_scores_to_metadata(turn)

        return _sort_and_number_turns(turns)
    except Exception:
        pass

    # --- 2차 폴백: trace.list() + trace.get() (소규모 세션용) ---
    all_traces = []
    page = 1
    while True:
        try:
            response = _api_call_with_retry(
                client.api.trace.list, session_id=session_id, limit=50, page=page
            )
        except Exception:
            return None
        all_traces.extend(response.data)
        if len(response.data) < 50:
            break
        page += 1

    if not all_traces:
        return []

    turns = []
    for trace_summary in all_traces:
        try:
            trace = _api_call_with_retry(
                client.api.trace.get, trace_id=trace_summary.id
            )
        except Exception:
            turns.append(_build_turn(trace_summary))
            continue
        turns.append(_build_turn(trace))

    return _sort_and_number_turns(turns)


# ═══════════════════════════════════════
# Observation 조회 (Lazy — 턴 선택 시)
# ═══════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_observations(trace_id: str) -> list[dict] | None:
    """단일 trace의 모든 observation (SPAN + GENERATION)을 조회한다.

    Tab 4 (LLM 호출 로그)와 Tab 5 (노드 Contribution)에서 사용.
    Lazy loading: 사용자가 턴을 선택할 때만 호출한다.

    Args:
        trace_id: Langfuse trace ID.

    Returns:
        observation 리스트. 각 항목:
        {id, name, type, input, output, model, metadata, start_time, end_time, usage, parent_observation_id}.
        실패 시 None.
    """
    client = _get_langfuse_client()
    if client is None:
        return None

    try:
        all_observations = []
        page = 1
        while True:
            response = _api_call_with_retry(
                client.api.observations.get_many,
                trace_id=trace_id,
                limit=100,
                page=page,
            )
            for obs in response.data:
                all_observations.append({
                    "id": obs.id,
                    "name": obs.name,
                    "type": obs.type,
                    "input": obs.input,
                    "output": obs.output,
                    "model": getattr(obs, "model", None),
                    "metadata": getattr(obs, "metadata", None) or {},
                    "start_time": str(obs.start_time) if obs.start_time else None,
                    "end_time": str(obs.end_time) if obs.end_time else None,
                    "usage": {
                        "input": getattr(obs.usage, "input", None) if obs.usage else None,
                        "output": getattr(obs.usage, "output", None) if obs.usage else None,
                        "total": getattr(obs.usage, "total", None) if obs.usage else None,
                    } if obs.usage else {},
                    "parent_observation_id": getattr(obs, "parent_observation_id", None),
                })
            if len(response.data) < 100:
                break
            page += 1

        # start_time 기준 정렬
        all_observations.sort(key=lambda o: o["start_time"] or "")
        return all_observations
    except Exception:
        return None


# ═══════════════════════════════════════
# Span metadata 병합 (Phase A 핵심)
# ═══════════════════════════════════════

def _extract_span_metadata(observations: list[dict]) -> dict:
    """SPAN observation의 metadata를 iteration-aware로 병합한다.

    각 노드는 서로 다른 namespace의 속성을 기록하므로 키 충돌이 없다.
    같은 노드가 여러 번 실행된 경우 (re-gather):
    - merged: 마지막 iteration 값 (최종 결정)
    - iterations: 모든 iteration의 전체 기록 (progression 추적)

    Args:
        observations: load_observations()의 반환값.

    Returns:
        {
            "merged": {attr_key: value, ...},  — 64개 속성의 최종 값
            "iterations": {node_name: [{iteration: 1, ...}, ...], ...}  — 다중 실행 기록
        }
    """
    merged: dict = {}
    node_iterations: dict[str, list[dict]] = {}
    node_counts: dict[str, int] = {}

    # SPAN만 추출, start_time 순 정렬
    spans = [o for o in observations if o["type"] == "SPAN"]
    spans = [s for s in spans if s.get("name") not in ("_execute_turn", None, "")]
    spans.sort(key=lambda s: s.get("start_time") or "")

    for span in spans:
        name = span.get("name", "unknown")
        metadata = span.get("metadata", {})

        # Langfuse 내부 키 제외
        filtered = {k: v for k, v in metadata.items() if k not in _SKIP_METADATA_KEYS}

        if not filtered:
            continue

        # iteration 추적
        node_counts[name] = node_counts.get(name, 0) + 1
        iteration = node_counts[name]

        # merged에 덮어쓰기 (마지막 iteration이 최종 값)
        merged.update(filtered)

        # 다중 실행 노드만 iterations에 기록
        if iteration >= 1:
            if name not in node_iterations:
                node_iterations[name] = []
            iter_record = {"iteration": iteration}
            iter_record.update(filtered)
            node_iterations[name].append(iter_record)

    # 단일 실행 노드는 iterations에서 제거 (불필요한 데이터)
    iterations = {
        name: records for name, records in node_iterations.items()
        if len(records) > 1
    }

    return {"merged": merged, "iterations": iterations}


@st.cache_data(ttl=300, show_spinner=False)
def load_enriched_session_data(session_id: str) -> list[dict] | None:
    """세션 데이터를 span metadata까지 병합하여 반환한다.

    load_session_data()로 trace-level 데이터(22개 속성)를 가져온 뒤,
    각 턴의 observations에서 span metadata를 추출하여 64개 속성으로 확장한다.

    Args:
        session_id: Langfuse session ID.

    Returns:
        enriched 턴별 데이터 리스트. 각 항목에 추가:
        - metadata: 64개 속성 (trace + span 병합)
        - iterations: 다중 실행 노드의 iteration별 기록
        - observations: raw observation 리스트 (Tab 3, 5에서 재사용)
        실패 시 None.
    """
    turns = load_session_data(session_id)
    if turns is None:
        return None

    for turn in turns:
        observations = load_observations(turn["trace_id"])
        if observations:
            span_data = _extract_span_metadata(observations)
            # span metadata를 trace metadata에 병합 (trace가 이미 가진 키는 유지)
            for key, value in span_data["merged"].items():
                if key not in turn["metadata"]:
                    turn["metadata"][key] = value
            turn["iterations"] = span_data["iterations"]
            turn["observations"] = observations
        else:
            turn["iterations"] = {}
            turn["observations"] = []

    return turns


# ═══════════════════════════════════════
# 다중 세션 데이터 (Tab 4 상관 분석용)
# ═══════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def load_multi_session_data(session_ids: list[str]) -> list[dict]:
    """여러 세션의 enriched 턴 데이터를 집계한다.

    Tab 4 Attribute Impact의 상관 분석에 사용.

    Args:
        session_ids: 세션 ID 리스트.

    Returns:
        모든 세션의 턴 데이터를 합친 리스트.
    """
    all_turns = []
    for sid in session_ids:
        turns = load_enriched_session_data(sid)
        if turns:
            for turn in turns:
                turn["session_id"] = sid
            all_turns.extend(turns)
    return all_turns
