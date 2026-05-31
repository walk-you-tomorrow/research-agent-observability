"""
dashboard/views/composition.py — Tab 3: 구성 (Composition)

관측 질문:
    "LLM에게 건네진 컨텍스트는 어떤 재료로, 어떤 비율로 구성되었고,
     무엇이 빠졌으며, 그 비율이 턴마다 어떻게 이동하는가?"

스토리 흐름 (Charter §3.1 / analysis/37 / Tab 3 검토 종합):
    KPI strip       — 5초 진단용 4개 핵심 지표 (항상 노출)
    재료 구성       — 100% stacked + Prev% delta sparkline (Hero)
    Plan vs 실행   — dot matrix + 5컬럼 표 + drill-down + reason 카테고리
    구성 품질 지표  — 완전성·효율성·관련성 badge (항상 펼침)
    소스 Risk Map   — 기여도 × 탈락률 2D scatter
    Retrieval 깊이   — chunk-level drilldown (placeholder)

데이터 흐름:
    입력: turns (list[dict]) — enriched session data
    출력: Streamlit UI

품질 차원: 완전성 (gather.* 비율), 효율성 (token 분해 + window), 관련성 (noise, exclusion).
일관성은 측정&진단 탭(Tab 6)에 위임.
"""
from collections import Counter

import pandas as pd
import streamlit as st

from agent.monitoring_schema import ATTRS, DASHBOARD_THRESHOLDS
from dashboard.charts import (
    STATUS_COLORS,
    plan_vs_actual_dot_matrix,
    previous_turns_delta_sparkline,
    source_contribution_scatter,
    token_composition,             # 절대값 stacked bar
    token_composition_normalized,  # 100% stacked bar (비율)
)
from dashboard.widgets.cross_tab_link import (
    alert_with_tab_link as _alert_with_tab_link,
    TAB_IDX as _TAB_IDX,
)

# --- 도구 → 소스 매핑 (Plan vs 실행 dot matrix 입력) ---
_TOOL_TO_SOURCE = {
    "pandas_query": "csv", "calculate": "csv",
    "rag_search": "rag", "rag_deep_read": "rag",
    "rag_global_summary": "rag", "rag_compare": "rag",
    "web_search": "web",
    "api_query": "api",
    "lookup_previous": "memory",
}

# --- 토큰 구성 4소스 라벨 ↔ attribute (v3 SSOT, 절대값 표 / KPI 계산 공통) ---
# v3에서 turn_conclusions_tokens는 previous_turns_tokens에 합산 (yaml 폐기 명시).
_SOURCE_KEYS: list[tuple[str, str]] = [
    ("System Prompt",    "context.source.system_prompt_tokens"),
    ("Query Analysis",   "context.source.query_analysis_tokens"),
    ("Gathered Data",    "context.source.gathered_data_tokens"),
    ("Previous Turns",   "context.source.previous_turns_tokens"),
]

# --- 탈락/잘림 사유 카테고리 분류 (자유 텍스트 → enum, P1) ---
_REASON_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("relevance",    ("noise", "irrelevant", "unrelated", "off-topic",
                       "low_relevance", "관련", "노이즈", "무관")),
    ("redundancy",   ("duplicate", "redundant", "overlap", "중복", "유사")),
    ("freshness",    ("stale", "outdated", "old", "expired", "오래", "낡은")),
    ("token_budget", ("token", "size", "budget", "limit", "truncat",
                       "토큰", "용량", "예산", "잘림")),
    ("error",        ("error", "fail", "missing", "unavailable",
                       "오류", "실패", "없음")),
]


def _categorize_reason(reason: str | None) -> str:
    """자유 텍스트 사유를 5개 카테고리 enum으로 분류한다 (대소문자 무시, 한·영 키워드)."""
    if not reason:
        return "other"
    r = reason.lower()
    for cat, kws in _REASON_KEYWORDS:
        if any(k in r for k in kws):
            return cat
    return "other"


def _severity(value: float | None, ok_max: float, warn_max: float,
              direction: str = "lower_better") -> str:
    """수치를 OK/WARN/CRIT/— 텍스트 severity로 변환한다.

    Args:
        value: 평가 대상 값.
        ok_max: OK 상한 (lower_better) 또는 OK 하한 (higher_better).
        warn_max: WARN 상한 (lower_better) 또는 WARN 하한 (higher_better).
        direction: "lower_better" — 값이 작을수록 좋음. "higher_better" — 클수록 좋음.

    Returns:
        "OK" / "WARN" / "CRIT" / "—"
    """
    if value is None or not isinstance(value, (int, float)):
        return "—"
    if direction == "lower_better":
        if value <= ok_max:
            return "OK"
        if value <= warn_max:
            return "WARN"
        return "CRIT"
    # higher_better
    if value >= ok_max:
        return "OK"
    if value >= warn_max:
        return "WARN"
    return "CRIT"


def _kpi_card(label: str, value: str, severity: str, hint: str = "") -> str:
    """KPI strip 한 칸 — HTML 카드. severity ∈ {ok, warn, crit, na}."""
    color = {
        "ok":   STATUS_COLORS["good"],
        "warn": STATUS_COLORS["warn"],
        "crit": STATUS_COLORS["bad"],
        "na":   STATUS_COLORS["na"],
    }.get(severity, STATUS_COLORS["na"])
    hint_html = (
        f'<div style="font-size:10px;color:#888;margin-top:2px;">{hint}</div>'
        if hint else ""
    )
    return (
        f'<div style="text-align:center;padding:12px 8px;background:#1e1e1e;'
        f'border-radius:8px;border-left:4px solid {color};">'
        f'<div style="font-size:11px;color:#999;letter-spacing:0.3px;">{label}</div>'
        f'<div style="font-size:24px;font-weight:700;color:{color};line-height:1.2;">{value}</div>'
        f'{hint_html}</div>'
    )


def _render_kpi_strip(turns: list[dict]) -> None:
    """5초 진단용 KPI strip — 완전성·Prev%·도구 갭·노이즈 (last turn 기준)."""
    last = turns[-1] if turns else {}
    last_meta = last.get("metadata", {}) if last else {}

    items = last_meta.get(ATTRS["gather.items_collected"], 0) or 0
    excl = last_meta.get(ATTRS["gather.items_excluded"], 0) or 0
    completeness = items / (items + excl) if (items + excl) > 0 else None

    total = last_meta.get(ATTRS["context.total_tokens"]) or 0
    prev = last_meta.get(ATTRS["context.source.previous_turns_tokens"]) or 0
    prev_ratio = (prev / total) if total > 0 else None

    gap_turns = 0
    for t in turns:
        meta = t.get("metadata", {})
        plan_set = set(meta.get(ATTRS["query.tool_plan"], []) or [])
        actual_set = set(meta.get(ATTRS["gather.tools_called"], []) or [])
        if plan_set and (plan_set ^ actual_set):
            gap_turns += 1

    noise = last_meta.get(ATTRS["context.noise_ratio"])

    warn = DASHBOARD_THRESHOLDS["previous_turns_warn"]
    danger = DASHBOARD_THRESHOLDS["previous_turns_danger"]
    noise_warn = DASHBOARD_THRESHOLDS["noise_warn"]

    comp_sev_str = _severity(completeness, 0.8, 0.6, direction="higher_better") if completeness is not None else "—"
    comp_sev = comp_sev_str.lower() if comp_sev_str != "—" else "na"

    if prev_ratio is None:
        prev_sev = "na"
    elif prev_ratio < warn:
        prev_sev = "ok"
    elif prev_ratio < danger:
        prev_sev = "warn"
    else:
        prev_sev = "crit"

    gap_sev = "ok" if gap_turns == 0 else ("warn" if gap_turns <= 2 else "crit")

    if noise is None:
        noise_sev = "na"
    elif noise <= 0.3:
        noise_sev = "ok"
    elif noise <= noise_warn:
        noise_sev = "warn"
    else:
        noise_sev = "crit"

    cards = [
        _kpi_card(
            "완전성 (last)",
            f"{completeness:.0%}" if completeness is not None else "—",
            comp_sev,
            "수집/(수집+제외)",
        ),
        _kpi_card(
            "Prev Turns % (last)",
            f"{prev_ratio:.0%}" if prev_ratio is not None else "—",
            prev_sev,
            f"warn ≥ {warn:.0%}",
        ),
        _kpi_card(
            "도구 갭 (세션)",
            f"{gap_turns} / {len(turns)}",
            gap_sev,
            "planned ≠ actual 턴 수",
        ),
        _kpi_card(
            "노이즈 (last)",
            f"{noise:.2f}" if isinstance(noise, (int, float)) else "—",
            noise_sev,
            f"warn ≥ {noise_warn}",
        ),
    ]
    cols = st.columns(4)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)


def _render_hero(turns: list[dict]) -> None:
    """토큰 구성 Hero — 비율/절대값 토글 + Prev% delta + 격상된 경고 + 절대값 expander."""
    st.markdown("### 토큰 구성")
    st.caption(
        "각 턴에서 LLM에 전달된 컨텍스트의 토큰 구성 — Charter §3.1 "
        "\"무엇이 들어오고, 빠지고, 비율이 어떻게 이동하는가\""
    )

    # 비율 ↔ 절대값 보기 토글
    view = st.radio(
        "보기",
        options=["비율 (%)", "절대값 (tokens)"],
        horizontal=True,
        label_visibility="collapsed",
        key="composition_token_view",
    )
    if view == "비율 (%)":
        st.plotly_chart(token_composition_normalized(turns), use_container_width=True)
    else:
        st.plotly_chart(token_composition(turns), use_container_width=True)

    warn = DASHBOARD_THRESHOLDS["previous_turns_warn"]
    danger = DASHBOARD_THRESHOLDS["previous_turns_danger"]
    st.plotly_chart(
        previous_turns_delta_sparkline(turns, warn=warn, danger=danger),
        use_container_width=True,
    )

    warn_turns, danger_turns = [], []
    for turn in turns:
        meta = turn.get("metadata", {})
        total = meta.get(ATTRS["context.total_tokens"]) or 0
        prev = meta.get(ATTRS["context.source.previous_turns_tokens"]) or 0
        if total > 0:
            ratio = prev / total
            if ratio >= danger:
                danger_turns.append((turn.get("turn_number", "?"), ratio))
            elif ratio >= warn:
                warn_turns.append((turn.get("turn_number", "?"), ratio))

    if danger_turns:
        worst_r = max(r for _, r in danger_turns)
        _alert_with_tab_link(
            "error",
            f"이전 턴 점유 최대 <b>{worst_r:.0%}</b> (danger ≥ {danger:.0%})",
            tab_key="impact",
            link_label="③ Rot 위험",
        )
    elif warn_turns:
        worst_r = max(r for _, r in warn_turns)
        _alert_with_tab_link(
            "warn",
            f"이전 턴 점유 최대 <b>{worst_r:.0%}</b> (warn ≥ {warn:.0%})",
            tab_key="impact",
            link_label="③ Rot 위험",
        )

    with st.expander("턴별 토큰 구성 절대값", expanded=False):
        rows = []
        for turn in turns:
            meta = turn.get("metadata", {})
            tn = turn.get("turn_number", "?")
            total_t = meta.get(ATTRS["context.total_tokens"]) or 0
            row: dict = {"턴": f"T{tn}"}
            col_sum = 0
            prev_val = 0
            for label, attr_key in _SOURCE_KEYS:
                val = meta.get(ATTRS[attr_key], 0) or 0
                row[label] = int(val)
                col_sum += val
                if attr_key == "context.source.previous_turns_tokens":
                    prev_val = val
            row["Total"] = int(total_t) if total_t else int(col_sum)
            denom = row["Total"]
            row["Prev%"] = f"{prev_val / denom:.0%}" if denom > 0 else "—"
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_plan_vs_actual_section(turns: list[dict]) -> None:
    """Plan vs 실행 — dot matrix + 5컬럼 표 + drill expander + reason 카테고리."""
    st.markdown("### Plan vs 실행")
    st.caption(
        "analyze_query 의도와 gather_data 실행을 소스 단위로 대조. "
        "갭은 \"계획했는데 못 썼다(missed)\" 또는 \"계획에 없는데 썼다(extra)\"."
    )

    st.plotly_chart(
        plan_vs_actual_dot_matrix(turns, _TOOL_TO_SOURCE),
        use_container_width=True,
    )

    no_plan_data = all(
        not (t.get("metadata", {}).get(ATTRS["query.tool_plan"]))
        for t in turns
    )
    if no_plan_data:
        st.info("query.tool_plan 데이터 없음 — analyze_query에서 tool_plan 수집 필요")

    rows, detail_rows = [], []
    any_gap = False
    exclusion_rows, truncation_rows = [], []

    for turn in turns:
        meta = turn.get("metadata", {})
        tn = turn.get("turn_number", "?")
        intent = (meta.get(ATTRS["query.intent"]) or "")
        intent_short = (intent[:40] + "…") if len(intent) > 40 else (intent or "—")

        tool_plan = meta.get(ATTRS["query.tool_plan"], []) or []
        tools_called = meta.get(ATTRS["gather.tools_called"], []) or []
        if not isinstance(tool_plan, list):
            tool_plan = []
        if not isinstance(tools_called, list):
            tools_called = []
        planned, actual = set(tool_plan), set(tools_called)
        matched, missing, extra = planned & actual, planned - actual, actual - planned
        has_gap = bool(missing or extra)
        if has_gap:
            any_gap = True

        items = meta.get(ATTRS["gather.items_collected"], 0) or 0

        gap_badge = "—" if not planned else ("OK" if not has_gap else f"WARN −{len(missing)} +{len(extra)}")

        rows.append({
            "턴": f"T{tn}",
            "Intent": intent_short,
            "갭": gap_badge,
            "수집": f"{items}건",
            "도구": f"plan {len(planned)} / actual {len(actual)}",
        })

        for t in sorted(matched):
            detail_rows.append({"턴": f"T{tn}", "도구": t, "상태": "matched",
                                "소스": _TOOL_TO_SOURCE.get(t, "—")})
        for t in sorted(missing):
            detail_rows.append({"턴": f"T{tn}", "도구": t, "상태": "missed",
                                "소스": _TOOL_TO_SOURCE.get(t, "—")})
        for t in sorted(extra):
            detail_rows.append({"턴": f"T{tn}", "도구": t, "상태": "extra",
                                "소스": _TOOL_TO_SOURCE.get(t, "—")})

        for r in (meta.get(ATTRS["gather.exclusion_reasons"], []) or []):
            if isinstance(r, dict):
                reason = r.get("reason", "")
                exclusion_rows.append({
                    "턴": f"T{tn}",
                    "소스": r.get("source", "—"),
                    "카테고리": _categorize_reason(reason),
                    "사유": reason or "—",
                })

        for item in (meta.get(ATTRS["context.truncated_items"], []) or []):
            if isinstance(item, dict):
                reason = item.get("reason", "")
                truncation_rows.append({
                    "턴": f"T{tn}",
                    "소스": item.get("source", "—"),
                    "카테고리": _categorize_reason(reason),
                    "사유": reason or "—",
                })

    if rows:
        if not no_plan_data and not any_gap:
            st.success("모든 턴에서 계획과 실행이 일치합니다.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if any_gap:
        st.caption("🔗 갭의 시점·노드 경로를 보려면 → 🔄 실행 흐름 탭의 Span 워터폴.")

    if any_gap and detail_rows:
        with st.expander(f"도구별 갭 상세 ({len(detail_rows)}건)", expanded=False):
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    if exclusion_rows or truncation_rows:
        with st.expander(
            f"제외·잘린 항목 (카테고리 분류) — 제외 {len(exclusion_rows)} · 잘림 {len(truncation_rows)}",
            expanded=False,
        ):
            if exclusion_rows:
                st.caption("**제외된 항목** — gather.exclusion_reasons")
                ex_df = pd.DataFrame(exclusion_rows)
                st.dataframe(ex_df, use_container_width=True, hide_index=True)
                freq = Counter(ex_df["카테고리"])
                if freq:
                    st.bar_chart(pd.Series(freq, name="제외 사유 빈도"), height=160)
            if truncation_rows:
                st.caption("**잘린 항목** — context.truncated_items (토큰 예산 초과 포함)")
                tr_df = pd.DataFrame(truncation_rows)
                st.dataframe(tr_df, use_container_width=True, hide_index=True)
                freq_t = Counter(tr_df["카테고리"])
                if freq_t:
                    st.bar_chart(pd.Series(freq_t, name="잘림 사유 빈도"), height=160)
                st.caption("🔗 잘림이 충분성 fail로 이어졌는지 → 🔍③ 전달 탭 ① 충분성 평가")


def _render_quality_section(turns: list[dict]) -> None:
    """구성 품질 지표 — 완전성·효율성·관련성 + 텍스트+색 severity badge (항상 펼침)."""
    st.markdown("### 구성 품질 지표")

    win_min = DASHBOARD_THRESHOLDS["window_optimal_min"]
    win_max = DASHBOARD_THRESHOLDS["window_optimal_max"]
    noise_warn = DASHBOARD_THRESHOLDS["noise_warn"]
    redund_warn = DASHBOARD_THRESHOLDS["redundancy_warn"]

    def _win_severity(u: float | None) -> str:
        if u is None or not isinstance(u, (int, float)):
            return "—"
        if u > win_max:
            return "CRIT"
        if u >= win_min:
            return "OK"
        return "WARN"

    # 컬럼 라벨은 [4D 차원] ([indicator]) 형식 — 4D framing 명시
    col_completeness = "완전성 (수집)"
    col_eff_window = "효율성 (윈도우)"
    col_eff_redund = "효율성 (중복)"
    col_relevance = "관련성 (노이즈)"
    badge_cols = [col_completeness, col_eff_window, col_eff_redund, col_relevance]

    def _fmt(value_str: str, sev: str) -> str:
        """`OK · 86%` 형식으로 값+판정을 한 셀에 합침. severity 없으면 '—'."""
        if sev == "—":
            return "—"
        return f"{sev} · {value_str}"

    rows = []
    any_low_complete, any_high_noise = False, False
    for turn in turns:
        meta = turn.get("metadata", {})
        tn = turn.get("turn_number", "?")
        items = meta.get(ATTRS["gather.items_collected"], 0) or 0
        excl = meta.get(ATTRS["gather.items_excluded"], 0) or 0
        completeness = items / (items + excl) if (items + excl) > 0 else None
        window_util = meta.get(ATTRS["context.window_utilization"])
        redundancy = meta.get(ATTRS["context.redundancy_ratio"])
        noise = meta.get(ATTRS["context.noise_ratio"])

        comp_sev = _severity(completeness, 0.8, 0.6, direction="higher_better")
        win_sev = _win_severity(window_util)
        redund_sev = _severity(redundancy, redund_warn * 0.7, redund_warn, direction="lower_better")
        noise_sev = _severity(noise, 0.3, noise_warn, direction="lower_better")

        if comp_sev == "CRIT":
            any_low_complete = True
        if noise_sev == "CRIT":
            any_high_noise = True

        rows.append({
            "턴": f"T{tn}",
            col_completeness: _fmt(
                f"{completeness:.0%}" if completeness is not None else "—",
                comp_sev,
            ),
            col_eff_window: _fmt(
                f"{window_util:.0%}" if isinstance(window_util, (int, float)) else "—",
                win_sev,
            ),
            col_eff_redund: _fmt(
                f"{redundancy:.2f}" if isinstance(redundancy, (int, float)) else "—",
                redund_sev,
            ),
            col_relevance: _fmt(
                f"{noise:.2f}" if isinstance(noise, (int, float)) else "—",
                noise_sev,
            ),
        })

    if not rows:
        st.info("품질 지표 데이터 없음")
        return

    df = pd.DataFrame(rows)

    def _color_severity_cell(val: str) -> str:
        """`OK · 86%` 셀에서 severity prefix로 색 결정. 접근성: 텍스트도 유지."""
        if not isinstance(val, str) or " · " not in val:
            return ""
        prefix = val.split(" · ", 1)[0]
        c = {
            "OK":   STATUS_COLORS["good"],
            "WARN": STATUS_COLORS["warn"],
            "CRIT": STATUS_COLORS["bad"],
        }.get(prefix, "")
        return f"color:{c};font-weight:600;" if c else ""

    styler = df.style.map(_color_severity_cell, subset=badge_cols)
    st.dataframe(styler, use_container_width=True, hide_index=True)
    st.caption(
        "- **완전성** = 수집 / (수집 + 제외)\n"
        f"- **효율성 (윈도우)**: OPTIMAL {win_min:.0%}~{win_max:.0%} "
        f"(초과 CRIT, 미만 WARN)\n"
        f"- **효율성 (중복)**: WARN ≥ {redund_warn}\n"
        f"- **관련성 (노이즈)**: WARN ≥ {noise_warn}"
    )

    if any_high_noise:
        st.caption("🔗 노이즈 CRIT 발견 — → 🔍③ 전달 탭 ③ Rot 위험 에서 누적 추이 확인")
    if any_low_complete:
        st.caption("🔗 완전성 CRIT 발견 — → 📊 측정 & 진단 탭 에서 진단 규칙 확인")


def _render_risk_map(turns: list[dict]) -> None:
    """소스 Risk Map — 기여도 × 탈락률 2D scatter (P3 / FIX-12)."""
    with st.expander("소스 Risk Map — 기여도 × 탈락률", expanded=False):
        st.caption("각 점 = (턴, 소스) · 점 크기 = 시도 항목 수")
        st.plotly_chart(source_contribution_scatter(turns), use_container_width=True)


def _render_retrieval_drilldown_placeholder(turns: list[dict]) -> None:
    """chunk-level retrieval drilldown — `source.retrieval_chunks` 가 있을 때만 렌더 (P3 / FIX-11).

    Phase 4 OTel 전환에서 `gen_ai.retrieval.documents.*` 표준 매핑과 함께
    attribute가 부착되면 자동 활성화된다. 그 전까지는 UI에 노출되지 않음.
    """
    has_chunk_data = any(
        turn.get("metadata", {}).get("source.retrieval_chunks")
        for turn in turns
    )
    if not has_chunk_data:
        return  # 미수집 — 섹션 자체 미렌더

    with st.expander("Retrieval 깊이 분석 (chunk × relevance)", expanded=False):
        st.caption("각 chunk의 relevance score + 최종 컨텍스트 포함 여부.")
        chunk_rows = []
        for turn in turns:
            meta = turn.get("metadata", {})
            chunks = meta.get("source.retrieval_chunks") or []
            if not isinstance(chunks, list):
                continue
            for c in chunks:
                if not isinstance(c, dict):
                    continue
                chunk_rows.append({
                    "턴": turn.get("turn_number", "?"),
                    "chunk_id": c.get("id", "—"),
                    "relevance": c.get("score", "—"),
                    "included": "✓" if c.get("included") else "✗",
                })
        if chunk_rows:
            st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)


def render(turns: list[dict]) -> None:
    """구성 탭을 렌더링한다 — P0~P3 통합 (Tab 3 검토 종합)."""
    from dashboard import tab_header
    tab_header.render("composition")

    _render_kpi_strip(turns)
    st.divider()

    _render_hero(turns)
    st.divider()

    _render_plan_vs_actual_section(turns)
    st.divider()

    _render_quality_section(turns)
    st.divider()

    _render_risk_map(turns)
    _render_retrieval_drilldown_placeholder(turns)
