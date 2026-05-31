"""
dashboard/views/process_observe.py — Tab 2: 실행 흐름

관측 질문: "실행 경로에서 컨텍스트는 어디서 형성·변형되었고,
           그 경로가 품질에 어떤 흔적을 남겼는가?"

역할:
    섹션 A — 컨텍스트: 누적된 Span
        세션 요약 카드 (재수집·검증실패·평균 wall_time) +
        DevTools Network 탭 스타일 Span 워터폴 (절대 타임라인)

    섹션 B — 턴별 드릴다운
        expander 헤더: 질문 요약 + 최저 4D 차원
        내부: 흐름도 → 품질 신호 요약 → 실행 이상 → Tab 7 링크

데이터 흐름:
    입력: turns (list[dict]) — enriched session data
    출력: Streamlit UI
"""
from collections import OrderedDict
from datetime import datetime, timezone

import graphviz
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.monitoring_schema import ATTRS
from dashboard.charts import STATUS_COLORS


# --- 프로세스 단계 → 노드 매핑 ---
STAGE_NODE_MAP = {
    "① Plan": "analyze_query",
    "② Collect": "gather_data",
    "③ Organize": "evaluate_context",
    "④ Generate": "generate_analysis",
    "⑤ Memory": "respond_to_user",
}

# --- 프로세스 노드 집합 (6개) ---
PROCESS_NODES = {
    "analyze_query", "gather_data", "evaluate_context",
    "generate_analysis", "verify_result", "respond_to_user",
}

# --- 프로세스 표시 순서 ---
_NODE_DISPLAY_ORDER = [
    "analyze_query", "gather_data", "evaluate_context",
    "generate_analysis", "verify_result", "respond_to_user",
]

# --- 노드 라벨 ---
_NODE_LABELS = {
    "analyze_query": "① Plan — 질의 분석",
    "gather_data": "② Collect — 데이터 수집",
    "evaluate_context": "③ Organize — 충분성 평가",
    "generate_analysis": "④ Generate — 분석 생성",
    "verify_result": "검증 checkpoint",
    "respond_to_user": "⑤ Memory — 응답 생성",
}

# --- 노드 단계 약칭 (워터폴 Y 라벨용) ---
_NODE_SHORT = {
    "analyze_query": "①Plan",
    "gather_data": "②Collect",
    "evaluate_context": "③Organize",
    "generate_analysis": "④Generate",
    "verify_result": "⑤Verify",
    "respond_to_user": "⑤Memory",
}

# --- 워터폴 단계별 색상 ---
_NODE_COLORS = {
    "analyze_query": "#4A90D9",
    "gather_data": "#5CB85C",
    "evaluate_context": "#F0AD4E",
    "generate_analysis": "#9B59B6",
    "verify_result": "#D9534F",
    "respond_to_user": "#1ABC9C",
}
# 재시도 span: 동일 색 반투명
_RETRY_OPACITY = 0.4

# --- 품질 신호 정의: (attr_key, 표시명, 생산노드, 관련탭, lower_is_better, 의미) ---
# evaluate_context 생산 지표는 iteration별 변화 추적 대상
_QUALITY_SIGNALS = [
    (
        ATTRS["context.fidelity_score"],
        "충실도", "evaluate_context", "🔍① 구성", False,
        "이전 턴 결론이 현재 컨텍스트에 얼마나 충실히 반영됐는지 (일관성 패턴 C)",
    ),
    (
        ATTRS["context.noise_ratio"],
        "노이즈비", "evaluate_context", "🔍① 구성", True,
        "수집된 데이터 중 질문과 무관한 비율 (관련성)",
    ),
    (
        ATTRS["context.window_utilization"],
        "윈도우 사용률", "evaluate_context", "🔍① 구성", True,
        "컨텍스트 윈도우 대비 총 토큰 비율 — 재수집 시 압박 증가 여부 확인",
    ),
    (
        ATTRS["analysis.conclusion_utilization"],
        "인용률", "generate_analysis", "🔍② 변형", False,
        "이전 턴 결론을 현재 분석에서 실제로 활용한 비율",
    ),
    (
        "analysis.query_alignment",
        "쿼리 정렬(분석)", "generate_analysis", "🔍② 변형", False,
        "분석 결과가 사용자 쿼리에 얼마나 정렬됐는지",
    ),
    (
        ATTRS["response.grounded_claim_ratio"],
        "근거율", "respond_to_user", "🔍③ 전달", False,
        "최종 응답의 주장 중 컨텍스트에 근거가 있는 비율",
    ),
    (
        "response.query_alignment",
        "쿼리 정렬(응답)", "respond_to_user", "🔍③ 전달", False,
        "최종 응답이 사용자 쿼리에 얼마나 정렬됐는지",
    ),
]

# evaluate_context가 생산하는 신호 — iteration별 변화 추적 대상
_EVAL_CONTEXT_SIGNAL_KEYS = {
    sig[0] for sig in _QUALITY_SIGNALS if sig[2] == "evaluate_context"
}

# --- context.source.* 속성 → 한국어 라벨 ---
_SOURCE_LABELS = {
    "context.source.system_prompt_tokens": "시스템 프롬프트",
    "context.source.query_analysis_tokens": "질의 분석",
    "context.source.gathered_data_tokens": "수집 데이터",
    "context.source.previous_turns_tokens": "이전 턴 결론",
}

# --- verify.overall_verdict → 재라우팅 라벨 ---
_REROUTE_MAP = {
    "fail_numeric": "→ gather_data (재수집)",
    "fail_interpretation": "→ generate_analysis (재생성)",
}


# ═══════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════

def _fmt_delta(d: float | None) -> str:
    if d is None:
        return "—"
    return f"+{d:.2f}" if d > 0 else f"{d:.2f}"


def _quality_badge(value: float, lower_is_better: bool = False) -> str:
    """품질 값에 색상 뱃지(이모지)를 반환한다."""
    if lower_is_better:
        if value < 0.2:
            return "🟢"
        if value < 0.4:
            return "🟡"
        return "🔴"
    else:
        if value >= 0.8:
            return "🟢"
        if value >= 0.6:
            return "🟡"
        return "🔴"


def _format_attr_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "[]"
    return str(value)


def _count_node_spans(observations: list[dict]) -> dict[str, int]:
    """프로세스 노드의 SPAN 실행 횟수를 카운트한다."""
    counts: dict[str, int] = {}
    for obs in observations:
        if obs.get("type") != "SPAN":
            continue
        name = obs.get("name", "")
        for known in PROCESS_NODES:
            if name == known or name.startswith(known + "_"):
                counts[known] = counts.get(known, 0) + 1
                break
    return counts


# ═══════════════════════════════════════
# 섹션 A — 컨텍스트: 누적된 Span
# ═══════════════════════════════════════

def _render_session_summary(turns: list[dict]) -> None:
    """세션 실행 요약 카드 3개를 렌더링한다."""
    regather_count = 0
    verify_fail_count = 0
    wall_times: list[float] = []

    for turn in turns:
        meta = turn.get("metadata", {})
        iteration = meta.get("gather.iteration") or meta.get(ATTRS.get("gather.iteration", "gather.iteration"), 1)
        if isinstance(iteration, (int, float)) and iteration > 1:
            regather_count += 1

        # top-level metadata 우선, 없으면 verify_result span에서 탐색
        verdict = meta.get(ATTRS["verify.overall_verdict"])
        if verdict and verdict != "pass":
            verify_fail_count += 1
        elif verdict is None:
            has_fail = any(
                o.get("metadata", {}).get("verify.overall_verdict") not in ("pass", None)
                for o in turn.get("observations", [])
                if o.get("type") == "SPAN" and o.get("name") == "verify_result"
            )
            if has_fail:
                verify_fail_count += 1

        wt = meta.get(ATTRS["turn.wall_time_ms"])
        if isinstance(wt, (int, float)):
            wall_times.append(float(wt))

    avg_wt = sum(wall_times) / len(wall_times) if wall_times else None
    avg_str = f"{avg_wt / 1000:.1f}s" if avg_wt is not None else "—"

    # 카드별 색상 — 이상 여부에 따라 강조
    regather_color = "#f39c12" if regather_count > 0 else "#2ecc71"
    verify_color = "#e74c3c" if verify_fail_count > 0 else "#2ecc71"
    time_color = "#e67e22" if avg_wt is not None and avg_wt > 120_000 else "#3498db"
    total_turns = len(turns)

    def _stat_card(title: str, main_value: str, sub_value: str, color: str) -> str:
        return (
            f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
            f'border-radius:8px;border-left:4px solid {color};">'
            f'<div style="font-size:13px;color:#999;">{title}</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};line-height:1.2;">{main_value}</div>'
            f'<div style="font-size:11px;color:#888;">{sub_value}</div>'
            f'</div>'
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_stat_card("재수집 발생", str(regather_count), f"/ {total_turns} 턴", regather_color), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card("검증 실패", str(verify_fail_count), f"/ {total_turns} 턴", verify_color), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card("평균 실행 시간", avg_str, "턴당 wall time", time_color), unsafe_allow_html=True)


def _render_span_waterfall(turns: list[dict]) -> None:
    """DevTools Network 탭 스타일 Span 워터폴 차트를 렌더링한다.

    X축: 세션 시작(첫 span) 기준 경과 시간(ms)
    Y축: "T{n} {단계약칭}" — 각 (turn, node) 조합이 한 행
    색상: 프로세스 단계별, 재시도 span은 반투명
    """
    # --- 1. 전체 span 수집 ---
    all_rows: list[dict] = []
    session_start: datetime | None = None

    for turn in turns:
        turn_num = turn.get("turn_number", "?")
        node_iter: dict[str, int] = {}

        spans = sorted(
            [o for o in turn.get("observations", [])
             if o.get("type") == "SPAN"
             and o.get("name") not in ("_execute_turn", None, "")
             and o.get("start_time") and o.get("end_time")],
            key=lambda s: s.get("start_time") or "",
        )

        for obs in spans:
            name = obs.get("name", "")
            node = None
            for known in PROCESS_NODES:
                if name == known or name.startswith(known + "_"):
                    node = known
                    break
            if node is None:
                continue

            try:
                start_dt = datetime.fromisoformat(str(obs["start_time"]))
                end_dt = datetime.fromisoformat(str(obs["end_time"]))
            except (ValueError, TypeError):
                continue

            # timezone 통일
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

            if session_start is None or start_dt < session_start:
                session_start = start_dt

            node_iter[node] = node_iter.get(node, 0) + 1
            iteration = node_iter[node]

            all_rows.append({
                "turn_num": turn_num,
                "node": node,
                "iteration": iteration,
                "is_retry": iteration > 1,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "wall_ms": max(1, round((end_dt - start_dt).total_seconds() * 1000)),
            })

    if not all_rows or session_start is None:
        st.info("Span 타임라인 데이터 없음 — observation start_time/end_time 미기록")
        return

    # --- 2. 상대 시간 계산 (session_start = 0ms) ---
    for row in all_rows:
        row["x_start_ms"] = (row["start_dt"] - session_start).total_seconds() * 1000

    # --- 3. Y 라벨 & 표시 순서 ---
    # (turn_num, node) 조합마다 하나의 Y 행 — retry는 같은 행에 이어붙음
    y_order: list[str] = []
    seen_y: set[str] = set()
    for turn in turns:
        tn = turn.get("turn_number", "?")
        for node in _NODE_DISPLAY_ORDER:
            label = f"T{tn}  {_NODE_SHORT.get(node, node)}"
            if label not in seen_y:
                y_order.append(label)
                seen_y.add(label)

    for row in all_rows:
        row["y_label"] = f"T{row['turn_num']}  {_NODE_SHORT.get(row['node'], row['node'])}"

    # --- 4. 노드별 traces 구성 (범례 1회 표시) ---
    fig = go.Figure()
    node_in_legend: set[str] = set()

    # 정상 실행 먼저, 재시도 뒤에
    for is_retry_pass in (False, True):
        for node in _NODE_DISPLAY_ORDER:
            rows_for = [r for r in all_rows
                        if r["node"] == node and r["is_retry"] == is_retry_pass]
            if not rows_for:
                continue

            color = _NODE_COLORS.get(node, "#888888")
            legend_name = _NODE_LABELS.get(node, node)
            show_legend = not is_retry_pass and legend_name not in node_in_legend
            if show_legend:
                node_in_legend.add(legend_name)

            customdata = [
                [r["turn_num"], r["wall_ms"], r["iteration"]]
                for r in rows_for
            ]

            fig.add_trace(go.Bar(
                name=legend_name,
                x=[r["wall_ms"] for r in rows_for],
                y=[r["y_label"] for r in rows_for],
                orientation="h",
                base=[r["x_start_ms"] for r in rows_for],
                marker=dict(
                    color=color,
                    opacity=_RETRY_OPACITY if is_retry_pass else 1.0,
                    pattern_shape="/" if is_retry_pass else "",
                    line=dict(
                        color=color,
                        width=1.5 if is_retry_pass else 0,
                    ),
                ),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{_NODE_LABELS.get(node, node)}"
                    + ("  [재시도]" if is_retry_pass else "")
                    + "</b><br>"
                    "Turn %{customdata[0]}<br>"
                    "소요: %{customdata[1]:,}ms<br>"
                    "실행 %{customdata[2]}회차<br>"
                    "세션 시작 후: %{base:,.0f}ms"
                    "<extra></extra>"
                ),
                showlegend=show_legend,
                legendgroup=legend_name,
            ))

    # --- 5. Turn 구분선 (더 명확한 색상) ---
    for i, turn in enumerate(turns[:-1]):
        tn = turn.get("turn_number", "?")
        last_label = f"T{tn}  {_NODE_SHORT.get('respond_to_user', '')}"
        if last_label in seen_y:
            idx = y_order.index(last_label)
            n_rows = len(y_order)
            y_pos = (n_rows - 1 - idx) + 0.5  # Y축 reversed 기준
            fig.add_hline(
                y=y_pos,
                line=dict(color="rgba(255,255,255,0.15)", width=1.5, dash="dot"),
            )
            # Turn 라벨 주석
            fig.add_annotation(
                x=0, y=y_pos + 0.5,
                xref="paper", yref="y",
                text=f"T{tn + 1}",
                showarrow=False,
                font=dict(size=9, color="rgba(255,255,255,0.35)"),
                xanchor="left",
            )

    # --- 6. 레이아웃 ---
    n_rows = len(y_order)
    chart_height = max(300, n_rows * 34 + 100)

    fig.update_layout(
        barmode="overlay",
        bargap=0.25,
        height=chart_height,
        xaxis=dict(
            title="세션 시작 후 경과 시간 (ms)",
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
            tickformat=",",
            tickfont=dict(size=11, color="rgba(255,255,255,0.6)"),
            title_font=dict(size=12, color="rgba(255,255,255,0.5)"),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(y_order)),  # T1이 위, T_last가 아래
            tickfont=dict(size=11, color="rgba(255,255,255,0.7)"),
            gridcolor="rgba(0,0,0,0)",
            title="",
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="단계", font=dict(size=11, color="rgba(255,255,255,0.5)")),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="rgba(255,255,255,0.75)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",  # 다크 테마 — 거의 투명
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("바 너비 = 소요 시간(ms)  ·  반투명 빗금 = 재시도 span  ·  마우스오버로 상세 확인")


# ═══════════════════════════════════════
# 섹션 B — 흐름도 관련
# ═══════════════════════════════════════

def _extract_node_metrics(observations: list[dict]) -> dict[str, dict]:
    """프로세스 노드별 핵심 메트릭을 추출한다."""
    from dashboard.analysis import _calc_latency_ms

    node_spans: dict[str, list[dict]] = {}
    for obs in observations:
        if obs.get("type") != "SPAN":
            continue
        name = obs.get("name", "")
        for known in PROCESS_NODES:
            if name == known or name.startswith(known + "_"):
                node_spans.setdefault(known, []).append(obs)
                break

    metrics: dict[str, dict] = {}
    for node, spans in node_spans.items():
        last = spans[-1]
        meta = last.get("metadata", {})
        m: dict = {}

        wt = _calc_latency_ms(last.get("start_time"), last.get("end_time"))
        if wt is not None:
            m["wall_time_ms"] = wt

        if node == "gather_data":
            total_tools = 0
            for span in spans:
                tools = span.get("metadata", {}).get(ATTRS["gather.tools_called"])
                if isinstance(tools, list):
                    total_tools += len(tools)
            if total_tools > 0:
                m["tools_called_count"] = total_tools
        elif node == "evaluate_context":
            fs = meta.get(ATTRS["context.fidelity_score"])
            if isinstance(fs, (int, float)):
                m["fidelity_score"] = float(fs)
            nr = meta.get(ATTRS["context.noise_ratio"])
            if isinstance(nr, (int, float)):
                m["noise_ratio"] = float(nr)
        elif node == "generate_analysis":
            cu = meta.get(ATTRS["analysis.conclusion_utilization"])
            if isinstance(cu, (int, float)):
                m["conclusion_utilization"] = float(cu)
        elif node == "respond_to_user":
            gr = meta.get(ATTRS["response.grounded_claim_ratio"])
            if isinstance(gr, (int, float)):
                m["grounded_claim_ratio"] = float(gr)

        metrics[node] = m

    return metrics


def _build_process_diagram(
    node_counts: dict[str, int],
    node_metrics: dict[str, dict] | None = None,
    verify_routing: str | None = None,
) -> graphviz.Digraph:
    """프로세스 흐름도를 생성한다."""
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", bgcolor="transparent", fontname="Helvetica",
             ranksep="0.6", nodesep="0.4")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
             fontsize="11", width="1.6", height="1.0")

    c_good = STATUS_COLORS["good"]
    c_warn = STATUS_COLORS["warn"]
    c_na = STATUS_COLORS["na"]
    c_muted = STATUS_COLORS["muted"]
    c_primary = STATUS_COLORS["primary"]

    has_regather = node_counts.get("gather_data", 0) > 1
    has_reverify = node_counts.get("verify_result", 0) > 1

    # ── 프로세스 영역 ──
    with dot.subgraph(name="cluster_process") as proc:
        proc.attr(label="프로세스 (Agent)", style="rounded", color=c_primary,
                  fontcolor=c_primary, fontsize="13", labeljust="l")
        for stage, node in STAGE_NODE_MAP.items():
            count = node_counts.get(node, 0)
            if count == 0:
                color = c_na
            elif count == 1:
                color = c_good
            else:
                color = c_warn
            label = f"{stage}\n({node})"
            if count > 1:
                label += f"\n×{count}"

            if node_metrics:
                wt = node_metrics.get(node, {}).get("wall_time_ms")
                if wt is not None:
                    label += f"\n{wt:,}ms"

            proc.node(node, label=label, fillcolor=color, fontcolor="white",
                      fontsize="11")

        proc.edge("analyze_query", "gather_data")
        proc.edge("gather_data", "evaluate_context")
        proc.edge("evaluate_context", "generate_analysis")
        proc.edge("generate_analysis", "respond_to_user")

    # ── 모니터링 checkpoint ──
    suf_color = c_good if node_counts.get("evaluate_context", 0) > 0 else c_na
    ver_count = node_counts.get("verify_result", 0)
    ver_color = c_good if ver_count > 0 else c_na
    if ver_count > 1:
        ver_color = c_warn
    ver_label = "검증" + (f"\n×{ver_count}" if ver_count > 1 else "")

    with dot.subgraph(name="cluster_monitor") as mon:
        mon.attr(label="모니터링 활동", style="dashed,rounded", color=c_muted,
                 fontcolor=c_muted, fontsize="12", labelloc="t", labeljust="l",
                 margin="20", rank="same")
        mon.node("checkpoint_suf", "충분성\n평가", fillcolor=suf_color,
                 fontcolor="white", shape="diamond", width="1.2", height="0.8",
                 fontsize="11")
        mon.node("checkpoint_ver", ver_label, fillcolor=ver_color,
                 fontcolor="white", shape="diamond", width="1.2", height="0.8",
                 fontsize="11")
        mon.edge("checkpoint_suf", "checkpoint_ver", style="invis")

    dot.edge("analyze_query", "checkpoint_suf", style="invis", minlen="2")
    dot.edge("gather_data", "checkpoint_suf", style="dashed", color=c_muted,
             arrowhead="none", constraint="false")
    dot.edge("checkpoint_suf", "evaluate_context", style="dashed", color=c_muted,
             arrowhead="none", constraint="false")
    dot.edge("generate_analysis", "checkpoint_ver", style="dashed", color=c_muted,
             arrowhead="none", constraint="false")

    if has_regather:
        regather_count = node_counts.get("gather_data", 1) - 1
        dot.edge("checkpoint_suf", "gather_data", style="bold,dashed", color=c_warn,
                 label=f"  부족 → 재수집 ×{regather_count}  ", fontcolor=c_warn,
                 fontsize="11", penwidth="2.5", constraint="false")

    if has_reverify:
        reverify_count = node_counts.get("verify_result", 1) - 1
        if verify_routing == "gather_data":
            dot.edge("checkpoint_ver", "gather_data", style="bold,dashed", color=c_warn,
                     label=f"  실패 → 재수집 ×{reverify_count}  ", fontcolor=c_warn,
                     fontsize="11", penwidth="2.5", constraint="false")
        else:
            dot.edge("checkpoint_ver", "generate_analysis", style="bold,dashed", color=c_warn,
                     label=f"  실패 → 재생성 ×{reverify_count}  ", fontcolor=c_warn,
                     fontsize="11", penwidth="2.5", constraint="false")

    return dot


# ═══════════════════════════════════════
# 섹션 B — 검증 iteration 테이블
# ═══════════════════════════════════════

def _render_verify_iterations(verify_spans: list[dict]) -> None:
    """검증(verify_result) iteration별 pivot 테이블을 렌더링한다.

    열 = 시도 번호 / 행 = 지표
    원인 영역: 수치 일치 여부 / 불일치 건수 / 해석 충실도 / 발견된 문제점
    결과 영역: 최종 판정 / 라우팅 결정
    단일 시도(pass 포함)도 항상 렌더링하여 통과 근거를 보여준다.
    """
    n = len(verify_spans)

    _TH = 'style="padding:7px 10px;text-align:center;background:#2a2a2a;color:#aaa;font-size:12px;font-weight:600;border-bottom:1px solid #444;"'
    _TH_LEFT = 'style="padding:7px 10px;text-align:left;background:#2a2a2a;color:#aaa;font-size:12px;font-weight:600;border-bottom:1px solid #444;"'
    _TD = 'style="padding:6px 10px;text-align:center;font-size:12px;color:#ccc;border-bottom:1px solid #333;vertical-align:top;"'
    _TD_LABEL = 'style="padding:6px 10px;text-align:left;font-size:12px;color:#888;border-bottom:1px solid #333;white-space:nowrap;"'
    _TD_SECTION = f'style="padding:5px 10px;text-align:left;font-size:11px;color:#555;background:#1a1a1a;font-weight:600;letter-spacing:0.05em;" colspan="{n + 1}"'

    def _verdict_cell(verdict: str) -> str:
        if verdict == "pass":
            return '<span style="color:#2ecc71;font-weight:600;">✓ pass</span>'
        if verdict == "fail_numeric":
            return '<span style="color:#e74c3c;font-weight:600;">✗ fail_numeric</span>'
        if verdict == "fail_interpretation":
            return '<span style="color:#f39c12;font-weight:600;">✗ fail_interp</span>'
        return verdict or "—"

    def _routing_cell(verdict: str) -> str:
        if verdict == "pass":
            return '<span style="color:#2ecc71;">→ respond_to_user</span>'
        if verdict == "fail_numeric":
            return '<span style="color:#e74c3c;">→ gather_data (재수집)</span>'
        if verdict == "fail_interpretation":
            return '<span style="color:#f39c12;">→ generate_analysis (재생성)</span>'
        return "—"

    def _bool_cell(val) -> str:
        if val is True:
            return '<span style="color:#2ecc71;">✓ 통과</span>'
        if val is False:
            return '<span style="color:#e74c3c;">✗ 실패</span>'
        return "—"

    def _score_cell(score) -> str:
        if not isinstance(score, (int, float)):
            return "—"
        color = "#2ecc71" if float(score) >= 0.6 else "#e74c3c"
        return f'<span style="color:{color};">{float(score):.2f}</span><span style="color:#555;font-size:10px;"> (임계값 0.6)</span>'

    def _issues_cell(issues) -> str:
        if not issues or not isinstance(issues, list):
            return '<span style="color:#555;">없음</span>'
        items = "".join(
            f'<div style="margin-bottom:3px;color:#bbb;">• {str(i)}</div>'
            for i in issues[:5]
        )
        return f'<div style="font-size:11px;">{items}</div>'

    # 데이터 수집
    iter_data = []
    for span in verify_spans:
        m = span.get("metadata", {})
        verdict = m.get("verify.overall_verdict") or m.get(ATTRS.get("verify.overall_verdict", "verify.overall_verdict"), "—")
        iter_data.append({
            "numeric_passed": m.get("verify.numeric_check_passed") or m.get(ATTRS.get("verify.numeric_check_passed", ""), None),
            "discrepancies": m.get("verify.numeric_discrepancies") or m.get(ATTRS.get("verify.numeric_discrepancies", ""), None),
            "interp_score": m.get("verify.interpretation_score") or m.get(ATTRS.get("verify.interpretation_score", ""), None),
            "issues": m.get("verify.issues") or m.get(ATTRS.get("verify.issues", ""), []),
            "verdict": verdict,
        })

    header_cells = "".join(f'<th {_TH}>시도 {i + 1}</th>' for i in range(n))

    def _row(label: str, cells: list[str]) -> str:
        tds = "".join(f'<td {_TD}>{c}</td>' for c in cells)
        return f'<tr><td {_TD_LABEL}>{label}</td>{tds}</tr>'

    rows_html = [
        f'<tr><th {_TH_LEFT}>지표</th>{header_cells}</tr>',
        f'<tr><td {_TD_SECTION}>판단 근거</td></tr>',
        _row("수치 일치 여부",    [_bool_cell(d["numeric_passed"]) for d in iter_data]),
        _row("불일치 건수",       [str(d["discrepancies"]) if d["discrepancies"] is not None else "—" for d in iter_data]),
        _row("해석 충실도 (0~1)", [_score_cell(d["interp_score"]) for d in iter_data]),
        _row("발견된 문제점",     [_issues_cell(d["issues"]) for d in iter_data]),
        f'<tr><td {_TD_SECTION}>판정 & 라우팅</td></tr>',
        _row("최종 판정",   [_verdict_cell(d["verdict"]) for d in iter_data]),
        _row("라우팅 결정", [_routing_cell(d["verdict"]) for d in iter_data]),
    ]

    html = (
        '<div style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        + "".join(rows_html) +
        '</table></div>'
    )

    last_verdict = iter_data[-1]["verdict"] if iter_data else ""
    if last_verdict == "pass":
        result_str = f"✓ 통과"
    elif last_verdict == "fail_numeric":
        result_str = "✗ fail_numeric (수치 불일치)"
    else:
        result_str = "✗ fail_interpretation (해석 충실도 미달)"

    st.caption(f"▸ 검증 루프  {n}회 시도 / 최종: {result_str}")
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════
# 섹션 B — 품질 신호 테이블
# ═══════════════════════════════════════

def _render_quality_signals(meta: dict, observations: list[dict]) -> None:
    """품질 신호 + 재수집 iteration 정보를 전치(pivot) HTML 테이블로 렌더링한다.

    열 = Iteration 번호 / 행 = 지표
    단일 iteration이면 열 헤더 없이 지표|값 2컬럼 flat 테이블.
    복수 iteration이면:
        충분성 여부 / 부족 항목 / 신뢰도 / 추가 도구 / 수집 수 / 토큰 증가량
        ── 품질 지표 변화 ──
        충실도 / 노이즈비 / 윈도우 사용률
    generate_analysis·respond_to_user 지표는 항상 마지막 값 단일 행 (단일 iteration 영역에 표시).
    """
    from dashboard.analysis import _calc_latency_ms  # noqa: F401 (사용 안 하지만 import 오류 방지)

    # ── span 목록 수집 ──
    eval_spans = sorted(
        [o for o in observations if o.get("type") == "SPAN" and o.get("name") == "evaluate_context"],
        key=lambda s: s.get("start_time") or "",
    )
    gather_spans = sorted(
        [o for o in observations if o.get("type") == "SPAN" and o.get("name") == "gather_data"],
        key=lambda s: s.get("start_time") or "",
    )

    eval_signals = [sig for sig in _QUALITY_SIGNALS if sig[2] == "evaluate_context"]
    # respond_to_user 지표(근거율·쿼리 정렬(응답))는 verify_result 이후 생산 → 검증 입력값 아님
    other_signals = [sig for sig in _QUALITY_SIGNALS if sig[2] == "generate_analysis"]

    n_iter = len(eval_spans)

    # ── CSS 공통 ──
    _TH = 'style="padding:7px 10px;text-align:center;background:#2a2a2a;color:#aaa;font-size:12px;font-weight:600;border-bottom:1px solid #444;"'
    _TH_LEFT = 'style="padding:7px 10px;text-align:left;background:#2a2a2a;color:#aaa;font-size:12px;font-weight:600;border-bottom:1px solid #444;"'
    _TD = 'style="padding:6px 10px;text-align:center;font-size:12px;color:#ccc;border-bottom:1px solid #333;vertical-align:top;"'
    _TD_LEFT = 'style="padding:6px 10px;text-align:left;font-size:12px;color:#ccc;border-bottom:1px solid #333;vertical-align:top;"'
    _TD_LABEL = 'style="padding:6px 10px;text-align:left;font-size:12px;color:#888;border-bottom:1px solid #333;white-space:nowrap;"'
    _TD_SECTION = 'style="padding:5px 10px;text-align:left;font-size:11px;color:#555;background:#1a1a1a;font-weight:600;letter-spacing:0.05em;" colspan="{}"'

    def _suf_cell(is_suf) -> str:
        if is_suf is True:
            return '<span style="color:#2ecc71;">✓</span>'
        if is_suf is False:
            return '<span style="color:#e74c3c;">✗</span>'
        return "—"

    def _quality_cell(fval: float | None, lower_is_better: bool) -> str:
        if fval is None:
            return "—"
        badge = _quality_badge(fval, lower_is_better=lower_is_better)
        return f"{badge}&nbsp;&nbsp;{fval:.2f}"

    # ════════════════════════════════════════
    # 복수 iteration: pivot 테이블
    # ════════════════════════════════════════
    if n_iter > 1:
        # iteration별 데이터 수집
        iter_data: list[dict] = []
        prev_tokens: float | None = None
        for i, (es, gs) in enumerate(zip(eval_spans, gather_spans + [{}] * n_iter)):
            em = es.get("metadata", {})
            gm = gs.get("metadata", {}) if gs else {}

            raw_missing = em.get("context.missing_info") or em.get(ATTRS.get("context.missing_info", "context.missing_info"), "")
            if isinstance(raw_missing, list):
                missing_str = " ".join(str(m) for m in raw_missing)
            else:
                missing_str = str(raw_missing) if raw_missing else "—"

            tools = gm.get("gather.tools_called") or gm.get(ATTRS.get("gather.tools_called", "gather.tools_called"), [])
            tools_str = ", ".join(str(t) for t in tools) if isinstance(tools, list) and tools else "—"

            tokens = em.get("context.total_tokens") or em.get(ATTRS.get("context.total_tokens", "context.total_tokens"))
            delta_tok: str
            if tokens is not None and prev_tokens is not None:
                d = int(tokens) - int(prev_tokens)
                delta_tok = f"+{d:,}" if d > 0 else f"{d:,}"
            else:
                delta_tok = "—"
            prev_tokens = tokens

            items = gm.get("gather.items_collected") or gm.get(ATTRS.get("gather.items_collected", "gather.items_collected"))

            noise_key = ATTRS.get("context.noise_ratio", "context.noise_ratio")
            window_key = ATTRS.get("context.window_utilization", "context.window_utilization")

            iter_data.append({
                "is_suf": em.get("context.is_sufficient"),
                "missing": missing_str,
                "conf": em.get("context.sufficiency_confidence") or em.get(ATTRS.get("context.sufficiency_confidence", ""), None),
                "tools": tools_str,
                "items": str(items) if items is not None else "—",
                "delta_tok": delta_tok,
                "noise_ratio": em.get(noise_key),
                "window_util": em.get(window_key),
            })

        n_cols = n_iter + 1  # 라벨 열 + iteration 열들
        header_cells = "".join(f'<th {_TH}>Iteration {i + 1}</th>' for i in range(n_iter))

        def _row(label: str, cells: list[str], label_style: str = _TD_LABEL) -> str:
            tds = "".join(f'<td {_TD}>{c}</td>' for c in cells)
            return f'<tr><td {label_style}>{label}</td>{tds}</tr>'

        def _section_row(title: str) -> str:
            return f'<tr><td {_TD_SECTION.format(n_cols)}>{title}</td></tr>'

        def _noise_cell(val) -> str:
            if not isinstance(val, (int, float)):
                return "—"
            badge = _quality_badge(float(val), lower_is_better=True)
            return f"{badge}&nbsp;&nbsp;{float(val):.2f}"

        def _window_cell(val) -> str:
            if not isinstance(val, (int, float)):
                return "—"
            badge = _quality_badge(float(val), lower_is_better=True)
            return f"{badge}&nbsp;&nbsp;{float(val):.2f}"

        rows_html = [
            f'<tr><th {_TH_LEFT}>지표</th>{header_cells}</tr>',
            # ── 원인: 충분성에 영향을 주는 요소 ──
            _section_row("충분성에 영향을 주는 요소"),
            _row("추가 도구",   [f'<span style="font-size:11px;">{d["tools"]}</span>' for d in iter_data]),
            _row("수집 수",     [d["items"] for d in iter_data]),
            _row("Δ토큰",       [d["delta_tok"] for d in iter_data]),
            _row("노이즈비",    [_noise_cell(d["noise_ratio"]) for d in iter_data]),
            # ── 판정: 충분성 직접 출력 ──
            _section_row("충분성 판정"),
            _row("충분성 여부", [_suf_cell(d["is_suf"]) for d in iter_data]),
            _row("신뢰도",      [f'{d["conf"]:.2f}' if isinstance(d["conf"], float) else "—" for d in iter_data]),
            _row("부족 항목",   [f'<span style="font-size:11px;color:#bbb;">{d["missing"]}</span>' for d in iter_data], _TD_LABEL),
            # ── 결과: 충분성 실패로 인해 영향받는 요소 ──
            _section_row("충분성 실패로 인해 영향받는 요소"),
            _row("윈도우 사용률", [_window_cell(d["window_util"]) for d in iter_data]),
        ]

        html = (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            + "".join(rows_html) +
            '</table></div>'
        )
        st.caption("▸ 재수집 루프 — 충분성 평가")
        st.markdown(html, unsafe_allow_html=True)

        # generate_analysis / respond_to_user 지표 — 검증의 입력값으로서 표시
        other_rows = []
        for attr_key, label, producer, tab_hint, lower_is_better, _ in other_signals:
            val = meta.get(attr_key)
            if not isinstance(val, (float, int)):
                continue
            fval = float(val)
            badge = _quality_badge(fval, lower_is_better=lower_is_better)
            other_rows.append({"지표": label, "생산 노드": producer, "값": f"{badge}  {fval:.2f}", "관련 탭": tab_hint})
        if other_rows:
            st.caption("▸ 생성 결과 (검증 입력값)")
            st.dataframe(
                pd.DataFrame(other_rows), use_container_width=True, hide_index=True,
                column_config={
                    "지표": st.column_config.TextColumn(width="medium"),
                    "생산 노드": st.column_config.TextColumn(width="medium"),
                    "값": st.column_config.TextColumn(width="small"),
                    "관련 탭": st.column_config.TextColumn(width="small"),
                },
            )
        return

    # ════════════════════════════════════════
    # 단일 iteration: 인과 구조 flat 테이블
    # ════════════════════════════════════════
    span_meta = eval_spans[0].get("metadata", {}) if eval_spans else {}

    def _fval(key: str) -> float | None:
        v = span_meta.get(key) or meta.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def _badge_row(label: str, key: str, lower: bool, section: str) -> dict | None:
        v = _fval(key)
        if v is None:
            return None
        return {"섹션": section, "지표": label, "값": f"{_quality_badge(v, lower)}  {v:.2f}"}

    rows_flat = [r for r in [
        {"섹션": "충분성에 영향을 주는 요소", "지표": "노이즈비",
         "값": f"{_quality_badge(_fval(ATTRS.get('context.noise_ratio','context.noise_ratio')) or 0, True)}  {_fval(ATTRS.get('context.noise_ratio','context.noise_ratio')):.2f}"
         } if _fval(ATTRS.get("context.noise_ratio","context.noise_ratio")) is not None else None,
        {"섹션": "충분성 판정", "지표": "충분성 여부",
         "값": "✓ 통과" if span_meta.get("context.is_sufficient") else "✗ 실패"},
        {"섹션": "충분성 판정", "지표": "신뢰도",
         "값": f"{_fval(ATTRS.get('context.sufficiency_confidence','context.sufficiency_confidence')):.2f}"
         } if _fval(ATTRS.get("context.sufficiency_confidence","context.sufficiency_confidence")) is not None else None,
        _badge_row("윈도우 사용률", ATTRS.get("context.window_utilization","context.window_utilization"),
                   True, "충분성 실패로 인해 영향받는 요소"),
    ] if r is not None]

    if not rows_flat:
        st.caption("컨텍스트 품질 데이터 없음")
        return

    st.caption("▸ 충분성 평가 (단일 실행)")
    st.dataframe(
        pd.DataFrame(rows_flat), use_container_width=True, hide_index=True,
        column_config={
            "섹션": st.column_config.TextColumn(width="large"),
            "지표": st.column_config.TextColumn(width="medium"),
            "값": st.column_config.TextColumn(width="small"),
        },
    )


# ═══════════════════════════════════════
# 섹션 B — 실행 이상 상세 (기존 유지)
# ═══════════════════════════════════════

def _render_span_meta_summary(contrib: dict) -> None:
    """span 상단에 duration·토큰 요약 1줄을 렌더링한다."""
    from dashboard.analysis import _calc_latency_ms

    parts = []
    duration_ms = _calc_latency_ms(contrib.get("start_time"), contrib.get("end_time"))
    if duration_ms is not None:
        parts.append(f"⏱ {duration_ms:,}ms")

    token_parts = []
    cost_val = None
    for attr in contrib.get("attributes", []):
        k, v = attr["key"], attr["value"]
        if k == "context.total_tokens" and isinstance(v, (int, float)):
            token_parts.append(f"컨텍스트 {int(v):,}tok")
        elif k == "response.token_count" and isinstance(v, (int, float)):
            token_parts.append(f"출력 {int(v):,}tok")
        elif k == "gather.tokens_gathered" and isinstance(v, (int, float)):
            token_parts.append(f"수집 {int(v):,}tok")
        elif k == "turn.total_cost_usd" and isinstance(v, (int, float)):
            cost_val = v
    if token_parts:
        parts.append("📊 " + " / ".join(token_parts))
    if cost_val is not None:
        parts.append(f"💰 ${cost_val:.4f}")

    if parts:
        st.caption("  •  ".join(parts))


def _render_tool_sequence(group: list[dict]) -> None:
    """gather_data 노드의 소스별 기여도를 렌더링한다."""
    for i, item in enumerate(group):
        attrs = {a["key"]: a["value"] for a in item["attributes"]}
        contrib = attrs.get("source.contribution")
        if isinstance(contrib, list) and contrib:
            header = f"**소스별 기여도 (Iter {i + 1})**" if len(group) > 1 else "**소스별 기여도**"
            st.markdown(header)
            rows = []
            for entry in contrib:
                if isinstance(entry, dict):
                    rows.append({
                        "소스": entry.get("source", "—"),
                        "기여도": _format_attr_value(entry.get("contribution")),
                        "토큰": entry.get("tokens", "—"),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.divider()


def _render_source_breakdown(group: list[dict]) -> None:
    """evaluate_context 노드의 컨텍스트 소스별 토큰 분포를 렌더링한다."""
    attrs = {a["key"]: a["value"] for a in group[-1]["attributes"]}
    total = attrs.get("context.total_tokens")
    rows = []
    for key, label in _SOURCE_LABELS.items():
        val = attrs.get(key)
        if val is None:
            continue
        pct = f"{val / total * 100:.1f}%" if total and isinstance(total, (int, float)) and total > 0 else "—"
        rows.append({"소스": label, "토큰": int(val), "비율": pct})
    if rows:
        st.markdown("**소스별 토큰 분포**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.divider()


# ═══════════════════════════════════════
# 메인 렌더 함수
# ═══════════════════════════════════════

def render(turns: list[dict]) -> None:
    """실행 흐름 탭을 렌더링한다."""
    # --- 탭 헤더 ---
    st.markdown(
        '<h2 style="margin-bottom:4px;">실행 흐름</h2>'
        '<p style="color:rgba(255,255,255,0.45);font-size:13px;margin-top:0;margin-bottom:20px;">'
        '관측 질문: <em>"실행 경로에서 컨텍스트는 어디서 형성·변형되었고, 그 경로가 품질에 어떤 흔적을 남겼는가?"</em>'
        '</p>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════
    # 섹션 A — 컨텍스트: 누적된 Span
    # ══════════════════════════════════════
    _render_session_summary(turns)
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    _render_span_waterfall(turns)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # 섹션 B — 턴별 드릴다운
    # ══════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        '<div style="width:3px;height:20px;background:#9B59B6;border-radius:2px;"></div>'
        '<span style="font-size:15px;font-weight:600;color:rgba(255,255,255,0.85);">턴별 드릴다운</span>'
        '<span style="font-size:12px;color:rgba(255,255,255,0.35);margin-left:8px;">흐름도 → 품질 신호 → 실행 이상 · 전체 속성은 📋 상세 탭</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for turn in turns:
        meta = turn.get("metadata", {})
        turn_num = turn.get("turn_number", meta.get(ATTRS["turn.number"], "?"))

        # --- expander 헤더 구성 ---
        full_user_query = _extract_user_query(turn)
        question_only = _extract_question_only(full_user_query)
        if question_only:
            preview = question_only[:60] + "…" if len(question_only) > 60 else question_only
        else:
            preview = "(질문 없음)"

        # 실행 이상 뱃지 — expander 내부에서 바로 추적 가능한 정보만
        observations = turn.get("observations", [])
        node_counts = _count_node_spans(observations)
        anomaly_parts = []
        if node_counts.get("gather_data", 0) > 1:
            anomaly_parts.append("🔄 재수집")
        failed_verify = any(
            o.get("metadata", {}).get("verify.overall_verdict") not in ("pass", None)
            for o in observations
            if o.get("type") == "SPAN" and o.get("name") == "verify_result"
        )
        if failed_verify:
            anomaly_parts.append("⚠️ 검증실패")
        anomaly_str = ("  " + "  ".join(anomaly_parts)) if anomaly_parts else ""

        header = f'Turn {turn_num} — "{preview}"{anomaly_str}'

        with st.expander(header, expanded=False):
            if not observations:
                st.info("관측 데이터 없음 (Langfuse observation 미로드)")
                continue

            # ── 공통 준비 ──
            node_metrics = _extract_node_metrics(observations)

            verify_spans = sorted(
                [o for o in observations
                 if o.get("type") == "SPAN" and o.get("name") == "verify_result"],
                key=lambda s: s.get("start_time") or "",
            )
            failed_verifies = [
                s for s in verify_spans
                if s.get("metadata", {}).get("verify.overall_verdict") not in ("pass", None)
            ]
            verify_routing: str | None = None
            if failed_verifies:
                last_verdict = failed_verifies[-1].get("metadata", {}).get("verify.overall_verdict")
                verify_routing = "gather_data" if last_verdict == "fail_numeric" else "generate_analysis"

            has_regather = node_counts.get("gather_data", 0) > 1

            # ── [1] 실행 경로 ──
            st.caption("▸ 실행 경로")
            diagram = _build_process_diagram(node_counts, node_metrics, verify_routing)
            st.graphviz_chart(diagram, use_container_width=True)

            # ── [2] 이상 없을 때만 성공 메시지 ──
            if not has_regather and not failed_verifies:
                st.success("실행 이상 없음 — 모든 체크포인트 통과", icon="✅")

            # ── [3] 재수집 루프 — 충분성 평가 & 검증 루프 ──
            _render_quality_signals(meta, observations)

            if verify_spans:
                _render_verify_iterations(verify_spans)

            # ── 상세 탭 안내 ──
            attn_tabs: list[str] = []
            for sig in _QUALITY_SIGNALS:
                attr_key, _lbl, _prod, tab_hint, lower_is_better, _desc = sig
                val = meta.get(attr_key)
                if isinstance(val, (int, float)) and tab_hint and tab_hint != "—":
                    if _quality_badge(float(val), lower_is_better=lower_is_better) == "🔴":
                        if tab_hint not in attn_tabs:
                            attn_tabs.append(tab_hint)
            if attn_tabs:
                st.warning(f"이 턴 주의 탭: {', '.join(attn_tabs)} — 🔴 신호 감지됨")
            else:
                st.caption("")


# ═══════════════════════════════════════
# 헬퍼 — 사용자 질문/답변 추출
# ═══════════════════════════════════════

_EVIDENCE_MARKERS = (
    "**근거", "## 근거", "### 근거", "**참고", "## 참고", "**출처", "## 출처",
    "**Evidence", "**References", "## Evidence", "## References",
)


def _extract_question_only(user_query: str | None) -> str | None:
    """user_query에서 부가 컨텍스트를 제거하고 질문 본체만 반환한다."""
    if not user_query:
        return None
    text = user_query.strip()
    if "질문:" in text:
        text = text.split("질문:", 1)[1]
    for marker in ("이전 턴 결론", "이전 턴", "이전 결론", "Previous turn", "Context:"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip().rstrip(":").strip()


def _extract_user_query(turn: dict) -> str | None:
    """turn에서 사용자 질문을 추출한다."""
    meta = turn.get("metadata", {})
    user_q = meta.get(ATTRS["query.user_query"]) or meta.get("user_query")
    if user_q:
        return str(user_q)
    _CANDIDATE_KEYS = ("query", "user_query", "question", "current_query", "input", "text")
    for obs in turn.get("observations", []):
        inp = obs.get("input")
        if isinstance(inp, dict):
            for k in _CANDIDATE_KEYS:
                val = inp.get(k)
                if isinstance(val, str) and val.strip():
                    return val
        if isinstance(inp, list):
            for msg in reversed(inp):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    if isinstance(content, list) and content:
                        first = content[0]
                        if isinstance(first, dict):
                            return first.get("text", "") or str(first)
    return None
