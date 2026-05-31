"""
dashboard/charts.py — Plotly 차트 생성 함수

역할:
    Context Monitoring Dashboard의 모든 차트를 Plotly로 생성한다.
    evaluation/generate_charts.py의 matplotlib 차트를 Plotly로 변환하고,
    신규 차트(상관 히트맵, LLM 타임라인, 노드 contribution)를 추가한다.

데이터 흐름:
    입력: turns (list[dict]), 분석 결과 (DataFrame, list[dict])
    출력: plotly.graph_objects.Figure
"""
import pandas as pd
import plotly.graph_objects as go

from agent.monitoring_schema import (
    ATTRS,
    DRIFT_ALIGNMENT_THRESHOLD,
    DRIFT_CONTINUITY_THRESHOLD,
    FIDELITY_SCORE_WEIGHTS,
    THRESHOLDS,
)

# --- 색상 (generate_charts.py에서 포팅) ---
COLORS = {
    "completeness": "#27ae60",   # 어두운 초록 — STATUS_COLORS["good"]과 구분
    "efficiency": "#2980b9",     # 어두운 파랑
    "relevance": "#d35400",      # 어두운 주황
    "consistency": "#8e44ad",    # 어두운 보라
    "threshold": "#e74c3c",
    "gathered": "#2ecc71",
    "previous_turns": "#e74c3c",
    "conclusions": "#f39c12",
    "system_prompt": "#95a5a6",
    "query_analysis": "#3498db",
}

# --- 상태 색상 (메트릭/테이블 공통) ---
STATUS_COLORS = {
    "good": "#2ecc71",     # 초록 — 정상
    "warn": "#f39c12",     # 노랑 — 주의
    "bad": "#e74c3c",      # 빨강 — 위험
    "na": "#bdc3c7",       # 회색 — 데이터 없음
    "change_bg": "#fff3cd",     # 연노랑 — 변경 셀 배경
    "change_text": "#856404",   # 진갈색 — 변경 셀 글자
    "muted": "#95a5a6",         # 회색 — 보조 요소 (다이어그램 점선 등)
    "primary": "#3498db",       # 파랑 — 주요 UI 강조
}

# --- 3축 그룹 색상 (여정 테이블 컬럼 구분) ---
AXIS_COLORS = {
    "composition": "#2ecc71",   # 구성 — 초록
    "transformation": "#9b59b6", # 변형 — 보라
    "delivery": "#3498db",       # 전달 — 파랑
    "quality": "#e67e22",        # 품질 — 주황
}

DIMENSIONS = ["completeness", "efficiency", "relevance", "consistency"]
DIMENSION_LABELS = {
    "completeness": "Completeness (완전성)",
    "efficiency": "Efficiency (효율성)",
    "relevance": "Relevance (관련성)",
    "consistency": "Consistency (일관성)",
}
SCORE_KEYS = {
    "completeness": "completeness_score",
    "efficiency": "efficiency_score",
    "relevance": "relevance_score",
    "consistency": "consistency_score",
}


# ═══════════════════════════════════════
# Tab 1: 세션 개요
# ═══════════════════════════════════════

def score_trend(turns: list[dict]) -> go.Figure:
    """4D 품질 점수 추이 차트를 생성한다.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        4D 점수 line chart + threshold hlines.
    """
    turn_numbers = [t["turn_number"] for t in turns]
    fig = go.Figure()

    # 각 차원의 점수 라인
    for dim in DIMENSIONS:
        score_key = SCORE_KEYS[dim]
        values = [t["scores"].get(score_key) for t in turns]
        fig.add_trace(go.Scatter(
            x=turn_numbers,
            y=values,
            mode="lines+markers",
            name=DIMENSION_LABELS[dim],
            line=dict(color=COLORS[dim], width=2),
            marker=dict(size=8),
            connectgaps=False,
        ))

    # Threshold 라인 — 고유 임계값마다 회색 점선 1개 (차원색 혼용 방지)
    for threshold_val in sorted(set(THRESHOLDS[d] for d in DIMENSIONS)):
        fig.add_hline(
            y=threshold_val,
            line_dash="dot",
            line_color="#888888",
            opacity=0.4,
            annotation_text=f"threshold {threshold_val}",
            annotation_position="bottom right",
            annotation_font_size=9,
            annotation_opacity=0.6,
        )

    # 위험 점수에 라벨 표시 (임계값 미만인 점수에 값 표시)
    for dim in DIMENSIONS:
        score_key = SCORE_KEYS[dim]
        threshold = THRESHOLDS[dim]
        for t in turns:
            val = t["scores"].get(score_key)
            if val is not None and val < threshold:
                fig.add_annotation(
                    x=t["turn_number"], y=val,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(size=10, color=COLORS["threshold"]),
                    yshift=12,
                )

    fig.update_layout(
        title="4D 품질 점수 추이",
        xaxis_title="Turn",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=60, b=40),
    )
    return fig


def radar_4d(turn: dict) -> go.Figure:
    """단일 턴의 4D 레이더 차트를 생성한다.

    Args:
        turn: 단일 턴 데이터.

    Returns:
        4D radar chart.
    """
    categories = [DIMENSION_LABELS[d] for d in DIMENSIONS]
    values = [turn["scores"].get(SCORE_KEYS[d], 0) or 0 for d in DIMENSIONS]
    thresholds = [THRESHOLDS[d] for d in DIMENSIONS]

    # 레이더 차트는 시작점으로 돌아가야 하므로 첫 값을 끝에 추가
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    thresholds_closed = thresholds + [thresholds[0]]

    fig = go.Figure()

    # Threshold 영역
    fig.add_trace(go.Scatterpolar(
        r=thresholds_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.1)",
        line=dict(color=COLORS["threshold"], dash="dot"),
        name="Threshold",
    ))

    # 실제 점수
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.2)",
        line=dict(color="#3498db", width=2),
        name=f"Turn {turn['turn_number']}",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"Turn {turn['turn_number']} — 4D 품질 레이더",
        height=350,
        margin=dict(t=60, b=20),
    )
    return fig


# ═══════════════════════════════════════
# Tab 2: 컨텍스트 진화
# ═══════════════════════════════════════

def token_composition(turns: list[dict]) -> go.Figure:
    """턴별 토큰 소스 구성 stacked bar 차트를 생성한다.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        토큰 소스별 stacked bar chart.
    """
    turn_numbers = [t["turn_number"] for t in turns]

    # v3: turn_conclusions_tokens는 previous_turns_tokens에 합산되어 폐기.
    # 역순으로 add_trace하여 stacked bar 시각 순서가
    # 아래→위 = Gathered Data → ... → System Prompt (조립 순서)가 되도록 한다
    sources = {
        "Gathered Data": ("context.source.gathered_data_tokens", "gathered"),
        "Query Analysis": ("context.source.query_analysis_tokens", "query_analysis"),
        "Previous Turns": ("context.source.previous_turns_tokens", "previous_turns"),
        "System Prompt": ("context.source.system_prompt_tokens", "system_prompt"),
    }

    fig = go.Figure()
    for label, (attr_key, color_key) in sources.items():
        values = [t["metadata"].get(ATTRS[attr_key], 0) or 0 for t in turns]
        fig.add_trace(go.Bar(
            x=turn_numbers,
            y=values,
            name=label,
            marker_color=COLORS.get(color_key, "#95a5a6"),
        ))

    fig.update_layout(
        title="턴별 토큰 구성",
        xaxis_title="Turn",
        yaxis_title="Tokens",
        barmode="stack",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=60, b=40),
    )
    return fig


def noise_evolution(turns: list[dict]) -> go.Figure:
    """Noise ratio + Rot risk + Window utilization 추이 차트를 생성한다.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        dual-axis line chart.
    """
    turn_numbers = [t["turn_number"] for t in turns]
    noise_values = [t["metadata"].get(ATTRS["context.noise_ratio"]) for t in turns]
    util_values = [t["metadata"].get(ATTRS["context.window_utilization"]) for t in turns]
    rot_values = [t["metadata"].get(ATTRS["context.rot_risk"]) for t in turns]
    # A1: rot_velocity — rot_risk의 턴 간 변화율
    rot_velocity_values = [
        t["metadata"].get(ATTRS.get("context.rot_velocity", "context.rot_velocity"))
        for t in turns
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=turn_numbers, y=noise_values,
        mode="lines+markers", name="Noise Ratio",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=turn_numbers, y=util_values,
        mode="lines+markers", name="Window Utilization",
        line=dict(color="#3498db", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=turn_numbers, y=rot_values,
        mode="lines+markers", name="Rot Risk",
        line=dict(color="#f39c12", width=2, dash="dash"),
    ))
    # Rot Velocity — 보조축(secondary y) 없이 같은 스케일에 표시
    # velocity가 있는 턴이 하나라도 있을 때만 trace 추가
    if any(v is not None for v in rot_velocity_values):
        fig.add_trace(go.Scatter(
            x=turn_numbers, y=rot_velocity_values,
            mode="lines+markers", name="Rot Velocity (Δ)",
            line=dict(color="#8e44ad", width=2, dash="dot"),
            marker=dict(size=6, symbol="diamond"),
        ))

    # 기준선: noise_ratio > 0.5 경계
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="noise 경계 (0.5)", annotation_position="bottom right",
                  annotation_font_size=9, annotation_opacity=0.5)
    # 기준선: rot_risk > 0.3 경계
    fig.add_hline(y=0.3, line_dash="dot", line_color="#f39c12", opacity=0.3,
                  annotation_text="rot risk 경계 (0.3)", annotation_position="top right",
                  annotation_font_size=9, annotation_opacity=0.5)

    fig.update_layout(
        title="",
        xaxis_title="Turn",
        yaxis_title="Ratio",
        yaxis_range=[0, 1],
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(t=60, b=40),
    )
    return fig


def source_selection_heatmap(turns: list[dict]) -> go.Figure:
    """턴 × 소스타입 히트맵을 생성한다.

    trace metadata에서 source.types_selected 또는 gather.tools_called를 읽어
    어떤 턴에서 어떤 소스를 사용했는지 표시한다.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        히트맵 Figure.
    """
    source_types = ["csv", "rag", "web", "api"]
    turn_numbers = [f"Turn {t['turn_number']}" for t in turns]

    # 각 턴에서 사용한 소스 타입 추출
    # trace-level metadata에는 source.types_selected가 없을 수 있으므로
    # gather.tools_called에서 추론
    matrix = []
    for source_type in source_types:
        row = []
        for turn in turns:
            metadata = turn["metadata"]
            # source.types_selected가 있으면 직접 사용
            selected = metadata.get(ATTRS.get("source.types_selected", ""), [])
            if selected and source_type in selected:
                row.append(1)
            else:
                # tools_called에서 추론
                tools = metadata.get(ATTRS.get("gather.tools_called", ""), [])
                if isinstance(tools, list):
                    tool_source_map = {
                        "csv": ["pandas_query", "calculate"],
                        "rag": ["rag_search", "rag_deep_read", "rag_global_summary", "rag_compare"],
                        "web": ["web_search"],
                        "api": ["api_query"],
                    }
                    if any(t in tools for t in tool_source_map.get(source_type, [])):
                        row.append(1)
                    else:
                        row.append(0)
                else:
                    row.append(0)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=turn_numbers,
        y=[s.upper() for s in source_types],
        colorscale=[[0, "#f0f0f0"], [1, "#2ecc71"]],
        showscale=False,
        text=matrix,
        texttemplate="%{text}",
        textfont=dict(size=14),
    ))

    fig.update_layout(
        title="소스 선택 히트맵",
        xaxis_title="",
        yaxis_title="",
        height=250,
        margin=dict(t=40, b=20, l=60),
    )
    return fig


# ═══════════════════════════════════════
# Tab 3: Attribute Impact
# ═══════════════════════════════════════

def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Attribute × 4D dimension 상관 히트맵을 생성한다.

    BH FDR 보정된 q-value < 0.05인 셀만 강조하고, 나머지는 흐리게 표시한다.

    Args:
        corr_df: compute_attribute_correlations()의 반환값.

    Returns:
        상관 히트맵 Figure.
    """
    if corr_df.empty:
        return _empty_figure("데이터 부족: 상관 분석을 수행할 수 없습니다")

    dimensions = ["completeness", "efficiency", "relevance", "consistency"]
    attributes = sorted(corr_df["attribute"].unique())

    # 상관계수 행렬 구성
    z_values = []
    hover_text = []
    for attr in attributes:
        row_z = []
        row_hover = []
        for dim in dimensions:
            match = corr_df[(corr_df["attribute"] == attr) & (corr_df["dimension"] == dim)]
            if len(match) > 0:
                r = match.iloc[0]
                corr = r["correlation"]
                p = r["p_value"]
                q = r["q_value"]
                n = r["n_samples"]
                sig = "✓" if r["is_significant"] else ""
                # 유의하지 않으면 값을 희석
                row_z.append(corr if r["is_significant"] else corr * 0.3)
                row_hover.append(
                    f"{attr} × {dim}<br>"
                    f"r={corr:.3f} {sig}<br>"
                    f"p={p:.4f}, q={q:.4f}<br>"
                    f"n={n}"
                )
            else:
                row_z.append(0)
                row_hover.append(f"{attr} × {dim}<br>데이터 부족")
        z_values.append(row_z)
        hover_text.append(row_hover)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[DIMENSION_LABELS.get(d, d) for d in dimensions],
        y=attributes,
        colorscale=[
            [0.0, "#e74c3c"],     # 강한 음의 상관
            [0.25, "#f5b7b1"],
            [0.5, "#f0f0f0"],     # 무상관
            [0.75, "#abebc6"],
            [1.0, "#2ecc71"],     # 강한 양의 상관
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=hover_text,
        hoverinfo="text",
        colorbar=dict(title="Spearman r"),
    ))

    fig.update_layout(
        title="Attribute → 4D 상관 히트맵 (BH FDR 보정, q < 0.05 강조)",
        xaxis_title="",
        yaxis_title="",
        height=max(400, len(attributes) * 30 + 100),
        margin=dict(t=60, b=40, l=280),
        yaxis=dict(tickfont=dict(size=14)),
        xaxis=dict(tickfont=dict(size=13)),
    )
    return fig


def impact_bar(corr_df: pd.DataFrame, dimension: str, top_n: int = 10) -> go.Figure:
    """특정 4D 차원의 top-N 영향 속성 bar chart를 생성한다.

    Args:
        corr_df: compute_attribute_correlations()의 반환값.
        dimension: 4D 차원명 (예: "completeness").
        top_n: 표시할 최대 속성 수.

    Returns:
        bar chart Figure.
    """
    if corr_df.empty:
        return _empty_figure(f"{dimension}: 데이터 부족")

    dim_df = corr_df[corr_df["dimension"] == dimension].copy()
    if dim_df.empty:
        return _empty_figure(f"{dimension}: 관련 데이터 없음")

    dim_df["abs_corr"] = dim_df["correlation"].abs()
    dim_df = dim_df.nlargest(top_n, "abs_corr")
    dim_df = dim_df.sort_values("correlation")

    colors = [
        COLORS["completeness"] if r["is_significant"] and r["correlation"] > 0
        else COLORS["threshold"] if r["is_significant"] and r["correlation"] < 0
        else "#bdc3c7"
        for _, r in dim_df.iterrows()
    ]

    fig = go.Figure(data=go.Bar(
        x=dim_df["correlation"],
        y=dim_df["attribute"],
        orientation="h",
        marker_color=colors,
        text=[f"r={r:.3f}" for r in dim_df["correlation"]],
        textposition="outside",
        hovertext=[
            f"q={q:.4f}, n={n}" for q, n in zip(dim_df["q_value"], dim_df["n_samples"])
        ],
        hoverinfo="text+y",
    ))

    dim_label = DIMENSION_LABELS.get(dimension, dimension)
    fig.update_layout(
        title=f"Top {top_n} 영향 속성 — {dim_label}",
        xaxis_title="Spearman r",
        xaxis_range=[-1, 1],
        yaxis_title="",
        height=max(300, len(dim_df) * 30 + 100),
        margin=dict(t=60, b=40, l=200),
    )
    return fig


def cross_turn_evolution(turns: list[dict]) -> go.Figure:
    """G3: 교차 턴 진화 — new_data_ratio vs inherited_ratio 스택 area 차트.

    턴이 진행되면서 새 데이터와 상속 데이터의 비율이 어떻게 변하는지 시각화한다.
    LLM 호출 없음 — 순수 데이터 시각화.

    Args:
        turns: 턴별 데이터 리스트 (metadata 포함).

    Returns:
        Plotly Figure (stacked area chart).
    """
    if len(turns) < 2:
        return _empty_figure("교차 턴 진화: 2턴 이상 필요")

    turn_nums = []
    new_ratios = []
    inherited_ratios = []

    for turn in turns:
        meta = turn.get("metadata", {})
        nr = meta.get("context.new_data_ratio")
        ir = meta.get("context.inherited_ratio")
        if nr is not None and ir is not None:
            turn_nums.append(meta.get("turn.number", len(turn_nums) + 1))
            new_ratios.append(nr)
            inherited_ratios.append(ir)

    if not turn_nums:
        return _empty_figure("교차 턴 진화: 데이터 없음")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turn_nums, y=new_ratios, name="새 데이터",
        fill="tozeroy", mode="lines",
        line=dict(color=COLORS["gathered"]),
    ))
    fig.add_trace(go.Scatter(
        x=turn_nums, y=inherited_ratios, name="상속 데이터",
        fill="tonexty", mode="lines",
        line=dict(color=COLORS["previous_turns"]),
    ))
    fig.update_layout(
        title="",
        xaxis_title="턴", yaxis_title="비율",
        yaxis=dict(range=[0, 1.05]),
        height=350, margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def causal_propagation_chart(turns: list[dict]) -> go.Figure:
    """Post-1: 인과 전파 히트맵 — 이전 턴이 현재 턴에 미친 영향.

    각 턴의 causal_sources에서 이전 턴별 impact 값을 추출하여 히트맵으로 시각화.
    LLM 호출 없음.

    Args:
        turns: 턴별 데이터 리스트 (metadata 포함).

    Returns:
        Plotly Figure (heatmap).
    """
    if len(turns) < 2:
        return _empty_figure("인과 전파: 2턴 이상 필요")

    # 턴 번호 수집
    all_turns = []
    for turn in turns:
        meta = turn.get("metadata", {})
        tn = meta.get("turn.number", len(all_turns) + 1)
        all_turns.append(tn)

    n = len(all_turns)
    # n x n 행렬 (source_turn x current_turn)
    impact_matrix = [[0.0] * n for _ in range(n)]

    for i, turn in enumerate(turns):
        meta = turn.get("metadata", {})
        causal = meta.get(ATTRS.get("context.causal_sources", ""), [])
        if not isinstance(causal, list):
            continue
        for cs in causal:
            src_turn = cs.get("turn", 0)
            impact = cs.get("impact", 0)
            # src_turn의 인덱스 찾기
            if src_turn in all_turns:
                j = all_turns.index(src_turn)
                impact_matrix[j][i] = impact

    fig = go.Figure(data=go.Heatmap(
        z=impact_matrix,
        x=[f"Turn {t}" for t in all_turns],
        y=[f"Turn {t}" for t in all_turns],
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        colorbar=dict(title="영향력"),
    ))
    fig.update_layout(
        title="",
        xaxis_title="현재 턴", yaxis_title="소스 턴",
        height=400, margin=dict(t=40, b=30),
    )
    return fig


def fidelity_trend(turns: list[dict]) -> go.Figure:
    """충실도 추세 — fidelity_score 라인 차트.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        Plotly Figure (line chart with threshold).
    """
    if len(turns) < 2:
        return _empty_figure("충실도 추세: 2턴 이상 필요")

    turn_nums = []
    fidelity_vals = []
    for turn in turns:
        meta = turn.get("metadata", {})
        f = meta.get("context.fidelity_score")
        if f is not None:
            turn_nums.append(meta.get("turn.number", turn.get("turn_number", len(turn_nums) + 1)))
            fidelity_vals.append(f)

    if not turn_nums:
        return _empty_figure("충실도 데이터 없음")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turn_nums, y=fidelity_vals, name="Fidelity Score",
        mode="lines+markers", line=dict(color=COLORS["consistency"], width=2),
        marker=dict(size=8),
    ))
    # 의미 손실 경계선
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", opacity=0.4,
                  annotation_text="의미 손실 경계 (0.5)", annotation_position="bottom right",
                  annotation_font_size=9, annotation_opacity=0.5)
    fig.update_layout(
        title="",
        xaxis_title="턴", yaxis_title="점수",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(dtick=1),
        height=350, margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def compression_detail(turns: list[dict]) -> go.Figure:
    """압축 충실도 — compression_ratio 바 차트 + conditions_preserved 마커.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        Plotly Figure (bar + markers).
    """
    # 조건보존/조건탈락을 별도 trace로 분리하여 범례 생성
    preserved_nums = []
    preserved_ratios = []
    lost_nums = []
    lost_ratios = []

    for turn in turns:
        meta = turn.get("metadata", {})
        ratio = meta.get("response.compression_ratio")
        preserved = meta.get("response.conditions_preserved", True)
        if ratio is not None:
            tn = meta.get("turn.number", turn.get("turn_number", 0))
            if preserved:
                preserved_nums.append(tn)
                preserved_ratios.append(ratio)
            else:
                lost_nums.append(tn)
                lost_ratios.append(ratio)

    if not preserved_nums and not lost_nums:
        return _empty_figure("압축 데이터 없음")

    fig = go.Figure()
    c_good = STATUS_COLORS["good"]
    c_bad = STATUS_COLORS["bad"]

    if preserved_nums:
        fig.add_trace(go.Bar(
            x=preserved_nums, y=preserved_ratios,
            name="조건 보존", marker_color=c_good,
            text=["✓"] * len(preserved_nums), textposition="outside",
        ))
    if lost_nums:
        fig.add_trace(go.Bar(
            x=lost_nums, y=lost_ratios,
            name="조건 탈락", marker_color=c_bad,
            text=["✗"] * len(lost_nums), textposition="outside",
        ))

    fig.add_hline(y=0.1, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="과도한 압축 (0.1)", annotation_position="bottom right",
                  annotation_font_size=9)
    fig.update_layout(
        xaxis_title="턴", yaxis_title="압축률 (결론/분석)",
        xaxis=dict(dtick=1),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300, margin=dict(t=40, b=30),
    )
    return fig


def density_trend(turns: list[dict]) -> go.Figure:
    """정보 밀도 추세 — information_density + redundancy_ratio 라인 차트.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        Plotly Figure (dual line chart).
    """
    turn_nums = []
    density_vals = []
    redundancy_vals = []
    for turn in turns:
        meta = turn.get("metadata", {})
        d = meta.get("context.information_density")
        r = meta.get("context.redundancy_ratio")
        tn = meta.get("turn.number", turn.get("turn_number", len(turn_nums) + 1))
        turn_nums.append(tn)
        density_vals.append(d)
        redundancy_vals.append(r)

    if not turn_nums:
        return _empty_figure("정보 밀도 데이터 없음")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turn_nums, y=density_vals, name="Information Density",
        mode="lines+markers", line=dict(color=COLORS["efficiency"], width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=turn_nums, y=redundancy_vals, name="Redundancy Ratio",
        mode="lines+markers", line=dict(color=COLORS["threshold"], width=2, dash="dash"),
        marker=dict(size=6),
    ))
    # y축 상한: 실제 최대값의 1.2배 또는 최소 0.6
    all_vals = [v for v in density_vals + redundancy_vals if v is not None]
    y_max = max(max(all_vals), 0.5) * 1.2 if all_vals else 0.6

    fig.update_layout(
        title="",
        xaxis_title="턴", yaxis_title="비율",
        yaxis=dict(range=[0, y_max]),
        xaxis=dict(dtick=1),
        height=350, margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ═══════════════════════════════════════
# Tab 1: 컨텍스트 여정 요약
# ═══════════════════════════════════════
# 모든 함수는 compute_session_aggregates()의 반환 dict를 입력으로 받는다.
# turns를 직접 순회하지 않음 (score_trend만 예외 — 기존 함수 재사용).


def source_contribution_summary(agg: dict) -> go.Figure:
    """§1: 소스별 기여 비율 수평 바 차트를 생성한다.

    Tab 3의 턴별 히트맵(on/off)과 달리 세션 평균 기여율(%)을 표시한다.

    Args:
        agg: compute_session_aggregates()의 반환값.

    Returns:
        수평 바 차트 Figure.
    """
    contrib = agg.get("source_contribution_agg", {})
    if not contrib:
        return _empty_figure("소스 기여 데이터 없음")

    # 기여율 내림차순 정렬
    sorted_items = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
    sources = [s for s, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # 소스별 색상
    source_colors = {
        "csv": "#2ecc71",    # 초록 — 정확한 수치
        "rag": "#3498db",    # 파랑 — 도메인 지식
        "web": "#e67e22",    # 주황 — 실시간 정보
        "api": "#9b59b6",    # 보라 — 공공 데이터
    }
    colors = [source_colors.get(s, "#95a5a6") for s in sources]

    fig = go.Figure(data=go.Bar(
        x=values,
        y=[s.upper() for s in sources],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="소스별 기여 비율",
        xaxis=dict(title="비율", range=[0, max(values) * 1.3] if values else [0, 1],
                   tickformat=".0%"),
        yaxis=dict(title=""),
        height=250,
        margin=dict(t=40, b=30, l=60, r=40),
    )
    return fig


def tool_usage_summary(agg: dict) -> go.Figure:
    """§1: 도구 사용 빈도 수평 바 차트를 생성한다.

    세션 전체에서 각 도구가 몇 번 호출되었는지 표시한다.

    Args:
        agg: compute_session_aggregates()의 반환값.

    Returns:
        수평 바 차트 Figure.
    """
    freq = agg.get("tool_frequency", {})
    if not freq:
        return _empty_figure("도구 사용 데이터 없음")

    # 빈도 내림차순 정렬
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    tools = [t for t, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    fig = go.Figure(data=go.Bar(
        x=counts,
        y=tools,
        orientation="h",
        marker_color="#3498db",
        text=[str(c) for c in counts],
        textposition="outside",
    ))
    fig.update_layout(
        title="도구 사용 빈도",
        xaxis=dict(title="호출 횟수", dtick=1),
        yaxis=dict(title="", autorange="reversed"),
        height=max(200, len(tools) * 30 + 80),
        margin=dict(t=40, b=30, l=120, r=40),
    )
    return fig


def collection_breakdown(agg: dict) -> go.Figure:
    """§2: 수집 항목 분해 수평 stacked bar를 생성한다.

    collected = delivered + excluded + truncated 관계를 한 줄 stacked bar로 표시.
    제외·절단은 병렬 활동이므로 펀넬이 아닌 분해(decomposition) 시각화를 사용한다.

    Args:
        agg: compute_session_aggregates()의 반환값.

    Returns:
        수평 stacked bar Figure (1줄).
    """
    collected = agg.get("items_collected_total", 0)
    excluded = agg.get("items_excluded_total", 0)
    truncated = agg.get("truncated_items_total", 0)
    delivered = collected - excluded - truncated

    if collected == 0:
        return _empty_figure("수집 데이터 없음")

    # 음수 방지 (데이터 불일치 시)
    delivered = max(delivered, 0)

    fig = go.Figure()

    # 순서: delivered → excluded → truncated (긍정→부정)
    segments = [
        ("LLM 도달", delivered, "#2ecc71"),
        ("제외", excluded, "#e74c3c"),
        ("절단", truncated, "#f39c12"),
    ]

    for label, value, color in segments:
        pct = value / collected * 100 if collected > 0 else 0
        fig.add_trace(go.Bar(
            x=[value],
            y=["수집 항목"],
            orientation="h",
            name=label,
            marker_color=color,
            text=[f"{label} {value}건 ({pct:.0f}%)"],
            textposition="inside",
            insidetextanchor="middle",
        ))

    fig.update_layout(
        title=f"수집 항목 분해 (총 {collected}건)",
        barmode="stack",
        xaxis=dict(title="건수"),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=180,
        margin=dict(t=50, b=30, l=10, r=10),
    )
    return fig


def token_distribution_summary(agg: dict) -> go.Figure:
    """§3: 세션 평균 토큰 구성 수평 stacked bar를 생성한다.

    Tab 3의 턴별 N줄 stacked bar와 달리 세션 평균 1줄로 요약한다.

    Args:
        agg: compute_session_aggregates()의 반환값.

    Returns:
        수평 stacked bar Figure (1줄).
    """
    token_avg = agg.get("token_sources_avg", {})
    if not token_avg:
        return _empty_figure("토큰 구성 데이터 없음")

    # 소스 정의 (표시 순서)
    sources = [
        ("system", "System Prompt", COLORS.get("system_prompt", "#95a5a6")),
        ("query", "Query Analysis", COLORS.get("query_analysis", "#3498db")),
        ("gathered", "Gathered Data", COLORS.get("gathered", "#2ecc71")),
        ("previous", "Previous Turns", COLORS.get("previous_turns", "#e74c3c")),
        ("conclusions", "Turn Conclusions", COLORS.get("conclusions", "#f39c12")),
    ]

    total = sum(token_avg.get(key, 0) for key, _, _ in sources)
    if total == 0:
        return _empty_figure("토큰 구성 데이터 없음")

    fig = go.Figure()
    for key, label, color in sources:
        val = token_avg.get(key, 0)
        pct = val / total * 100 if total > 0 else 0
        fig.add_trace(go.Bar(
            x=[val],
            y=["토큰 구성"],
            orientation="h",
            name=label,
            marker_color=color,
            text=[f"{val/1000:.1f}K ({pct:.0f}%)"] if val > 0 else [""],
            textposition="inside",
            insidetextanchor="middle",
        ))

    fig.update_layout(
        title=f"세션 평균 토큰 구성 (총 {total/1000:.1f}K)",
        barmode="stack",
        xaxis=dict(title="토큰"),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=180,
        margin=dict(t=50, b=30, l=10, r=10),
    )
    return fig


def fidelity_detail_chart(turns: list[dict]) -> go.Figure:
    """F4+A7: fidelity_detail 3요소 분리 그룹 막대 차트.

    fidelity_score를 구성하는 3가지 요소 (조건보존, 주장비율, 압축적절성)를
    턴별로 분리하여 어떤 요소가 충실도 저하의 원인인지 식별한다.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        Plotly Figure (grouped bar chart).
    """
    turn_nums = []
    cond_scores = []
    claims_ratios = []
    compression_penalties = []

    for turn in turns:
        meta = turn.get("metadata", {})
        detail = meta.get("context.fidelity_detail")
        if not isinstance(detail, dict):
            continue
        tn = meta.get("turn.number", turn.get("turn_number", len(turn_nums) + 1))
        turn_nums.append(tn)
        cond_scores.append(detail.get("cond_score", 0))
        claims_ratios.append(detail.get("claims_ratio", 0))
        compression_penalties.append(detail.get("compression_penalty", 0))

    if not turn_nums:
        return _empty_figure("Fidelity 상세 데이터 없음")

    fig = go.Figure()
    # 3요소를 그룹 막대로 표시
    _fw = FIDELITY_SCORE_WEIGHTS
    fig.add_trace(go.Bar(
        x=turn_nums, y=cond_scores,
        name=f"조건보존 ({round(_fw['cond_score'] * 100)}%)", marker_color="#9b59b6",
    ))
    fig.add_trace(go.Bar(
        x=turn_nums, y=claims_ratios,
        name=f"주장비율 ({round(_fw['claims_ratio'] * 100)}%)", marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        x=turn_nums, y=compression_penalties,
        name=f"압축적절성 ({round(_fw['compression_penalty'] * 100)}%)", marker_color="#e67e22",
    ))

    fig.update_layout(
        xaxis_title="턴", yaxis_title="점수",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(dtick=1),
        barmode="group",
        height=320, margin=dict(t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """메시지만 표시하는 빈 차트를 생성한다."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#7f8c8d"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=200, margin=dict(t=20, b=20),
    )
    return fig


# ═══════════════════════════════════════
# Phase B 신규 차트 (변별성 검증된 4종)
# ═══════════════════════════════════════

# --- 4 sources 색상 ---
_SOURCE_COLORS = {
    "csv": "#3498db",   # 파랑
    "rag": "#2ecc71",   # 초록
    "web": "#f39c12",   # 노랑
    "api": "#e67e22",   # 주황
    "memory": "#9b59b6",  # 보라
}

_TOOL_TO_SOURCE = {
    "pandas_query": "csv", "calculate": "csv",
    "rag_search": "rag", "rag_deep_read": "rag",
    "rag_global_summary": "rag", "rag_compare": "rag",
    "web_search": "web",
    "api_query": "api",
    "lookup_previous": "memory",
}


def source_contribution_stream(turns: list[dict]) -> go.Figure:
    """소스 기여도 stream graph — 턴별 소스 mix 변화 (Phase B #1).

    데이터 우선순위: source.contribution (보강 후) → gather.source_distribution
    (실측 std=0.35) → tools_called 빈도. 100% 정규화 stacked area.
    """
    rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        tn = turn.get("turn_number")
        contrib = meta.get("source.contribution") or {}
        if not isinstance(contrib, dict) or not contrib:
            contrib = meta.get("gather.source_distribution") or {}
        if not isinstance(contrib, dict) or not contrib:
            tools = meta.get("gather.tools_called") or []
            if isinstance(tools, list) and tools:
                counts: dict[str, int] = {}
                for t in tools:
                    src = _TOOL_TO_SOURCE.get(t, t)
                    counts[src] = counts.get(src, 0) + 1
                total = sum(counts.values())
                contrib = {k: v / total for k, v in counts.items()}
        if not contrib:
            continue
        rows.append({"turn": tn, **contrib})

    if not rows:
        return _empty_figure("소스 기여도 데이터 없음")

    df = pd.DataFrame(rows).fillna(0)
    df = df.sort_values("turn")

    fig = go.Figure()
    sources_in_data = [c for c in df.columns if c != "turn"]
    for src in sources_in_data:
        color = _SOURCE_COLORS.get(src, "#7f8c8d")
        fig.add_trace(go.Scatter(
            x=df["turn"], y=df[src],
            mode="lines",
            stackgroup="one", groupnorm="percent",
            name=src.upper(),
            line=dict(width=0.5, color=color),
            fillcolor=color,
            hovertemplate=f"{src.upper()}: %{{y:.0f}}%<extra>Turn %{{x}}</extra>",
        ))
    fig.update_layout(
        title=None,
        xaxis_title="Turn", yaxis_title="기여도 (%)",
        xaxis=dict(dtick=1), yaxis=dict(range=[0, 100], ticksuffix="%"),
        height=320, margin=dict(t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def contributing_flow_sankey(turns: list[dict]) -> go.Figure:
    """기여 흐름 Sankey — source turn → target turn (변형 축 핵심).

    Charter §3.2 "어디서 어디로 살아남는가"를 직접 시각화한다.
    각 링크: (이전 턴 → 현재 턴), 두께 = 활용된 결론 수 또는 1.

    데이터 우선순위:
        1. analysis.referenced_turns + analysis.utilized_conclusions (정확) → Sankey
        2. analysis.referenced_turns 단독 (값=1) → Sankey
        3. fallback: context.contributing_turns 합계만 → 막대 차트
    """
    flows: dict[tuple[int, int], float] = {}
    contributing_only: list[tuple[int, int]] = []
    all_turns: set[int] = set()

    for turn in turns:
        meta = turn.get("metadata", {})
        tn = turn.get("turn_number")
        if not isinstance(tn, int):
            continue
        all_turns.add(tn)

        ref_turns = meta.get("analysis.referenced_turns") or []
        utilized = meta.get("analysis.utilized_conclusions") or []
        contributing = meta.get("context.contributing_turns")

        if isinstance(ref_turns, list) and ref_turns:
            ut_count = len(utilized) if isinstance(utilized, list) else 0
            per_ref = (ut_count / len(ref_turns)) if ut_count > 0 else 1.0
            for src in ref_turns:
                if isinstance(src, int) and src < tn:
                    flows[(src, tn)] = flows.get((src, tn), 0) + per_ref
                    all_turns.add(src)
        elif isinstance(contributing, (int, float)) and contributing > 0:
            contributing_only.append((tn, int(contributing)))

    if not flows and not contributing_only:
        return _empty_figure(
            "기여 흐름 데이터 없음 (첫 턴은 비교 대상 없음, "
            "또는 referenced_turns/contributing_turns 미기록)"
        )

    # 명시 흐름이 없고 contributing_turns만 있는 경우 — 막대로 폴백
    if not flows and contributing_only:
        x_vals = [t for t, _ in contributing_only]
        y_vals = [c for _, c in contributing_only]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_vals, y=y_vals,
            marker_color="#9b59b6",
            text=y_vals, textposition="outside",
            hovertemplate="Turn %{x}: 이전 %{y}개 턴 기여<extra></extra>",
        ))
        fig.update_layout(
            title=None,
            xaxis_title="Turn (referenced_turns 미기록 — 막대 폴백)",
            yaxis_title="기여 이전 턴 수",
            xaxis=dict(dtick=1),
            height=280, margin=dict(t=20, b=30),
            showlegend=False,
        )
        return fig

    # Sankey: 각 턴을 단일 노드로 — src/tgt 양쪽에서 재사용 가능
    sorted_turns = sorted(all_turns)
    turn_to_idx = {t: i for i, t in enumerate(sorted_turns)}
    labels = [f"T{t}" for t in sorted_turns]

    sources, targets, values, link_colors = [], [], [], []
    for (src, tgt), val in flows.items():
        sources.append(turn_to_idx[src])
        targets.append(turn_to_idx[tgt])
        values.append(val)
        link_colors.append("rgba(155,89,182,0.45)")  # 보라 반투명

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=18,
            line=dict(color="#888", width=0.5),
            label=labels,
            color="#9b59b6",
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            hovertemplate="T%{source.label} → T%{target.label}: "
                          "%{value:.1f} 결론 흐름<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=None,
        height=340, margin=dict(t=20, b=20, l=20, r=20),
        font=dict(size=12),
    )
    return fig


def confidence_delta_bar(turns: list[dict]) -> go.Figure:
    """confidence_delta 막대 차트 (Phase B #3).

    iteration 간 sufficiency_confidence 변화 — 패턴 A(재수집 효과) 직접 측정.
    양수=개선(초록), 음수=악화(빨강).
    """
    x_vals, y_vals, colors = [], [], []
    for turn in turns:
        meta = turn.get("metadata", {})
        tn = turn.get("turn_number")
        delta = meta.get("context.confidence_delta")
        if isinstance(delta, (int, float)):
            x_vals.append(tn)
            y_vals.append(float(delta))
            colors.append(STATUS_COLORS["good"] if delta >= 0 else STATUS_COLORS["bad"])

    if not x_vals:
        return _empty_figure("confidence_delta 데이터 없음")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals, y=y_vals,
        marker_color=colors,
        text=[f"{v:+.2f}" for v in y_vals], textposition="outside",
        hovertemplate="Turn %{x}: Δ %{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="#888", line_width=1)
    fig.update_layout(
        title=None,
        xaxis_title="Turn", yaxis_title="Δ confidence",
        xaxis=dict(dtick=1),
        height=280, margin=dict(t=20, b=30),
        showlegend=False,
    )
    return fig


def verify_gate_sankey(turns: list[dict]) -> go.Figure:
    """검증 게이트 Sankey (Phase B #4).

    generate_analysis → verify_numeric → verify_interpretation → respond_to_user
    경로별 흐름. fail 시 분기(fail_numeric → 재수집 / fail_interpretation → 재생성).
    """
    counts = {
        "total": 0,
        "num_pass": 0, "num_fail": 0,
        "interp_pass": 0, "interp_fail": 0,
        "final_pass": 0,
    }
    for turn in turns:
        meta = turn.get("metadata", {})
        verdict = meta.get("verify.overall_verdict")
        num_passed = meta.get("verify.numeric_check_passed")
        if verdict is None and num_passed is None:
            continue
        counts["total"] += 1
        if num_passed:
            counts["num_pass"] += 1
        elif num_passed is False:
            counts["num_fail"] += 1
        if verdict == "pass":
            counts["interp_pass"] += 1
            counts["final_pass"] += 1
        elif verdict == "fail_interpretation":
            counts["interp_fail"] += 1
        elif verdict == "fail_numeric":
            pass  # 수치 단계에서 이미 fail

    if counts["total"] == 0:
        return _empty_figure("검증 데이터 없음")

    # 노드 정의: 0=Generate / 1=수치 검증 / 2=수치 PASS / 3=수치 FAIL→재수집 /
    #            4=해석 검증 / 5=해석 PASS / 6=해석 FAIL→재생성 / 7=Respond
    labels = [
        f"Generate ({counts['total']})",
        "수치 검증",
        f"수치 PASS ({counts['num_pass']})",
        f"수치 FAIL ({counts['num_fail']})",
        "해석 검증",
        f"해석 PASS ({counts['interp_pass']})",
        f"해석 FAIL ({counts['interp_fail']})",
        f"Respond ({counts['final_pass']})",
    ]
    sources, targets, values, colors = [], [], [], []

    def add(s, t, v, color):
        if v > 0:
            sources.append(s); targets.append(t); values.append(v); colors.append(color)

    add(0, 1, counts["total"], "rgba(52,152,219,0.3)")  # Gen→수치
    add(1, 2, counts["num_pass"], "rgba(46,204,113,0.4)")  # 수치 PASS
    add(1, 3, counts["num_fail"], "rgba(231,76,60,0.4)")   # 수치 FAIL
    add(2, 4, counts["num_pass"], "rgba(52,152,219,0.3)")  # 수치 PASS→해석
    add(4, 5, counts["interp_pass"], "rgba(46,204,113,0.4)")  # 해석 PASS
    add(4, 6, counts["interp_fail"], "rgba(231,76,60,0.4)")   # 해석 FAIL
    add(5, 7, counts["final_pass"], "rgba(46,204,113,0.6)")   # 최종 통과

    fig = go.Figure(go.Sankey(
        name="검증 흐름",
        node=dict(
            pad=15, thickness=18,
            line=dict(color="#888", width=0.5),
            label=labels,
            color=["#3498db", "#7f8c8d", "#2ecc71", "#e74c3c",
                   "#7f8c8d", "#2ecc71", "#e74c3c", "#3498db"],
            hovertemplate="%{label}<br>흐름량: %{value:.0f} 턴<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values, color=colors,
            hovertemplate="%{source.label} → %{target.label}<br>%{value:.0f} 턴<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(
            text="검증 흐름 (Generate → 수치/해석 검증 → Respond)",
            font=dict(size=13, color="#ddd"),
            x=0.0, xanchor="left",
        ),
        height=400, margin=dict(t=50, b=20, l=20, r=20),
        font=dict(size=12, color="#ddd"),
    )
    return fig


# ═══════════════════════════════════════
# Tab 3 신규 차트 (구성 검토 — P1/P2/P3 산출물)
# ═══════════════════════════════════════

def token_composition_normalized(turns: list[dict]) -> go.Figure:
    """100% stacked bar — 토큰 비율 이동(% mix) 시각화.

    Charter §3.1 "비율이 어떻게 이동하는가" 정면 충족용 차트.
    절대 토큰량은 hover에서 확인.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        100% stacked bar Figure.
    """
    turn_numbers = [t["turn_number"] for t in turns]
    # v3: turn_conclusions_tokens는 previous_turns_tokens에 합산되어 폐기.
    sources = {
        "Gathered Data":    ("context.source.gathered_data_tokens",  "gathered"),
        "Query Analysis":   ("context.source.query_analysis_tokens", "query_analysis"),
        "Previous Turns":   ("context.source.previous_turns_tokens", "previous_turns"),
        "System Prompt":    ("context.source.system_prompt_tokens",  "system_prompt"),
    }

    abs_values: dict[str, list[float]] = {}
    for label, (attr_key, _) in sources.items():
        abs_values[label] = [t["metadata"].get(ATTRS[attr_key], 0) or 0 for t in turns]

    totals = [sum(abs_values[label][i] for label in sources) for i in range(len(turns))]

    fig = go.Figure()
    for label, (_, color_key) in sources.items():
        pct = [
            (abs_values[label][i] / totals[i] * 100) if totals[i] > 0 else 0
            for i in range(len(turns))
        ]
        fig.add_trace(go.Bar(
            x=turn_numbers,
            y=pct,
            name=label,
            marker_color=COLORS.get(color_key, "#95a5a6"),
            customdata=[[abs_values[label][i], totals[i]] for i in range(len(turns))],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Turn %{x}<br>"
                "비율 %{y:.1f}%<br>"
                "토큰 %{customdata[0]:,} / 총 %{customdata[1]:,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="턴별 토큰 구성 (비율)",
        xaxis_title="Turn",
        yaxis_title="비율 (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        barmode="stack",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=60, b=40),
    )
    return fig


def previous_turns_delta_sparkline(turns: list[dict], warn: float, danger: float) -> go.Figure:
    """Previous Turns 토큰 점유율의 턴별 추세 — 임계값 밴드 포함.

    Args:
        turns: 턴별 데이터 리스트.
        warn:   warning 격상 임계값 (예: 0.25)
        danger: danger 격상 임계값 (예: 0.50)

    Returns:
        라인 차트 Figure.
    """
    turn_numbers = [t["turn_number"] for t in turns]
    prev_ratios: list[float | None] = []
    for t in turns:
        meta = t.get("metadata", {})
        total = meta.get(ATTRS["context.total_tokens"]) or 0
        prev = meta.get(ATTRS["context.source.previous_turns_tokens"]) or 0
        prev_ratios.append((prev / total) if total > 0 else None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turn_numbers,
        y=[(r * 100 if r is not None else None) for r in prev_ratios],
        mode="lines+markers",
        line=dict(color=COLORS["previous_turns"], width=2),
        marker=dict(size=8),
        name="Prev Turns %",
        hovertemplate="Turn %{x}<br>Prev Turns %{y:.1f}%<extra></extra>",
    ))
    fig.add_hrect(y0=warn * 100, y1=danger * 100,
                  fillcolor=STATUS_COLORS["warn"], opacity=0.10, line_width=0,
                  annotation_text="warn", annotation_position="top left",
                  annotation_font_size=10)
    fig.add_hrect(y0=danger * 100, y1=100,
                  fillcolor=STATUS_COLORS["bad"], opacity=0.10, line_width=0,
                  annotation_text="danger", annotation_position="top left",
                  annotation_font_size=10)

    fig.update_layout(
        title="Previous Turns 점유율 추세 (Rot 1차 신호)",
        xaxis_title="Turn",
        yaxis_title="비율 (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        xaxis=dict(dtick=1),
        height=240,
        margin=dict(t=40, b=30),
        showlegend=False,
    )
    return fig


def plan_vs_actual_dot_matrix(turns: list[dict], tool_to_source: dict[str, str]) -> go.Figure:
    """Plan vs 실행 — dot matrix (소스 × 턴, planned/actual 상태 인코딩).

    상태 코드:
        ● (filled green)   : matched — planned & used
        ○ (open red)       : missed — planned & not used
        ◆ (open blue)      : extra — used & not planned

    Args:
        turns: 턴별 데이터 리스트.
        tool_to_source: 도구→소스 매핑 dict.

    Returns:
        Plotly scatter Figure (matrix 형태).
    """
    sources_order = ["csv", "rag", "web", "api", "memory"]
    turn_labels = [f"T{t.get('turn_number', '?')}" for t in turns]

    xs, ys, symbols, colors, hovers = [], [], [], [], []
    for ti, turn in enumerate(turns):
        meta = turn.get("metadata", {})
        plan = meta.get(ATTRS["query.tool_plan"], []) or []
        actual = meta.get(ATTRS.get("gather.tools_called", "gather.tools_called"), []) or []
        if not isinstance(plan, list):
            plan = []
        if not isinstance(actual, list):
            actual = []
        planned_sources = {tool_to_source.get(t, "—") for t in plan}
        actual_sources = {tool_to_source.get(t, "—") for t in actual}

        for src in sources_order:
            p, a = src in planned_sources, src in actual_sources
            if not p and not a:
                continue
            if p and a:
                sym, col, lab = "circle", STATUS_COLORS["good"], "matched"
            elif p and not a:
                sym, col, lab = "circle-open", STATUS_COLORS["bad"], "missed"
            else:
                sym, col, lab = "diamond-open", STATUS_COLORS["primary"], "extra"
            xs.append(turn_labels[ti])
            ys.append(src)
            symbols.append(sym)
            colors.append(col)
            hovers.append(f"{turn_labels[ti]} · {src}<br>{lab}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(symbol=symbols, color=colors, size=18,
                    line=dict(color=colors, width=2)),
        text=hovers, hoverinfo="text",
        showlegend=False,
    ))
    # 범례용 가짜 trace
    for lab, sym, col in [
        ("matched (planned & used)", "circle", STATUS_COLORS["good"]),
        ("missed (planned, not used)", "circle-open", STATUS_COLORS["bad"]),
        ("extra (used, not planned)", "diamond-open", STATUS_COLORS["primary"]),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=sym, color=col, size=12,
                        line=dict(color=col, width=2)),
            name=lab, showlegend=True,
        ))

    fig.update_layout(
        title="Plan vs 실행 (소스 단위)",
        xaxis=dict(title="Turn", categoryorder="array", categoryarray=turn_labels),
        yaxis=dict(title="소스", categoryorder="array",
                   categoryarray=list(reversed(sources_order))),
        height=max(220, 50 * len(sources_order) + 80),
        margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def source_contribution_scatter(turns: list[dict]) -> go.Figure:
    """소스 기여도 × 탈락률 2D scatter — "위험 소스" 식별.

    각 점 = (turn, source). X=기여 비율, Y=탈락률, 크기=시도 항목 수(proxy).
    오른쪽 위 사분면이 "위험" (비중 크고 탈락도 큰 소스).

    데이터 없으면 빈 차트 + 안내.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        scatter Figure.
    """
    points = []
    for turn in turns:
        meta = turn.get("metadata", {})
        contrib = meta.get(ATTRS.get("source.contribution", "source.contribution"))
        if not isinstance(contrib, dict):
            continue
        excl_reasons = meta.get(ATTRS.get("gather.exclusion_reasons",
                                          "gather.exclusion_reasons"), []) or []
        excl_per_source: dict[str, int] = {}
        for r in excl_reasons:
            if isinstance(r, dict):
                src = r.get("source")
                if src:
                    excl_per_source[src] = excl_per_source.get(src, 0) + 1

        items_total = meta.get(ATTRS.get("gather.items_collected",
                                         "gather.items_collected"), 0) or 0
        excl_total = sum(excl_per_source.values())
        items_attempted_total = items_total + excl_total

        for src, ratio in contrib.items():
            if not isinstance(ratio, (int, float)):
                continue
            src_excl = excl_per_source.get(src, 0)
            attempted_src = max(1, int(items_attempted_total * ratio))
            excl_rate = src_excl / attempted_src if attempted_src > 0 else 0
            points.append({
                "turn": turn.get("turn_number", "?"),
                "source": src,
                "contrib": ratio,
                "excl_rate": min(excl_rate, 1.0),
                "attempted": attempted_src,
            })

    if not points:
        return _empty_figure(
            "source.contribution 미수집 — Phase 3.8 데이터 수집 후 활성화"
        )

    src_color = {
        "csv": COLORS.get("efficiency", "#2980b9"),
        "rag": COLORS.get("relevance", "#d35400"),
        "web": COLORS.get("consistency", "#8e44ad"),
        "api": COLORS.get("completeness", "#27ae60"),
        "memory": STATUS_COLORS["muted"],
    }
    fig = go.Figure()
    for src in {p["source"] for p in points}:
        sps = [p for p in points if p["source"] == src]
        fig.add_trace(go.Scatter(
            x=[p["contrib"] * 100 for p in sps],
            y=[p["excl_rate"] * 100 for p in sps],
            mode="markers+text",
            marker=dict(
                color=src_color.get(src, "#95a5a6"),
                size=[max(8, min(28, p["attempted"] * 2)) for p in sps],
                line=dict(color="white", width=1),
                opacity=0.75,
            ),
            text=[f"T{p['turn']}" for p in sps],
            textposition="top center",
            textfont=dict(size=9),
            name=src,
            hovertemplate=(
                "<b>%{fullData.name}</b> · %{text}<br>"
                "기여 %{x:.1f}%<br>"
                "탈락 %{y:.1f}%<extra></extra>"
            ),
        ))
    fig.add_vline(x=50, line_dash="dot", line_color="#ccc")
    fig.add_hline(y=50, line_dash="dot", line_color="#ccc")
    # 4사분면 라벨 — 각 코너에 작게 배치 (캡션 대신 차트가 self-document)
    _quadrant_labels = [
        # (x, y, xanchor, yanchor, text, color)
        (95, 95, "right", "top",    "⚠ 위험 (큰 비중·높은 탈락)",  STATUS_COLORS["bad"]),
        (5,  95, "left",  "top",    "스팸 (작은 비중·높은 탈락)",   STATUS_COLORS["warn"]),
        (95, 5,  "right", "bottom", "일꾼 (큰 비중·낮은 탈락)",     STATUS_COLORS["good"]),
        (5,  5,  "left",  "bottom", "경량 (작은 비중·낮은 탈락)",   STATUS_COLORS["muted"]),
    ]
    for qx, qy, xa, ya, txt, col in _quadrant_labels:
        fig.add_annotation(
            x=qx, y=qy, text=txt, showarrow=False,
            font=dict(size=10, color=col),
            xanchor=xa, yanchor=ya,
        )

    fig.update_layout(
        title="소스 기여도 × 탈락률 (Risk Map)",
        xaxis=dict(title="기여 비율 (%)", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title="탈락률 (%)", range=[0, 100], ticksuffix="%"),
        height=380,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═══════════════════════════════════════
# Tab 5 신규 차트 (전달 — Impact)
# ═══════════════════════════════════════


def density_noise_combined(turns: list[dict]) -> go.Figure:
    """정보 밀도 ↔ 노이즈 ↔ Rot Risk 통합 multi-line + 임계 밴드 + 패턴 annotation.

    Tab 5 ③ Rot 위험 — 기존 density_trend + noise_evolution 2개 차트를 통합.
    한 화면에서 세 라인의 상호 관계 (밀도 ↑ + 노이즈 ↓ = 정상, 역 = Rot)를 비교.

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        Plotly Figure (multi-line + threshold bands + 4 패턴 annotation).
    """
    try:
        from agent.monitoring_schema import (
            ATTRS as _ATTRS, DASHBOARD_THRESHOLDS as _DT,
            ROT_GATE_THRESHOLD as _ROT,
        )
    except ImportError:
        _ATTRS = {"context.information_density": "context.information_density",
                  "context.noise_ratio": "context.noise_ratio",
                  "context.rot_risk": "context.rot_risk"}
        _DT = {"noise_warn": 0.5, "noise_good": 0.3,
               "density_good": 0.5, "density_warn": 0.2}
        _ROT = 0.3

    turn_nums, density_vals, noise_vals, rot_vals = [], [], [], []
    for turn in turns:
        meta = turn.get("metadata", {})
        tn = meta.get(_ATTRS.get("turn.number", "turn.number"),
                      turn.get("turn_number", len(turn_nums) + 1))
        turn_nums.append(tn)
        density_vals.append(meta.get(_ATTRS["context.information_density"]))
        noise_vals.append(meta.get(_ATTRS["context.noise_ratio"]))
        rot_vals.append(meta.get(_ATTRS["context.rot_risk"]))

    if not turn_nums or all(v is None for v in density_vals + noise_vals + rot_vals):
        return _empty_figure("밀도·노이즈·Rot 데이터 없음")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turn_nums, y=density_vals, name="Information Density (높을수록 좋음)",
        mode="lines+markers", line=dict(color="#2ecc71", width=2.5),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="T%{x} · 밀도 %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=turn_nums, y=noise_vals, name="Noise Ratio (낮을수록 좋음)",
        mode="lines+markers", line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=8, symbol="square"),
        hovertemplate="T%{x} · 노이즈 %{y:.2f}<extra></extra>",
    ))
    if any(v is not None for v in rot_vals):
        fig.add_trace(go.Scatter(
            x=turn_nums, y=rot_vals, name="Rot Risk (낮을수록 좋음)",
            mode="lines+markers", line=dict(color="#f39c12", width=2, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="T%{x} · Rot %{y:.2f}<extra></extra>",
        ))

    noise_warn = _DT.get("noise_warn", 0.5)
    fig.add_hline(y=noise_warn, line_dash="dot", line_color="#e74c3c", opacity=0.35,
                  annotation_text=f"noise WARN ≥ {noise_warn:.0%}",
                  annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#e74c3c")
    fig.add_hline(y=_ROT, line_dash="dot", line_color="#f39c12", opacity=0.35,
                  annotation_text=f"Rot Gate ≥ {_ROT:.0%}",
                  annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="#f39c12")

    fig.add_annotation(
        text=("<b>4 패턴 해석</b><br>"
              "● 밀도↑ + 노이즈↓ = 정상<br>"
              "● 밀도↓ + 노이즈↑ = Rot 신호<br>"
              "● 밀도↑ + 노이즈↑ = 컨텍스트 포화<br>"
              "● 밀도↓ + 노이즈↓ = 수집 부족"),
        xref="paper", yref="paper", x=0.01, y=0.98,
        xanchor="left", yanchor="top", showarrow=False,
        bgcolor="rgba(30,30,30,0.85)", bordercolor="#888", borderwidth=1,
        font=dict(size=10, color="#ddd"), align="left",
    )

    fig.update_layout(
        title="",
        xaxis_title="턴", yaxis_title="비율 (0~1)",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(dtick=1),
        height=380, margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    return fig


def causal_source_flow(turns: list[dict]) -> go.Figure:
    """이전 턴 → 현재 턴 인과 기여도 Sankey.

    Tab 5 ⑤ Causal Source Coverage — context.causal_sources의 (source_turn, impact)를
    Sankey로 시각화. Phoenix retrieval lineage 패턴 정합 (OTel `gen_ai.retrieval.*`).

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        Plotly Sankey Figure (또는 데이터 부재 시 _empty_figure).
    """
    links = []
    for turn in turns:
        cur_tn = turn.get("turn_number", "?")
        sources = turn.get("metadata", {}).get("context.causal_sources") or []
        if not isinstance(sources, list):
            continue
        for s in sources:
            if not isinstance(s, dict):
                continue
            src_tn = s.get("turn")
            if src_tn is None:
                continue
            retained = s.get("claims_retained", 0) or 0
            impact = s.get("impact")
            value = max(1, int(retained) if isinstance(retained, (int, float)) else 1)
            links.append((f"T{src_tn} (source)", f"T{cur_tn} (current)", value, impact))

    if not links:
        return _empty_figure(
            "causal_sources 데이터 없음 — Phase 4 OTel `gen_ai.retrieval.documents.*` 매핑 후 활성화 예정"
        )

    labels: list[str] = []
    label_to_idx: dict[str, int] = {}
    for src, tgt, _, _ in links:
        for lab in (src, tgt):
            if lab not in label_to_idx:
                label_to_idx[lab] = len(labels)
                labels.append(lab)

    src_idx = [label_to_idx[s] for s, _, _, _ in links]
    tgt_idx = [label_to_idx[t] for _, t, _, _ in links]
    values = [v for _, _, v, _ in links]

    def _color_for_impact(impact):
        if not isinstance(impact, (int, float)):
            return "rgba(150,150,150,0.4)"
        if impact >= 0.7:
            return "rgba(46,204,113,0.55)"
        if impact >= 0.4:
            return "rgba(243,156,18,0.55)"
        return "rgba(231,76,60,0.55)"

    link_colors = [_color_for_impact(i) for _, _, _, i in links]
    hover = [
        f"{s} → {t}<br>보존 주장: {v}" + (f"<br>impact: {i:.2f}" if isinstance(i, (int, float)) else "")
        for s, t, v, i in links
    ]
    node_colors = ["#3498db" if "current" in lab else "#7f8c8d" for lab in labels]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=14, thickness=18, label=labels, color=node_colors,
            line=dict(color="#1e1e1e", width=0.5),
        ),
        link=dict(source=src_idx, target=tgt_idx, value=values,
                  color=link_colors, customdata=hover,
                  hovertemplate="%{customdata}<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(text="이전 턴 → 현재 턴 인과 흐름",
                   font=dict(size=13, color="#ddd")),
        height=320, margin=dict(t=40, b=20, l=10, r=10),
        font=dict(size=11, color="#ddd"),
    )
    return fig


# ═══════════════════════════════════════
# Tab 4/5/6: 이탈 감지 시각화 (Phase 3.7)
# ═══════════════════════════════════════
#
# 3개 속성 분산 흡수 (Tab 8 폐기, 2026-05-31):
#   - query.session_continuity   → Tab 4 변형 ⑥ Intent Continuity
#   - analysis.query_alignment   → Tab 4 변형 ⑦ Query Alignment (분석)
#   - response.query_alignment   → Tab 5 전달 ⑧ Query Alignment (응답)
#   - Pattern I/II/III + 자연어 요약 → Tab 6 측정&진단 §4

# 패턴 표시 정보 (Pattern I/II/III)
DRIFT_PATTERN_INFO = {
    "I":   {"label": "Pattern I",   "icon": "✅", "name": "User Pivot",  "color": "#2ecc71"},
    "II":  {"label": "Pattern II",  "icon": "⚠️", "name": "Agent Drift", "color": "#f39c12"},
    "III": {"label": "Pattern III", "icon": "🚨", "name": "이중 실패",   "color": "#e74c3c"},
    "-":   {"label": "정상",        "icon": "✓",  "name": "정상",        "color": "#95a5a6"},
}


def _drift_get_float(meta: dict, attr_key: str) -> float | None:
    """metadata에서 float 값을 안전하게 꺼낸다 (ATTRS 키 변환 포함)."""
    raw = meta.get(ATTRS.get(attr_key, attr_key))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def classify_drift_pattern(continuity: float | None, alignment: float | None) -> str:
    """턴의 이탈 패턴을 분류한다.

    Args:
        continuity: query.session_continuity 값 (None 허용 — Turn 1).
        alignment: analysis/response.query_alignment 대표값 (보수적: 최솟값).

    Returns:
        패턴 키 "I" / "II" / "III" / "-".
    """
    if continuity is None:
        if alignment is not None and alignment < DRIFT_ALIGNMENT_THRESHOLD:
            return "II"
        return "-"
    low_continuity = continuity < DRIFT_CONTINUITY_THRESHOLD
    low_alignment = (
        alignment < DRIFT_ALIGNMENT_THRESHOLD if alignment is not None else False
    )
    if low_continuity and not low_alignment:
        return "I"
    if not low_continuity and low_alignment:
        return "II"
    if low_continuity and low_alignment:
        return "III"
    return "-"


def build_drift_stats(turns: list[dict]) -> list[dict]:
    """각 턴의 이탈 관련 지표를 추출한다.

    Returns:
        턴별 dict 리스트: {turn_number, continuity, analysis_alignment,
        response_alignment, min_alignment, pattern}.
    """
    stats: list[dict] = []
    for turn in turns:
        meta = turn.get("metadata", {})
        tn = turn.get("turn_number", 0)
        continuity = _drift_get_float(meta, "query.session_continuity")
        a_align = _drift_get_float(meta, "analysis.query_alignment")
        r_align = _drift_get_float(meta, "response.query_alignment")
        aligns = [v for v in (a_align, r_align) if v is not None]
        min_align = min(aligns) if aligns else None
        stats.append({
            "turn_number": tn,
            "continuity": continuity,
            "analysis_alignment": a_align,
            "response_alignment": r_align,
            "min_alignment": min_align,
            "pattern": classify_drift_pattern(continuity, min_align),
        })
    return sorted(stats, key=lambda s: s["turn_number"])


def continuity_trend(turns: list[dict]) -> go.Figure:
    """query.session_continuity 턴별 추이 차트.

    - Turn 1 (null) 은 차트에서 제외.
    - continuity < 임계값(0.5) 인 턴에 Pivot 마커 (빨간 점) 표시.
    - 임계값 수평선 표시.

    Args:
        turns: enriched 턴별 데이터.

    Returns:
        Plotly Figure.
    """
    stats = build_drift_stats(turns)
    valid = [(s["turn_number"], s["continuity"]) for s in stats
             if s["continuity"] is not None]

    if not valid:
        return _empty_figure(
            "query.session_continuity 데이터 없음 (Turn 1은 항상 null)"
        )

    x_vals = [v[0] for v in valid]
    y_vals = [v[1] for v in valid]
    pivot_x = [x for x, y in valid if y < DRIFT_CONTINUITY_THRESHOLD]
    pivot_y = [y for x, y in valid if y < DRIFT_CONTINUITY_THRESHOLD]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines+markers",
        name="session_continuity",
        line=dict(color="#3498db", width=2),
        marker=dict(size=7, color="#3498db"),
        connectgaps=False,
        hovertemplate="Turn %{x}<br>continuity: %{y:.3f}<extra></extra>",
    ))
    if pivot_x:
        fig.add_trace(go.Scatter(
            x=pivot_x, y=pivot_y, mode="markers",
            name=f"Pivot (< {DRIFT_CONTINUITY_THRESHOLD})",
            marker=dict(size=12, color=STATUS_COLORS["bad"], symbol="circle",
                        line=dict(width=2, color="white")),
            hovertemplate="Turn %{x}<br>⚠ Pivot — continuity: %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(
        y=DRIFT_CONTINUITY_THRESHOLD,
        line_dash="dot", line_color="#e74c3c", opacity=0.5,
        annotation_text=f"Pivot 임계값 ({DRIFT_CONTINUITY_THRESHOLD})",
        annotation_position="bottom right",
        annotation_font_size=10, annotation_opacity=0.7,
    )
    fig.update_layout(
        title="session_continuity 추이 (Turn 1 제외)",
        xaxis_title="Turn", yaxis_title="Continuity Score",
        yaxis_range=[0, 1.05], xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=320, margin=dict(t=60, b=40),
    )
    return fig


def query_alignment_trend(turns: list[dict], kind: str = "both") -> go.Figure:
    """query_alignment 턴별 추이 차트.

    Args:
        turns: enriched 턴별 데이터.
        kind: "analysis" / "response" / "both".

    Returns:
        Plotly Figure. 둘 다 데이터 없으면 빈 figure.
    """
    stats = build_drift_stats(turns)
    x_all = [s["turn_number"] for s in stats]
    y_a = [s["analysis_alignment"] for s in stats]
    y_r = [s["response_alignment"] for s in stats]

    show_a = kind in ("analysis", "both")
    show_r = kind in ("response", "both")

    has_data = (
        (show_a and any(v is not None for v in y_a))
        or (show_r and any(v is not None for v in y_r))
    )
    if not has_data:
        return _empty_figure(
            "query_alignment 데이터 없음 (run_evaluation.py 실행 후 생성)"
        )

    fig = go.Figure()
    if show_a:
        fig.add_trace(go.Scatter(
            x=x_all, y=y_a, mode="lines+markers",
            name="analysis.query_alignment",
            line=dict(color="#3498db", width=2),
            marker=dict(size=7), connectgaps=False,
            hovertemplate="Turn %{x}<br>analysis: %{y:.3f}<extra></extra>",
        ))
        drift_x = [s["turn_number"] for s in stats
                   if s["analysis_alignment"] is not None
                   and s["analysis_alignment"] < DRIFT_ALIGNMENT_THRESHOLD]
        drift_y = [s["analysis_alignment"] for s in stats
                   if s["analysis_alignment"] is not None
                   and s["analysis_alignment"] < DRIFT_ALIGNMENT_THRESHOLD]
        if drift_x:
            fig.add_trace(go.Scatter(
                x=drift_x, y=drift_y, mode="markers",
                name="Drift — analysis",
                marker=dict(size=11, color="#f39c12", symbol="x",
                            line=dict(width=2, color="#f39c12")),
                hovertemplate="Turn %{x}<br>⚠ Drift — analysis: %{y:.3f}<extra></extra>",
            ))
    if show_r:
        fig.add_trace(go.Scatter(
            x=x_all, y=y_r, mode="lines+markers",
            name="response.query_alignment",
            line=dict(color="#2ecc71", width=2),
            marker=dict(size=7), connectgaps=False,
            hovertemplate="Turn %{x}<br>response: %{y:.3f}<extra></extra>",
        ))
        drift_x = [s["turn_number"] for s in stats
                   if s["response_alignment"] is not None
                   and s["response_alignment"] < DRIFT_ALIGNMENT_THRESHOLD]
        drift_y = [s["response_alignment"] for s in stats
                   if s["response_alignment"] is not None
                   and s["response_alignment"] < DRIFT_ALIGNMENT_THRESHOLD]
        if drift_x:
            fig.add_trace(go.Scatter(
                x=drift_x, y=drift_y, mode="markers",
                name="Drift — response",
                marker=dict(size=11, color="#e67e22", symbol="x",
                            line=dict(width=2, color="#e67e22")),
                hovertemplate="Turn %{x}<br>⚠ Drift — response: %{y:.3f}<extra></extra>",
            ))

    fig.add_hline(
        y=DRIFT_ALIGNMENT_THRESHOLD,
        line_dash="dot", line_color="#e74c3c", opacity=0.5,
        annotation_text=f"Agent Drift 임계값 ({DRIFT_ALIGNMENT_THRESHOLD})",
        annotation_position="bottom right",
        annotation_font_size=10, annotation_opacity=0.7,
    )
    title_map = {
        "analysis": "analysis.query_alignment 추이",
        "response": "response.query_alignment 추이",
        "both":     "query_alignment 추이 (analysis / response)",
    }
    fig.update_layout(
        title=title_map.get(kind, "query_alignment 추이"),
        xaxis_title="Turn", yaxis_title="Alignment Score",
        yaxis_range=[0, 1.05], xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=320, margin=dict(t=60, b=40),
    )
    return fig


def drift_pattern_matrix_html(stats: list[dict]) -> str:
    """Pattern I/II/III 진단 매트릭스 HTML을 반환한다.

    각 턴마다 continuity/alignment 값과 패턴 판정을 표시한다.
    Streamlit에서 st.markdown(..., unsafe_allow_html=True)로 렌더한다.

    Args:
        stats: build_drift_stats() 반환값.

    Returns:
        테이블 + 범례 HTML 문자열.
    """
    _ROW_COLORS = {
        "I":   "#fff9e6", "II":  "#fff3e0",
        "III": "#fde8e8", "-":   "transparent",
    }
    rows_html = []
    for s in stats:
        tn = s["turn_number"]
        cont = s["continuity"]
        a_align = s["analysis_alignment"]
        r_align = s["response_alignment"]
        pattern = s["pattern"]
        pinfo = DRIFT_PATTERN_INFO[pattern]

        is_first = (s is stats[0])
        cont_str = (
            f"{cont:.3f}" if cont is not None
            else ("— ⓘ T1 기준없음" if is_first else "—")
        )
        a_str = f"{a_align:.3f}" if a_align is not None else "—"
        r_str = f"{r_align:.3f}" if r_align is not None else "—"

        cont_color = (
            STATUS_COLORS["bad"]
            if cont is not None and cont < DRIFT_CONTINUITY_THRESHOLD
            else "#333333"
        )
        a_color = (
            STATUS_COLORS["warn"]
            if a_align is not None and a_align < DRIFT_ALIGNMENT_THRESHOLD
            else "#333333"
        )
        r_color = (
            STATUS_COLORS["warn"]
            if r_align is not None and r_align < DRIFT_ALIGNMENT_THRESHOLD
            else "#333333"
        )
        bg = _ROW_COLORS.get(pattern, "transparent")
        label_html = (
            f'<span style="color:{pinfo["color"]};font-weight:600;">'
            f'{pinfo["icon"]} {pinfo["label"]} — {pinfo["name"]}</span>'
        ) if pattern != "-" else '<span style="color:#9aa0a6;">─ 정상</span>'

        rows_html.append(
            f'<tr style="background:{bg};">'
            f'<td style="text-align:center;font-weight:600;">Turn {tn}</td>'
            f'<td style="text-align:center;color:{cont_color};">{cont_str}</td>'
            f'<td style="text-align:center;color:{a_color};">{a_str}</td>'
            f'<td style="text-align:center;color:{r_color};">{r_str}</td>'
            f'<td style="text-align:left;padding-left:8px;">{label_html}</td>'
            f'</tr>'
        )

    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
        '<thead>'
        '<tr style="border-bottom:2px solid #e0e0e0;background:#f8f9fa;">'
        '<th style="text-align:center;padding:8px 4px;">턴</th>'
        '<th style="text-align:center;padding:8px 4px;">session_continuity<br>'
        f'<small style="color:#888;font-weight:400;">(임계값 {DRIFT_CONTINUITY_THRESHOLD})</small></th>'
        '<th style="text-align:center;padding:8px 4px;">analysis<br>alignment<br>'
        f'<small style="color:#888;font-weight:400;">(임계값 {DRIFT_ALIGNMENT_THRESHOLD})</small></th>'
        '<th style="text-align:center;padding:8px 4px;">response<br>alignment<br>'
        f'<small style="color:#888;font-weight:400;">(임계값 {DRIFT_ALIGNMENT_THRESHOLD})</small></th>'
        '<th style="text-align:left;padding:8px 8px;">패턴 판정</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )
    legend_html = (
        '<div style="margin-top:12px;font-size:12px;color:#555;">'
        f'<strong>패턴 정의</strong> (판정 기준: continuity &lt; {DRIFT_CONTINUITY_THRESHOLD} = LOW, '
        f'alignment &lt; {DRIFT_ALIGNMENT_THRESHOLD} = LOW)&nbsp; | &nbsp;'
        '<span style="color:#2ecc71;font-weight:600;">✅ Pattern I</span>: '
        'LOW continuity + HIGH alignment → User Pivot (정상)&nbsp; | &nbsp;'
        '<span style="color:#f39c12;font-weight:600;">⚠️ Pattern II</span>: '
        'HIGH continuity + LOW alignment → Agent Drift&nbsp; | &nbsp;'
        '<span style="color:#e74c3c;font-weight:600;">🚨 Pattern III</span>: '
        'LOW continuity + LOW alignment → 이중 실패'
        '</div>'
    )
    return table_html + legend_html
