"""
evaluation/generate_charts.py — Phase 1 보고용 시각화 차트 생성

역할:
    Langfuse에서 3개 검증 세션 데이터를 가져와 6개 차트를 PNG로 생성한다.
    Langfuse 접속 불가 시 보고서 실측 데이터(fallback)로 차트를 생성한다.
    Phase 1 Progress Report(§5)의 핵심 메시지를 시각적으로 뒷받침한다.

데이터 흐름:
    입력: Langfuse API (3개 세션 ID) 또는 보고서 실측 데이터 (fallback)
    출력: docs/analysis/charts/*.png (6개 차트)

의존:
    - evaluation.visualize_session.fetch_session_data() — Langfuse API 호출 + 파싱
    - agent.monitoring_schema — ATTRS, THRESHOLDS
    - matplotlib

사용 방법:
    python -m evaluation.generate_charts              # Langfuse 우선, fallback 자동
    python -m evaluation.generate_charts --offline     # fallback 데이터만 사용
"""
import argparse
import os
import sys

import truststore

truststore.inject_into_ssl()

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 동작

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from dotenv import load_dotenv

from agent.monitoring_schema import ATTRS, THRESHOLDS

load_dotenv()

# --- 세션 ID ---
SESSION_IDS = {
    "happy_path": "sess_60281712",      # Happy Path 7턴
    "insufficient": "sess_3175c1df",    # Insufficient Re-gather
    "numeric_verify": "sess_fa588f9f",  # Numeric Verify Fail
}

# --- 출력 경로 ---
CHART_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "analysis", "charts"
)

# --- 4D 차원 정의 ---
DIMENSIONS = ["completeness", "efficiency", "relevance", "consistency"]
DIMENSION_LABELS = {
    "completeness": "Completeness",
    "efficiency": "Efficiency",
    "relevance": "Relevance",
    "consistency": "Consistency",
}
SCORE_KEYS = {
    "completeness": "completeness_score",
    "efficiency": "efficiency_score",
    "relevance": "relevance_score",
    "consistency": "consistency_score",
}

# --- 공통 스타일 ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# 차트별 색상
COLORS = {
    "completeness": "#2ecc71",
    "efficiency": "#3498db",
    "relevance": "#e67e22",
    "consistency": "#9b59b6",
    "threshold": "#e74c3c",
    "gathered": "#2ecc71",
    "previous_turns": "#e74c3c",
    "conclusions": "#f39c12",
    "system_prompt": "#95a5a6",
    "query_analysis": "#3498db",
}

SCENARIO_COLORS = {
    "happy_path": "#3498db",
    "insufficient": "#e74c3c",
    "numeric_verify": "#f39c12",
}

SCENARIO_LABELS = {
    "happy_path": "Happy Path (7 turns)",
    "insufficient": "Insufficient",
    "numeric_verify": "Numeric Verify",
}


# ═══════════════════════════════════════
# FALLBACK 데이터 — Progress Report §5.2 실측값
# ═══════════════════════════════════════
# Langfuse 접속 불가 시 보고서에 기록된 실측 데이터로 차트를 생성한다.

def _build_fallback_turn(
    turn_number: int, scores: dict, metadata: dict,
) -> dict:
    """fallback 턴 데이터를 fetch_session_data() 반환 형식과 동일하게 구성한다."""
    return {
        "turn_number": turn_number,
        "trace_id": f"fallback_turn_{turn_number}",
        "scores": scores,
        "metadata": metadata,
        "events": [],
        "wall_time_ms": metadata.get(ATTRS["turn.wall_time_ms"]),
    }


def _build_happy_path_fallback() -> list[dict]:
    """Happy Path 7턴 실측 데이터 (Progress Report §5.2 표에서 추출)."""
    # --- 턴별 4D 점수 ---
    turn_scores = [
        {"completeness_score": 0.92, "efficiency_score": 0.95, "relevance_score": 0.85, "consistency_score": 0.70},
        {"completeness_score": 0.40, "efficiency_score": 0.92, "relevance_score": 0.85, "consistency_score": 0.40},
        {"completeness_score": 0.95, "efficiency_score": 0.85, "relevance_score": 0.75, "consistency_score": 0.70},
        {"completeness_score": 0.92, "efficiency_score": 0.20, "relevance_score": 0.45, "consistency_score": 0.75},
        {"completeness_score": 0.35, "efficiency_score": 0.65, "relevance_score": 0.65, "consistency_score": 0.70},
        {"completeness_score": 0.35, "efficiency_score": 0.65, "relevance_score": 0.72, "consistency_score": 0.70},
        {"completeness_score": 0.92, "efficiency_score": 0.25, "relevance_score": 0.45, "consistency_score": 0.75},
    ]

    # --- 턴별 컨텍스트 구성 (토큰) ---
    turn_tokens = [
        {"total": 21000, "gathered": 20734, "previous": 1, "conclusions": 1, "system": 200, "query": 64,
         "noise": 0.000, "continuity": 1.000, "wall_ms": 34600},
        {"total": 11900, "gathered": 10954, "previous": 709, "conclusions": 246, "system": 200, "query": 64,
         "noise": 0.060, "continuity": 1.000, "wall_ms": 38000},
        {"total": 6900, "gathered": 5342, "previous": 1341, "conclusions": 456, "system": 200, "query": 64,
         "noise": 0.194, "continuity": 1.000, "wall_ms": 25500},
        {"total": 3700, "gathered": 1397, "previous": 2094, "conclusions": 686, "system": 200, "query": 64,
         "noise": 0.565, "continuity": 1.000, "wall_ms": 27700},
        {"total": 9200, "gathered": 6028, "previous": 2941, "conclusions": 931, "system": 200, "query": 64,
         "noise": 0.319, "continuity": 1.000, "wall_ms": 51100},
        {"total": 13700, "gathered": 9653, "previous": 3852, "conclusions": 1202, "system": 200, "query": 64,
         "noise": 0.280, "continuity": 1.000, "wall_ms": 39000},
        {"total": 7500, "gathered": 2638, "previous": 4697, "conclusions": 1460, "system": 200, "query": 64,
         "noise": 0.622, "continuity": 0.833, "wall_ms": 77200},
    ]

    # --- 모순 감지 이벤트 (Pattern B) ---
    # Turn 4~7에서 모순 감지, 모두 해결됨
    contradiction_turns = {4, 5, 6, 7}

    turns = []
    for i in range(7):
        tn = i + 1
        tok = turn_tokens[i]

        metadata = {
            ATTRS["turn.number"]: tn,
            ATTRS["turn.wall_time_ms"]: tok["wall_ms"],
            ATTRS["context.total_tokens"]: tok["total"],
            ATTRS["context.window_utilization"]: tok["total"] / 180000,
            ATTRS["context.source.gathered_data_tokens"]: tok["gathered"],
            ATTRS["context.source.previous_turns_tokens"]: tok["previous"],
            ATTRS.get("context.source.turn_conclusions_tokens", "context.source.turn_conclusions_tokens"): tok["conclusions"],
            ATTRS["context.source.system_prompt_tokens"]: tok["system"],
            ATTRS["context.source.query_analysis_tokens"]: tok["query"],
            ATTRS["context.noise_ratio"]: tok["noise"],
            ATTRS["context.rot_risk"]: tok["noise"] * (tok["total"] / 180000),
            ATTRS.get("context.continuity_score", "context.continuity_score"): tok["continuity"],
            ATTRS.get("analysis.contradicts_previous", "analysis.contradicts_previous"): tn in contradiction_turns,
            ATTRS.get("analysis.contradiction_resolved", "analysis.contradiction_resolved"): tn in contradiction_turns,
            ATTRS.get("analysis.previous_conclusion", "analysis.previous_conclusion"): "Previous turn conclusion" if tn in contradiction_turns else "",
        }

        turns.append(_build_fallback_turn(tn, turn_scores[i], metadata))

    return turns


def _build_insufficient_fallback() -> list[dict]:
    """Insufficient Re-gather 시나리오 최종 턴 데이터."""
    scores = {"completeness_score": 0.20, "efficiency_score": 0.95,
              "relevance_score": 0.90, "consistency_score": 0.40}
    metadata = {
        ATTRS["turn.number"]: 1,
        ATTRS["context.total_tokens"]: 2873,
        ATTRS["context.window_utilization"]: 2873 / 180000,
        ATTRS["context.noise_ratio"]: 0.05,
        ATTRS["context.rot_risk"]: 0.0008,
        ATTRS["context.source.gathered_data_tokens"]: 2600,
        ATTRS["context.source.previous_turns_tokens"]: 0,
        ATTRS.get("context.source.turn_conclusions_tokens", "context.source.turn_conclusions_tokens"): 0,
        ATTRS["context.source.system_prompt_tokens"]: 200,
        ATTRS["context.source.query_analysis_tokens"]: 73,
        ATTRS.get("analysis.contradicts_previous", "analysis.contradicts_previous"): False,
        ATTRS.get("analysis.contradiction_resolved", "analysis.contradiction_resolved"): False,
        ATTRS.get("analysis.previous_conclusion", "analysis.previous_conclusion"): "",
    }
    return [_build_fallback_turn(1, scores, metadata)]


def _build_numeric_verify_fallback() -> list[dict]:
    """Numeric Verify Fail 시나리오 최종 턴 데이터."""
    scores = {"completeness_score": 0.85, "efficiency_score": 0.95,
              "relevance_score": 0.90, "consistency_score": 0.70}
    metadata = {
        ATTRS["turn.number"]: 1,
        ATTRS["context.total_tokens"]: 5200,
        ATTRS["context.window_utilization"]: 5200 / 180000,
        ATTRS["context.noise_ratio"]: 0.08,
        ATTRS["context.rot_risk"]: 0.002,
        ATTRS["context.source.gathered_data_tokens"]: 4800,
        ATTRS["context.source.previous_turns_tokens"]: 0,
        ATTRS.get("context.source.turn_conclusions_tokens", "context.source.turn_conclusions_tokens"): 0,
        ATTRS["context.source.system_prompt_tokens"]: 200,
        ATTRS["context.source.query_analysis_tokens"]: 200,
        ATTRS.get("analysis.contradicts_previous", "analysis.contradicts_previous"): False,
        ATTRS.get("analysis.contradiction_resolved", "analysis.contradiction_resolved"): False,
        ATTRS.get("analysis.previous_conclusion", "analysis.previous_conclusion"): "",
    }
    return [_build_fallback_turn(1, scores, metadata)]


def get_fallback_sessions() -> dict[str, list[dict]]:
    """보고서 실측 데이터로 구성한 fallback 세션 데이터."""
    return {
        "happy_path": _build_happy_path_fallback(),
        "insufficient": _build_insufficient_fallback(),
        "numeric_verify": _build_numeric_verify_fallback(),
    }


def _get_score(turn: dict, dimension: str) -> float | None:
    """턴에서 특정 차원의 점수를 가져온다."""
    return turn["scores"].get(SCORE_KEYS[dimension])


def _get_meta(turn: dict, attr_name: str, default=None):
    """턴의 metadata에서 속성값을 가져온다."""
    return turn["metadata"].get(ATTRS[attr_name], default)


def _save_chart(fig: plt.Figure, filename: str) -> str:
    """차트를 PNG로 저장하고 경로를 반환한다."""
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장됨: {path}")
    return path


# ═══════════════════════════════════════
# CHART 1: 4D Score Radar
# ═══════════════════════════════════════

def chart_radar_4d(turns: list[dict], output_path: str) -> None:
    """턴별 4D 점수 레이더 차트 — 대표 턴 오버레이.

    Args:
        turns: Happy Path 세션의 턴 데이터.
        output_path: 출력 PNG 파일명.
    """
    # 대표 턴 선택: Turn 1 (최초), Turn 4 (효율성 최저), Turn 7 (요약)
    representative_turns = [1, 4, 7]
    turn_colors = ["#2ecc71", "#e74c3c", "#3498db"]
    turn_styles = ["-", "--", "-."]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 다각형

    # 임계값 다각형
    threshold_values = [THRESHOLDS[d] for d in DIMENSIONS] + [THRESHOLDS[DIMENSIONS[0]]]
    ax.plot(angles, threshold_values, ":", color=COLORS["threshold"],
            linewidth=2, label="Threshold", alpha=0.7)
    ax.fill(angles, threshold_values, color=COLORS["threshold"], alpha=0.05)

    for i, turn_num in enumerate(representative_turns):
        turn = next((t for t in turns if t["turn_number"] == turn_num), None)
        if not turn:
            continue

        values = []
        for dim in DIMENSIONS:
            score = _get_score(turn, dim)
            values.append(score if score is not None else 0.0)
        values += values[:1]  # 닫힌 다각형

        ax.plot(angles, values, turn_styles[i], color=turn_colors[i],
                linewidth=2.5, label=f"Turn {turn_num}", marker="o", markersize=6)
        ax.fill(angles, values, color=turn_colors[i], alpha=0.08)

    # 축 라벨
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in DIMENSIONS], fontsize=11)

    # 반지름 눈금
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.6)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("4D Quality Score Radar\n(Happy Path — Turn 1, 4, 7)", fontsize=14, pad=20)

    _save_chart(fig, output_path)


# ═══════════════════════════════════════
# CHART 2: Token Composition Stacked Bar
# ═══════════════════════════════════════

def chart_token_composition(turns: list[dict], output_path: str) -> None:
    """토큰 구성 Stacked Bar — 소스별 토큰 비율 변화.

    턴이 진행될수록 previous_turns 비중이 증가하는 패턴을 시각화.

    Args:
        turns: Happy Path 세션의 턴 데이터.
        output_path: 출력 PNG 파일명.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    turn_numbers = [t["turn_number"] for t in turns]
    x = np.arange(len(turn_numbers))

    # 소스별 토큰 수 추출
    gathered = []
    previous_turns = []
    conclusions = []
    system_prompt = []
    query_analysis = []

    for t in turns:
        meta = t["metadata"]
        total = meta.get(ATTRS["context.total_tokens"], 1)
        g = meta.get(ATTRS["context.source.gathered_data_tokens"], 0)
        p = meta.get(ATTRS["context.source.previous_turns_tokens"], 0)
        c = meta.get(ATTRS.get("context.source.turn_conclusions_tokens", "context.source.turn_conclusions_tokens"), 0)
        s = meta.get(ATTRS["context.source.system_prompt_tokens"], 0)
        q = meta.get(ATTRS["context.source.query_analysis_tokens"], 0)

        # 비율로 변환
        gathered.append(g / total * 100 if total else 0)
        previous_turns.append(p / total * 100 if total else 0)
        conclusions.append(c / total * 100 if total else 0)
        system_prompt.append(s / total * 100 if total else 0)
        query_analysis.append(q / total * 100 if total else 0)

    bar_width = 0.6

    # 아래부터 쌓기
    ax.bar(x, gathered, bar_width, label="Gathered Data", color=COLORS["gathered"], alpha=0.85)
    ax.bar(x, previous_turns, bar_width, bottom=gathered,
           label="Previous Turns", color=COLORS["previous_turns"], alpha=0.85)

    bottom2 = [g + p for g, p in zip(gathered, previous_turns)]
    ax.bar(x, conclusions, bar_width, bottom=bottom2,
           label="Turn Conclusions", color=COLORS["conclusions"], alpha=0.85)

    bottom3 = [b + c for b, c in zip(bottom2, conclusions)]
    ax.bar(x, query_analysis, bar_width, bottom=bottom3,
           label="Query Analysis", color=COLORS["query_analysis"], alpha=0.85)

    bottom4 = [b + q for b, q in zip(bottom3, query_analysis)]
    ax.bar(x, system_prompt, bar_width, bottom=bottom4,
           label="System Prompt", color=COLORS["system_prompt"], alpha=0.85)

    # 총 토큰 수 텍스트 표시
    for i, t in enumerate(turns):
        total = t["metadata"].get(ATTRS["context.total_tokens"], 0)
        if total >= 1000:
            label = f"{total / 1000:.1f}K"
        else:
            label = str(total)
        ax.text(i, 102, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Token Composition (%)", fontsize=12)
    ax.set_title("Context Token Composition by Turn\n(Happy Path 7-Turn Session)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{n}" for n in turn_numbers])
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    # 주요 인사이트 어노테이션
    # Turn 4: previous_turns 50%
    if len(turns) >= 4:
        t4_prev = previous_turns[3]
        t4_gathered = gathered[3]
        ax.annotate(
            "previous_turns\n50%!",
            xy=(3, t4_gathered + t4_prev / 2),
            xytext=(4.5, 85),
            fontsize=9, color=COLORS["previous_turns"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["previous_turns"], lw=1.5),
        )

    fig.tight_layout()
    _save_chart(fig, output_path)


# ═══════════════════════════════════════
# CHART 3: 4D Score Trend (Line Chart)
# ═══════════════════════════════════════

def chart_score_trend(turns: list[dict], output_path: str) -> None:
    """4D 점수 추이 Line Chart — threshold 수평선 포함.

    Args:
        turns: Happy Path 세션의 턴 데이터.
        output_path: 출력 PNG 파일명.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    turn_numbers = [t["turn_number"] for t in turns]

    for dim in DIMENSIONS:
        scores = [_get_score(t, dim) for t in turns]
        color = COLORS[dim]
        label = dim.capitalize()

        # None을 NaN으로 (라인이 끊기지 않게)
        plot_scores = [s if s is not None else float("nan") for s in scores]

        ax.plot(turn_numbers, plot_scores, "-o", color=color,
                linewidth=2.5, markersize=8, label=label, alpha=0.9)

        # threshold 수평선
        threshold = THRESHOLDS[dim]
        ax.axhline(y=threshold, color=color, linestyle=":", linewidth=1, alpha=0.4)

    # PASS/FAIL 영역 배경
    ax.axhspan(0, min(THRESHOLDS.values()), color="#e74c3c", alpha=0.03)
    ax.axhspan(min(THRESHOLDS.values()), 1.0, color="#2ecc71", alpha=0.03)

    # PASS/FAIL 마크
    for t in turns:
        tn = t["turn_number"]
        all_pass = all(
            (_get_score(t, d) or 0) >= THRESHOLDS[d]
            for d in DIMENSIONS
        )
        marker = "PASS" if all_pass else "FAIL"
        color = "#2ecc71" if all_pass else "#e74c3c"
        ax.text(tn, -0.08, marker, ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("4D Quality Score Trends\n(Happy Path 7-Turn Session)", fontsize=14)
    ax.set_xticks(turn_numbers)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_chart(fig, output_path)


# ═══════════════════════════════════════
# CHART 4: Noise Ratio Evolution
# ═══════════════════════════════════════

def chart_noise_evolution(turns: list[dict], output_path: str) -> None:
    """Noise ratio 추이 + rot_risk 영역 차트.

    Args:
        turns: Happy Path 세션의 턴 데이터.
        output_path: 출력 PNG 파일명.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    turn_numbers = [t["turn_number"] for t in turns]

    # noise_ratio
    noise_ratios = [
        _get_meta(t, "context.noise_ratio", 0.0) for t in turns
    ]
    ax.plot(turn_numbers, noise_ratios, "-o", color="#e74c3c",
            linewidth=2.5, markersize=8, label="Noise Ratio", zorder=3)
    ax.fill_between(turn_numbers, noise_ratios, alpha=0.15, color="#e74c3c")

    # rot_risk
    rot_risks = [
        _get_meta(t, "context.rot_risk", 0.0) for t in turns
    ]
    ax.plot(turn_numbers, rot_risks, "--s", color="#9b59b6",
            linewidth=2, markersize=6, label="Rot Risk", alpha=0.8)

    # window_utilization
    utilizations = [
        _get_meta(t, "context.window_utilization", 0.0) for t in turns
    ]
    ax.plot(turn_numbers, utilizations, ":^", color="#3498db",
            linewidth=1.5, markersize=5, label="Window Utilization", alpha=0.6)

    # 위험 구간 표시
    ax.axhline(y=0.5, color="#e74c3c", linestyle="--", linewidth=1.5,
               alpha=0.5, label="Noise Danger Zone (50%)")

    # 값 표시
    for i, (tn, nr) in enumerate(zip(turn_numbers, noise_ratios)):
        ax.annotate(f"{nr:.1%}", xy=(tn, nr), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#e74c3c", fontweight="bold")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Ratio", fontsize=12)
    ax.set_title("Noise Ratio & Context Rot Risk Evolution\n(Happy Path 7-Turn Session)", fontsize=14)
    ax.set_xticks(turn_numbers)
    ax.set_ylim(-0.05, max(max(noise_ratios), 0.7) + 0.1)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    _save_chart(fig, output_path)


# ═══════════════════════════════════════
# CHART 5: Cross-Scenario Comparison
# ═══════════════════════════════════════

def chart_cross_scenario(all_sessions: dict[str, list[dict]], output_path: str) -> None:
    """3개 시나리오 평균 점수 Grouped Bar.

    Args:
        all_sessions: {scenario_name: turns} 딕셔너리.
        output_path: 출력 PNG 파일명.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = list(all_sessions.keys())
    x = np.arange(len(DIMENSIONS))
    bar_width = 0.25

    for i, scenario in enumerate(scenarios):
        turns = all_sessions[scenario]

        # 마지막 턴의 점수 사용 (최종 결과)
        last_turn = turns[-1]
        scores = []
        for dim in DIMENSIONS:
            score = _get_score(last_turn, dim)
            scores.append(score if score is not None else 0.0)

        bars = ax.bar(
            x + i * bar_width, scores, bar_width,
            label=SCENARIO_LABELS[scenario],
            color=SCENARIO_COLORS[scenario], alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

        # 값 표시
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{score:.2f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold",
            )

    # threshold 마커
    for j, dim in enumerate(DIMENSIONS):
        threshold = THRESHOLDS[dim]
        ax.plot(
            [j - 0.15, j + len(scenarios) * bar_width - 0.1],
            [threshold, threshold],
            "--", color=COLORS["threshold"], linewidth=1.5, alpha=0.6,
        )
        if j == 0:
            ax.plot([], [], "--", color=COLORS["threshold"], label="Threshold")

    ax.set_xlabel("Quality Dimension", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Cross-Scenario 4D Score Comparison\n(Final Turn Results)", fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([DIMENSION_LABELS[d].replace("\n", " ") for d in DIMENSIONS])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save_chart(fig, output_path)


# ═══════════════════════════════════════
# CHART 6: Contradiction Events Timeline
# ═══════════════════════════════════════

def chart_contradiction_timeline(turns: list[dict], output_path: str) -> None:
    """교차 턴 모순 감지(Pattern B) 이벤트 시각화.

    Args:
        turns: Happy Path 세션의 턴 데이터.
        output_path: 출력 PNG 파일명.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    turn_numbers = [t["turn_number"] for t in turns]

    # 일관성 점수 라인
    consistency_scores = [_get_score(t, "consistency") for t in turns]
    plot_scores = [s if s is not None else float("nan") for s in consistency_scores]
    ax.plot(turn_numbers, plot_scores, "-o", color=COLORS["consistency"],
            linewidth=2.5, markersize=8, label="Consistency Score", zorder=2)

    # threshold
    ax.axhline(y=THRESHOLDS["consistency"], color=COLORS["threshold"],
               linestyle="--", linewidth=1.5, alpha=0.5, label="Threshold (0.7)")

    # 모순 감지 이벤트 마커 (v2/v3 trace 양립)
    from agent.monitoring_schema import (
        get_contradicts_from_metadata,
        get_contradiction_resolved_from_metadata,
        get_previous_conclusion_from_metadata,
    )
    for t in turns:
        tn = t["turn_number"]
        meta = t["metadata"]

        contradicts = get_contradicts_from_metadata(meta) or False
        resolved = get_contradiction_resolved_from_metadata(meta) or False

        if contradicts:
            color = "#2ecc71" if resolved else "#e74c3c"
            marker = "^" if resolved else "v"
            label_text = "Resolved" if resolved else "Unresolved"

            score = _get_score(t, "consistency") or 0.5
            ax.scatter(
                tn, score, s=200, c=color, marker=marker,
                edgecolors="black", linewidth=1.5, zorder=4,
            )

            # 어노테이션
            prev_conclusion = get_previous_conclusion_from_metadata(meta) or ""
            short_conclusion = prev_conclusion[:30] + "..." if len(prev_conclusion) > 30 else prev_conclusion
            if short_conclusion:
                ax.annotate(
                    f"T{tn}: Contradiction\n({label_text})",
                    xy=(tn, score), xytext=(0, 25),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                )

    # confidence_delta 표시 (서브플롯 대신 텍스트)
    for t in turns:
        tn = t["turn_number"]
        delta = _get_meta(t, "context.confidence_delta")
        if delta is not None and delta != 0:
            ax.text(
                tn, -0.08, f"delta={delta:+.2f}",
                ha="center", fontsize=7, color="#7f8c8d", style="italic",
            )

    # 범례에 이벤트 마커 추가
    ax.scatter([], [], s=100, c="#2ecc71", marker="^",
               edgecolors="black", label="Contradiction Resolved")
    ax.scatter([], [], s=100, c="#e74c3c", marker="v",
               edgecolors="black", label="Contradiction Unresolved")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Consistency Score", fontsize=12)
    ax.set_title("Contradiction Events Timeline (Pattern B)\n(Happy Path 7-Turn Session)", fontsize=14)
    ax.set_xticks(turn_numbers)
    ax.set_ylim(-0.15, 1.1)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save_chart(fig, output_path)


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def fetch_all_sessions(offline: bool = False) -> dict[str, list[dict]]:
    """3개 세션 데이터를 가져온다. Langfuse 실패 시 fallback 사용.

    Args:
        offline: True면 Langfuse 호출 없이 fallback 데이터만 사용.

    Returns:
        {scenario_name: turns} 딕셔너리.
    """
    if offline:
        print("\n[오프라인 모드] 보고서 실측 데이터 사용")
        return get_fallback_sessions()

    # Langfuse에서 가져오기 시도
    try:
        from evaluation.visualize_session import fetch_session_data
    except ImportError:
        print("\n[경고] visualize_session 모듈 import 실패 → fallback 사용")
        return get_fallback_sessions()

    all_sessions = {}
    for name, session_id in SESSION_IDS.items():
        print(f"\n[{name}] 세션 데이터 조회 중: {session_id}")
        try:
            turns = fetch_session_data(session_id)
            all_sessions[name] = turns
            print(f"  → {len(turns)}개 턴 조회 완료")
        except SystemExit:
            print(f"  → 세션 '{session_id}' 조회 실패 (건너뜀)")
        except Exception as e:
            print(f"  → 세션 '{session_id}' 오류: {e} (건너뜀)")

    if not all_sessions:
        print("\n[경고] Langfuse 조회 실패 → 보고서 실측 데이터(fallback)로 대체")
        return get_fallback_sessions()

    return all_sessions


def main() -> None:
    """전체 차트 생성 → docs/analysis/charts/ 저장."""
    parser = argparse.ArgumentParser(description="Phase 1 보고용 시각화 차트 생성")
    parser.add_argument(
        "--offline", action="store_true",
        help="Langfuse 호출 없이 보고서 실측 데이터로 차트 생성",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1 보고용 시각화 차트 생성")
    print("=" * 60)

    # 세션 데이터 조회
    all_sessions = fetch_all_sessions(offline=args.offline)

    if not all_sessions:
        print("\n오류: 데이터를 가져올 수 없습니다.")
        sys.exit(1)

    print(f"\n총 {len(all_sessions)}개 세션 데이터 준비 완료")
    print("-" * 60)
    print("차트 생성 시작...\n")

    # Happy Path 데이터 (차트 1~4, 6)
    happy_turns = all_sessions.get("happy_path")

    if happy_turns:
        print("[1/6] 4D Score Radar Chart")
        chart_radar_4d(happy_turns, "01_radar_4d.png")

        print("[2/6] Token Composition Stacked Bar")
        chart_token_composition(happy_turns, "02_token_composition.png")

        print("[3/6] 4D Score Trend Line Chart")
        chart_score_trend(happy_turns, "03_score_trend.png")

        print("[4/6] Noise Ratio Evolution")
        chart_noise_evolution(happy_turns, "04_noise_evolution.png")
    else:
        print("[1-4/6] Happy Path 세션 없음 — 건너뜀")

    # 교차 시나리오 비교 (차트 5)
    if len(all_sessions) >= 2:
        print("[5/6] Cross-Scenario Comparison")
        chart_cross_scenario(all_sessions, "05_cross_scenario.png")
    else:
        print("[5/6] 비교할 세션 부족 — 건너뜀")

    # 모순 감지 타임라인 (차트 6)
    if happy_turns:
        print("[6/6] Contradiction Events Timeline")
        chart_contradiction_timeline(happy_turns, "06_contradiction_timeline.png")
    else:
        print("[6/6] Happy Path 세션 없음 — 건너뜀")

    print("\n" + "=" * 60)
    print("차트 생성 완료!")
    print(f"출력 경로: {os.path.abspath(CHART_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
