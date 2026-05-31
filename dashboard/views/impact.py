"""
dashboard/views/impact.py — Tab 5: 전달 (Impact)

관측 질문 (Charter §3.3):
    "누적 컨텍스트가 결과 품질에 미치는 영향"

3영역 구조 (2026-06-01 Session 4 — 4 agent 정체성 비평 후 전면 축소):
    [§3.3 종합 판정 카드] — 누적 영향 한 줄 판정 (HEALTHY/WATCH/ROT)

    ① 인과 전파 (Causal Propagation)  ⭐ 본업
       어떤 이전 컨텍스트가 현재 답변에 인과 기여했는가
       — context.causal_sources Sankey + drill-down

    ② 누적 신호 (Accumulation State)
       컨텍스트가 어떻게 쌓이고 있는가
       — density × noise × rot_risk 통합 차트 + Rot Gate 판정

    ③ 결과 흔적 (Outcome Footprint)
       누적이 결과 품질에 남긴 흔적
       — 이상 패턴 요약 + Query Alignment 시계열 + Tab 4 ⑤ Groundedness 인라인 링크

제거 (Session 4, 4 agent 비평 결과):
    - ① ② 게이트 (Tab 2 process 탭이 SSOT)
    - ⑦ Post-loop 진단 (Eval/Experiment 트랙으로, Tab 7~8)
    - 표면 4종 (8영역 개요표 / 전달 여정 표 / 한눈 카드 3장 / 게이트 비용)
    - Verify Gate Sankey (Tab 2로 흡수)
    - ⑥ Groundedness 영역 → Tab 4 ⑤ SSOT 인라인 링크로 강등

데이터 흐름:
    입력: turns (list[dict])
    출력: Streamlit UI
"""
import pandas as pd
import streamlit as st

from agent.monitoring_schema import ATTRS, DASHBOARD_THRESHOLDS, ROT_GATE_THRESHOLD
from dashboard.analysis import compute_attribute_trends, detect_anomaly_patterns
from dashboard.charts import (
    STATUS_COLORS,
    causal_source_flow,
    density_noise_combined,
    query_alignment_trend,
)
from dashboard.widgets.cross_tab_link import alert_with_tab_link as _alert_with_tab_link

# --- 전달 축에 해당하는 이상 패턴 이름 (③ 결과 흔적 영역에서 필터) ---
_DELIVERY_ANOMALY_NAMES = {"context_rot", "token_bloat"}


_SEVERITY_LABEL: dict[str, str] = {
    "good": "OK", "warn": "WARN", "bad": "CRIT", "na": "N/A",
}


def _color_severity_cell(val: str) -> str:
    """Styler.map: 'OK ·' / 'WARN ·' / 'CRIT ·' prefix로 색상 결정."""
    if not isinstance(val, str) or " · " not in val:
        return ""
    prefix = val.split(" · ", 1)[0]
    c = {
        "OK":   STATUS_COLORS["good"],
        "WARN": STATUS_COLORS["warn"],
        "CRIT": STATUS_COLORS["bad"],
    }.get(prefix, "")
    return f"color:{c};font-weight:600;" if c else ""


# ═══════════════════════════════════════════════════════════════
# 진입 헬퍼 — §3.3 종합 판정 + 핵심 용어
# ═══════════════════════════════════════════════════════════════


def _compute_session_verdict(turns: list[dict]) -> tuple[str, str, str]:
    """세션 전체의 §3.3 종합 판정 — (severity, label, 사유).

    판정 규칙 (간단한 휴리스틱, Phase 4에서 정밀화 예정):
        - ROT  (CRIT): rot_trig ≥ 2  OR  delivery anomaly ≥ 2
        - WATCH (WARN): rot_trig == 1  OR  delivery anomaly ≥ 1
                       OR  최종 응답 grounded_ratio < 0.7 (데이터 있을 때)
        - HEALTHY (OK): 그 외
    """
    rot_trig = sum(
        1 for t in turns
        if t.get("metadata", {}).get(ATTRS["context.rot_gate_triggered"])
    )
    anomalies = detect_anomaly_patterns(turns)
    delivery_anom = sum(1 for a in anomalies if a["name"] in _DELIVERY_ANOMALY_NAMES)

    grounded_ratios = [
        t.get("metadata", {}).get("response.grounded_claim_ratio")
        for t in turns
    ]
    grounded_ratios = [g for g in grounded_ratios if isinstance(g, (int, float))]
    grounded_avg = sum(grounded_ratios) / len(grounded_ratios) if grounded_ratios else None

    reasons = []
    if rot_trig:
        reasons.append(f"Rot 트리거 {rot_trig}턴")
    if delivery_anom:
        reasons.append(f"이상 패턴 {delivery_anom}건")
    if grounded_avg is not None and grounded_avg < 0.7:
        reasons.append(f"Grounded 평균 {grounded_avg:.0%}")
    if not reasons:
        reasons.append("Rot·이상·근거 신호 모두 정상")

    if rot_trig >= 2 or delivery_anom >= 2:
        return "bad", "ROT", " / ".join(reasons)
    if rot_trig or delivery_anom or (grounded_avg is not None and grounded_avg < 0.7):
        return "warn", "WATCH", " / ".join(reasons)
    return "good", "HEALTHY", " / ".join(reasons)


def _render_summary_verdict_card(turns: list[dict]) -> None:
    """§3.3 종합 판정 한 카드 — 한눈 카드 3장의 통합 대체."""
    sev, label, reason = _compute_session_verdict(turns)
    color = STATUS_COLORS[sev]
    text_label = _SEVERITY_LABEL[sev]
    st.markdown(
        f'<div role="status" '
        f'aria-label="§3.3 종합 판정: {label} — {reason}" '
        f'style="padding:16px 20px;background:#1e1e1e;border-radius:8px;'
        f'border-left:6px solid {color};margin-bottom:8px;">'
        f'<div style="font-size:11px;color:#bbb;letter-spacing:0.4px;">§3.3 누적 영향 종합 판정</div>'
        f'<div style="font-size:28px;font-weight:700;color:{color};line-height:1.2;'
        f'margin-top:4px;">{label} · {text_label}</div>'
        f'<div style="font-size:13px;color:#ccc;margin-top:6px;">{reason}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_terminology() -> None:
    """핵심 용어 2개 — Rot · Causal Source. 진입 시 접힘."""
    with st.expander("🗂 핵심 용어 (2)", expanded=False):
        st.markdown("""
- **Rot 위험 (Context Rot)**: 누적 컨텍스트에서 유효 정보 비율이 떨어지는 현상.
  *밀도 ↓ + 노이즈 ↑* 동시 발생이 핵심 신호. `rot_risk = window_util × noise`.
- **Causal Source Coverage**: 이전 턴 주장이 현재 컨텍스트에 보존된 정도. `impact ∈ [0, 1]`.
  본 탭의 §3.3 정면 답 — 어떤 이전 컨텍스트가 현재 결과에 인과 기여했는가.
        """)


# ═══════════════════════════════════════════════════════════════
# ① 인과 전파 (Causal Propagation) — 본업
# ═══════════════════════════════════════════════════════════════


def _render_causal_propagation(turns: list[dict]) -> None:
    """① 인과 전파 — §3.3 정면 답 영역. Sankey + drill-down."""
    st.markdown("### ① 인과 전파 (Causal Propagation)  ⭐ 본업")
    st.markdown("#### 어떤 이전 컨텍스트가 현재 답변에 인과 기여했는가 _(누적 → 결과)_")
    st.markdown(
        "- **무엇**: `context.causal_sources` — 이전 턴 주장이 현재 답변에 보존된 정도\n"
        "- **읽기**: Sankey 두께 = 보존 주장 수 · 색 = impact (녹 ≥ 0.7 / 주 ≥ 0.4 / 적 미만)\n"
        "- **§3.3 정합**: 누적 컨텍스트 → 결과 품질의 인과 화살표를 직접 시각화"
    )

    st.plotly_chart(causal_source_flow(turns), use_container_width=True)

    # 상세 표 drill-down
    causal_rows = []
    for turn in turns:
        sources = turn.get("metadata", {}).get(ATTRS["context.causal_sources"]) or []
        if not isinstance(sources, list):
            continue
        for s in sources:
            if not isinstance(s, dict):
                continue
            impact_val = s.get("impact")
            if isinstance(impact_val, (int, float)):
                if impact_val >= 0.7:
                    impact_sev = "good"
                elif impact_val >= 0.4:
                    impact_sev = "warn"
                else:
                    impact_sev = "bad"
                impact_label = f"{_SEVERITY_LABEL[impact_sev]} · {impact_val:.2f}"
            else:
                impact_label = "—"
            causal_rows.append({
                "현재 턴": turn.get("turn_number", "?"),
                "기여 턴": s.get("turn", "?"),
                "주장 총수": s.get("claims_total", "—"),
                "보존 주장": s.get("claims_retained", "—"),
                "impact": impact_label,
            })
    if causal_rows:
        with st.expander(f"상세 표 ({len(causal_rows)}건)", expanded=False):
            df = pd.DataFrame(causal_rows)
            styler = df.style.map(_color_severity_cell, subset=["impact"])
            st.dataframe(styler, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# ② 누적 신호 (Accumulation State)
# ═══════════════════════════════════════════════════════════════


def _render_accumulation_state(turns: list[dict]) -> None:
    """② 누적 신호 — density × noise × rot_risk + Rot Gate 판정 (compact)."""
    st.markdown("### ② 누적 신호 (Accumulation State)")
    st.markdown("#### 컨텍스트가 어떻게 쌓이고 있는가 _(누적 단계)_")
    st.markdown(
        "- **무엇**: 정보 밀도 ↓ + 노이즈 ↑ 가 동시 발생하면 Rot 신호\n"
        f"- **읽기**: `rot_risk > {ROT_GATE_THRESHOLD:.0%}` & `dead_weight > 0` → "
        "자동 Rot Gate 트리거 (prune + 결론 윈도우 축소)\n"
        "- **§3.3 정합**: 누적 *상태*가 결과 품질의 잠재 원인. ①과 함께 봐야 인과 완성"
    )

    st.plotly_chart(density_noise_combined(turns), use_container_width=True)

    # Rot Gate 판정 표 (compact — 트리거 있을 때만)
    rot_gate_rows = []
    triggered_count = 0
    for turn in turns:
        meta = turn.get("metadata", {})
        triggered = meta.get(ATTRS["context.rot_gate_triggered"])
        dead_weight = meta.get(ATTRS["context.dead_weight_tokens"]) or 0
        window_size = meta.get(ATTRS["context.conclusion_window_size"])
        rot_risk = meta.get(ATTRS["context.rot_risk"])

        if triggered:
            triggered_count += 1
        if triggered is not None or window_size is not None:
            judgment = (
                "CRIT · ROT DETECTED" if triggered
                else ("OK · HEALTHY" if triggered is False else "—")
            )
            rot_gate_rows.append({
                "턴": turn.get("turn_number", "?"),
                "rot_risk": f"{rot_risk:.3f}" if isinstance(rot_risk, (int, float)) else "—",
                "판정": judgment,
                "Pruned 토큰": f"{dead_weight:,}" if triggered and dead_weight else "—",
                "결론 윈도우": str(window_size) if window_size is not None else "—",
            })

    if rot_gate_rows:
        with st.expander(
            f"Rot Gate 판정 ({triggered_count}턴 트리거 / {len(rot_gate_rows)}턴 측정)",
            expanded=bool(triggered_count),
        ):
            df = pd.DataFrame(rot_gate_rows)
            styler = df.style.map(_color_severity_cell, subset=["판정"])
            st.dataframe(styler, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# ③ 결과 흔적 (Outcome Footprint)
# ═══════════════════════════════════════════════════════════════


def _render_outcome_footprint(turns: list[dict]) -> None:
    """③ 결과 흔적 — 이상 패턴 + Query Alignment + Grounded 인라인 링크."""
    st.markdown("### ③ 결과 흔적 (Outcome Footprint)")
    st.markdown("#### 누적이 결과 품질에 남긴 흔적 _(이상·정렬·근거)_")
    st.markdown(
        "- **무엇**: 이상 패턴 / Query Alignment 시계열 / Grounded (Tab 4 SSOT)\n"
        "- **읽기**: 어느 턴이 이상 신호 / 응답이 쿼리에 정렬됐는가\n"
        "- **§3.3 정합**: 누적 영향의 *결과* 측면 — 본격 진단은 Tab 4 / 측정 & 진단 탭"
    )

    # 이상 패턴 — 전달 축만 + verify_fail 통합 (compact)
    all_anomalies = detect_anomaly_patterns(turns)
    delivery_anomalies = [a for a in all_anomalies if a["name"] in _DELIVERY_ANOMALY_NAMES]

    verify_fail_anom = []
    for turn in turns:
        verdict = turn.get("metadata", {}).get(ATTRS["verify.overall_verdict"])
        if verdict and verdict != "pass":
            verify_fail_anom.append({
                "name": "verify_fail",
                "turn_number": turn.get("turn_number", "?"),
                "severity": "error" if verdict == "fail_numeric" else "warning",
                "message": f"검증 게이트 fail — {verdict} (상세: Tab 2 실행흐름)",
            })

    anomalies = delivery_anomalies + verify_fail_anom
    if anomalies:
        st.markdown("**이상 신호**")
        for a in anomalies[:5]:  # 최대 5건만, 더 있으면 안내
            line = f"Turn {a['turn_number']}: {a['message']}"
            if a["severity"] == "error":
                st.error(line)
            else:
                st.warning(line)
        if len(anomalies) > 5:
            st.caption(f"… 추가 {len(anomalies) - 5}건 — 전체 진단은 측정 & 진단 탭")
    else:
        st.success("OK · 이상 신호 없음")

    # 전달 속성 추세 (있을 때만, compact)
    all_trends = compute_attribute_trends(turns)
    delivery_trends = [
        t for t in all_trends
        if t["attribute"] in {
            "context.noise_ratio", "context.rot_risk",
            "context.information_density", "context.redundancy_ratio",
        }
    ]
    if delivery_trends:
        with st.expander(f"전달 속성 추세 감지 ({len(delivery_trends)}건)", expanded=False):
            for t in delivery_trends:
                direction_label = "상승" if t["direction"] == "up" else "하락"
                st.info(
                    f"**{t['attribute']}** — "
                    f"Turn {t['start_turn']}~{t['end_turn']}에서 "
                    f"{t['consecutive_turns']}턴 연속 {direction_label}"
                )

    # Query Alignment (응답) — 시계열 차트
    st.markdown("**Query Alignment (응답)**")
    st.caption(
        "응답이 사용자 쿼리에 정렬되어 있는가 — `response.query_alignment` 시계열. "
        "분석 단계 정렬은 → 변형 탭 (Tab 4) §⑦ Query Alignment (분석)"
    )
    st.plotly_chart(query_alignment_trend(turns, kind="response"),
                    use_container_width=True)

    # Groundedness — Tab 4 SSOT 인라인 링크 (영역 격하)
    _alert_with_tab_link(
        "info",
        body_html=(
            "<b>Groundedness (패턴 D)</b> — 답변↔컨텍스트 근거 비율. SSOT는"
        ),
        tab_key="fidelity",
        link_label="변형 탭 (Tab 4) → ⑤ Groundedness",
        height=58,
    )


# ═══════════════════════════════════════════════════════════════
# render — main entrypoint
# ═══════════════════════════════════════════════════════════════


def render(turns: list[dict]) -> None:
    """전달 탭을 렌더링한다 (Session 4 — ⑤ 중심 3영역 축소).

    구조:
        [헤더]
        [§3.3 종합 판정 카드]
        [핵심 용어 expander (조건부 접힘)]
        ① 인과 전파 (본업)
        ② 누적 신호
        ③ 결과 흔적
    """
    from dashboard import tab_header
    tab_header.render("impact")

    # §3.3 종합 판정 카드
    _render_summary_verdict_card(turns)

    # 용어집 (조건부 접힘)
    _render_terminology()

    st.divider()

    # ① 인과 전파 (본업)
    _render_causal_propagation(turns)
    st.divider()

    # ② 누적 신호
    _render_accumulation_state(turns)
    st.divider()

    # ③ 결과 흔적
    _render_outcome_footprint(turns)
