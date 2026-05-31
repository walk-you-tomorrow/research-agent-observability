"""
dashboard/views/measure_diagnose.py — Tab 6: 측정 & 진단

관측 질문: "품질 점수는 어떤가? 뭘 고쳐야 하나?"

역할:
    4D 품질 현황 파악 → 3축 진단 매트릭스(heatmap 형태) →
    턴별 진단 (증상→원인→액션 3-step, 근거 속성값 포함) →
    §4 쿼리 정렬 진단 (Pattern I/II/III 통합 — Tab 8 이탈 추적 흡수 SSOT).

논리적 전개 (인지 흐름):
    섹션 1 (현황)     — 턴별 4D 점수 카드 (임계값 게이지 + 이전 턴 대비 변화량)
                        + Judge 평가 근거 / 입력 속성 / Accuracy (expander)
    섹션 2 (매트릭스) — 3축 × 4D heatmap (어느 축×차원이 문제인가 색상으로 즉시 파악)
                        + 드릴다운 안내 (문제 교차점 → 해당 탭)
    섹션 3 (진단)     — (증상) → (원인 후보) → (권장 액션) 3-step + 근거 속성 테이블
    섹션 4 (쿼리 정렬) — PASS/FAIL 뱃지 + Pattern I/II/III 매트릭스 + 자연어 요약
                        (2026-05-31 Tab 8 이탈 추적 흡수)

데이터 흐름:
    입력: turns (list[dict]), session_id (str)
    출력: Streamlit UI
"""
import pandas as pd
import streamlit as st

from agent.monitoring_schema import (
    ATTR_META,
    ATTRS,
    DRIFT_ALIGNMENT_THRESHOLD,
    DRIFT_CONTINUITY_THRESHOLD,
    THRESHOLDS,
    get_judge_attributes,
)
from dashboard.analysis import SCORE_KEYS, detect_attention_events
from dashboard.charts import (
    DIMENSIONS,
    STATUS_COLORS,
    build_drift_stats,
    continuity_trend,
    drift_pattern_matrix_html,
    query_alignment_trend,
)
from dashboard.utils import _render_metric_card, _score_status
from evaluation.diagnosis import diagnose_quality


def _format_value(value) -> str:
    """속성 값을 테이블 표시용 문자열로 변환한다."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "[]"
    return str(value)


def _delta_badge_html(current: float | None, previous: float | None) -> str:
    """이전 턴 대비 변화량 인라인 HTML을 반환한다."""
    if current is None or previous is None:
        return ""
    delta = current - previous
    if abs(delta) < 0.01:
        return '<span style="font-size:11px;color:#888;">→ 변화없음</span>'
    arrow = "▲" if delta > 0 else "▼"
    color = STATUS_COLORS["good"] if delta > 0 else STATUS_COLORS["bad"]
    return (
        f'<span style="font-size:11px;color:{color};font-weight:600;">'
        f'{arrow} {abs(delta):.2f}</span>'
    )


# ═══ 섹션 1: 4D 카드 렌더 ═══

def _render_4d_cards(selected_turn: dict, turns: list[dict]) -> None:
    """4D 점수 카드를 임계값 게이지 + 이전 턴 변화량과 함께 렌더링한다."""
    scores = selected_turn.get("scores", {})
    meta = selected_turn.get("metadata", {})
    verdicts = selected_turn.get("score_verdicts", {})
    turn_num = selected_turn.get("turn_number", 1)

    prev_scores: dict = {}
    if turn_num > 1:
        prev = next((t for t in turns if t["turn_number"] == turn_num - 1), None)
        if prev:
            prev_scores = prev.get("scores", {})

    dim_info = {
        "completeness": {
            "icon": "📦", "label": "완전성",
            "q": "답변에 필요한 데이터가 모두 수집됐는가?",
        },
        "efficiency": {
            "icon": "⚡", "label": "효율성",
            "q": "토큰·시간·비용을 적절히 사용했는가?",
        },
        "relevance": {
            "icon": "🎯", "label": "관련성",
            "q": "노이즈 없이 관련 데이터만 컨텍스트에 들어왔는가?",
        },
        "consistency": {
            "icon": "🔄", "label": "일관성",
            "q": "이전 결론·패턴 A/B/C/D가 일관적으로 유지됐는가?",
        },
    }

    win_util = meta.get("context.window_utilization")
    eff_label: tuple[str, str] | None = None
    if isinstance(win_util, (int, float)):
        if win_util > 0.85:
            eff_label = ("OVER-USED", STATUS_COLORS["bad"])
        elif win_util < 0.40:
            eff_label = ("UNDER-USED", "#f4a900")
        else:
            eff_label = ("OPTIMAL", STATUS_COLORS["good"])

    cols = st.columns(4)
    for i, dim_key in enumerate(DIMENSIONS):
        info = dim_info[dim_key]
        score_val = scores.get(SCORE_KEYS[dim_key])
        threshold = THRESHOLDS[dim_key]
        verdict = verdicts.get(dim_key)
        prev_val = prev_scores.get(SCORE_KEYS[dim_key])

        with cols[i]:
            _render_metric_card(f"{info['icon']} {info['label']}", score_val, threshold)

            # 임계값 게이지
            if score_val is not None:
                pct = min(score_val * 100, 100)
                bar_color = STATUS_COLORS["good"] if score_val >= threshold else STATUS_COLORS["bad"]
                st.markdown(
                    f'<div style="margin:2px 0 4px;">'
                    f'<div style="font-size:10px;color:#888;text-align:center;">'
                    f'{score_val:.2f} / 임계 {threshold:.2f}</div>'
                    f'<div style="background:#333;border-radius:3px;height:4px;margin-top:2px;">'
                    f'<div style="background:{bar_color};width:{pct:.0f}%;height:4px;'
                    f'border-radius:3px;"></div></div></div>',
                    unsafe_allow_html=True,
                )

            # PASS/FAIL 뱃지
            if verdict:
                v_color = STATUS_COLORS["good"] if verdict == "PASS" else STATUS_COLORS["bad"]
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<span style="background:{v_color};color:white;padding:2px 10px;'
                    f'border-radius:4px;font-size:12px;font-weight:600;">{verdict}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 이전 턴 대비 변화량
            delta_html = _delta_badge_html(score_val, prev_val)
            if delta_html:
                st.markdown(
                    f'<div style="text-align:center;margin-top:2px;">{delta_html}</div>',
                    unsafe_allow_html=True,
                )

            # Efficiency 자원 판정 라벨
            if dim_key == "efficiency" and eff_label:
                lbl, color = eff_label
                st.markdown(
                    f'<div style="text-align:center;margin-top:4px;">'
                    f'<span style="background:{color}20;color:{color};'
                    f'padding:2px 8px;border-radius:4px;font-size:11px;'
                    f'font-weight:500;border:1px solid {color};">{lbl}</span>'
                    f' <span style="font-size:10px;color:#777">'
                    f'(window {win_util:.0%})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.caption(info["q"])

    # Judge 평가 근거
    score_comments = selected_turn.get("score_comments", {})
    if any(score_comments.values()):
        with st.expander("🧑‍⚖️ 이 점수가 어떻게 계산됐나 (Judge 평가 근거)", expanded=False):
            for dim_key in DIMENSIONS:
                comment = score_comments.get(SCORE_KEYS[dim_key], "")
                if not comment:
                    continue
                info = dim_info[dim_key]
                reasoning = comment.split("] ", 1)[-1] if "] " in comment else comment
                verdict_tag = comment.split("]")[0].strip("[") if "]" in comment else ""
                v_color = STATUS_COLORS["good"] if "PASS" in verdict_tag else STATUS_COLORS["bad"]
                st.markdown(
                    f"**{info['icon']} {info['label']}** "
                    f'<span style="color:{v_color};font-size:12px;">({verdict_tag})</span>',
                    unsafe_allow_html=True,
                )
                st.caption(reasoning)

    # Judge 입력 속성
    with st.expander("📥 Judge 입력 속성 (어떤 값들을 보고 평가했나)", expanded=False):
        for dim_key in DIMENSIONS:
            judge_attrs = get_judge_attributes(dim_key)
            if not judge_attrs:
                continue
            info = dim_info[dim_key]
            st.markdown(f"**{info['icon']} {info['label']}** ({len(judge_attrs)}개 입력)")
            rows = []
            for attr_key in sorted(judge_attrs):
                attr_info = ATTR_META.get(attr_key, {})
                rows.append({
                    "속성": attr_key,
                    "설명": attr_info.get("description", ""),
                    "값": _format_value(meta.get(attr_key)),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 검증 체크포인트 속성 (verify.* 네임스페이스)
    # 검증(verify_result)은 4D quality dimension이 아닌 별도 모니터링 체크포인트이므로
    # (CLAUDE.md: "검증은 프로세스 단계가 아닌 모니터링 활동"), 4D quality 태그가 아니라
    # verify.* 네임스페이스로 수집한다. (이전엔 quality=="accuracy"로 필터했으나
    # 그런 quality 값을 가진 속성이 YAML에 없어 항상 0건 → 섹션 미표시 버그였음.)
    verify_attrs = [
        (key, am) for key, am in ATTR_META.items()
        if key.startswith("verify.")
    ]
    if verify_attrs:
        with st.expander(
            f"🔍 검증 체크포인트 (verify.*) — {len(verify_attrs)}개 속성  ·  상세: Tab 5 전달",
            expanded=False,
        ):
            rows = []
            for attr_key, am in sorted(verify_attrs, key=lambda x: x[0]):
                rows.append({
                    "속성": attr_key,
                    "설명": am.get("description", ""),
                    "값": _format_value(meta.get(attr_key)),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══ 섹션 2: 3축 × 4D heatmap 매트릭스 ═══

def _render_axis_quality_matrix(scores: dict, meta: dict) -> None:
    """3축 × 4D 진단 heatmap — 어느 교차점에 문제가 있는지 색상으로 즉시 파악.

    관여하는 교차점은 점수 색상(good/warn/bad)으로, 미관여는 회색 선으로 표시.
    """
    mapping = {
        "completeness": {"구성": True,  "변형": False, "전달": False},
        "efficiency":   {"구성": True,  "변형": False, "전달": True},
        "relevance":    {"구성": True,  "변형": False, "전달": True},
        "consistency":  {"구성": False, "변형": True,  "전달": False},
    }
    dim_labels = {
        "completeness": "완전성",
        "efficiency":   "효율성",
        "relevance":    "관련성",
        "consistency":  "일관성",
    }

    rows_html = []
    fail_cells: list[tuple[str, str, float]] = []

    for dim_key, dim_label in dim_labels.items():
        score = scores.get(SCORE_KEYS[dim_key])
        threshold = THRESHOLDS.get(dim_key, 0.7)

        if score is None:
            cell_color = "#555"
            cell_text = "—"
            score_color = "#9aa0a6"
            score_str = "—"
        elif score >= threshold:
            cell_color = STATUS_COLORS["good"]
            cell_text = f"{score:.2f}"
            score_color = STATUS_COLORS["good"]
            score_str = f"✓ {score:.2f}"
        elif score >= 0.5:
            cell_color = STATUS_COLORS["warn"]
            cell_text = f"{score:.2f}"
            score_color = STATUS_COLORS["warn"]
            score_str = f"△ {score:.2f}"
        else:
            cell_color = STATUS_COLORS["bad"]
            cell_text = f"{score:.2f}"
            score_color = STATUS_COLORS["bad"]
            score_str = f"⚠ {score:.2f}"

        cells = []
        for axis_label in ["구성", "변형", "전달"]:
            if mapping[dim_key][axis_label]:
                cells.append(
                    f'<td style="text-align:center;background:{cell_color}22;'
                    f'color:{cell_color};font-size:13px;font-weight:700;'
                    f'border:1px solid {cell_color}44;padding:6px 8px;">'
                    f'{cell_text}</td>'
                )
                if score is not None and score < threshold:
                    fail_cells.append((axis_label, dim_label, score))
            else:
                cells.append(
                    '<td style="text-align:center;color:#444;font-size:14px;'
                    'border:1px solid #333;padding:6px 8px;">─</td>'
                )

        rows_html.append(
            f'<tr>'
            f'<td style="font-weight:600;padding:6px 10px;white-space:nowrap;">{dim_label}</td>'
            f'{"".join(cells)}'
            f'<td style="text-align:right;font-weight:700;color:{score_color};'
            f'padding:6px 10px;white-space:nowrap;">{score_str}</td>'
            f'<td style="text-align:right;font-size:10px;color:#888;padding:6px 6px;">'
            f'임계 {threshold:.0%}</td>'
            f'</tr>'
        )

    table_html = (
        '<table style="width:100%;border-collapse:collapse;margin:8px 0;">'
        '<thead><tr style="border-bottom:2px solid #444;">'
        '<th style="text-align:left;padding:6px 10px;color:#888;font-size:11px;">차원</th>'
        '<th style="text-align:center;padding:6px 8px;color:#4a9eff;">🔍① 구성</th>'
        '<th style="text-align:center;padding:6px 8px;color:#a78bfa;">🔍② 변형</th>'
        '<th style="text-align:center;padding:6px 8px;color:#34d399;">🔍③ 전달</th>'
        '<th style="text-align:right;padding:6px 10px;color:#888;font-size:11px;">Score</th>'
        '<th style="text-align:right;padding:6px 6px;color:#888;font-size:11px;">임계</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.caption(
        f'셀 색상: '
        f'<span style="color:{STATUS_COLORS["good"]}">■ 양호(≥임계)</span>  '
        f'<span style="color:{STATUS_COLORS["warn"]}">■ 주의(0.5~임계)</span>  '
        f'<span style="color:{STATUS_COLORS["bad"]}">■ 위험(&lt;0.5)</span>  '
        f'<span style="color:#555">─ 미관여</span>',
        unsafe_allow_html=True,
    )

    _AXIS_DRILL = {
        "구성": "**Tab 3 구성** — 토큰 구성 / 소스 분포 / Plan & Query",
        "변형": "**Tab 4 변형** — 결론 압축 / 참조 추적 / 모순",
        "전달": "**Tab 5 전달** — 충분성 게이트 / 검증 게이트 / Rot 위험",
    }
    if fail_cells:
        triggered_axes: set[str] = set()
        lines = []
        for axis_label, dim_label, score in fail_cells:
            lines.append(f"**{axis_label} × {dim_label}** ({score:.2f})")
            triggered_axes.add(axis_label)
        st.warning("⚠ 문제 교차점: " + " · ".join(lines))
        st.markdown("**드릴다운 →**")
        for axis_name in ["구성", "변형", "전달"]:
            if axis_name in triggered_axes:
                st.markdown(f"- {_AXIS_DRILL[axis_name]}")
    else:
        st.success("✓ 모든 교차점 PASS")


# ═══ 섹션 3: 3-step 진단 ═══

_DIM_ICON = {
    "completeness": "📦", "efficiency": "⚡",
    "relevance": "🎯", "consistency": "🔄",
}
_DIM_KR = {
    "completeness": "완전성", "efficiency": "효율성",
    "relevance": "관련성", "consistency": "일관성",
}
_CAUSE_HINT = {
    "completeness": "수집 항목 부족 또는 충분성 게이트 미통과",
    "efficiency":   "토큰 과다 사용, 윈도우 포화, 또는 Rot 누적",
    "relevance":    "노이즈 비율 상승 또는 소스 기여도 불균형",
    "consistency":  "이전 결론 압축 손실, 모순 미해결, 또는 Groundedness 저하",
}


def _render_3step_diagnosis(turns: list[dict], selected_turn_num: int) -> None:
    """모든 턴의 진단을 (증상 → 원인 후보 → 권장 액션) 3-step으로 렌더링한다."""
    has_diagnosis = False

    for turn in turns:
        t_scores = {}
        for dim in DIMENSIONS:
            val = turn.get("scores", {}).get(SCORE_KEYS[dim])
            if val is not None:
                t_scores[dim] = float(val)

        if not t_scores:
            continue

        trace_data = {"metadata": turn.get("metadata", {})}
        diag_results = diagnose_quality(t_scores, trace_data)

        if diag_results:
            has_diagnosis = True
            with st.expander(
                f"Turn {turn['turn_number']} — {len(diag_results)}건 진단",
                expanded=(turn["turn_number"] == selected_turn_num),
            ):
                for idx, d in enumerate(diag_results, 1):
                    dim_key = d["dimension"]
                    icon = _DIM_ICON.get(dim_key, "")
                    dim_kr = _DIM_KR.get(dim_key, dim_key)
                    score_val = d.get("score", 0)
                    threshold = THRESHOLDS.get(dim_key, 0.7)
                    sev_color = (
                        STATUS_COLORS["bad"] if score_val < 0.5
                        else STATUS_COLORS["warn"]
                    )

                    st.markdown(
                        f'<div style="border-left:3px solid {sev_color};'
                        f'padding:8px 12px;margin:6px 0;background:#1e1e1e;'
                        f'border-radius:0 6px 6px 0;">'
                        f'<span style="font-weight:700;color:{sev_color};">'
                        f'#{idx} {icon} {dim_kr}</span>'
                        f' <span style="font-size:11px;color:#888;">'
                        f'({score_val:.2f} / 임계 {threshold:.2f})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Step 1: 증상
                    st.markdown("**① 증상**")
                    st.warning(d["diagnosis"])

                    # Step 2: 원인 후보
                    st.markdown("**② 원인 후보**")
                    st.info(_CAUSE_HINT.get(dim_key, ""))

                    # Step 3: 권장 액션
                    st.markdown("**③ 권장 액션**")
                    st.success(f"💡 {d['suggestion']}")

                    # 근거 속성 테이블
                    related = d.get("related_attrs", [])
                    if related:
                        t_meta = turn.get("metadata", {})
                        evidence_rows = [
                            {
                                "속성": attr_key,
                                "설명": ATTR_META.get(attr_key, {}).get("description", ""),
                                "값": _format_value(t_meta.get(attr_key)),
                            }
                            for attr_key in related
                        ]
                        with st.expander("📎 근거 속성값", expanded=False):
                            st.dataframe(
                                pd.DataFrame(evidence_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                    if idx < len(diag_results):
                        st.markdown("---")

    if not has_diagnosis:
        st.success("모든 턴의 4D 품질이 임계값 이상입니다. 진단 대상 없음.")

    # 이벤트 기반 진단 안내
    events = detect_attention_events(turns)
    if events:
        from collections import Counter
        ev_counts = Counter(e["event_type"] for e in events)
        _EV_LABELS = {
            "regather": "재수집", "contradiction": "모순 감지",
            "source_conflict": "소스 충돌", "verify_fail": "검증 실패",
        }
        _EV_TARGETS = {
            "regather": "**Tab 5 전달 § 충분성 게이트**",
            "contradiction": "**Tab 4 변형 § 턴 간 모순**",
            "source_conflict": "**Tab 4 변형 § 턴 간 모순** (소스 충돌 행)",
            "verify_fail": "**Tab 5 전달 § 검증 게이트** + § 이상 패턴 종합",
        }
        st.markdown("#### 이벤트 기반 진단")
        for etype, cnt in ev_counts.most_common():
            label = _EV_LABELS.get(etype, etype)
            target = _EV_TARGETS.get(etype, "—")
            (st.error if etype == "verify_fail" else st.warning)(
                f"**{label}** {cnt}건 → {target}에서 상세 확인"
            )


# ═══ §4 쿼리 정렬 진단 (Tab 8 이탈 추적 흡수, 2026-05-31) ═══

def _render_drift_pass_fail_table(turns: list[dict]) -> None:
    """턴별 3개 속성의 PASS/WARN/FAIL 뱃지 테이블 (현황 요약)."""
    drift_rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        continuity = meta.get(ATTRS["query.session_continuity"])
        a_align = meta.get(ATTRS["analysis.query_alignment"])
        r_align = meta.get(ATTRS["response.query_alignment"])

        def _badge(val, threshold: float) -> str:
            if val is None:
                return '<span style="color:#555;font-size:12px;">—</span>'
            status = _score_status(val, threshold)
            color = STATUS_COLORS[status]
            label = "PASS" if status == "good" else ("WARN" if status == "warn" else "FAIL")
            return (
                f'<span style="background:{color};color:white;'
                f'padding:1px 7px;border-radius:3px;font-size:11px;">{label}</span>'
                f' <span style="font-size:11px;color:#888;">({val:.2f})</span>'
            )

        drift_rows.append({
            "턴": f"T{turn.get('turn_number', '?')}",
            "c": _badge(continuity, DRIFT_CONTINUITY_THRESHOLD),
            "a": _badge(a_align, DRIFT_ALIGNMENT_THRESHOLD),
            "r": _badge(r_align, DRIFT_ALIGNMENT_THRESHOLD),
        })

    if not drift_rows:
        st.caption("이탈 감지 데이터 없음")
        return

    header = (
        '<tr>'
        '<th style="padding:4px 8px;color:#888;font-size:11px;text-align:center;">턴</th>'
        '<th style="padding:4px 8px;color:#888;font-size:11px;">session_continuity<br>'
        '<span style="font-weight:400;font-size:10px;">(사용자 전환 감지)</span></th>'
        '<th style="padding:4px 8px;color:#888;font-size:11px;">analysis.query_alignment<br>'
        '<span style="font-weight:400;font-size:10px;">(분석 단계 이탈)</span></th>'
        '<th style="padding:4px 8px;color:#888;font-size:11px;">response.query_alignment<br>'
        '<span style="font-weight:400;font-size:10px;">(응답 단계 이탈)</span></th>'
        '</tr>'
    )
    rows_html = "".join(
        f'<tr style="border-bottom:1px solid #333;">'
        f'<td style="text-align:center;padding:4px 8px;font-weight:600;">{r["턴"]}</td>'
        f'<td style="padding:4px 8px;">{r["c"]}</td>'
        f'<td style="padding:4px 8px;">{r["a"]}</td>'
        f'<td style="padding:4px 8px;">{r["r"]}</td>'
        f'</tr>'
        for r in drift_rows
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead>{header}</thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"임계값: session_continuity ≥ {DRIFT_CONTINUITY_THRESHOLD} / "
        f"alignment ≥ {DRIFT_ALIGNMENT_THRESHOLD}. "
        "시계열 차트: 🔍② 변형 §⑥⑦ · 🔍③ 전달 §⑧"
    )


def _render_drift_pattern_summary(stats: list[dict]) -> None:
    """4 메트릭 카드 + 자연어 텍스트 진단 — trajectory.py _render_summary 흡수."""
    pivot_turns = [s["turn_number"] for s in stats if s["pattern"] == "I"]
    drift_turns = [s["turn_number"] for s in stats if s["pattern"] == "II"]
    double_fail_turns = [s["turn_number"] for s in stats if s["pattern"] == "III"]

    # 가장 큰 이탈 턴 — continuity + alignment 합산 최솟값 (Pattern II/III 대상)
    worst_turn = None
    worst_score = 2.0
    for s in stats:
        if s["pattern"] in ("II", "III"):
            score = (s["continuity"] if s["continuity"] is not None else 1.0) + \
                    (s["min_alignment"] if s["min_alignment"] is not None else 1.0)
            if score < worst_score:
                worst_score = score
                worst_turn = s["turn_number"]

    def _metric_html(label: str, value: str, sub: str, color: str) -> str:
        return (
            f'<div style="text-align:center;padding:8px 0;">'
            f'<div style="font-size:13px;color:#888;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:28px;font-weight:700;color:{color};">{value}</div>'
            f'<div style="font-size:11px;color:#aaa;margin-top:2px;">{sub}</div>'
            f'</div>'
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cnt = len(pivot_turns)
        turns_str = f"Turn {', '.join(str(t) for t in pivot_turns)}" if pivot_turns else "없음"
        st.markdown(
            _metric_html("✅ User Pivot", str(cnt), turns_str,
                         STATUS_COLORS["good"] if cnt == 0 else "#f39c12"),
            unsafe_allow_html=True,
        )
    with col2:
        cnt = len(drift_turns)
        turns_str = f"Turn {', '.join(str(t) for t in drift_turns)}" if drift_turns else "없음"
        st.markdown(
            _metric_html("⚠️ Agent Drift", str(cnt), turns_str,
                         STATUS_COLORS["good"] if cnt == 0 else STATUS_COLORS["warn"]),
            unsafe_allow_html=True,
        )
    with col3:
        cnt = len(double_fail_turns)
        turns_str = f"Turn {', '.join(str(t) for t in double_fail_turns)}" if double_fail_turns else "없음"
        st.markdown(
            _metric_html("🚨 이중 실패", str(cnt), turns_str,
                         STATUS_COLORS["good"] if cnt == 0 else STATUS_COLORS["bad"]),
            unsafe_allow_html=True,
        )
    with col4:
        worst_str = f"Turn {worst_turn}" if worst_turn is not None else "없음"
        st.markdown(
            _metric_html("최대 이탈 턴", worst_str, "continuity + alignment 최솟값 기준",
                         STATUS_COLORS["bad"] if worst_turn is not None else STATUS_COLORS["good"]),
            unsafe_allow_html=True,
        )

    st.markdown("")  # 여백

    # 텍스트 진단
    data_turns = [s for s in stats if s["continuity"] is not None or s["min_alignment"] is not None]
    total_anomaly = len(pivot_turns) + len(drift_turns) + len(double_fail_turns)

    if not data_turns:
        st.info(
            "이탈 감지 속성 데이터가 없습니다. "
            "`evaluation/run_evaluation.py`를 실행하여 "
            "`analysis.query_alignment` / `response.query_alignment` 점수를 기록하고, "
            "`analyze_query` 노드에서 `query.session_continuity`를 계산하세요."
        )
        return

    if total_anomaly == 0:
        st.success(
            f"이번 세션 ({len(stats)}턴) 에서 이탈 이벤트가 감지되지 않았습니다. "
            f"모든 턴의 session_continuity ≥ {DRIFT_CONTINUITY_THRESHOLD} "
            f"또는 query_alignment ≥ {DRIFT_ALIGNMENT_THRESHOLD} 입니다."
        )
        return

    if drift_turns:
        st.warning(
            f"**Agent Drift** {len(drift_turns)}건 — "
            f"Turn {', '.join(str(t) for t in drift_turns)}: "
            f"에이전트의 분석/응답이 사용자 쿼리에서 이탈했습니다 "
            f"(alignment < {DRIFT_ALIGNMENT_THRESHOLD}). "
            f"`generate_analysis` 프롬프트의 쿼리 정렬 강화를 검토하세요."
        )
    if pivot_turns:
        st.info(
            f"**User Pivot** {len(pivot_turns)}건 — "
            f"Turn {', '.join(str(t) for t in pivot_turns)}: "
            f"사용자가 이전 방향에서 벗어난 새 주제를 요청했습니다 (정상 이탈). "
            f"`analyze_query` 노드가 새 intent를 정확히 감지했는지 확인하세요."
        )
    if double_fail_turns:
        st.error(
            f"**이중 실패** {len(double_fail_turns)}건 — "
            f"Turn {', '.join(str(t) for t in double_fail_turns)}: "
            f"session_continuity와 query_alignment 모두 임계값 미달. "
            f"세션 히스토리 관리와 쿼리 정렬을 동시에 점검하세요."
        )
    if worst_turn is not None:
        st.caption(
            f"**가장 큰 이탈**: Turn {worst_turn} — continuity + alignment 합산 최솟값 기준. "
            "💡 시계열 차트: 🔍② 변형 §⑥⑦ · 🔍③ 전달 §⑧"
        )


def _render_drift_summary(turns: list[dict]) -> None:
    """§4 쿼리 정렬 진단 — 3개 속성의 PASS/FAIL × Pattern I/II/III 통합 진단.

    구성 (Tab 8 이탈 추적 흡수):
        1. 턴별 PASS/WARN/FAIL 뱃지 테이블 (현황)
        2. Pattern I/II/III 진단 매트릭스 (판정)
        3. 4 메트릭 카드 + 자연어 텍스트 진단 (요약)
    """
    # ── 1. 현황 뱃지 테이블 ──
    st.markdown("#### 1️⃣ 턴별 PASS/FAIL 현황")
    _render_drift_pass_fail_table(turns)

    # 데이터 유무 판정
    stats = build_drift_stats(turns)
    has_any_data = any(
        s["continuity"] is not None or s["min_alignment"] is not None
        for s in stats
    )
    if not stats or not has_any_data:
        return

    # ── 2. Pattern I/II/III 매트릭스 ──
    st.markdown("")
    st.markdown("#### 2️⃣ Pattern I / II / III 진단 매트릭스")
    st.caption(
        "각 턴의 continuity × alignment 조합으로 이탈 패턴을 분류합니다. "
        "alignment는 analysis/response 두 값 중 최솟값(보수적 판정)을 사용합니다."
    )
    st.markdown(drift_pattern_matrix_html(stats), unsafe_allow_html=True)

    # ── 3. 4 카드 + 자연어 요약 ──
    st.markdown("")
    st.markdown("#### 3️⃣ 세션 이탈 요약")
    _render_drift_pattern_summary(stats)


# ═══ 메인 렌더 ═══

def render(turns: list[dict], session_id: str) -> None:
    """측정 & 진단 탭을 렌더링한다."""
    st.subheader("📊 측정 & 진단")
    st.caption(
        "AI Agent의 4D 품질 점수를 확인하고, 문제가 있는 교차점을 찾아 원인과 조치를 안내합니다."
    )

    # ═══ 섹션 1: 현황 ═══
    st.markdown("### 📏 현황: 4D 품질")
    st.caption(
        "4D 점수 카드. 점수 아래 게이지는 임계값 대비 위치, "
        "화살표는 이전 턴 대비 변화량을 나타냅니다."
    )

    turn_options = {t["turn_number"]: f"Turn {t['turn_number']}" for t in turns}
    selected_turn_num = st.selectbox(
        "턴 선택", options=list(turn_options.keys()),
        format_func=lambda x: turn_options[x],
        key="measure_turn_select",
    )
    selected_turn = next(t for t in turns if t["turn_number"] == selected_turn_num)

    _render_4d_cards(selected_turn, turns)

    st.divider()

    # ═══ 섹션 2: 3축 × 4D heatmap 매트릭스 ═══
    st.markdown("### 🧭 어느 축·차원이 문제인가?")
    st.caption(
        "4D 점수를 3관측 축(구성·변형·전달)에 매핑. "
        "색상 있는 셀 = 해당 축에서 이 차원을 측정. "
        "빨간/주황 셀 = 문제 교차점 → 드릴다운 탭으로 이동."
    )
    scores = selected_turn.get("scores", {})
    meta = selected_turn.get("metadata", {})
    _render_axis_quality_matrix(scores, meta)

    st.divider()

    # ═══ 섹션 3: 3-step 진단 ═══
    st.markdown("### 🩺 진단: 증상 → 원인 → 액션")
    st.caption(
        "임계값 미달 차원을 자동 진단. "
        "각 건별로 ① 증상 ② 원인 후보 ③ 권장 액션 순으로 전개. "
        "근거 속성값은 '📎 근거 속성값'을 펼쳐서 확인하세요."
    )
    _render_3step_diagnosis(turns, selected_turn_num)

    st.divider()

    # ═══ §4 쿼리 정렬 진단 (Tab 8 이탈 추적 흡수, 2026-05-31) ═══
    st.divider()
    st.markdown("### §4 🔀 쿼리 정렬 진단 (Drift Detection)")
    st.caption(
        "User Pivot(사용자 전환), Agent Drift(에이전트 이탈), 이중 실패를 "
        "**임계값 PASS/FAIL × Pattern I/II/III** 두 축으로 통합 진단합니다. "
        "시계열 시각화 SSOT: **🔍② 변형 §⑥⑦** (continuity·analysis alignment) / "
        "**🔍③ 전달 §⑧** (response alignment)."
    )
    _render_drift_summary(turns)
