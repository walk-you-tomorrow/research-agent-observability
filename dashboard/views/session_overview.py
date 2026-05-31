"""
dashboard/views/session_overview.py — Tab 1: 한눈에

관측자의 질문: "이 세션에서 컨텍스트가 어떻게 구성·변형·전달되었고, 품질은 어떤가?"

역할:
    세션 요약(총 턴 + Health), 턴별 4D 품질 여정 테이블,
    4D 품질 추이 차트, 주의 필요 이벤트를 표시한다.
    3축(구성/변형/전달) 관측은 각 탭(Tab 3/4/5)에서 담당한다.

    인지 흐름: "괜찮아?" → "자세히 보자" → "품질 변화" → "어디가 문제야?"

데이터 흐름:
    입력: turns (list[dict]) — enriched session data (83개 속성)
    출력: Streamlit UI
"""
import streamlit as st
import streamlit.components.v1 as stc

from agent.monitoring_schema import ATTRS, THRESHOLDS
from dashboard.analysis import (
    SCORE_KEYS,
    _compute_turn_health,
    detect_attention_events,
    extract_journey_timeline,
)
from dashboard.charts import STATUS_COLORS, score_trend
from dashboard.utils import _render_metric_card, _score_status


# --- 4D 차원 풀네임 ---
_DIM_FULL_NAMES = {
    "C": "Completeness (완전성)",
    "E": "Efficiency (효율성)",
    "R": "Relevance (관련성)",
    "S": "Consistency (일관성)",
}



def _detect_session_issues(turns: list[dict]) -> list[dict]:
    """세션 데이터에서 감지 가능한 이슈를 반환한다.

    대시보드에서 N/A가 "정상적 데이터 없음"인지 "에러로 인한 공백"인지
    구분할 수 있도록 이슈를 수집한다.

    Args:
        turns: enriched 턴별 데이터 리스트.

    Returns:
        이슈 리스트. 각 항목: {"level": "error"|"warn", "msg": str}.
    """
    issues = []

    # ① 4D 평가 점수 유무 확인
    turns_with_scores = [t for t in turns if any(v is not None for v in t["scores"].values())]
    missing_score_turns = [t["turn_number"] for t in turns if not any(v is not None for v in t["scores"].values())]

    if not turns_with_scores:
        issues.append({
            "level": "error",
            "msg": "4D 평가 데이터 없음 — 에이전트 실행 중 평가가 실패했거나, 실행되지 않았습니다.",
        })
    elif missing_score_turns:
        issues.append({
            "level": "warn",
            "msg": f"일부 턴 4D 평가 누락 — T{', T'.join(str(n) for n in missing_score_turns)}의 점수가 없습니다.",
        })

    # ② 노드 에러 감지 (safe_node가 error 상태를 verify.overall_verdict에 기록)
    error_turns = [
        t["turn_number"] for t in turns
        if t.get("metadata", {}).get(ATTRS["verify.overall_verdict"]) == "error"
    ]
    if error_turns:
        issues.append({
            "level": "error",
            "msg": f"노드 실행 에러 — T{', T'.join(str(n) for n in error_turns)}에서 예외가 발생했습니다.",
        })

    # ③ 핵심 메타데이터 자체가 비어있는 턴 (노드가 아예 실행 안 됨)
    _KEY_GATHER = ATTRS["gather.items_collected"]
    _KEY_TOKENS = ATTRS["context.total_tokens"]
    empty_meta_turns = [
        t["turn_number"] for t in turns
        if t.get("metadata", {}).get(_KEY_GATHER) is None
        and t.get("metadata", {}).get(_KEY_TOKENS) is None
    ]
    if empty_meta_turns:
        issues.append({
            "level": "warn",
            "msg": f"노드 기록 없음 — T{', T'.join(str(n) for n in empty_meta_turns)}의 모니터링 데이터가 비어 있습니다.",
        })

    return issues


def _render_session_status_banner(issues: list[dict]) -> None:
    """세션 이슈를 에러/경고 배너로 렌더링한다.

    이슈가 없으면 표시하지 않는다.

    Args:
        issues: _detect_session_issues()의 반환값.
    """
    if not issues:
        return

    errors = [i for i in issues if i["level"] == "error"]
    warns  = [i for i in issues if i["level"] == "warn"]

    if errors:
        lines = "\n".join(f"- {i['msg']}" for i in errors)
        st.error(f"**세션 데이터 이상 감지**\n{lines}", icon="🚨")
    if warns:
        lines = "\n".join(f"- {i['msg']}" for i in warns)
        st.warning(f"**데이터 누락 주의**\n{lines}", icon="⚠️")


def _render_count_card(label: str, value: int | str) -> None:
    """점수 없는 카운트 메트릭을 메트릭 카드와 동일한 스타일로 렌더링한다.

    Args:
        label: 지표 이름.
        value: 표시할 값.
    """
    html = f"""
    <div style="text-align:center; padding:8px 0;">
        <div style="font-size:14px; color:#888; margin-bottom:4px;">{label}</div>
        <div style="font-size:32px; font-weight:700; color:#ccc;">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_quality_table(timeline: list[dict], turns: list[dict] | None = None) -> None:
    """4D 품질 점수 테이블을 HTML로 렌더링한다.

    Args:
        timeline: extract_journey_timeline()의 반환값.
        turns: enriched 턴별 데이터 (verdict 표시용, 없으면 verdict 생략).
    """
    if not timeline:
        st.info("데이터 없음")
        return

    # 어두운 배경 + 글자색 변경 (메트릭 카드와 동일 톤)
    c_good = STATUS_COLORS["good"]   # #2ecc71
    c_warn = STATUS_COLORS["warn"]   # #f39c12
    c_bad = STATUS_COLORS["bad"]     # #e74c3c
    c_na = STATUS_COLORS["na"]       # #bdc3c7

    _CELL_STYLES = {
        "good": f"color:{c_good}; font-weight:700;",
        "warn": f"color:{c_warn}; font-weight:700;",
        "bad": f"color:{c_bad}; font-weight:700;",
        "na": f"color:#555;",
    }

    style = """
    <style>
    .qt { width:100%; border-collapse:collapse; font-size:15px;
          font-family:-apple-system,BlinkMacSystemFont,sans-serif; }
    .qt th { padding:10px 12px; text-align:center; font-weight:600; font-size:12px;
             color:#999; border-bottom:1px solid #444; background:#1e1e1e; }
    .qt td { padding:10px 12px; text-align:center; border-bottom:1px solid #333;
             background:#1e1e1e; font-size:16px; }
    .qt tr:hover td { background:#2a2a2a; }
    .qt .tn { font-weight:600; text-align:left; color:#ccc; }
    .qt .avg td { border-top:2px solid #666; font-weight:700; background:#252525; }
    </style>
    """

    html = style + '<table class="qt">'

    # 컬럼 헤더
    html += '<tr>'
    html += '<th style="width:60px;">턴</th>'
    for dk in ["C", "E", "R", "S"]:
        html += f'<th>{_DIM_FULL_NAMES[dk]}</th>'
    html += '<th>Health</th></tr>'

    # 데이터 행 + 합계용 누적
    dim_keys = ["C", "E", "R", "S"]
    dim_sums: dict[str, float] = {dk: 0.0 for dk in dim_keys}
    dim_counts: dict[str, int] = {dk: 0 for dk in dim_keys}
    health_sum = 0.0
    health_count = 0

    # verdict lookup 구축
    verdict_lookup: dict[int, dict] = {}
    if turns:
        for t in turns:
            verdict_lookup[t.get("turn_number")] = t.get("score_verdicts", {})

    def _cell(val: float | None, threshold: float, verdict: str | None = None) -> str:
        """점수 셀 HTML을 반환한다."""
        if val is None:
            return f'<td style="{_CELL_STYLES["na"]}">—</td>'
        status = "good" if val >= threshold else ("warn" if val >= 0.5 else "bad")
        return f'<td style="{_CELL_STYLES[status]}">{val * 100:.0f}</td>'

    _DK_TO_DIM = {"C": "completeness", "E": "efficiency",
                  "R": "relevance", "S": "consistency"}

    for row in timeline:
        qual = row["quality"]
        tn = row["turn_number"]
        turn_verdicts = verdict_lookup.get(tn, {})
        html += f'<tr><td class="tn">T{tn}</td>'

        for dk in dim_keys:
            val = qual[dk]
            dn = _DK_TO_DIM[dk]
            verdict = turn_verdicts.get(dn)
            html += _cell(val, THRESHOLDS.get(dn, 0.7), verdict)
            if val is not None:
                dim_sums[dk] += val
                dim_counts[dk] += 1

        h = qual["H"]
        html += _cell(h, 0.7)
        if h is not None:
            health_sum += h
            health_count += 1

        html += '</tr>'

    # 평균 행
    html += '<tr class="avg"><td style="text-align:left; color:#ccc;">평균</td>'
    for dk in dim_keys:
        if dim_counts[dk] > 0:
            avg = dim_sums[dk] / dim_counts[dk]
            dn = {"C": "completeness", "E": "efficiency",
                  "R": "relevance", "S": "consistency"}[dk]
            html += _cell(avg, THRESHOLDS.get(dn, 0.7))
        else:
            html += f'<td style="{_CELL_STYLES["na"]}">—</td>'
    if health_count > 0:
        html += _cell(health_sum / health_count, 0.7)
    else:
        html += f'<td style="{_CELL_STYLES["na"]}">—</td>'
    html += '</tr>'

    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)


def render(turns: list[dict], session_id: str) -> None:
    """한눈에 탭을 렌더링한다 (session 전체 요약).

    레이아웃: 메트릭 카드 → 4D 추이 → 주의 필요(카드 형태)
    Turn별 상세는 Tab 2 + Tab 7에서 확인.

    Args:
        turns: enriched 턴별 데이터 리스트.
        session_id: Langfuse session ID.
    """
    # ═══ 탭 목적 안내 ═══
    st.caption(
        "이 세션에서 컨텍스트가 어떻게 구성·변형·전달되었고, 품질은 어떤가? "
        "건강도 요약 → 4D 추이 → 주의 필요 이벤트 순으로 확인합니다. "
        "각 탭(구성/변형/전달)에서 3축 상세를 드릴다운하세요."
    )

    # ═══ 세션 상태 배너 (이상 감지 시만 표시) ═══
    issues = _detect_session_issues(turns)
    _render_session_status_banner(issues)

    timeline = extract_journey_timeline(turns)
    events = detect_attention_events(turns)

    # ═══ 메트릭 카드: 총 턴 + 4D + Health ═══
    # 세션 평균 4D 점수
    dim_avgs: dict[str, float | None] = {}
    for dim in ["completeness", "efficiency", "relevance", "consistency"]:
        vals = [t["scores"].get(SCORE_KEYS[dim]) for t in turns]
        valid = [v for v in vals if v is not None]
        dim_avgs[dim] = sum(valid) / len(valid) if valid else None

    # 다른 4D 카드와 동일하게 세션 평균 사용 (테이블 평균 행과 일치)
    turn_health = [_compute_turn_health(t["scores"]) for t in turns]
    valid_health = [h for h in turn_health if h is not None]
    session_health = sum(valid_health) / len(valid_health) if valid_health else None

    cols = st.columns(6)
    with cols[0]:
        _render_count_card("총 턴", len(turns))
    with cols[1]:
        _render_metric_card("완전성", dim_avgs["completeness"], THRESHOLDS["completeness"])
    with cols[2]:
        _render_metric_card("효율성", dim_avgs["efficiency"], THRESHOLDS["efficiency"])
    with cols[3]:
        _render_metric_card("관련성", dim_avgs["relevance"], THRESHOLDS["relevance"])
    with cols[4]:
        _render_metric_card("일관성", dim_avgs["consistency"], THRESHOLDS["consistency"])
    with cols[5]:
        _render_metric_card("Health", session_health)

    st.divider()

    # ═══ 4D 점수 추이 ═══
    st.markdown("### 4D 품질 추이")
    st.plotly_chart(score_trend(turns), use_container_width=True)

    with st.expander("턴별 4D 품질 (펼쳐보기)", expanded=False):
        _render_quality_table(timeline, turns)

    st.divider()

    # ═══ 주의 필요 — 한눈 카드 (집계 → 상세) ═══
    st.markdown("### 주의 필요")

    _DIM_LABELS = {
        "completeness": "완전성",
        "efficiency": "효율성",
        "relevance": "관련성",
        "consistency": "일관성",
    }
    _DIM_TAB = {
        "completeness": "Tab 3 구성",
        "efficiency": "Tab 5 전달",
        "relevance": "Tab 5 전달",
        "consistency": "Tab 4 변형",
    }
    _EVENT_LABELS = {
        "regather": "재수집",
        "contradiction": "모순 감지",
        "source_conflict": "소스 충돌",
        "verify_fail": "검증 실패",
    }
    _EVENT_TAB = {
        "regather": "Tab 2 실행 흐름",
        "contradiction": "Tab 4 변형",
        "source_conflict": "Tab 4 변형",
        "verify_fail": "Tab 6 측정&진단",
    }

    # 집계: 차원별 / 이벤트 유형별 카운트
    dim_alerts: dict[str, list[dict]] = {}
    for turn in turns:
        tn = turn.get("turn_number", "?")
        scores = turn.get("scores", {})
        for dim in _DIM_LABELS:
            val = scores.get(SCORE_KEYS[dim])
            if val is None:
                continue
            if val < THRESHOLDS[dim]:
                dim_alerts.setdefault(dim, []).append({"turn": tn, "score": val})

    proc_groups: dict[str, list] = {}
    for evt in events:
        proc_groups.setdefault(evt["event_type"], []).append(evt)

    total_quality_issues = sum(len(v) for v in dim_alerts.values())
    total_proc_events = sum(len(v) for v in proc_groups.values())

    # 위험 카운트 — Rot Gate 트리거 턴 수 (Tab 5 SSOT 위임, 여기는 카운트만)
    rot_triggered_count = sum(
        1 for t in turns
        if t.get("metadata", {}).get(ATTRS["context.rot_gate_triggered"])
    )

    # 이탈 카운트 — Phase 3.7 이탈 감지 (Tab 6 측정&진단 §4 SSOT, 여기는 카운트만)
    _CONTINUITY_THRESHOLD = 0.5
    _ALIGNMENT_THRESHOLD = 0.7
    drift_count = sum(
        1 for t in turns
        if (
            # User Pivot: session_continuity가 있고 임계값 미달 (Turn 1 = None 제외)
            (
                t.get("metadata", {}).get(ATTRS["query.session_continuity"]) is not None
                and t.get("metadata", {}).get(ATTRS["query.session_continuity"]) < _CONTINUITY_THRESHOLD
            )
            # Agent Drift: 분석 또는 응답이 쿼리와 정렬 안 됨
            or (
                t.get("metadata", {}).get(ATTRS["analysis.query_alignment"]) is not None
                and t.get("metadata", {}).get(ATTRS["analysis.query_alignment"]) < _ALIGNMENT_THRESHOLD
            )
            or (
                t.get("metadata", {}).get(ATTRS["response.query_alignment"]) is not None
                and t.get("metadata", {}).get(ATTRS["response.query_alignment"]) < _ALIGNMENT_THRESHOLD
            )
        )
    )

    # 위험/이탈 턴 집합 — 테이블 행 구성 + 영향 턴 집계용
    rot_turns = {
        t.get("turn_number")
        for t in turns
        if t.get("metadata", {}).get(ATTRS["context.rot_gate_triggered"])
    }
    pivot_turns = {
        t.get("turn_number")
        for t in turns
        if (
            t.get("metadata", {}).get(ATTRS["query.session_continuity"]) is not None
            and t.get("metadata", {}).get(ATTRS["query.session_continuity"]) < _CONTINUITY_THRESHOLD
        )
    }
    agent_drift_turns = {
        t.get("turn_number")
        for t in turns
        if (
            (
                t.get("metadata", {}).get(ATTRS["analysis.query_alignment"]) is not None
                and t.get("metadata", {}).get(ATTRS["analysis.query_alignment"]) < _ALIGNMENT_THRESHOLD
            )
            or (
                t.get("metadata", {}).get(ATTRS["response.query_alignment"]) is not None
                and t.get("metadata", {}).get(ATTRS["response.query_alignment"]) < _ALIGNMENT_THRESHOLD
            )
        )
    }

    # 영향 턴 집합 — 품질·이벤트·위험·이탈 모두 포함
    affected_turns = set()
    for alerts in dim_alerts.values():
        for a in alerts:
            affected_turns.add(a["turn"])
    for evts in proc_groups.values():
        for e in evts:
            affected_turns.add(e["turn_number"])
    affected_turns |= rot_turns | pivot_turns | agent_drift_turns

    # 한눈 카운트 카드 — 항상 5개 표시 (이슈 없으면 green으로)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        color = STATUS_COLORS["bad"] if total_quality_issues > 0 else STATUS_COLORS["good"]
        st.markdown(
            f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
            f'border-radius:8px;border-left:4px solid {color};">'
            f'<div style="font-size:13px;color:#999;">품질 이슈</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};">'
            f'{total_quality_issues}</div>'
            f'<div style="font-size:11px;color:#888;">차원 {len(dim_alerts)}개</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        color = STATUS_COLORS["warn"] if total_proc_events > 0 else STATUS_COLORS["good"]
        st.markdown(
            f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
            f'border-radius:8px;border-left:4px solid {color};">'
            f'<div style="font-size:13px;color:#999;">프로세스</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};">'
            f'{total_proc_events}</div>'
            f'<div style="font-size:11px;color:#888;">유형 {len(proc_groups)}개</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        color = STATUS_COLORS["warn"] if affected_turns else STATUS_COLORS["good"]
        st.markdown(
            f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
            f'border-radius:8px;border-left:4px solid {color};">'
            f'<div style="font-size:13px;color:#999;">영향 턴</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};">'
            f'{len(affected_turns)}</div>'
            f'<div style="font-size:11px;color:#888;">/ {len(turns)} 턴</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        color = STATUS_COLORS["bad"] if rot_triggered_count > 0 else STATUS_COLORS["good"]
        st.markdown(
            f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
            f'border-radius:8px;border-left:4px solid {color};">'
            f'<div style="font-size:13px;color:#999;">Rot gate 활성화</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};">'
            f'{rot_triggered_count}</div>'
            f'<div style="font-size:11px;">'
            f'<a href="javascript:void(0)" '
            f'onclick="(function(){{var tabs=document.querySelectorAll(\'[data-baseweb=\\\"tab\\\"]\');'
            f'for(var i=0;i<tabs.length;i++){{if(tabs[i].textContent.includes(\'전달\')){{tabs[i].click();window.scrollTo(0,0);break;}}}}}})();" '
            f'style="color:#888;text-decoration:none;border-bottom:1px dotted #666;cursor:pointer;">'
            f'→ ③ 전달</a></div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        color = STATUS_COLORS["warn"] if drift_count > 0 else STATUS_COLORS["good"]
        st.markdown(
            f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
            f'border-radius:8px;border-left:4px solid {color};">'
            f'<div style="font-size:13px;color:#999;">이탈 (Drift)</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};">'
            f'{drift_count}</div>'
            f'<div style="font-size:11px;">'
            f'<a href="javascript:void(0)" '
            f'onclick="(function(){{var tabs=document.querySelectorAll(\'[data-baseweb=\\\"tab\\\"]\');'
            f'for(var i=0;i<tabs.length;i++){{if(tabs[i].textContent.includes(\'측정\')){{tabs[i].click();window.scrollTo(0,0);break;}}}}}})();" '
            f'style="color:#888;text-decoration:none;border-bottom:1px dotted #666;cursor:pointer;">'
            f'→ 측정 & 진단 §4</a></div></div>',
            unsafe_allow_html=True,
        )

    # ═══ 이슈 테이블 — 유형(품질/실행이상/위험/이탈) 셀 병합 + 상세위치 탭 이동 ═══
    if total_quality_issues > 0 or total_proc_events > 0 or rot_turns or pivot_turns or agent_drift_turns:
        all_turns_nums = sorted({t.get("turn_number", 0) for t in turns})

        # 탭 레이블 키워드 매핑: (검색 키워드, 표시 레이블)
        # 2026-05-31: Tab 8 '이탈 추적' 폐기 → 이탈 행은 Tab 6 측정&진단 §4로 라우팅
        _TAB_KEYWORD: dict[str, tuple[str, str]] = {
            "Tab 2 실행 흐름":  ("실행 흐름", "→ 실행 흐름"),
            "Tab 3 구성":       ("구성",       "→ ① 구성"),
            "Tab 4 변형":       ("변형",       "→ ② 변형"),
            "Tab 5 전달":       ("전달",       "→ ③ 전달"),
            "Tab 6 측정&진단":  ("측정",       "→ 측정 & 진단"),
        }

        # --- 셀 스타일 상수 ---
        _TD_CENTER = "padding:6px 10px;text-align:center;"
        _TD_LEFT   = "padding:6px 10px;text-align:left;"
        _TD_DIM    = "padding:6px 10px;color:#bbb;"
        _TD_GROUP  = (
            "padding:8px 10px;font-weight:700;color:#ccc;"
            "border-right:1px solid #444;vertical-align:middle;text-align:center;"
        )

        def _score_cells(alerts_by_turn: dict) -> str:
            """품질 행의 점수 셀 HTML을 반환한다."""
            out = []
            for tn in all_turns_nums:
                a = alerts_by_turn.get(tn)
                if a:
                    c = STATUS_COLORS["bad"] if a["score"] < 0.5 else STATUS_COLORS["warn"]
                    t = "심각" if a["score"] < 0.5 else "주의"
                    out.append(
                        f'<td style="{_TD_CENTER}color:{c};font-weight:600;">'
                        f'{t} ({a["score"]:.2f})</td>'
                    )
                else:
                    out.append(f'<td style="{_TD_CENTER}color:#555;">—</td>')
            return "".join(out)

        def _event_cells(evt_turns: set, color: str, sev_text: str) -> str:
            """이벤트 행의 셀 HTML을 반환한다."""
            out = []
            for tn in all_turns_nums:
                if tn in evt_turns:
                    out.append(
                        f'<td style="{_TD_CENTER}color:{color};font-weight:600;">'
                        f'{sev_text}</td>'
                    )
                else:
                    out.append(f'<td style="{_TD_CENTER}color:#555;">—</td>')
            return "".join(out)

        def _tab_link_td(tab_ref: str) -> str:
            """JS 탭 이동 링크 셀을 반환한다."""
            entry = _TAB_KEYWORD.get(tab_ref)
            kw, label = entry if entry else ("", tab_ref)
            return (
                f'<td style="{_TD_LEFT}">'
                f'<a href="javascript:void(0)" onclick="goToTab(\'{kw}\')" '
                f'style="color:#aaa;text-decoration:none;'
                f'border-bottom:1px dotted #666;cursor:pointer;">'
                f'{label}</a></td>'
            )

        # --- 행 구성 ---
        rows_html: list[str] = []
        quality_list = list(dim_alerts.items())   # [(dim, alerts), …]
        event_list   = list(proc_groups.items())  # [(etype, evts), …]

        # 품질 그룹 — "품질" 셀을 rowspan으로 병합
        for i, (dim, alerts) in enumerate(quality_list):
            alerts_by_turn = {a["turn"]: a for a in alerts}
            group_td = (
                f'<td rowspan="{len(quality_list)}" style="{_TD_GROUP}">품질</td>'
                if i == 0 else ""
            )
            rows_html.append(
                f'<tr style="border-bottom:1px solid #333;">'
                f'{group_td}'
                f'<td style="{_TD_DIM}">{_DIM_LABELS[dim]}</td>'
                f'{_score_cells(alerts_by_turn)}'
                f'{_tab_link_td(_DIM_TAB[dim])}'
                f'</tr>'
            )

        # 프로세스 그룹 — "프로세스" 셀을 rowspan으로 병합
        for i, (etype, evts) in enumerate(event_list):
            evt_turns = {e["turn_number"] for e in evts}
            color    = STATUS_COLORS["bad"] if etype == "verify_fail" else STATUS_COLORS["warn"]
            sev_text = "심각" if etype == "verify_fail" else "주의"
            group_td = (
                f'<td rowspan="{len(event_list)}" style="{_TD_GROUP}">프로세스</td>'
                if i == 0 else ""
            )
            rows_html.append(
                f'<tr style="border-bottom:1px solid #333;">'
                f'{group_td}'
                f'<td style="{_TD_DIM}">{_EVENT_LABELS.get(etype, etype)}</td>'
                f'{_event_cells(evt_turns, color, sev_text)}'
                f'{_tab_link_td(_EVENT_TAB.get(etype, "—"))}'
                f'</tr>'
            )

        # 위험 그룹 — Rot Gate 트리거
        if rot_turns:
            rot_color = STATUS_COLORS["bad"]
            rows_html.append(
                f'<tr style="border-bottom:1px solid #333;">'
                f'<td rowspan="1" style="{_TD_GROUP}">위험</td>'
                f'<td style="{_TD_DIM}">Rot gate 활성화</td>'
                f'{_event_cells(rot_turns, rot_color, "활성화")}'
                f'{_tab_link_td("Tab 5 전달")}'
                f'</tr>'
            )

        # 이탈 그룹 — User Pivot + Agent Drift (2026-05-31 Tab 8 폐기 후 Tab 6으로 라우팅)
        drift_rows: list[tuple[str, set, str]] = []
        if pivot_turns:
            drift_rows.append(("사용자 전환 (Pivot)", pivot_turns, STATUS_COLORS["warn"]))
        if agent_drift_turns:
            drift_rows.append(("에이전트 이탈 (Drift)", agent_drift_turns, STATUS_COLORS["warn"]))
        for i, (label, turns_set, d_color) in enumerate(drift_rows):
            group_td = (
                f'<td rowspan="{len(drift_rows)}" style="{_TD_GROUP}">이탈</td>'
                if i == 0 else ""
            )
            rows_html.append(
                f'<tr style="border-bottom:1px solid #333;">'
                f'{group_td}'
                f'<td style="{_TD_DIM}">{label}</td>'
                f'{_event_cells(turns_set, d_color, "감지")}'
                f'{_tab_link_td("Tab 6 측정&진단")}'
                f'</tr>'
            )

        # --- JS + 테이블 (stc.html로 script 실행) ---
        turn_headers = "".join(
            f'<th style="padding:6px 10px;text-align:center;color:#888;font-size:11px;">T{tn}</th>'
            for tn in all_turns_nums
        )
        js = """
        <script>
        function goToTab(keyword) {
            if (!keyword) return;
            const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
            for (const tab of tabs) {
                if (tab.textContent.includes(keyword)) {
                    tab.click();
                    window.parent.scrollTo(0, 0);
                    break;
                }
            }
        }
        </script>
        """
        table_html = (
            '<table style="width:100%;margin-top:4px;border-collapse:collapse;'
            'font-size:13px;color:#ccc;background:#111;">'
            '<thead><tr style="border-bottom:2px solid #555;">'
            '<th style="padding:6px 10px;text-align:center;color:#888;font-size:11px;'
            'border-right:1px solid #444;">유형</th>'
            '<th style="padding:6px 10px;text-align:left;color:#888;font-size:11px;">세부</th>'
            f'{turn_headers}'
            '<th style="padding:6px 10px;text-align:left;color:#888;font-size:11px;">상세 위치</th>'
            '</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody>'
            '</table>'
        )
        num_rows = len(quality_list) + len(event_list) + (1 if rot_turns else 0) + len(drift_rows)
        stc.html(js + table_html, height=56 + num_rows * 40)
