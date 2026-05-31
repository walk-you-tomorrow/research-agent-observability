"""
dashboard/views/fidelity.py — Tab 4: 변형 (Fidelity)

관측 질문 (Charter §3.2): "이전 턴 정보가 다음 턴으로 얼마나 살아남고 활용되는가?"

역할:
    변형 축의 상세 관측. 일관성 4패턴(A/B/C/D) + 의도 변형(I) + 분석-쿼리 정렬(Q)을
    정보 흐름의 인과 캐스케이드 순서로 배치.

    논리적 전개 — "컨텍스트 구축 → 저장 → 재호출 → 충돌 → 출력 → 의도 추적 → 분석 정렬":
    ① Iteration Consistency: 동일 턴 내 재수집 안정성 (패턴 A)
    ② Memory Fidelity: 결론 압축 + 보존 통합 (패턴 C-1)
    ③ Conclusion Propagation: 압축된 결론이 다음 턴에 얼마나 살아남는가 (패턴 C-2)
    ④ Contradiction Detection: 변형 과정에서 발생한 이전 결론 모순 (패턴 B)
    ⑤ Groundedness: 답변이 컨텍스트에 실제로 근거하는가 (패턴 D)
    ⑥ Intent Continuity: 사용자 의도가 턴 간 어떻게 변형되는가 (cross-turn intent cosine)
    ⑦ Query Alignment (분석): 분석 결과가 사용자 쿼리에 정렬되는가

    ※ 인과 전파(causal propagation) 섹션은 analysis/40에서 폐기 결정 — Tab 5 전달 탭으로 이전.
    ※ 패턴 A는 본래 Tab 5에 있었으나 일관성 4패턴 통일성을 위해 본 탭으로 이전 (충분성 평가 표는 Tab 5 유지).
    ※ ⑥⑦은 2026-05-31 Tab 8 '이탈 추적' 폐기에 따른 흡수
      — query.session_continuity / analysis.query_alignment 시계열 SSOT 이전.
        response.query_alignment 는 Tab 5 전달(⑧), Pattern I/II/III 진단은 Tab 6 측정&진단.

데이터 흐름:
    입력: turns (list[dict]) — enriched session data
    출력: Streamlit UI
"""
import pandas as pd
import streamlit as st

from agent.monitoring_schema import (
    ATTRS,
    DASHBOARD_THRESHOLDS,
    get_contradicts_from_metadata,
)
from dashboard.charts import (
    STATUS_COLORS,
    confidence_delta_bar,
    continuity_trend,
    contributing_flow_sankey,
    fidelity_detail_chart,
    query_alignment_trend,
)


# Fallback for stale Streamlit module cache — yaml/loader 갱신 전 import된 세션 보호용.
# 정상 재시작 후엔 DASHBOARD_THRESHOLDS["fidelity_good"] 등이 yaml SSOT를 따른다.
_FIDELITY_THR_FALLBACK = {"fidelity_good": 0.80, "fidelity_warn": 0.50}


def _status_for_fidelity(val: float | None) -> tuple[str, str, str]:
    """fidelity 값을 (color, icon, label)로 변환 — DASHBOARD_THRESHOLDS SSOT 사용."""
    if val is None:
        return STATUS_COLORS["na"], "—", "데이터 없음"
    good_thr = DASHBOARD_THRESHOLDS.get("fidelity_good", _FIDELITY_THR_FALLBACK["fidelity_good"])
    warn_thr = DASHBOARD_THRESHOLDS.get("fidelity_warn", _FIDELITY_THR_FALLBACK["fidelity_warn"])
    if val >= good_thr:
        return STATUS_COLORS["good"], "✓", "양호"
    if val >= warn_thr:
        return STATUS_COLORS["warn"], "⚠", "주의"
    return STATUS_COLORS["bad"], "✗", "경고"


def _render_fidelity_at_a_glance(turns: list[dict]) -> None:
    """변형 한눈 카드 4개 — fidelity 추세 + 인용/모순/환각 카운트.

    A4: 임계값 마이크로카피, ✓/⚠/✗ 아이콘 (WCAG 색약 보강).
    A5: '이전 턴 인용 발생'을 analysis.referenced_turns 기준으로 계산
        (이전 contributing_turns 기준은 "기여"이지 "인용"이 아님).
    """
    fid_attr = ATTRS["context.fidelity_score"]
    ref_attr = ATTRS["analysis.referenced_turns"]
    halluc_attr = ATTRS["response.hallucination_detected"]

    # --- ① 평균 Fidelity ---
    fids_ts: list[float] = []
    for t in turns:
        v = t.get("metadata", {}).get(fid_attr)
        if isinstance(v, (int, float)):
            fids_ts.append(float(v))
    avg_fid = sum(fids_ts) / len(fids_ts) if fids_ts else None
    fid_color, fid_icon, fid_label = _status_for_fidelity(avg_fid)

    # --- ② '이전 턴 인용 발생' (A5: referenced_turns 기준) ---
    referenced_count = sum(
        1 for t in turns
        if isinstance(t.get("metadata", {}).get(ref_attr), list)
        and t["metadata"][ref_attr]
    )
    # T≥2에서는 인용 0건이 결함 신호 (warn). 단일 턴 세션은 na.
    if len(turns) < 2:
        ref_color, ref_icon = STATUS_COLORS["na"], "—"
    elif referenced_count == 0:
        ref_color, ref_icon = STATUS_COLORS["warn"], "⚠"
    else:
        ref_color, ref_icon = STATUS_COLORS["good"], "✓"

    # --- ③ 모순 ---
    contradiction_count = sum(
        1 for t in turns
        if get_contradicts_from_metadata(t.get("metadata", {}))
    )
    contra_color = STATUS_COLORS["bad"] if contradiction_count else STATUS_COLORS["good"]
    contra_icon = "✗" if contradiction_count else "✓"

    # --- ④ 환각 ---
    halluc_count = sum(
        1 for t in turns
        if t.get("metadata", {}).get(halluc_attr)
    )
    halluc_color = STATUS_COLORS["bad"] if halluc_count else STATUS_COLORS["good"]
    halluc_icon = "✗" if halluc_count else "✓"

    fid_thr_good = DASHBOARD_THRESHOLDS.get("fidelity_good", _FIDELITY_THR_FALLBACK["fidelity_good"])
    fid_thr_warn = DASHBOARD_THRESHOLDS.get("fidelity_warn", _FIDELITY_THR_FALLBACK["fidelity_warn"])
    total_turns = len(turns)

    cards = [
        {
            "label": "평균 Fidelity",
            "value": f"{avg_fid:.2f}" if avg_fid is not None else "—",
            "color": fid_color,
            "icon": fid_icon,
            "status": fid_label,
            "hint": f"임계: ≥{fid_thr_good:.2f} 양호 · ≥{fid_thr_warn:.2f} 주의",
            "extra": "",
        },
        {
            "label": "이전 턴 인용 발생",
            "value": f"{referenced_count}/{total_turns}",
            "color": ref_color,
            "icon": ref_icon,
            "status": "정상" if ref_color == STATUS_COLORS["good"]
                      else "주의 (인용 0)" if ref_color == STATUS_COLORS["warn"]
                      else "—",
            "hint": "analysis.referenced_turns 비어있지 않은 턴 수",
            "extra": "",
        },
        {
            "label": "모순 감지",
            "value": str(contradiction_count),
            "color": contra_color,
            "icon": contra_icon,
            "status": "없음" if contradiction_count == 0 else "검토 필요",
            "hint": "이전 결론과 충돌 (패턴 B)",
            "extra": "",
        },
        {
            "label": "환각 감지",
            "value": str(halluc_count),
            "color": halluc_color,
            "icon": halluc_icon,
            "status": "없음" if halluc_count == 0 else "검토 필요",
            "hint": "근거 없는 주장 (패턴 D)",
            "extra": "",
        },
    ]

    cols = st.columns(4)
    for col, c in zip(cols, cards):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:14px;background:#1e1e1e;'
                f'border-radius:8px;border-left:4px solid {c["color"]};">'
                f'<div style="font-size:13px;color:#999;">{c["label"]}</div>'
                f'<div style="font-size:28px;font-weight:700;color:{c["color"]};'
                f'line-height:1.1;">{c["value"]}</div>'
                f'<div style="font-size:11px;color:#aaa;margin-top:2px;">{c["status"]}</div>'
                f'{c["extra"]}'
                f'<div style="font-size:10px;color:#777;margin-top:4px;">{c["hint"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render(turns: list[dict]) -> None:
    """변형 탭을 렌더링한다 — 일관성 4패턴 + 의도/쿼리 정렬 통합 7영역.

    영역 순서 (정보 흐름의 인과 캐스케이드 — 구축→저장→재호출→충돌→출력→의도→정렬):
        ① Iteration Consistency (재수집 안정성) — 패턴 A
        ② Memory Fidelity (압축 + 보존 통합) — 패턴 C-1
        ③ Conclusion Propagation (재호출 생존율) — 패턴 C-2
        ④ Contradiction Detection (모순) — 패턴 B
        ⑤ Groundedness (답변 근거성) — 패턴 D
        ⑥ Intent Continuity (의도 변형) — query.session_continuity 시계열
        ⑦ Query Alignment (분석) — analysis.query_alignment 시계열
    """
    from dashboard import tab_header
    tab_header.render("fidelity")

    if len(turns) < 2:
        st.info(
            "**첫 턴이므로 변형 관측 대상이 없습니다.**\n\n"
            "변형은 '이전 턴의 결론이 다음 턴에 전달되면서 어떻게 변하는가'를 관측합니다. "
            "2턴 이상의 세션에서 의미 있는 데이터가 생성됩니다."
        )
        return

    # 변형 한눈 카드 (heading 없이 카드만 노출)
    # 5.4 피드백: 변형 여정 표는 변별성 부족(new_data_ratio 0/25, fidelity_score unique=1.0)
    # 으로 폐기 → 카운트 카드로 대체.
    _render_fidelity_at_a_glance(turns)

    st.divider()

    # ═══ 변형 7영역 개요 테이블 (영역 진입 전 멘탈 모델 제시) ═══
    st.markdown("#### 변형 7영역 개요")
    pattern_overview = pd.DataFrame([
        {"패턴": "A", "영역": "① Iteration Consistency (A: 재수집)",
         "시점": "턴 N · Organize 내부 iteration",
         "관측 질문": "재수집 전후 신뢰도가 안정적으로 수렴하는가"},
        {"패턴": "C-1", "영역": "② Memory Fidelity (C-1: 압축)",
         "시점": "턴 N · Memory (결론 저장)",
         "관측 질문": "분석이 결론으로 압축될 때 무엇이 사라지는가"},
        {"패턴": "C-2", "영역": "③ Conclusion Propagation (C-2: 재호출)",
         "시점": "턴 N+1 · Organize (이전 결론 재호출)",
         "관측 질문": "저장된 결론이 다음 턴에 인용·활용되는가"},
        {"패턴": "B", "영역": "④ Contradiction Detection (B: 모순)",
         "시점": "턴 N+1 · Generate",
         "관측 질문": "이전 결론과 충돌이 발생했나, 해결됐나"},
        {"패턴": "D", "영역": "⑤ Groundedness (D: 근거)",
         "시점": "턴 N+1 · Generate/verify",
         "관측 질문": "답변이 컨텍스트에 실제로 근거하는가"},
        {"패턴": "I", "영역": "⑥ Intent Continuity (의도 변형)",
         "시점": "턴 N+1 · Plan",
         "관측 질문": "사용자 의도가 이전 턴 대비 얼마나 연속되는가"},
        {"패턴": "Q", "영역": "⑦ Query Alignment (분석 정렬)",
         "시점": "턴 N · Generate",
         "관측 질문": "분석 결과가 사용자 쿼리에 정렬되어 있는가"},
    ])
    st.dataframe(pattern_overview, use_container_width=True, hide_index=True)
    st.caption(
        "정보 흐름의 인과 캐스케이드: **구축(A) → 저장(C-1) → 재호출(C-2) → 충돌(B) → 출력(D) → 의도(I) → 정렬(Q)**. "
        "각 영역은 다음 영역의 전제가 되며, 앞 단계의 결함이 다음 단계로 전파된다. "
        "Pattern I/II/III 통합 진단은 **📊 측정 & 진단 탭 §4 쿼리 정렬 진단**에서 본다."
    )

    st.divider()

    # ═══ ① Iteration Consistency (재수집 안정성, 패턴 A) ═══
    # B2/B4: 패턴 A 신설 — Tab 5 충분성 평가 섹션에서 confidence_delta_bar 이전.
    # Tab 5에는 충분성 게이트 표가 남아있으며 패턴 A 진단은 이쪽 SSOT.
    st.markdown("### ① Iteration Consistency (A: 재수집)")
    st.markdown("#### 컨텍스트 구축이 안정적이었나 _(턴 N · Organize)_")
    # 패턴 매핑: 본 탭은 A(①) / C-1(②) / C-2(③) / B(④) / D(⑤) — 일관성 4패턴 전체 SSOT.
    st.markdown(
        "- **무엇**: 동일 턴 안에서 evaluate_context가 재수집을 반복할 때 신뢰도(`sufficiency_confidence`)가 어떻게 변하는가\n"
        "- **읽기**: **양수(+)** = 재수집 효과 / **음수(−)** = 추가 수집이 신뢰도를 흔듦 / 0 부근 = 수렴\n"
        "- **→ 다음**: 이 컨텍스트가 결론으로 압축될 때 **② Memory Fidelity**로 이어진다"
    )
    st.plotly_chart(confidence_delta_bar(turns), use_container_width=True)

    st.divider()

    # ═══ ② Memory Fidelity (압축 + 보존 통합, 패턴 C-1) ═══
    # A2: fidelity_trend 폐기 (fidelity_detail이 3요소로 이미 분해 표시).
    # 압축 표에서도 fidelity_score 컬럼 제거 (차트가 SSOT).
    st.markdown("### ② Memory Fidelity (C-1: 압축)")
    st.markdown("#### 결론이 저장될 때 무엇이 사라지는가 _(턴 N · Memory)_")
    st.markdown(
        "- **무엇**: 분석을 결론으로 압축할 때 손실(`lost_claims`) · 보존(`key_claims_preserved`) · 압축 적절성을 종합한 `fidelity_score`\n"
        "- **읽기**: 차트 3요소(조건보존 / 주장비율 / 압축적절성) 중 낮은 막대가 충실도 저하의 원인. 손실 주장 표에서 무엇이 빠졌는지 확인\n"
        "- **→ 다음**: 살아남은 결론은 **③ Conclusion Propagation**에서 다음 턴 인용 여부를 본다"
    )
    st.plotly_chart(fidelity_detail_chart(turns), use_container_width=True)

    # 압축 핵심 표 — fidelity_score 컬럼은 위 차트가 SSOT이므로 제거 (A2)
    compress_rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        claims = meta.get(ATTRS["response.key_claims"]) or []
        preserved = meta.get(ATTRS["response.key_claims_preserved"])
        lost = meta.get(ATTRS["response.lost_claims"]) or []
        summary = meta.get(ATTRS["response.conclusion_summary"]) or ""
        fid = meta.get(ATTRS["context.fidelity_score"])
        if not claims and preserved is None and not lost and not summary and fid is None:
            continue
        claims_count = len(claims) if isinstance(claims, list) else 0
        lost_count = len(lost) if isinstance(lost, list) else 0
        compress_rows.append({
            "턴": turn.get("turn_number", "?"),
            "key_claims 수": claims_count or "—",
            "보존 비율": f"{preserved:.2f}" if isinstance(preserved, (int, float)) else "—",
            "손실 주장 수": lost_count or "—",
            "결론 요약 길이": len(summary) if summary else "—",
        })
    if compress_rows:
        st.dataframe(pd.DataFrame(compress_rows), use_container_width=True, hide_index=True)

    # key_claims drill-down — 다음 턴에 무엇을 넘기는가 (5.6 피드백 명시)
    has_claims = any(
        isinstance(turn.get("metadata", {}).get(ATTRS["response.key_claims"]), list)
        and turn["metadata"][ATTRS["response.key_claims"]]
        for turn in turns
    )
    if has_claims:
        with st.expander("이 턴이 다음 턴에 넘기는 key_claims", expanded=False):
            st.caption(
                "압축 후 다음 턴의 컨텍스트로 상속되는 핵심 주장. "
                "다음 턴에서 이 항목들이 인용되면 ③ Conclusion Propagation 영역에 카운트됨."
            )
            for turn in turns:
                meta = turn.get("metadata", {})
                claims = meta.get(ATTRS["response.key_claims"]) or []
                if not isinstance(claims, list) or not claims:
                    continue
                turn_num = turn.get("turn_number", "?")
                st.markdown(f"**Turn {turn_num}** — {len(claims)}개 주장")
                for i, claim in enumerate(claims, 1):
                    st.markdown(f"{i}. {claim}")

    has_summary = any(
        turn.get("metadata", {}).get(ATTRS["response.conclusion_summary"])
        for turn in turns
    )
    if has_summary:
        with st.expander("턴별 결론 요약 (압축된 결론)", expanded=False):
            for turn in turns:
                meta = turn.get("metadata", {})
                summary = meta.get(ATTRS["response.conclusion_summary"]) or ""
                if not summary:
                    continue
                turn_num = turn.get("turn_number", "?")
                st.markdown(f"**Turn {turn_num}**")
                st.markdown(summary)
                st.divider()

    # 손실 주장
    lost_claims_rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        lost = meta.get(ATTRS["response.lost_claims"])
        if isinstance(lost, list) and lost:
            for claim in lost:
                lost_claims_rows.append({"턴": turn.get("turn_number", "?"), "손실된 주장": claim})
    if lost_claims_rows:
        st.markdown("**손실된 주장** (분석엔 있었으나 결론에서 탈락)")
        st.dataframe(pd.DataFrame(lost_claims_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ═══ ③ Conclusion Propagation (재호출, 패턴 C-2) ═══
    st.markdown("### ③ Conclusion Propagation (C-2: 재호출)")
    st.markdown("#### 저장된 결론이 다음 턴에 얼마나 살아남는가 _(턴 N+1 · Organize)_")
    st.markdown(
        "- **무엇**: ②에서 살아남은 결론을 다음 턴이 **명시 인용**(`referenced_turns`)하고 **실제 활용**(`utilized_conclusions`)했는가\n"
        "- **읽기**: Sankey 두께 = 흐른 결론 양 / 활용률(`conclusion_utilization`) ↓ = 인용은 했지만 분석에 도달 못함\n"
        "- **→ 다음**: 인용된 결론이 이전 결론과 충돌하면 **④ Contradiction**에 기록된다  \n"
        "_보충: 인과 기여도 상세는 🔍③ 전달 탭 ⑦ 전달 기여도 섹션 참조._"
    )

    # A1: 흐름 Sankey — Charter §3.2 "어디서 어디로 살아남는가"를 직접 시각화
    st.plotly_chart(contributing_flow_sankey(turns), use_container_width=True)

    ref_rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        contributing = meta.get(ATTRS["context.contributing_turns"])
        ref_turns = meta.get(ATTRS["analysis.referenced_turns"]) or []
        utilized = meta.get(ATTRS["analysis.utilized_conclusions"]) or []
        utilization = meta.get(ATTRS["analysis.conclusion_utilization"])
        if (contributing is None and not ref_turns
                and not utilized and utilization is None):
            continue
        ref_str = (
            ", ".join(f"T{t}" for t in ref_turns)
            if isinstance(ref_turns, list) and ref_turns else "—"
        )
        ut_count = len(utilized) if isinstance(utilized, list) else 0
        ut_str = f"{utilization:.2f}" if isinstance(utilization, (int, float)) else "—"
        ref_rows.append({
            "턴": turn.get("turn_number", "?"),
            "기여 이전 턴 수": contributing if contributing is not None else "—",
            "참조된 턴": ref_str,
            "활용된 결론 수": ut_count if ut_count else "—",
            "활용률": ut_str,
        })
    if ref_rows:
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
        st.caption(
            "**참조된 턴 vs 활용된 결론**: 참조는 명시 인용, 활용은 분석 단계 실제 사용. "
            "활용률(conclusion_utilization)이 낮으면 인용은 했지만 정보가 분석에 도달 못함."
        )
    else:
        st.info("Conclusion Propagation 데이터 없음 (이전 턴 결론을 인용한 흔적 없음)")

    st.divider()

    # ═══ ④ Contradiction Detection (모순, 패턴 B) ═══
    st.markdown("### ④ Contradiction Detection (B: 모순)")
    st.markdown("#### 이전 결론과 충돌했나, 해결됐나 _(턴 N+1 · Generate)_")
    st.markdown(
        "- **무엇**: 이전 턴 결론(t-1)과 현재 턴 분석(t) 사이의 모순 감지 + 해결 흐름\n"
        "- **읽기**: 표의 \"해결\" 컬럼 — **✓ 해결됨** = 충돌을 명시적으로 설명함 / **✗ 미해결** = 누수 위험. \"유형\"이 \"소스 간 충돌\"이면 수집 단계에서 이미 데이터가 어긋남\n"
        "- **→ 다음**: 미해결 모순은 **⑤ Groundedness** 저하로 이어질 수 있다"
    )
    from agent.monitoring_schema import (
        get_contradiction_resolved_from_metadata,
        get_previous_conclusion_from_metadata,
    )
    conflict_rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        if not get_contradicts_from_metadata(meta):
            continue
        resolved = get_contradiction_resolved_from_metadata(meta)
        prev_conclusion = get_previous_conclusion_from_metadata(meta) or "—"
        if len(prev_conclusion) > 80:
            prev_conclusion = prev_conclusion[:80] + "…"
        ref_turns = meta.get(ATTRS["analysis.referenced_turns"], [])
        ref_str = ", ".join(str(t) for t in ref_turns) if isinstance(ref_turns, list) and ref_turns else "—"
        conflict_rows.append({
            "턴": turn.get("turn_number", "?"),
            "유형": "이전 턴 모순",
            "해결": "✓ 해결됨" if resolved else "✗ 미해결",
            "참조 턴": ref_str,
            "요약": prev_conclusion,
        })
        # 소스 간 충돌 (분리 attribute)
        if meta.get(ATTRS["source.conflict_detected"]):
            conflict_rows.append({
                "턴": turn.get("turn_number", "?"),
                "유형": "소스 간 충돌",
                "해결": "—",
                "참조 턴": "—",
                "요약": "수집 데이터 간 수치 불일치 감지",
            })
    if conflict_rows:
        st.dataframe(pd.DataFrame(conflict_rows), use_container_width=True, hide_index=True)
    else:
        st.success("이번 세션은 턴 간 모순 또는 소스 충돌 없음")

    # ═══ ⑤ Groundedness — 답변이 컨텍스트에 근거하는가 (Pattern D) ═══
    # 5.7 피드백: 데이터 0/25 미수집 시 영역 자체 미표시 (조건부)
    grounded_rows = []
    for turn in turns:
        meta = turn.get("metadata", {})
        ratio = meta.get(ATTRS["response.grounded_claim_ratio"])
        halluc = meta.get(ATTRS["response.hallucination_detected"])
        ungrounded = meta.get(ATTRS["response.ungrounded_claims"]) or []
        if ratio is None and halluc is None and not ungrounded:
            continue
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "—"
        halluc_str = "⚠ 감지" if halluc else "✓ 없음" if halluc is False else "—"
        grounded_rows.append({
            "턴": turn.get("turn_number", "?"),
            "Grounded 비율": ratio_str,
            "환각": halluc_str,
            "근거 없는 주장 수": len(ungrounded) if isinstance(ungrounded, list) else 0,
        })
    # ⑤ Groundedness 섹션 — 5영역 통일 위계
    st.divider()
    st.markdown("### ⑤ Groundedness (D: 근거)")
    st.markdown("#### 답변이 컨텍스트에 근거했나 _(턴 N+1 · Generate/verify)_")
    st.markdown(
        "- **무엇**: 답변의 각 주장이 수집된 컨텍스트(`gathered_data`)에 실제로 근거하는지 검증\n"
        "- **읽기**: `grounded_claim_ratio` ↓ 또는 \"환각 ⚠ 감지\" = 컨텍스트와 답변 사이 누수. \"근거 없는 주장\" 컬럼은 정확히 어떤 주장이 검증 실패했는지 카운트\n"
        "- **위치**: 일관성 4패턴 캐스케이드의 **종착점** — ①~④의 모든 결함이 최종적으로 여기서 표출된다  \n"
        "_평가자: Cross-model (Agent와 다른 모델)로 self-referential bias 회피._"
    )
    if grounded_rows:
        st.dataframe(pd.DataFrame(grounded_rows), use_container_width=True, hide_index=True)
    else:
        st.info("미수집 — P0' Groundedness Checker 보강 후 활성화 예정 (analysis/39)")

    # ═══ ⑥ Intent Continuity (의도 변형, Tab 8 흡수) ═══
    st.divider()
    st.markdown("### ⑥ Intent Continuity (I: 의도 변형)")
    st.markdown("#### 사용자 의도가 턴 간 어떻게 변형되는가 _(턴 N+1 · Plan)_")
    st.markdown(
        "- **무엇**: `query.session_continuity` — 이전 N턴 intent ↔ 현재 intent의 cosine 유사도 (`analyze_query` 노드 산출)\n"
        "- **읽기**: 1.0에 가까울수록 의도 연속, **0.5 미만** = 사용자 Pivot 감지 (빨간 점). Turn 1은 비교 대상이 없어 항상 null → 차트 제외\n"
        "- **→ 다음**: Pivot이 발생한 턴에서 `analyze_query`가 새 intent를 정확히 감지했는지 **⑦ Query Alignment**로 확인.\n"
        "  Pattern I/II/III 통합 판정은 **📊 측정 & 진단 탭 §4 쿼리 정렬 진단**"
    )
    st.plotly_chart(continuity_trend(turns), use_container_width=True,
                    key="tab4_intent_continuity_trend")

    # ═══ ⑦ Query Alignment 분석 (Tab 8 흡수) ═══
    st.divider()
    st.markdown("### ⑦ Query Alignment (Q: 분석 정렬)")
    st.markdown("#### 분석 결과가 사용자 쿼리에 정렬되어 있는가 _(턴 N · Generate)_")
    st.markdown(
        "- **무엇**: `analysis.query_alignment` — 분석 산출물이 사용자 쿼리 intent와 얼마나 정렬됐는지 (`generate_analysis` 노드 산출)\n"
        "- **읽기**: **0.7 미만** = Agent Drift 의심 (주황 × 마커). 분석은 했지만 사용자가 물은 것에서 벗어남\n"
        "- **응답 단계 정렬**: `response.query_alignment` 추이는 **🔍③ 전달 탭 §⑧ Query Alignment (응답)** 에서 본다  \n"
        "_평가자: run_evaluation.py 실행 시 채워짐._"
    )
    st.plotly_chart(query_alignment_trend(turns, kind="analysis"),
                    use_container_width=True,
                    key="tab4_query_alignment_analysis_trend")
