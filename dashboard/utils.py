"""dashboard/utils.py — Dashboard 공통 유틸리티

역할:
    여러 view에서 중복 사용되는 공통 함수를 제공한다.
    session_overview, measure_diagnose 등에서 동일한 로직이 반복되지 않도록 통합.

데이터 흐름:
    입력: 점수값, 라벨 등
    출력: 상태 문자열, Streamlit HTML 렌더링
"""
import streamlit as st

from dashboard.charts import STATUS_COLORS


def _score_status(val: float | None, threshold: float = 0.7) -> str:
    """점수의 상태를 반환한다.

    Args:
        val: 0~1 점수 (None이면 na).
        threshold: 정상/주의 경계값 (기본 0.7).

    Returns:
        "good" / "warn" / "bad" / "na"
    """
    if val is None:
        return "na"
    if val >= threshold:
        return "good"
    if val >= 0.5:
        return "warn"
    return "bad"


def _render_metric_card(label: str, val: float | None, threshold: float = 0.7) -> None:
    """색상이 적용된 메트릭 카드를 HTML로 렌더링한다.

    Args:
        label: 지표 이름.
        val: 0~1 점수 (None이면 N/A 표시).
        threshold: 정상/주의 경계값.
    """
    status = _score_status(val, threshold)
    color = STATUS_COLORS[status]
    display = f"{val * 100:.0f}" if val is not None else "N/A"
    html = f"""
    <div style="text-align:center; padding:8px 0;">
        <div style="font-size:14px; color:#888; margin-bottom:4px;">{label}</div>
        <div style="font-size:32px; font-weight:700; color:{color};">{display}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
