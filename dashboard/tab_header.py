"""
dashboard/tab_header.py — 3축 탭 공통 헤더 컴포넌트

★ Charter 정렬 — 자연스러운 컨셉 흐름 ★

역할:
    Tab 3/4/5 (구성/변형/전달) 헤더에 자연스러운 축 라벨을 표시한다.
    4D 차원은 콘텐츠 자체가 자연스럽게 represent하므로 명시 라벨 X.

데이터 흐름:
    입력: axis 키 ("composition" / "fidelity" / "impact")
    출력: Streamlit UI (제목 + Charter § 인용 한 줄)

원칙:
    - 4D 명시 라벨/배지는 강요하지 않음
    - 콘텐츠 흐름이 자연스럽게 차원 represent
    - 사용자가 5초 안에 "이 탭은 무엇을 관측하는가" 인지
"""
import streamlit as st


_AXIS_INFO = {
    "composition": {
        "icon": "🔍①",
        "label": "구성 (Composition)",
        "charter_ref": "§ 3.1",
        "charter_quote": "컨텍스트 입력 구성과 소스 분포의 변동 추적",
    },
    "fidelity": {
        "icon": "🔍②",
        "label": "변형 (Fidelity)",
        "charter_ref": "§ 3.2",
        "charter_quote": "이전 턴 정보가 다음 턴으로 어떻게 변형되며 살아남는가",
    },
    "impact": {
        "icon": "🔍③",
        "label": "전달 (Impact)",
        "charter_ref": "§ 3.3",
        "charter_quote": "누적 컨텍스트가 결과 품질에 미치는 영향",
    },
}


def render(axis: str) -> None:
    """3축 탭 헤더 렌더링.

    Args:
        axis: "composition" / "fidelity" / "impact"

    예시 출력:
        🔍① 구성 (Composition)
        Charter § 3.1: "무엇이 들어오고, 빠지고, 비율이 어떻게 이동하는가"
    """
    info = _AXIS_INFO.get(axis)
    if not info:
        return
    st.markdown(f"### {info['icon']} {info['label']}")
    st.caption(f"Charter {info['charter_ref']}: \"{info['charter_quote']}\"")
