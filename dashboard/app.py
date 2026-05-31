"""
dashboard/app.py — Context Monitoring Dashboard v3 진입점

차터 정의:
    "AI Agent의 실행 과정 전체에 걸쳐 컨텍스트가 어떻게 구성·변형·전달되는지를
    관측하고, 그 품질을 측정·진단할 수 있게 하는 것"

8탭 구조 (차터 행위 순서, 2026-05-31 Tab 8 이탈 추적 흡수 + Tab 9 분석 폐기):
    1. 한눈에 (Entry Point)
    2. 실행 흐름 — "실행 과정 전체에 걸쳐" (뼈대)
    3. 구성 — "구성" (관측 축 1)
    4. 변형 — "변형" (관측 축 2, ⑥ Intent Continuity + ⑦ Query Alignment 분석 흡수)
    5. 전달 — "전달" (관측 축 3, ⑧ Query Alignment 응답 흡수)
    6. 측정 & 진단 — "품질을 측정·진단" + §4 쿼리 정렬 진단 (Pattern I/II/III 흡수)
    7. 상세 — LLM 호출 로그 + 노드 Contribution (세션 운영 데이터)
    8. 속성 카탈로그 — YAML SSOT 기반 70개 속성 정의 (세션 독립)

    ※ 이전 Tab 8 '이탈 추적'은 2026-05-31에 Tab 4/5/6 분산 흡수로 폐기.
       각 결함의 시계열 추적은 해당 관측 축 탭에서, Pattern 통합 판정은
       측정 & 진단 탭에서 한다 (SSOT 원칙).
    ※ 이전 Tab 9 '분석'(Attribute ↔ 4D Spearman 상관)은 2026-05-31 사용자 요청으로 폐기.

실행 방법:
    cd observable-research-agent && source venv/bin/activate
    streamlit run dashboard/app.py --server.port 8501
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from dashboard.data_loader import (
    check_connection,
    list_recent_sessions,
    load_enriched_session_data,
)
from dashboard.views import (
    composition,
    detail_log,
    fidelity,
    impact,
    measure_diagnose,
    process_observe,
    schema_browser,
    session_overview,
)

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Context Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════
# 사이드바: Langfuse 연결 + 세션 필터 + 선택
# ═══════════════════════════════════════

with st.sidebar:
    st.title("Context Monitoring")
    st.caption(
        "컨텍스트의 구성·변형·전달을 관측하고, "
        "품질을 측정·진단합니다."
    )

    # Langfuse 연결 상태
    connected = check_connection()
    if connected:
        st.success("Langfuse 연결됨", icon="✅")
    else:
        st.error("Langfuse 연결 실패", icon="❌")
        st.info("환경변수: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST")

    st.divider()

    # 세션 필터링
    sessions = list_recent_sessions(limit=50) if connected else []

    if sessions:
        # ID 검색 필터
        search_id = st.text_input("세션 ID 검색", placeholder="sess_...")

        # 필터 적용
        filtered = sessions
        if search_id:
            filtered = [s for s in filtered if search_id.lower() in s["id"].lower()]

        if filtered:
            selected_session = st.selectbox(
                "세션 선택",
                options=[s["id"] for s in filtered],
                format_func=lambda x: x,
            )
        else:
            selected_session = None
            st.warning("검색 결과 없음")
    else:
        selected_session = None
        if connected:
            st.warning("세션이 없습니다")

    # 새로고침 버튼
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════
# 메인 영역: enriched 데이터 로드 + 9개 탭
# ═══════════════════════════════════════

if selected_session:
    with st.spinner(f"세션 데이터 로딩 중... ({selected_session})"):
        turns = load_enriched_session_data(selected_session)

    if turns is None:
        st.error(f"세션 '{selected_session}'의 데이터를 로드할 수 없습니다.")
    elif len(turns) == 0:
        st.warning(f"세션 '{selected_session}'에 턴 데이터가 없습니다.")
    else:
        # 8개 탭 — 차터 행위 순서: 관측 → 측정 → 진단 → 원본 참조
        # (Tab 8 '이탈 추적' = 2026-05-31 Tab 4/5/6 분산 흡수로 폐기)
        # (Tab 9 '분석'      = 2026-05-31 사용자 요청으로 폐기)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "👁️ 한눈에",
            "🔄 실행 흐름",
            "🔍① 구성",
            "🔍② 변형",
            "🔍③ 전달",
            "📊 측정 & 진단",
            "📋 상세",
            "🗂 속성 카탈로그",
        ])

        with tab1:
            session_overview.render(turns, selected_session)

        with tab2:
            process_observe.render(turns)

        with tab3:
            composition.render(turns)

        with tab4:
            fidelity.render(turns)

        with tab5:
            impact.render(turns)

        with tab6:
            measure_diagnose.render(turns, selected_session)

        with tab7:
            detail_log.render(turns)

        with tab8:
            schema_browser.render()

else:
    st.info("사이드바에서 세션을 선택하세요.")
