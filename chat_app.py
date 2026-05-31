"""
chat_app.py — 서울 상권 분석 에이전트 고객용 채팅 앱 (Streamlit)

역할:
    고객이 채팅창에서 상권/창업 질문을 입력하고 즉시 응답을 받는 대화형 웹 UI.
    관측 대시보드(dashboard/app.py)와 분리된 별도 Streamlit 프로세스로,
    고객 화면에는 내부 모니터링 용어(rot/fidelity/4D)를 노출하지 않는다.

실행 방법:
    streamlit run chat_app.py --server.port 8501
    (관측 대시보드는 별도: streamlit run dashboard/app.py --server.port 8502)

데이터 흐름:
    st.chat_input(질문)
      → turn_runner.run_turn(graph, agent_session, query, turn_number, config)  # 응답 생성
      → st.chat_message(응답 표시) + 참고 소스 캡션
      → 백그라운드 스레드: turn_runner.evaluate_and_diagnose(...)  # 4D 평가/진단 비동기 기록

설계 원칙:
    - 응답을 먼저 보여주고 4D 평가는 백그라운드로 미뤄 고객 체감 지연을 최소화한다.
    - 멀티턴 컨텍스트는 turn_runner가 carry-over한 session_state를 st.session_state에 보관해 유지한다.
"""
import threading
import uuid

import truststore
from dotenv import load_dotenv

# macOS 시스템 키체인 인증서 사용 (회사 프록시 SSL 호환) — agent 모듈 import 전에 실행
truststore.inject_into_ssl()
load_dotenv()

import streamlit as st  # noqa: E402

from agent.graph import build_turn_graph  # noqa: E402
from agent.turn_runner import (  # noqa: E402
    evaluate_and_diagnose,
    init_session_state,
    run_turn,
)
from main import load_config  # noqa: E402

# --- 도구 → 고객 친화적 소스 라벨 매핑 ---
# 내부 도구명을 고객이 이해할 수 있는 데이터 출처로 변환한다 (원본 데이터는 노출하지 않음).
_TOOL_SOURCE_LABELS: dict[str, str] = {
    "rag_search": "상권분석보고서",
    "rag_deep_read": "상권분석보고서",
    "rag_global_summary": "상권분석보고서",
    "rag_compare": "상권분석보고서",
    "pandas_query": "공공 통계 데이터",
    "calculate": "공공 통계 데이터",
    "web_search": "웹 검색(실시간)",
    "api_query": "서울시 상권 API",
    "lookup_previous": "이전 분석 결과",
}

WELCOME = (
    "안녕하세요! 서울 상권 분석 상담 챗봇입니다. 🏙️\n\n"
    "창업 입지, 업종 트렌드, 유동인구, 임대료 등 궁금한 점을 자유롭게 물어보세요.\n"
    "예: *\"강남구에서 카페 창업하기 좋은 동네는?\"*"
)


@st.cache_resource(show_spinner="에이전트 그래프를 준비하는 중...")
def _get_graph():
    """LangGraph 그래프를 1회만 빌드하여 캐시한다 (세션 간 공유)."""
    return build_turn_graph()


@st.cache_resource(show_spinner=False)
def _get_config() -> dict:
    """agent_config.yaml을 1회만 로드하여 캐시한다."""
    return load_config()


def _source_caption(result: dict) -> str | None:
    """이번 턴이 참고한 데이터 출처를 고객 친화적 캡션으로 만든다.

    tools_called를 친화적 라벨로 변환하고 중복을 제거한다. 원본 데이터/행 수는 노출하지 않는다.

    Args:
        result: graph.invoke()의 반환값.

    Returns:
        "참고 소스: …" 캡션 문자열, 또는 표시할 소스가 없으면 None.
    """
    tools = result.get("tools_called", []) or []
    seen: list[str] = []
    for tool in tools:
        label = _TOOL_SOURCE_LABELS.get(tool)
        if label and label not in seen:
            seen.append(label)
    if not seen:
        return None
    return "참고 소스: " + ", ".join(seen)


def _start_background_eval(config: dict, trace_id: str, result: dict, prev_diagnosis: list) -> None:
    """4D 평가/진단을 백그라운드 스레드에서 비동기로 수행한다 (fire-and-forget).

    스레드는 st.*에 접근하지 않고 Langfuse 기록만 수행하므로 ScriptRunContext가 필요 없다.
    결과는 관측 대시보드(dashboard/app.py)에서 동일 session_id로 사후 확인한다.
    """
    if not trace_id:
        return

    def _worker() -> None:
        try:
            evaluate_and_diagnose(config, trace_id, result, prev_diagnosis)
        except Exception as exc:  # 고객 경로를 막지 않도록 스레드 내 예외는 삼킨다 (로그만)
            print(f"[chat_app] 백그라운드 평가 실패: {exc}")

    threading.Thread(target=_worker, name="eval_and_diagnose", daemon=True).start()


def _init_state() -> None:
    """st.session_state를 초기화한다 (최초 진입 또는 '새 대화' 시)."""
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    st.session_state.session_id = session_id
    st.session_state.agent_session = init_session_state(session_id)
    st.session_state.chat_history = []  # 표시용 [{role, content, caption}]
    st.session_state.turn_number = 0


def main() -> None:
    """채팅 앱 진입점 — 페이지 구성 + 채팅 루프."""
    st.set_page_config(page_title="서울 상권 분석 상담", page_icon="🏙️", layout="centered")

    if "agent_session" not in st.session_state:
        _init_state()

    # --- 사이드바: 세션 정보 + 새 대화 ---
    with st.sidebar:
        st.markdown("### 서울 상권 분석 상담")
        st.caption(f"세션 ID: `{st.session_state.session_id}`")
        st.caption(f"진행 턴: {st.session_state.turn_number}")
        if st.button("🆕 새 대화 시작", use_container_width=True):
            _init_state()
            st.rerun()
        st.divider()
        st.caption(
            "공공 통계 · 상권분석보고서 · 서울시 API · 실시간 웹 검색을 종합해 답변합니다."
        )

    st.title("🏙️ 서울 상권 분석 상담")

    # --- 지난 대화 렌더링 ---
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(WELCOME)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("caption"):
                st.caption(msg["caption"])

    # --- 사용자 입력 처리 ---
    prompt = st.chat_input("상권/창업에 대해 물어보세요")
    if not prompt:
        return

    # 사용자 메시지 즉시 표시 + 기록
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 턴 실행 (응답 생성)
    config = _get_config()
    graph = _get_graph()
    st.session_state.turn_number += 1
    turn_number = st.session_state.turn_number
    prev_diagnosis = st.session_state.agent_session.get("previous_diagnosis", [])

    with st.chat_message("assistant"):
        with st.spinner("상권 데이터를 분석하는 중..."):
            turn_out = run_turn(
                graph,
                st.session_state.agent_session,
                prompt,
                turn_number,
                config,
            )
        response = turn_out["response"]
        caption = _source_caption(turn_out["result"])
        st.markdown(response)
        if caption:
            st.caption(caption)

    # 다음 턴으로 컨텍스트 이월 (carry-over된 새 session_state 보관)
    st.session_state.agent_session = turn_out["session_state"]
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response, "caption": caption}
    )

    # 4D 평가/진단은 백그라운드로 (고객 응답을 막지 않음)
    _start_background_eval(config, turn_out["trace_id"], turn_out["result"], prev_diagnosis)


if __name__ == "__main__":
    main()
