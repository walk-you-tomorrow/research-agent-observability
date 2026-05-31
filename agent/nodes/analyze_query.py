"""
agent/nodes/analyze_query.py — 노드 1: 사용자 질의 분석

역할:
    사용자의 자연어 질문을 분석하여 의도(intent), 필요한 데이터, 호출할 도구 계획을 생성한다.
    이 노드의 출력(query_analysis)은 이후 모든 노드에서 참조된다.

프로세스 단계: ① Plan (입력 분석 단계, 아직 컨텍스트를 수집하기 전)
품질 차원: — (직접 측정 없음, 다만 query.intent 등이 관련성 평가에 간접 활용)

데이터 흐름:
    입력: state["user_query"], state["turn_conclusions"] (이전 턴 결론),
          state["session_intent_history"] (이탈 감지용 intent 이력)
    출력: state["query_analysis"] → {intent, required_data, required_docs, tool_plan, ...}
          state["session_intent_history"] → turn별 intent 누적 기록

Langfuse 기록:
    query.intent: 사용자 의도
    query.required_data_types: 필요한 CSV 데이터 키
    query.tool_plan: 호출 예정 도구 목록
    query.references_previous_turn: 이전 턴 참조 여부
    query.user_query: 사용자 질문 원문 (alignment judge 입력용)
    query.session_continuity: 이전 N턴 intent 대비 현재 intent 방향 유사도 (User Pivot 감지)
"""
import json

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, observe

from agent.llm import create_llm, invoke_with_retry
from agent.models import QueryAnalysis
from agent.monitoring_schema import ATTRS, _SCHEMA
from agent.parser import parse_llm_json
from agent.redundancy_checker import _try_embed

# --- session_continuity 계산 윈도우 ---
# monitoring_schema.yaml constants.session_continuity_window 값을 SSOT로 사용한다.
# 미설정 시 기본값 5 (일반적인 대화 맥락 추적 범위).
_SESSION_CONTINUITY_WINDOW: int = (
    _SCHEMA.get("constants", {}).get("session_continuity_window", 5)
)


def compute_session_continuity(
    session_intent_history: list[dict],
    window: int = 5,
) -> float | None:
    """이전 N턴 intent centroid 대비 현재 intent의 cosine similarity를 계산한다.

    이탈 감지(User Pivot): 사용자가 갑자기 전혀 다른 주제로 전환했는지 확인한다.
    낮은 값(0에 가까움) → 이탈 가능성, 높은 값(1에 가까움) → 이전 맥락 연속.

    Args:
        session_intent_history: [{turn_number, intent}, ...] 형식의 intent 이력.
        window: 비교 대상 최대 이전 턴 수 (yaml constants.session_continuity_window).

    Returns:
        cosine similarity (float, 0~1). Turn 1이거나 Ollama 미가용 시 None.
    """
    if len(session_intent_history) < 2:
        # Turn 1: 비교 대상 없음 → null
        return None
    history = session_intent_history[-window:]
    texts = [h["intent"] for h in history]
    embeddings = _try_embed(texts)
    if embeddings is None:
        # Ollama 미가용 → fallback 없음, null 반환
        return None
    current_vec = np.array(embeddings[-1])
    centroid = np.array(embeddings[:-1]).mean(axis=0)
    similarity = float(
        np.dot(current_vec, centroid)
        / (np.linalg.norm(current_vec) * np.linalg.norm(centroid) + 1e-8)
    )
    return round(similarity, 4)


# --- 시스템 프롬프트 ---
# LLM에게 "질문 분석 모듈" 역할을 부여하고, 사용 가능한 데이터/도구 목록을 제공한다.
# LLM은 이 정보를 바탕으로 어떤 도구를 호출하고 어떤 데이터가 필요한지 계획한다.
# 응답은 반드시 JSON 형식이어야 한다 (QueryAnalysis 모델로 파싱하기 위해).
SYSTEM_PROMPT = """당신은 서울 상권 데이터 분석 Agent의 질문 분석 모듈입니다.

사용자의 질문을 분석하여 필요한 데이터와 tool 계획을 수립하세요.

**4개 데이터 소스:**
1. CSV 데이터 (pandas_query로 직접 쿼리, Knowledge Graph 비경유):
   - store_info.csv: 상가정보 (5개 구: 강남, 마포, 서초, 종로, 영등포)
   - foot_traffic.csv: 유동인구 (시간대·연령대)
   - rent.csv: 임대료 (구 수준)
   - demographics.csv: 인구통계
   - business_codes.csv: 업종코드
   - dong_summary.csv: 동별 집계
   - card_consumption.csv: 카드 소비
   - estimated_sales.csv: 추정 매출
   - subway_ridership.csv: 지하철 승하차

2. 도메인 문서 (LightRAG Knowledge Graph로 시맨틱 검색):
   - 상권분석보고서 PDF (구별 상권분석, 소상공인 정책, 금융리포트 등)
   - 분석 방법론, 정책 맥락, 트렌드 인사이트 제공

3. 웹 검색 (web_search로 실시간 정보):
   - 최신 뉴스, 시장 트렌드, 정책 변화
   - CSV/KG에 없는 최신 정보 보충

4. 서울시 공공 API (api_query로 런타임 조회):
   - estimated_sales, commercial_change, store_openclose, crowd_facility, resident_population

사용 가능한 tool (9종):
- rag_search: LightRAG 시맨틱 검색. 도메인 지식 탐색에 사용. (mix 모드)
- rag_deep_read: 특정 동/구/업종에 대한 상세 검색. (local 모드)
- rag_global_summary: 전역 패턴, 트렌드, 공통점 요약. (global 모드)
- rag_compare: 두 개 이상의 구/동/업종 비교 분석. (hybrid 모드)
- pandas_query: CSV에서 정확한 수치/집계가 필요할 때 pandas로 직접 쿼리.
- calculate: 수식 계산, 수치 검증.
- lookup_previous: 이전 턴 결론 조회.
- api_query: 서울시 상권분석서비스 API 호출 (CSV에 없는 데이터를 런타임 조회)
  api_query 사용 시 api_params를 반드시 포함:
  {"api": "commercial_change", "params": {"STDR_YYQU_CD": "20244"}}
- web_search: 서울 상권 관련 최신 뉴스, 트렌드, 시장 변화를 실시간 검색.
  CSV/API에 없는 최신 정보가 필요할 때 사용.

**도구 선택 가이드:**
- 도메인 지식, 분석 방법론, 정책 맥락 → rag_search, rag_deep_read
- 전체 트렌드, 패턴, 요약 → rag_global_summary
- 구 vs 구, 동 vs 동 비교 → rag_compare
- 정확한 수치/통계 집계 → pandas_query
- 수식 계산이나 파생 지표 → calculate
- 이전 분석 결과 참조 → lookup_previous
- CSV에 없는 실시간 공공데이터 → api_query
- 최신 뉴스, 트렌드, 정책 변화 → web_search

**소스 선택 규칙:**
각 질문에 대해 어떤 소스 유형이 필요한지 판단하세요:
- csv: 정확한 수치, 집계, 통계 → pandas_query, calculate
- rag: 도메인 지식, 분석 방법론, 정책 맥락 → rag_search, rag_deep_read, rag_global_summary, rag_compare
- web: 최신 뉴스, 트렌드, 실시간 정보 → web_search
- api: CSV에 없는 공공 데이터 → api_query
source_types에 필요한 소스 유형을, source_reasoning에 선택 근거를 기록하세요.

**이전 턴 참조 감지 규칙:**
다음 표현이 질문에 포함되면 반드시 references_previous=true로 설정하세요:
- 시간 참조: "아까", "방금", "전에", "앞에서", "위에서", "이전에", "처음에"
- 결론 참조: "~라고 했는데", "~다고 했잖아", "~라고 했지", "~라고 분석했는데"
- 수정 요청: "다시 보면", "재고하면", "고려하면 달라지지", "잠깐", "근데"
- 대명사 참조: "그거", "그때", "거기", "그 데이터"
이전 턴 결론이 존재하고 질문이 기존 결론에 이의를 제기하거나 수정을 요청하면,
해당 턴 번호를 referenced_turns에 포함하세요.

JSON으로만 응답:
{
  "intent": "compare_districts",
  "required_data": ["foot_traffic", "store_info"],
  "required_docs": [],
  "tool_plan": ["rag_compare", "pandas_query"],
  "references_previous": true,
  "referenced_turns": [2],
  "api_params": null,
  "source_types": ["csv", "rag"],
  "source_reasoning": "구 비교에 정확한 수치(csv)와 도메인 맥락(rag)이 필요"
}

web_search가 필요한 경우의 예시:
{
  "intent": "cafe_market_trend",
  "required_data": ["store_info"],
  "required_docs": [],
  "tool_plan": ["pandas_query", "rag_search", "web_search"],
  "references_previous": false,
  "referenced_turns": [],
  "api_params": null,
  "source_types": ["csv", "rag", "web"],
  "source_reasoning": "카페 시장 현황은 csv, 도메인 분석은 rag, 최신 트렌드는 web에서 확인"
}
"""

# --- 한국어 이전 턴 참조 표현 패턴 ---
# LLM이 한국어 맥락 단서를 놓칠 수 있으므로, 키워드 기반 폴백으로 보완한다.
# 이 패턴에 매칭되면 references_previous=True로 강제 설정한다.
KOREAN_REFERENCE_PATTERNS = [
    "아까",        # "아까 합정이 낫다고"
    "방금",        # "방금 말한"
    "앞에서",      # "앞에서 분석한"
    "위에서",      # "위에서 말한"
    "이전에",      # "이전에 추천한"
    "처음에",      # "처음에 말했던"
    "라고 했는데",  # "~라고 했는데"
    "다고 했는데",  # "~다고 했는데"
    "라고 했지",   # "~라고 했지"
    "다시 보면",   # "다시 보면"
    "재고하면",    # "재고하면"
    "잠깐",        # "잠깐, 근데"
]


@observe(name="analyze_query")
def analyze_query(state: dict) -> dict:
    """사용자 질의를 분석하여 intent, 필요 데이터, 도구 계획을 생성한다.

    Args:
        state: 현재 AgentState. user_query와 turn_conclusions를 참조한다.

    Returns:
        {"query_analysis": dict} — QueryAnalysis 모델의 dict 변환 결과.

    처리 과정:
        1. 이전 턴 결론이 있으면 맥락 정보로 추가 (최근 3턴까지)
        2. LLM에 시스템 프롬프트 + 사용자 질문을 전달
        3. JSON 응답을 QueryAnalysis 모델로 파싱
        4. 파싱 실패 시 1회 재시도 (LLM에게 JSON 형식을 다시 요청)
        5. 2회 모두 실패 시 기본값(general_query)으로 폴백
        6. Langfuse에 분석 결과를 메타데이터로 기록
    """
    llm = create_llm()

    # --- 이전 턴 결론을 맥락으로 제공 ---
    # 다중 턴 대화에서 이전 턴의 결론을 참조할 수 있도록 LLM에 컨텍스트를 제공한다.
    # 최근 3턴만 포함하여 프롬프트 길이를 관리한다.
    turn_context = ""
    if state.get("turn_conclusions"):
        conclusions = state["turn_conclusions"][-3:]  # 최근 3턴만
        turn_context = "\n\n이전 턴 결론:\n" + json.dumps(
            conclusions, ensure_ascii=False, indent=2
        )

    # LLM에 전달할 메시지 구성
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"질문: {state['user_query']}{turn_context}"),
    ]

    # --- JSON 파싱 재시도 로직 ---
    # LLM이 가끔 비정상적인 JSON을 반환할 수 있으므로 최대 2회 시도한다.
    # 1차 실패 시: "JSON으로만 응답하세요" 메시지를 추가하여 재시도
    # 2차 실패 시: 기본값(general_query)으로 폴백하여 그래프가 계속 진행되도록 한다
    for attempt in range(2):
        response = invoke_with_retry(llm, messages, generation_name="analyze_query.intent_plan")
        try:
            analysis = parse_llm_json(response.content, QueryAnalysis)
            break
        except ValueError as e:
            if attempt == 0:
                # 1차 실패: LLM에게 JSON 형식을 다시 요청
                messages.append(HumanMessage(
                    content=f"Invalid JSON. Error: {e}\nJSON으로만 응답하세요."
                ))
            else:
                # 2차 실패: 안전한 기본값으로 폴백.
                # doc_search + data_query를 기본 도구 계획으로 설정하여
                # 최소한의 데이터 수집이 이루어지도록 한다.
                analysis = QueryAnalysis(
                    intent="general_query",
                    required_data=[],
                    required_docs=[],
                    tool_plan=["rag_search", "pandas_query"],
                    references_previous=False,
                    referenced_turns=[],
                )

    # --- 한국어 참조 표현 폴백 감지 ---
    # LLM(Haiku)이 한국어 맥락 단서를 놓칠 수 있으므로
    # 키워드 기반 패턴 매칭으로 references_previous를 보완한다.
    if not analysis.references_previous and state.get("turn_conclusions"):
        user_query = state.get("user_query", "")
        if any(pattern in user_query for pattern in KOREAN_REFERENCE_PATTERNS):
            # 가장 최근 턴을 참조 대상으로 설정
            latest_turn = state["turn_conclusions"][-1].get("turn_number", 0)
            analysis = QueryAnalysis(
                intent=analysis.intent,
                required_data=analysis.required_data,
                required_docs=analysis.required_docs,
                tool_plan=analysis.tool_plan if "lookup_previous" in analysis.tool_plan
                    else analysis.tool_plan + ["lookup_previous"],
                references_previous=True,
                referenced_turns=[latest_turn] if latest_turn else [],
            )

    # --- 2-b. session_intent_history append ---
    # 현재 턴의 intent를 이력에 누적한다. session_continuity 계산과
    # 향후 alignment judge 입력에 사용된다.
    turn_number: int = state.get("current_turn", len(state.get("session_intent_history", [])) + 1)
    updated_history: list[dict] = list(state.get("session_intent_history", []))
    updated_history.append({
        "turn_number": turn_number,
        "intent": analysis.intent,
    })

    # --- 2-c. query.session_continuity 계산 ---
    # 이전 N턴 intent 대비 현재 intent의 cosine similarity (User Pivot 감지).
    # updated_history를 사용하므로 현재 턴 intent가 이미 포함된 상태에서 계산한다.
    session_continuity: float | None = compute_session_continuity(
        updated_history, window=_SESSION_CONTINUITY_WINDOW
    )

    # --- Langfuse 메타데이터 기록 ---
    # analyze_query 스팬에 질의 분석 결과를 Layer 2 attribute로 기록한다.
    # 이 데이터는 Langfuse 대시보드에서 질의 패턴 분석, 도구 사용 빈도 파악,
    # 이전 턴 참조 비율, User Pivot 감지 등을 확인하는 데 활용된다.
    get_client().update_current_span(
        metadata={
            ATTRS["query.intent"]: analysis.intent,                          # 사용자 의도
            ATTRS["query.required_data_types"]: analysis.required_data,       # 필요 데이터 유형
            ATTRS["query.tool_plan"]: analysis.tool_plan,                     # 도구 호출 계획
            ATTRS["query.references_previous_turn"]: analysis.references_previous,  # 이전 턴 참조 여부
            # v3 폐기: query.referenced_turn_numbers (boolean references_previous_turn으로 충분)

            # 소스 선택 관측 (Source Selection)
            # v3 폐기: source.types_selected (gather.tools_called에서 도출)
            ATTRS["source.selection_reasoning"]: analysis.source_reasoning,    # 소스 선택 근거

            # 이탈 감지 (User Pivot Detection) — Phase 3.7 신규
            ATTRS["query.user_query"]: state["user_query"],                   # alignment judge 입력용
            ATTRS["query.session_continuity"]: session_continuity,            # User Pivot 감지 (null 허용)
        }
    )

    # QueryAnalysis 모델을 dict로 변환하여 AgentState에 저장.
    # session_intent_history는 updated_history로 교체한다 (현재 턴 intent 누적).
    # session_continuity는 _build_trace_data()를 통해 diagnose_quality()에 전달된다.
    return {
        "query_analysis": analysis.model_dump(),
        "session_intent_history": updated_history,
        "session_continuity": session_continuity,
    }
