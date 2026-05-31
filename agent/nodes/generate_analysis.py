"""
agent/nodes/generate_analysis.py — 노드 4: 분석 생성 + 교차 턴 일관성 추적

역할:
    수집된 데이터를 기반으로 사용자 질문에 대한 분석 결과를 생성한다.
    동시에 이전 턴의 결론과 비교하여 모순 여부를 감지한다 (일관성 패턴 B).

프로세스 단계: ④ Generate
품질 차원: 일관성(B) — 교차 턴 모순 감지 및 해결

데이터 흐름:
    입력: gathered_data, query_analysis, user_query, turn_conclusions (이전 턴 결론)
    출력: analysis_result, referenced_turns, conflict_tracking
          (conflict_tracking은 모순 감지/해결/요약을 단일 dict로 통합 — v3 2026-04-29)

Langfuse 기록:
    analysis.referenced_turns: 참조한 이전 턴 번호
    analysis.conflict_tracking: 모순 감지/해결/이전결론/소스충돌해결 통합 dict (v3 2026-04-29)
    source.conflict_detected: 소스 간 충돌 감지 여부
    analysis.conclusion_utilization: 이전 결론 활용 비율 (Post-2)
    analysis.utilized_conclusions: 실제 활용한 이전 결론 목록 (Post-2)
    context.conclusion_window_size: rot_risk 기반 적응적 결론 참조 윈도우 크기 (A6)
    analysis.summary: 분석 결과 요약 텍스트 (alignment judge 입력용, 최대 3000자)
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, observe

from agent.llm import create_llm, invoke_with_retry
from agent.models import AnalysisResult
from agent.monitoring_schema import ATTRS
from agent.parser import parse_llm_json

# --- 분석 생성 프롬프트 ---
# LLM에게 "서울 상권 데이터 분석 Agent" 역할을 부여한다.
# 핵심: 이전 턴의 결론이 있을 때 모순을 명시적으로 감지하고 설명하도록 지시한다.
# contradicts_previous, contradiction_explanation, referenced_turns 필드가
# 일관성 패턴 B의 핵심이다.
ANALYSIS_PROMPT = """당신은 서울 상권 데이터 분석 Agent입니다.

수집된 데이터를 기반으로 분석 결과를 생성하세요.

**모순 감지 규칙 (매우 중요):**
1. 이전 턴의 결론(previous_conclusions)이 있다면, 현재 분석 결과와 **반드시 비교**하세요.
2. 다음 상황은 모순(contradicts_previous=true)으로 판정하세요:
   - 이전에 A 지역을 추천했는데, 새로운 요소(임대료, 경쟁 등)를 고려하면 결론이 달라지는 경우
   - 이전 결론의 핵심 주장이 현재 데이터로 뒤집히는 경우
   - 사용자가 이전 결론에 이의를 제기하고, 그 이의가 데이터로 뒷받침되는 경우
3. 모순이 감지되면 contradiction_explanation에 "이전에는 X라고 했지만, Y를 고려하면 Z"
   형식으로 명확히 설명하세요.
4. 모순이 감지되면 referenced_turns에 해당 이전 턴 번호를 포함하세요.

**소스 간 충돌 감지:**
수집된 데이터에서 서로 다른 소스(CSV, 도메인 문서, 웹 검색, API)가 모순된 정보를 제공하면:
- source_conflict: true
- source_conflict_resolution: 어떤 소스를 우선했고 왜 그런지 설명

**이전 결론 활용 추적 (Post-2):**
이전 턴 결론을 참조했다면, 실제로 분석에 반영된 결론을 utilized_previous에 명시하세요.
각 항목은 {"turn": 턴번호, "claim": "활용한 결론 내용", "used_in": "분석 어디에 반영했는지"} 형식입니다.

JSON으로만 응답:
{
  "summary": "분석 결론 한 문장",
  "claims": [
    {"text": "주장", "source": "데이터 출처", "value": "구체적 수치"}
  ],
  "data_references": ["사용한 데이터 소스 목록"],
  "caveats": ["주의사항/한계"],
  "contradicts_previous": false,
  "contradiction_explanation": "",
  "referenced_turns": [2, 3],
  "source_conflict": false,
  "source_conflict_resolution": "",
  "utilized_previous": []
}
"""


@observe(name="generate_analysis")
def generate_analysis(state: dict) -> dict:
    """수집된 데이터를 기반으로 분석을 생성하고 교차 턴 일관성을 추적한다.

    Args:
        state: 현재 AgentState. gathered_data, query_analysis, user_query,
               turn_conclusions를 참조.

    Returns:
        {
            "analysis_result": dict,           # {summary, claims, data_references, caveats}
            "referenced_turns": list[int],     # 참조한 이전 턴 번호
            "contradicts_previous": bool,      # 이전 턴과 모순 여부
            "contradiction_resolved": bool,    # 모순 해결 여부
            "previous_conclusion": str,        # 모순된 이전 결론 원문
        }

    처리 과정:
        1. rot_risk 기반 적응적 윈도우로 이전 턴 결론 범위 결정 (A6)
        2. LLM에 수집 데이터 + 이전 결론을 전달하여 분석 생성
        3. JSON 응답을 AnalysisResult 모델로 파싱
        4. 일관성 패턴 B attribute 추출 (모순 여부, 참조 턴 등)
        5. Langfuse에 분석 통계와 일관성 지표 기록
    """
    llm = create_llm()

    # ── A6: 적응적 결론 윈도우 ──
    # rot_risk가 높을수록 이전 결론의 신뢰도가 낮으므로 참조 범위를 줄인다.
    # rot_risk < 0.1 → 7턴 (최대, 낮은 rot), >= 0.3 → 2턴 (최소, 높은 rot)
    previous_conclusions = state.get("turn_conclusions", [])
    context_metadata = state.get("context_metadata", {})
    rot_risk = context_metadata.get("rot_risk", 0.0)

    if rot_risk < 0.1:
        conclusion_window = 7
    elif rot_risk < 0.2:
        conclusion_window = 5
    elif rot_risk < 0.3:
        conclusion_window = 3
    else:
        conclusion_window = 2

    # LLM에 전달할 분석 입력 구성:
    # - query: 사용자 질문 원문
    # - query_analysis: 질의 분석 결과 (intent, required_data 등)
    # - gathered_data: 수집된 데이터의 source + summary
    # - previous_conclusions: 이전 턴 결론 (rot_risk 기반 적응적 윈도우)
    analysis_input = json.dumps({
        "query": state.get("user_query", ""),
        "query_analysis": state.get("query_analysis", {}),
        "gathered_data": [
            {"source": d["source"], "summary": d["data_summary"]}
            for d in state.get("gathered_data", [])
        ],
        "previous_conclusions": previous_conclusions[-conclusion_window:],
    }, ensure_ascii=False)

    response = invoke_with_retry(
        llm,
        [
            SystemMessage(content=ANALYSIS_PROMPT),
            HumanMessage(content=analysis_input),
        ],
        generation_name="generate_analysis.compose",
    )
    result = parse_llm_json(response.content, AnalysisResult)
    result_dict = result.model_dump()

    # --- 일관성 패턴 B: 교차 턴 모순 감지 ---
    # LLM이 이전 결론과 현재 분석 사이의 모순을 감지했는지 확인한다.
    # contradicts_previous=true이면 모순이 감지된 것이며,
    # contradiction_explanation이 있으면 모순이 "해결"된 것으로 간주한다.
    contradicts = result_dict.get("contradicts_previous", False)
    referenced = result_dict.get("referenced_turns", [])

    # 모순 감지 시: 가장 최근 이전 결론의 요약을 추출
    # respond_to_user에서 "이전에는 X라고 했지만, Y를 고려하면 Z입니다"와 같이
    # 명시적 설명을 생성하는 데 사용된다.
    prev_conclusion = ""
    if contradicts and previous_conclusions:
        prev_conclusion = previous_conclusions[-1].get("conclusion_summary", "")

    # --- 소스 간 충돌 감지 (v3 REDEFINE 2026-04-29) ---
    # 정량 측정(수치 비교) + LLM 산출의 OR 결합.
    # 정량: source_conflict_checker.detect_source_conflict — 모델 독립
    # LLM: result_dict.source_conflict — 의미적 충돌 (단위 다름, 다른 차원 등)
    from agent.source_conflict_checker import detect_source_conflict
    from agent.monitoring_schema import get_customizable_threshold
    _num_diff = get_customizable_threshold("numeric_diff_threshold", 0.05) or 0.05
    _conflict_th = get_customizable_threshold("conflict_detection_threshold", 0.1) or 0.1
    quant_conflict, quant_summary = detect_source_conflict(
        state.get("gathered_data", []),
        numeric_diff_threshold=_num_diff,
        conflict_detection_threshold=_conflict_th,
    )
    llm_conflict = result_dict.get("source_conflict", False)
    llm_resolution = result_dict.get("source_conflict_resolution", "")
    source_conflict = bool(quant_conflict or llm_conflict)
    # resolution 텍스트: LLM 설명 우선, 없으면 정량 측정 근거.
    source_conflict_resolution = llm_resolution or quant_summary

    # --- Post-2: 기여도 추적 (Contribution Tracking) ---
    # v3 REDEFINE (2026-04-29): 조건부 측정.
    # referenced_turns가 비어있으면 "측정 불가능"이 자연스러우므로 None 반환 (Judge에서 N/A 처리).
    # AS-IS: 0.0 반환 → 0%로 오해 가능 (낮은 활용률처럼 보임)
    # TO-BE: None 반환 → "참조 자체가 없으므로 비율 무의미" 명시
    utilized = result_dict.get("utilized_previous", [])
    if referenced:
        conclusion_utilization = round(len(utilized) / len(referenced), 3)
    else:
        conclusion_utilization = None  # N/A: 참조 턴 없음

    has_explanation = contradicts and bool(result_dict.get("contradiction_explanation"))

    # v3 통합 (2026-04-29): 모순 추적을 단일 dict로 압축.
    # 기존 3개 attribute(contradicts_previous, contradiction_resolved, previous_conclusion) +
    # source.conflict_resolution을 하나로 묶어 다중공선성과 라벨 분산을 차단한다.
    conflict_tracking = {
        "detected": contradicts,
        "resolution": {
            "has_explanation": has_explanation,
            "conflict_summary": prev_conclusion,
            "source_resolution": source_conflict_resolution,
        },
    }

    # --- Langfuse 메타데이터 기록 (프로세스 단계 ④ Generate) ---
    get_client().update_current_span(
        metadata={
            # 일관성 패턴 B (Consistency - Pattern B) 지표
            ATTRS["analysis.referenced_turns"]: referenced,        # 참조한 이전 턴 번호
            # v3 REDEFINE: contradicts_previous/contradiction_resolved/previous_conclusion → conflict_tracking dict
            ATTRS["analysis.conflict_tracking"]: conflict_tracking,

            # v3 폐기: claims_count (response.grounded_claim_ratio가 더 직접)
            # v3 폐기: data_references_count (source.contribution이 더 강함)

            # 소스 간 충돌 관측 (Source Conflict)
            ATTRS["source.conflict_detected"]: source_conflict,
            # v3 통합: source.conflict_resolution → analysis.conflict_tracking.resolution.source_resolution

            # Post-2: 기여도 추적
            ATTRS["analysis.conclusion_utilization"]: conclusion_utilization,
            ATTRS["analysis.utilized_conclusions"]: utilized,

            # A6: 적응적 결론 윈도우 — rot_risk 기반 참조 범위
            ATTRS["context.conclusion_window_size"]: conclusion_window,

            # alignment judge 입력용 — query_alignment judge가 쿼리↔분석 일치도 평가에 사용
            ATTRS["analysis.summary"]: result_dict.get("summary", "")[:3000],
        }
    )

    return {
        # analysis_result: verify_result에서 수치/해석 검증에 사용
        "analysis_result": {
            "summary": result_dict.get("summary", ""),
            "claims": result_dict.get("claims", []),
            "data_references": result_dict.get("data_references", []),
            "caveats": result_dict.get("caveats", []),
        },
        # 일관성 패턴 B 관련 필드들: respond_to_user에서 참조
        "referenced_turns": referenced,
        # v3 통합: 3 + 1 attribute → conflict_tracking dict
        "conflict_tracking": conflict_tracking,
    }
