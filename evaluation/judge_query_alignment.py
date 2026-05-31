"""
evaluation/judge_query_alignment.py — 쿼리 정렬(Query Alignment) 평가 Judge

이 모듈은 "분석 결과 또는 최종 응답이 사용자 쿼리의 핵심 요구에 실제로 대답하고 있는가?"를
평가한다. 두 개의 속성을 담당한다:
    - analysis.query_alignment: 분석 결과(analysis.summary)가 쿼리에 정렬되어 있는가
    - response.query_alignment: 최종 응답(response.final_text)이 쿼리에 정렬되어 있는가

관여하는 프로세스 단계: ④ Generate (analysis), ⑤ Memory (response)
품질 차원: Query Alignment (쿼리 정렬) — 이탈 감지 Phase 3.7

평가 기준 (3가지 축):
    관련성 (Relevance): 분석/응답의 내용이 질문의 주제와 관련 있는가
    직접성 (Directness): 질문이 요구하는 것에 직접 답하는가 (우회하지 않는가)
    완결성 (Completeness): 질문의 핵심 요구가 충족되었는가 (부분적/전체적)

점수 기준:
    1.0: 분석/응답이 쿼리 핵심 요구에 직접·완전하게 대답함
    0.7: 핵심 요구에 대답하나 일부 항목 미흡 또는 우회적 표현
    0.4: 쿼리와 관련 있으나 핵심 요구에 직접적으로 대답하지 못함
    0.0: 분석/응답이 쿼리의 핵심 요구와 실질적으로 무관

NOTE: Phase 3.8에서 ≥30세션 데이터 기반으로 ALIGNMENT_PASS_THRESHOLD 보정 예정.
"""

from agent.monitoring_schema import ATTRS

# --- Pass/Fail 임계값 (Phase 3.8에서 데이터 기반 보정 예정) ---
ALIGNMENT_PASS_THRESHOLD = 0.7

# --- 쿼리 정렬 루브릭 (공통) ---
# 두 judge(분석/응답)가 동일한 기준을 사용하여 비교 가능성을 보장한다.
ALIGNMENT_RUBRIC = """[쿼리 정렬 기준 — 관련성·직접성·완결성]

1.0 (완전 정렬):
  - 분석/응답이 사용자 쿼리의 핵심 요구에 직접적으로 대답함
  - 쿼리가 요구한 모든 주요 항목이 충족됨
  - 우회적 표현 없이 명확한 결론을 제시함

0.7 (부분 정렬):
  - 핵심 요구에 대답하나 일부 항목이 미흡하거나 간접적
  - 쿼리의 부수적 요구는 충족하지 못하나 주요 요구는 충족
  - 결론이 다소 모호하거나 조건부임

0.4 (약한 정렬):
  - 쿼리와 주제상 관련 있으나 핵심 요구에 직접 대답하지 못함
  - 배경 정보·맥락만 제공하고 실질적 답변이 없음
  - 질문의 핵심을 회피하거나 다른 방향으로 전환됨

0.0 (비정렬):
  - 분석/응답이 사용자 쿼리의 핵심 요구와 실질적으로 무관
  - 쿼리와 완전히 다른 주제를 다룸
  - 질문을 잘못 이해하여 엉뚱한 정보를 제공함"""

# --- 분석 정렬 평가 프롬프트 ---
ANALYSIS_ALIGNMENT_PROMPT = """당신은 AI Agent의 분석 결과가 사용자 쿼리에 정렬되어 있는지 평가하는 평가자입니다.

[사용자 쿼리]
{user_query}

[분석 결과 요약]
{analysis_summary}

{rubric}

평가 기준: 분석 결과 요약이 사용자 쿼리의 핵심 요구에 실제로 대답하고 있는가?
- 관련성: 분석 내용이 질문 주제와 관련 있는가
- 직접성: 질문이 요구하는 것에 직접 답하는가 (우회하지 않는가)
- 완결성: 질문의 핵심 요구가 충족되었는가

이 분석의 쿼리 정렬 점수를 0.0~1.0 사이로 평가하고, 근거를 한 문장으로 설명하세요.

JSON으로만 응답:
{{"score": 0.85, "reasoning": "근거 설명"}}
"""

# --- 응답 정렬 평가 프롬프트 ---
RESPONSE_ALIGNMENT_PROMPT = """당신은 AI Agent의 최종 응답이 사용자 쿼리에 정렬되어 있는지 평가하는 평가자입니다.

[사용자 쿼리]
{user_query}

[최종 응답 텍스트]
{response_text}

{rubric}

평가 기준: 최종 응답이 사용자 쿼리의 핵심 요구에 실제로 대답하고 있는가?
- 관련성: 응답 내용이 질문 주제와 관련 있는가
- 직접성: 질문이 요구하는 것에 직접 답하는가 (우회하지 않는가)
- 완결성: 질문의 핵심 요구가 충족되었는가

이 응답의 쿼리 정렬 점수를 0.0~1.0 사이로 평가하고, 근거를 한 문장으로 설명하세요.

JSON으로만 응답:
{{"score": 0.85, "reasoning": "근거 설명"}}
"""


def build_analysis_alignment_input(trace_data: dict) -> str:
    """트레이스 데이터에서 분석 쿼리 정렬 평가에 필요한 입력을 구성한다.

    analysis.summary와 query.user_query를 결합하여 LLM Judge가
    분석 결과가 쿼리에 정렬되어 있는지 평가할 수 있도록 프롬프트를 생성한다.

    Args:
        trace_data: Langfuse 트레이스 데이터. metadata 키 아래에 attribute를 포함.

    Returns:
        완성된 평가 프롬프트 문자열. LLM에 직접 전달 가능.
    """
    metadata = trace_data.get("metadata", {})
    user_query = metadata.get(ATTRS["query.user_query"], "")
    analysis_summary = metadata.get(ATTRS["analysis.summary"], "")

    return ANALYSIS_ALIGNMENT_PROMPT.format(
        user_query=user_query or "(사용자 쿼리 없음)",
        analysis_summary=analysis_summary or "(분석 요약 없음)",
        rubric=ALIGNMENT_RUBRIC,
    )


def build_response_alignment_input(trace_data: dict) -> str:
    """트레이스 데이터에서 응답 쿼리 정렬 평가에 필요한 입력을 구성한다.

    response.final_text와 query.user_query를 결합하여 LLM Judge가
    최종 응답이 쿼리에 정렬되어 있는지 평가할 수 있도록 프롬프트를 생성한다.

    Args:
        trace_data: Langfuse 트레이스 데이터. metadata 키 아래에 attribute를 포함.

    Returns:
        완성된 평가 프롬프트 문자열. LLM에 직접 전달 가능.
    """
    metadata = trace_data.get("metadata", {})
    user_query = metadata.get(ATTRS["query.user_query"], "")
    response_text = metadata.get(ATTRS["response.final_text"], "")

    return RESPONSE_ALIGNMENT_PROMPT.format(
        user_query=user_query or "(사용자 쿼리 없음)",
        response_text=response_text or "(최종 응답 텍스트 없음)",
        rubric=ALIGNMENT_RUBRIC,
    )
