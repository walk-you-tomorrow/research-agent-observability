"""
evaluation/judge_relevance.py — 관련성(Relevance) 평가 Judge

이 모듈은 "수집한 데이터가 질문과 관련이 있는가? 노이즈는 적절한 수준인가?"를 평가한다.

관여하는 프로세스 단계: ① Plan, ② Collect
품질 차원: Relevance (관련성)

G4 리팩토링: 스키마 기반 동적 로딩 — YAML의 judge_input 필드에서 속성 자동 추출.
새 속성 추가 시 YAML에 judge_input: [relevance]만 설정하면 Judge 코드 수정 없이 자동 반영.

점수 기준:
    1.0: 모든 데이터가 질문과 직접 관련, 불필요 데이터 없음, 도구 선택 정확
    0.7: 핵심 데이터는 관련 있으나 경미한 노이즈, 도구 선택 대부분 적절
    0.4: 관련 데이터도 있으나 상당한 노이즈, 도구 선택 중 오류
    0.0: 대부분 무관한 데이터, 심각한 노이즈, 도구 선택 기준 부재
"""

from agent.monitoring_schema import ATTR_META, ATTRS, THRESHOLDS, extract_judge_metadata

# --- Pass/Fail 임계값 ---
PASS_THRESHOLD = THRESHOLDS["relevance"]

# --- 관련성 루브릭 (고정) ---
RELEVANCE_RUBRIC = """[관련성 기준]
1.0: 모든 데이터가 질문과 직접 관련, 불필요한 데이터 없음, tool 선택 정확
0.7: 핵심 데이터는 관련성 있으나 경미한 noise 포함, tool 선택 대부분 적절
0.4: 관련성 있는 데이터도 있으나 상당한 noise 포함, tool 선택 중 오류 발생
0.0: 대부분의 데이터가 질문과 무관, 심각한 noise, tool 선택 기준 부재"""

# --- 관련성 평가 프롬프트 (데이터 섹션 동적 생성) ---
RELEVANCE_PROMPT = """당신은 AI Agent의 컨텍스트 관련성을 평가하는 평가자입니다.

다음은 한 턴에서 수집된 관측 데이터입니다:
{data_section}

{rubric}

이 턴의 관련성 점수를 0.0~1.0 사이로 평가하고, 근거를 한 문장으로 설명하세요.

JSON으로만 응답:
{{"score": 0.85, "reasoning": "근거 설명"}}
"""


def build_relevance_input(trace_data: dict) -> str:
    """트레이스 데이터에서 관련성 평가에 필요한 입력을 구성한다.

    G4: YAML 스키마의 judge_input 필드에서 relevance judge가 사용하는 속성을
    자동으로 추출하고, ATTR_META의 description으로 라벨링한다.

    Args:
        trace_data: Langfuse 트레이스 데이터. metadata 키 아래에 attribute를 포함.

    Returns:
        완성된 평가 프롬프트 문자열. LLM에 직접 전달 가능.
    """
    metadata = trace_data.get("metadata", {})
    judge_data = extract_judge_metadata("relevance", metadata)

    # 속성명 → 한국어 설명으로 데이터 섹션 생성 (스키마 기반)
    data_lines = []
    for attr_name, value in judge_data.items():
        desc = ATTR_META.get(attr_name, {}).get("description", attr_name)
        data_lines.append(f"- {desc}: {value}")
    data_section = "\n".join(data_lines) if data_lines else "- (관측 데이터 없음)"

    return RELEVANCE_PROMPT.format(
        data_section=data_section,
        rubric=RELEVANCE_RUBRIC,
    )
