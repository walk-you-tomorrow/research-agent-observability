"""
evaluation/judge_completeness.py — 완전성(Completeness) 평가 Judge

이 모듈은 "질문에 답하기 위해 필요한 데이터가 모두 수집·포함되었는가?"를 평가한다.

관여하는 프로세스 단계: ② Collect, ③ Organize + 충분성 평가 checkpoint
품질 차원: Completeness (완전성)

G4 리팩토링: 스키마 기반 동적 로딩 — YAML의 judge_input 필드에서 속성 자동 추출.
새 속성 추가 시 YAML에 judge_input: [completeness]만 설정하면 Judge 코드 수정 없이 자동 반영.

점수 기준 (v2: H5 오진단 패턴 반영):
    1.0: 보유 데이터가 모두 수집됨, 또는 범위 외 질문에 정직한 한계 안내
    0.7: 핵심 데이터 포함, 보조 데이터 일부 누락/잘림
    0.4: 보유한 핵심 데이터 일부 미수집으로 불완전
    0.0: 보유한 핵심 데이터 대부분 미수집
"""

from agent.monitoring_schema import ATTR_META, ATTRS, THRESHOLDS, extract_judge_metadata

# --- Pass/Fail 임계값 ---
PASS_THRESHOLD = THRESHOLDS["completeness"]

# --- 완전성 루브릭 (v2: H5 오진단 패턴 반영) ---
# v1→v2 변경: (1) 데이터 범위 외 질문 고려, (2) 수집 가능한 데이터 기준으로 평가,
# (3) "데이터가 없다"고 정직하게 답한 경우는 FAIL이 아님
COMPLETENESS_RUBRIC = """[완전성 기준]
중요: "필요한 데이터"란 이 시스템이 접근 가능한 데이터(CSV 5개구, RAG 문서, 웹 검색, API) 중에서
질문에 관련된 것을 의미합니다. 시스템이 보유하지 않는 데이터(예: 다른 구, 미래 예측 등)의
부재는 감점 사유가 아닙니다. Agent가 "해당 데이터가 없습니다"라고 정직하게 안내한 경우는
완전성이 높은 것입니다.

1.0: 질문에 필요하고 시스템이 보유한 데이터가 모두 수집·포함됨. 또는 데이터 범위 외 질문에 대해 정직하게 한계를 안내함.
0.7: 핵심 데이터는 포함되었으나 보조 데이터 일부 누락 또는 잘림 발생. 답변의 핵심은 충족.
0.4: 시스템이 보유한 핵심 데이터 중 일부를 수집하지 못하여 답변이 불완전함.
0.0: 시스템이 보유한 핵심 데이터를 대부분 수집하지 못하여 의미 있는 답변 불가."""

# --- 완전성 평가 프롬프트 (데이터 섹션 동적 생성) ---
COMPLETENESS_PROMPT = """당신은 AI Agent의 컨텍스트 완전성을 평가하는 평가자입니다.

다음은 한 턴에서 수집된 관측 데이터입니다:
{data_section}

{rubric}

평가 시 주의사항:
- "충분성 판단 결과"(is_sufficient)가 True이고 confidence가 0.7 이상이면, 핵심 데이터는 수집된 것으로 간주하세요.
- 수집 항목 수(items_collected)가 0이더라도, 질문이 시스템 보유 데이터 범위 밖이라면 감점하지 마세요.
- "잘린 항목"(truncated)이 있더라도 핵심 정보가 포함되어 있으면 0.7 이상을 줄 수 있습니다.

이 턴의 완전성 점수를 0.0~1.0 사이로 평가하고, 근거를 한 문장으로 설명하세요.

JSON으로만 응답:
{{"score": 0.85, "reasoning": "근거 설명"}}
"""


def build_completeness_input(trace_data: dict) -> str:
    """트레이스 데이터에서 완전성 평가에 필요한 입력을 구성한다.

    G4: YAML 스키마의 judge_input 필드에서 completeness judge가 사용하는 속성을
    자동으로 추출하고, ATTR_META의 description으로 라벨링한다.

    Args:
        trace_data: Langfuse 트레이스 데이터. metadata 키 아래에 attribute를 포함.

    Returns:
        완성된 평가 프롬프트 문자열. LLM에 직접 전달 가능.
    """
    metadata = trace_data.get("metadata", {})
    judge_data = extract_judge_metadata("completeness", metadata)

    # 속성명 → 한국어 설명으로 데이터 섹션 생성 (스키마 기반)
    data_lines = []
    for attr_name, value in judge_data.items():
        desc = ATTR_META.get(attr_name, {}).get("description", attr_name)
        data_lines.append(f"- {desc}: {value}")
    data_section = "\n".join(data_lines) if data_lines else "- (관측 데이터 없음)"

    return COMPLETENESS_PROMPT.format(
        data_section=data_section,
        rubric=COMPLETENESS_RUBRIC,
    )
