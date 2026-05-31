"""
evaluation/judge_consistency.py — 일관성(Consistency) 평가 Judge

이 모듈은 "이전 턴의 결론과 모순이 있는가? 정보가 충실하게 전달되는가?"를 평가한다.

관여하는 프로세스 단계: ③ Organize (Pattern A), ④ Generate (Pattern B), ⑤ Memory → ③ Organize (Pattern C), ⑤ Memory (Pattern D)
품질 차원: Consistency (일관성)

★ 2026-04-27 재정의 (analysis/31): 패턴 D 추가 — Groundedness 통합 ★
일관성을 "컨텍스트 내부"에서 "답변까지 포함한 전체 일관성"으로 확장.
4개 패턴 모두 "두 표현/시점이 사실적으로 일치하는가" 동일 구조.

4가지 일관성 패턴:

패턴 A — Iteration 내 일관성:
    같은 턴 내에서 evaluate_context → gather_data → evaluate_context 루프를 돌 때,
    이전 iteration에서 부족했던 정보가 해결되었는지, 신뢰도가 개선되었는지 추적.
    - missing_info_resolved: 부족 정보 해결 여부
    - confidence_delta: 신뢰도 변화량 (+양수 = 개선)

패턴 B — 턴 간 일관성:
    서로 다른 턴의 결론이 모순되는지, 모순이 있다면 해결되었는지 추적.
    - contradicts_previous: 이전 턴과 모순 여부
    - contradiction_resolved: 모순 해결 여부
    - previous_conclusion: 모순된 이전 결론
    - referenced_turns: 참조한 이전 턴 번호

패턴 C — 전달 충실도 (G1):
    이전 턴의 결론이 다음 턴 컨텍스트에 의미적으로 보존되는지 측정.
    - fidelity_score: G2 attributes 기반 의미적 충실도 (0.0~1.0)
    - continuity_score: 키워드 기반 보존도 (0.0~1.0)

패턴 D — 답변↔컨텍스트 일관성 (Groundedness, 2026-04-27 신규):
    LLM 답변이 수집된 컨텍스트에 사실적으로 근거하는지 측정.
    환각(hallucination)이 있으면 답변과 컨텍스트가 불일치 = 일관성 위반.
    - response.grounded_claim_ratio: 근거 있는 주장 비율 (0.0~1.0)
    - response.hallucination_detected: 환각 감지 여부
    - response.ungrounded_claims: 근거 없는 주장 리스트

G4 리팩토링: 스키마 기반 동적 로딩 — YAML의 judge_input 필드에서 속성 자동 추출.
단, 첫 턴 감지 로직(first_turn_note)은 consistency 특유의 처리로 유지.
"""

from agent.monitoring_schema import ATTR_META, ATTRS, THRESHOLDS, extract_judge_metadata

# --- Pass/Fail 임계값 ---
PASS_THRESHOLD = THRESHOLDS["consistency"]

# --- 일관성 루브릭 (4패턴, 2026-04-27 패턴 D 추가) ---
CONSISTENCY_RUBRIC = """[일관성 기준 — 4가지 패턴]

1.0 (매우 일관):
  - 패턴 A: missing_info 해결, confidence 개선
  - 패턴 B: 이전 턴과 모순 없음 또는 명시적으로 해결
  - 패턴 C: fidelity_score ≥ 0.8 (충실도 높음)
  - 패턴 D: grounded_claim_ratio ≥ 0.9 (환각 없음)

0.7 (보통):
  - 일부 패턴에서 경미한 불일치
  - fidelity_score 0.5~0.8 또는 grounded_claim_ratio 0.7~0.9

0.4 (불일치):
  - 패턴 D 환각 감지 (grounded_claim_ratio < 0.7) 또는
  - 패턴 B 모순 미해결 또는
  - 패턴 C 충실도 < 0.5

0.0 (심각):
  - 다중 패턴 실패. 답변에 컨텍스트 무관 사실 다수.
  - hallucination_detected=True + 다수의 ungrounded_claims"""

# --- 일관성 평가 프롬프트 (데이터 섹션 동적 생성) ---
CONSISTENCY_PROMPT = """당신은 AI Agent의 컨텍스트 일관성을 평가하는 평가자입니다.

다음은 한 턴에서 수집된 관측 데이터입니다:
{data_section}
{first_turn_note}
{rubric}

이 턴의 일관성 점수를 0.0~1.0 사이로 평가하고, 근거를 한 문장으로 설명하세요.

JSON으로만 응답:
{{"score": 0.85, "reasoning": "근거 설명"}}
"""


def build_consistency_input(trace_data: dict) -> str:
    """트레이스 데이터에서 일관성 평가에 필요한 입력을 구성한다.

    G4: YAML 스키마의 judge_input 필드에서 consistency judge가 사용하는 속성을
    자동으로 추출하고, ATTR_META의 description으로 라벨링한다.

    첫 턴 감지: 이전 결론과 참조 턴이 모두 없으면 세션의 첫 턴으로 간주하고,
    프롬프트에 "패턴 B·C는 해당 없음" 안내를 추가하여 LLM Judge가
    이전 턴 부재를 부정적으로 평가하지 않도록 한다.

    Args:
        trace_data: Langfuse 트레이스 데이터. metadata 키 아래에 attribute를 포함.

    Returns:
        완성된 평가 프롬프트 문자열. LLM에 직접 전달 가능.
    """
    metadata = trace_data.get("metadata", {})
    judge_data = extract_judge_metadata("consistency", metadata)

    # 속성명 → 한국어 설명으로 데이터 섹션 생성 (스키마 기반)
    data_lines = []
    for attr_name, value in judge_data.items():
        desc = ATTR_META.get(attr_name, {}).get("description", attr_name)
        data_lines.append(f"- {desc}: {value}")
    data_section = "\n".join(data_lines) if data_lines else "- (관측 데이터 없음)"

    # 첫 턴 감지: 이전 결론과 참조 턴이 모두 없으면 첫 턴으로 간주 (v2/v3 trace 양립)
    from agent.monitoring_schema import get_previous_conclusion_from_metadata
    prev_conclusion = get_previous_conclusion_from_metadata(metadata) or ""
    ref_turns = metadata.get(ATTRS["analysis.referenced_turns"], [])
    is_first_turn = not prev_conclusion and not ref_turns

    # 첫 턴이면 패턴 B·C를 N/A로 안내하여 "참조 부족" 감점 방지
    first_turn_note = ""
    if is_first_turn:
        first_turn_note = (
            "\n[참고]\n"
            "이 턴은 세션의 첫 번째 턴입니다. 이전 턴이 없으므로 패턴 B(턴 간 일관성)와 "
            "패턴 C(전달 충실도)는 해당 없음(N/A)입니다. 패턴 A(Iteration 내 일관성)만 평가하세요. "
            "이전 턴이 없는 상태에서 missing_info_resolved=False, referenced_turns=[], "
            "fidelity_score=1.0은 정상입니다.\n"
        )

    return CONSISTENCY_PROMPT.format(
        data_section=data_section,
        first_turn_note=first_turn_note,
        rubric=CONSISTENCY_RUBRIC,
    )
