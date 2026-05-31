"""
evaluation/judge_efficiency.py — 효율성(Efficiency) 평가 Judge

이 모듈은 "컨텍스트 윈도우와 자원(토큰/시간/비용)을 적절히 사용하는가?"를 평가한다.

관여하는 프로세스 단계: ③ Organize
품질 차원: Efficiency (효율성) — Resource Consumption 관점

G4 리팩토링: 스키마 기반 동적 로딩 — YAML의 judge_input 필드에서 속성 자동 추출.
새 속성 추가 시 YAML에 judge_input: [efficiency]만 설정하면 Judge 코드 수정 없이 자동 반영.

★ 2026-04-27 재정의 (analysis/31) ★
구공식: "이전 턴 토큰 비율 (noise_ratio)" 중심 → relevance와 신호 공유로 r=0.86 발생
신공식: 자원 관점으로 회복 (Charter 원래 의도 = "컨텍스트 윈도우를 적절히 사용하는가")

핵심 지표 4가지 (자원 관점):
    1. 윈도우 사용률 (window_utilization): 0~1, 0.7 초과 시 비효율
    2. 응답/시간/비용 자원 (response.token_count, turn.wall_time_ms, turn.total_cost_usd)
    3. 토큰 분배 균형 (context.source.* 5개)
    4. 자원 낭비 지표 (redundancy_ratio, rot_risk, information_density)

NOTE: noise_ratio 평가는 relevance Judge의 책임으로 이관됨.
"""

from agent.monitoring_schema import ATTR_META, ATTRS, THRESHOLDS, extract_judge_metadata

# --- Pass/Fail 임계값 ---
PASS_THRESHOLD = THRESHOLDS["efficiency"]

# --- 효율성 루브릭 (자원 관점, 2026-04-27 재정의) ---
# Charter 원래 의도 회복: "컨텍스트 윈도우와 자원을 적절히 사용하는가"
# noise_ratio 평가는 relevance Judge로 이관 (analysis/31).
EFFICIENCY_RUBRIC = """[효율성 기준 — 자원(토큰·시간·비용) 관점]

1.0 (매우 효율):
  - 윈도우 사용률(window_utilization) < 0.5
  - 응답 토큰(response.token_count) < 1500
  - 비용(turn.total_cost_usd) < $0.03
  - 토큰 분배 균형 (system 20% / data 60% / 결론 20% 근사)
  - rot_risk < 0.2, redundancy_ratio < 0.1

0.7 (보통):
  - 윈도우 사용률 0.5~0.7
  - 응답 토큰 1500~3000
  - 비용 $0.03~0.05
  - 약간의 중복 또는 rot 누적

0.4 (비효율):
  - 윈도우 사용률 0.7~0.9 (한계 접근)
  - 응답 토큰 > 3000 (과다)
  - 비용 > $0.05
  - rot_risk > 0.3 또는 redundancy_ratio > 0.3

0.0 (심각):
  - 윈도우 사용률 > 0.9 (한계 근접, 답변 잘림 위험)
  - 자원 낭비 심각 (불필요한 토큰/시간/비용 폭증)
  - tool_call_count 과다 (재시도 폭주)

NOTE: 노이즈 비율 평가는 relevance Judge가 담당. 이 Judge는 자원 사용에 집중."""

# --- 효율성 평가 프롬프트 (데이터 섹션 동적 생성) ---
EFFICIENCY_PROMPT = """당신은 AI Agent의 컨텍스트 효율성을 평가하는 평가자입니다.

다음은 한 턴에서 수집된 관측 데이터입니다:
{data_section}

{rubric}

이 턴의 효율성 점수를 0.0~1.0 사이로 평가하고, 근거를 한 문장으로 설명하세요.

JSON으로만 응답:
{{"score": 0.85, "reasoning": "근거 설명"}}
"""


def build_efficiency_input(trace_data: dict) -> str:
    """트레이스 데이터에서 효율성 평가에 필요한 입력을 구성한다.

    G4: YAML 스키마의 judge_input 필드에서 efficiency judge가 사용하는 속성을
    자동으로 추출하고, ATTR_META의 description으로 라벨링한다.

    Args:
        trace_data: Langfuse 트레이스 데이터. metadata 키 아래에 attribute를 포함.

    Returns:
        완성된 평가 프롬프트 문자열. LLM에 직접 전달 가능.
    """
    metadata = trace_data.get("metadata", {})
    judge_data = extract_judge_metadata("efficiency", metadata)

    # 속성명 → 한국어 설명으로 데이터 섹션 생성 (스키마 기반)
    data_lines = []
    for attr_name, value in judge_data.items():
        desc = ATTR_META.get(attr_name, {}).get("description", attr_name)
        data_lines.append(f"- {desc}: {value}")
    data_section = "\n".join(data_lines) if data_lines else "- (관측 데이터 없음)"

    return EFFICIENCY_PROMPT.format(
        data_section=data_section,
        rubric=EFFICIENCY_RUBRIC,
    )
