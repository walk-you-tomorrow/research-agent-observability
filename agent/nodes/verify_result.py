"""
agent/nodes/verify_result.py — 노드 5: 이중 검증 (수치 + 해석)

역할:
    generate_analysis가 생성한 분석 결과의 정확성을 이중으로 검증한다.
    STEP 1: pandas 기반 수치 검증 (claims의 수치를 원본 데이터와 대조)
    STEP 2: LLM-as-a-Judge 해석 검증 (분석이 데이터에 기반한 정확한 해석인지 평가)

프로세스 단계: — (검증 checkpoint)
품질 차원: 정확성

데이터 흐름:
    입력: analysis_result (분석 결과), gathered_data (수집 데이터), query_analysis
    출력: verification (검증 결과), verify_retry_count (재시도 카운터)

분기 ② 연결:
    - overall_verdict="pass" → respond_to_user (응답 생성)
    - overall_verdict="fail_numeric" → gather_data (데이터 재수집)
    - overall_verdict="fail_interpretation" → generate_analysis (분석 재생성)

Langfuse 기록:
    verify.numeric_check_passed: 수치 검증 통과 여부
    verify.numeric_discrepancies: 수치 불일치 건수
    verify.interpretation_score: 해석 충실도 점수 (0.0~1.0)
    verify.overall_verdict: 최종 판정
    verify.issues: 발견된 문제점 (최대 5건)
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, observe

from agent.config_loader import get_config
from agent.llm import create_llm, invoke_with_retry
from agent.models import InterpretationCheck
from agent.monitoring_schema import ATTRS
from agent.parser import parse_llm_json
from agent.tools.result_tools import calculate

# --- 해석 검증 프롬프트 ---
# LLM에게 "검증자" 역할을 부여한다.
# 분석 결과가 수집된 데이터에 기반한 정확한 해석인지 0.0~1.0으로 평가한다.
# 0.6 미만이면 "fail_interpretation"으로 판정하여 분석 재생성을 트리거한다.
INTERPRETATION_PROMPT = """당신은 분석 결과의 품질을 검증하는 검증자입니다.

분석 결과가 수집된 데이터에 기반한 정확한 해석인지 판단하세요.

JSON으로만 응답:
{
  "score": 0.91,
  "issues": []
}

score: 0.0~1.0 (데이터에 기반한 정확한 해석일수록 높음)
issues: 발견된 문제점 리스트
"""


def _rule_based_interp_check(claims: list[dict]) -> tuple[float, list[str]]:
    """규칙 기반 해석 품질 추정 (LLM 미사용).

    claims의 출처(source)와 수치(value) 보유 비율로 점수를 계산한다.
    출처 있는 claim이 많을수록 데이터에 기반한 해석으로 판단한다.

    Args:
        claims: generate_analysis가 생성한 claims 리스트.
                각 claim은 {"text": str, "source": str, "value": str} 구조.

    Returns:
        (score, issues) 튜플. score는 0.0~1.0, issues는 문제점 리스트.
    """
    total = len(claims)
    if total == 0:
        return 0.7, []  # claims 없으면 중립 통과

    with_source = sum(1 for c in claims if c.get("source"))
    with_value = sum(1 for c in claims if c.get("value"))

    # 출처 비율(35%)과 수치 비율(25%)로 점수 산출. 기본 0.4부터 시작.
    source_ratio = with_source / total
    value_ratio = with_value / total
    score = round(0.4 + source_ratio * 0.35 + value_ratio * 0.25, 2)
    score = min(1.0, max(0.0, score))

    issues = []
    if source_ratio < 0.5:
        issues.append(f"출처 없는 주장 비율 높음 ({total - with_source}/{total}건)")

    return score, issues


@observe(name="verify_result")
def verify_result(state: dict) -> dict:
    """생성된 분석의 수치 정확성과 해석 충실도를 이중 검증한다.

    Args:
        state: 현재 AgentState. analysis_result, gathered_data, query_analysis을 참조.

    Returns:
        {
            "verification": dict,          # {numeric_check, interpretation_check, overall_verdict, issues}
            "verify_retry_count": int,     # 검증 재시도 카운터 (실패 시 +1)
        }

    처리 과정:
        1. STEP 1: 수치 검증 — claims의 각 수치를 calculate() 도구로 원본 데이터와 대조
        2. STEP 2: 해석 검증 — LLM에게 분석 결과와 수집 데이터를 주고 해석 정확도 평가
        3. 종합 판정 — 수치 실패 시 fail_numeric, 해석 점수 < 0.6 시 fail_interpretation, 그 외 pass
        4. Langfuse에 검증 결과 기록
    """
    analysis = state.get("analysis_result", {})
    claims = analysis.get("claims", [])

    # ════════════════════════════════════════════════
    # STEP 1: 수치 검증 (pandas 대조)
    # ════════════════════════════════════════════════
    # 분석의 claims에서 구체적 수치(value)가 있는 항목을 추출하고,
    # calculate() 도구를 사용하여 원본 데이터와 대조한다.
    # 불일치(discrepancy)가 하나라도 있으면 numeric_passed=False가 된다.
    numeric_passed = True
    discrepancies = []

    for claim in claims:
        # 수치(value)와 출처(source)가 모두 있는 claim만 검증 대상
        if claim.get("value") and claim.get("source"):
            try:
                # calculate() 도구로 원본 데이터와 대조
                calc_result = calculate(
                    query_analysis=state.get("query_analysis", {}),
                    state=state,
                )
                data_items = calc_result.get("data", [])

                # verified=False인 항목이 있으면 수치 불일치
                for item in data_items:
                    if not item.get("verified", True):
                        numeric_passed = False
                        discrepancies.append({
                            "claim": claim["text"],
                            "expected": claim["value"],
                            "got": item.get("stated_value", "mismatch"),
                        })
            except Exception:
                # 검증 도구 실행 실패 시: 검증 불가 = 통과로 처리
                # (검증 인프라 오류로 분석을 무효화하지 않기 위함)
                pass

    # ════════════════════════════════════════════════
    # STEP 2: 해석 검증 (조건부 — LLM 또는 규칙 기반)
    # ════════════════════════════════════════════════
    # 수치 검증 실패 시: 해석 검증 불필요 (verdict는 어차피 fail_numeric)
    # 수치 검증 통과 + bypass=true: 규칙 기반 해석 품질 추정 (LLM 미호출)
    # 수치 검증 통과 + bypass=false: 기존 LLM-as-a-Judge 해석 검증
    _opt = get_config().get("optimization", {})
    bypass_interp = _opt.get("bypass_verify_interp", False)

    if not numeric_passed:
        # 수치 실패 → 해석 검증 불필요 (어차피 fail_numeric 판정)
        interp_score = 0.0
        interp_issues = [d["claim"] for d in discrepancies]
    elif not claims:
        # v3 REDEFINE bypass branch (2026-04-29): 분석에 claim이 없으면 해석 검증 자체가 무의미.
        # 이 경우 score = 1.0 (검증 불필요로 통과 처리). issues 없음.
        interp_score = 1.0
        interp_issues = []
    elif bypass_interp:
        # 바이패스: 규칙 기반 해석 품질 추정 (LLM 미호출)
        interp_score, interp_issues = _rule_based_interp_check(claims)
    else:
        # v3 REDEFINE cross-model (2026-04-29): self-referential 차단.
        # generate_analysis가 사용한 모델(claude-haiku)과 다른 모델(claude-sonnet-4-6)로 검증.
        # 같은 모델이면 동일 편향이 분석+검증 양쪽에 작용 → 검증의 독립성 상실.
        from agent.monitoring_schema import get_cross_model_evaluator
        evaluator_model = get_cross_model_evaluator("interpretation_judge")
        llm = create_llm(model_override=evaluator_model) if evaluator_model else create_llm()

        verify_input = json.dumps({
            "analysis": analysis,
            "data_sources": [
                d["data_summary"][:500] for d in state.get("gathered_data", [])
            ],
        }, ensure_ascii=False)

        response = invoke_with_retry(
            llm,
            [
                SystemMessage(content=INTERPRETATION_PROMPT),
                HumanMessage(content=verify_input),
            ],
            generation_name="verify_result.interpretation",
        )
        interp = parse_llm_json(response.content, InterpretationCheck)
        interp_score = interp.score
        interp_issues = interp.issues

    # --- 종합 판정 ---
    # 우선순위: 수치 오류 > 해석 오류 > 통과
    # fail_numeric: 데이터 자체가 잘못되었을 가능성 → 데이터 재수집부터
    # fail_interpretation: 데이터는 맞지만 해석이 틀림 → 분석 재생성만
    if not numeric_passed:
        verdict = "fail_numeric"
    elif interp_score < 0.6:
        verdict = "fail_interpretation"
    else:
        verdict = "pass"

    # 모든 문제점을 하나의 리스트로 통합 (수치 불일치 + 해석 문제)
    all_issues = [d["claim"] for d in discrepancies] + interp_issues

    verification = {
        "numeric_check": {"passed": numeric_passed, "discrepancies": discrepancies},
        "interpretation_check": {"score": interp_score, "issues": interp_issues},
        "overall_verdict": verdict,
        "issues": all_issues,
    }

    # --- Langfuse 메타데이터 기록 (검증 checkpoint) ---
    get_client().update_current_span(
        metadata={
            ATTRS["verify.numeric_check_passed"]: numeric_passed,       # 수치 검증 통과 여부
            ATTRS["verify.numeric_discrepancies"]: len(discrepancies),   # 수치 불일치 건수
            ATTRS["verify.interpretation_score"]: interp_score,          # 해석 점수 (0.0~1.0)
            ATTRS["verify.overall_verdict"]: verdict,                    # 최종 판정
            ATTRS["verify.issues"]: all_issues[:5],                     # 문제점 (최대 5건)
        }
    )

    # --- 재시도 카운터 관리 ---
    # 검증 실패 시 카운터를 증가시킨다.
    # route_after_verify 분기에서 MAX_VERIFY_RETRIES와 비교하여
    # 재시도를 계속할지 강제 진행할지 결정한다.
    retry_count = state.get("verify_retry_count", 0)
    if verdict != "pass":
        retry_count += 1

    return {
        "verification": verification,
        "verify_retry_count": retry_count,
    }
