"""
evaluation/error_injection.py — 오류 주입 실험: gathered_data 변조 시뮬레이션

역할:
    H0 가설("Context Monitoring은 응답이 수집 데이터에서 벗어날 때 감지할 수 있다")의
    실험적 검증을 위해, gathered_data에 의도적 오류를 주입하고
    groundedness가 하락하는지 시뮬레이션한다.

3가지 오류 유형:
    - numeric_inversion: 숫자를 100배 축소 (예: 12,000 → 120)
    - source_swap: 지역명 치환 (예: 강남구 → 마포구)
    - null_injection: data_summary를 빈 문자열로 교체

설계 원칙:
    - 실제 에이전트 파이프라인을 재실행하지 않음 (비용 회피)
    - 기존 세션의 response + gathered_data를 활용한 오프라인 시뮬레이션
    - groundedness_checker를 통해 control/treatment 비교

데이터 흐름:
    입력: 기존 세션의 gathered_data (list[dict]), response 텍스트
    출력: {control_scores, treatment_scores, effect_size, p_value_proxy}
"""
import copy
import re
from typing import Literal

# --- 오류 주입 유형 ---
PoisonType = Literal["numeric_inversion", "source_swap", "null_injection"]

# --- 지역명 치환 매핑 ---
# source_swap에서 사용: 원래 지역명 → 무관한 지역명으로 교체
_DISTRICT_SWAP_MAP = {
    "강남구": "마포구",
    "마포구": "강남구",
    "서초구": "종로구",
    "종로구": "서초구",
    "영등포구": "강남구",
    "역삼동": "합정동",
    "합정동": "역삼동",
    "논현동": "상암동",
    "상암동": "논현동",
    "서초동": "광화문",
    "광화문": "서초동",
}

# --- 숫자 패턴 ---
# 쉼표 구분 숫자(예: 12,000) 또는 일반 숫자(예: 3500)를 매칭
_NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,}")


# ═══════════════════════════════════════
# STEP 1: 오류 주입 함수
# ═══════════════════════════════════════

def _invert_numbers(text: str) -> str:
    """텍스트 내 숫자를 100으로 나눈 값으로 교체한다.

    예: "유동인구 12,000명" → "유동인구 120명"

    Args:
        text: 원본 텍스트.

    Returns:
        숫자가 축소된 텍스트.
    """
    def _shrink(match: re.Match) -> str:
        raw = match.group(0).replace(",", "")
        try:
            val = int(raw)
            # 100으로 나누되, 최소 1 보장
            shrunk = max(1, val // 100)
            return str(shrunk)
        except ValueError:
            return match.group(0)

    return _NUMBER_PATTERN.sub(_shrink, text)


def _swap_districts(text: str) -> str:
    """텍스트 내 지역명을 다른 지역명으로 치환한다.

    예: "강남구 역삼동" → "마포구 합정동"

    2단계 치환: 먼저 플레이스홀더로 교체한 뒤, 플레이스홀더를 최종값으로 교체한다.
    이렇게 하면 "강남구→마포구" 후 "마포구→강남구"로 되돌아가는 이중 치환을 방지한다.

    Args:
        text: 원본 텍스트.

    Returns:
        지역명이 치환된 텍스트.
    """
    result = text
    # 1단계: 원본 → 플레이스홀더 (긴 키부터 치환하여 부분 매칭 방지)
    placeholders = {}
    for i, (original, replacement) in enumerate(
        sorted(_DISTRICT_SWAP_MAP.items(), key=lambda x: -len(x[0]))
    ):
        placeholder = f"\x00SWAP_{i}\x00"
        placeholders[placeholder] = replacement
        result = result.replace(original, placeholder)

    # 2단계: 플레이스홀더 → 최종 치환값
    for placeholder, replacement in placeholders.items():
        result = result.replace(placeholder, replacement)

    return result


def create_poisoned_gathered_data(
    real_data: list[dict],
    poison_type: PoisonType,
) -> list[dict]:
    """수집 데이터에 의도적 오류를 주입한다.

    원본 리스트를 깊은 복사(deep copy)하여 변조하므로 원본은 변경되지 않는다.
    각 항목의 data_summary 필드를 대상으로 오류를 주입한다.

    Args:
        real_data: 원본 gathered_data 리스트. 각 항목은 {source, tool_used, data_summary, ...}.
        poison_type: "numeric_inversion" | "source_swap" | "null_injection"

    Returns:
        오류가 주입된 gathered_data 사본 (원본 변경 없음).

    Raises:
        ValueError: 지원하지 않는 poison_type.
    """
    if poison_type not in ("numeric_inversion", "source_swap", "null_injection"):
        raise ValueError(f"지원하지 않는 poison_type: {poison_type}")

    poisoned = copy.deepcopy(real_data)

    for item in poisoned:
        summary = item.get("data_summary", "")
        if not isinstance(summary, str):
            summary = str(summary) if summary is not None else ""

        if poison_type == "numeric_inversion":
            item["data_summary"] = _invert_numbers(summary)
        elif poison_type == "source_swap":
            item["data_summary"] = _swap_districts(summary)
        elif poison_type == "null_injection":
            item["data_summary"] = ""

    return poisoned


# ═══════════════════════════════════════
# STEP 2: 오류 주입 효과 측정
# ═══════════════════════════════════════

def measure_injection_effect(
    response_text: str,
    real_gathered_data: list[dict],
    poisoned_gathered_data: list[dict],
    groundedness_fn: callable,
) -> dict:
    """단일 턴에 대해 control vs treatment groundedness를 비교한다.

    Control: response가 real_gathered_data에 얼마나 근거하는지
    Treatment: response가 poisoned_gathered_data에 얼마나 근거하는지

    Treatment에서 groundedness가 하락하면, 응답이 원본 데이터에 충실했다는 증거다.
    이는 곧 모니터링이 이탈을 감지할 수 있다는 H0의 간접 증거가 된다.

    Args:
        response_text: 에이전트가 생성한 응답 텍스트.
        real_gathered_data: 원본 수집 데이터.
        poisoned_gathered_data: 오류 주입된 수집 데이터.
        groundedness_fn: check_groundedness(response_text, gathered_data) → dict 형태의 함수.

    Returns:
        {
            "control": groundedness 결과 dict,
            "treatment": groundedness 결과 dict,
            "grounded_ratio_delta": treatment - control (음수이면 하락),
        }
    """
    control_result = groundedness_fn(response_text, real_gathered_data)
    treatment_result = groundedness_fn(response_text, poisoned_gathered_data)

    # grounded_claim_ratio 추출 (groundedness_checker 반환 형식에 따라)
    control_ratio = control_result.get("grounded_claim_ratio", 0.0)
    treatment_ratio = treatment_result.get("grounded_claim_ratio", 0.0)

    return {
        "control": control_result,
        "treatment": treatment_result,
        "grounded_ratio_delta": treatment_ratio - control_ratio,
    }


# ═══════════════════════════════════════
# STEP 3: 실험 실행기 (오프라인 시뮬레이션)
# ═══════════════════════════════════════

def run_injection_experiment(
    turn_data_list: list[dict],
    groundedness_fn: callable,
    poison_types: list[PoisonType] | None = None,
) -> dict:
    """통제군(정상) vs 실험군(오류주입) 비교 실험을 실행한다.

    실제 에이전트 파이프라인을 재실행하지 않는다.
    기존 세션 데이터(response + gathered_data)에 대해 오프라인으로 groundedness를 측정한다.

    각 턴에 대해:
    1. Control: groundedness(response, real_gathered_data)
    2. Treatment: groundedness(response, poisoned_gathered_data)
    3. 비교: treatment에서 grounded_claim_ratio가 하락하는지 확인

    Args:
        turn_data_list: 턴별 데이터 리스트. 각 항목은:
            {
                "response_text": str,     — 에이전트 응답
                "gathered_data": list[dict], — 수집 데이터
                "turn_id": str (선택),    — 식별용
            }
        groundedness_fn: check_groundedness(response, gathered_data) → dict.
        poison_types: 테스트할 오류 유형 리스트. None이면 3가지 전부.

    Returns:
        {
            "n_turns": int,
            "by_poison_type": {
                "numeric_inversion": {
                    "control_scores": [float, ...],
                    "treatment_scores": [float, ...],
                    "deltas": [float, ...],
                    "mean_delta": float,
                    "degradation_rate": float,  — 하락한 턴 비율
                },
                ...
            },
            "overall_verdict": "PASS" | "FAIL",
            "verdict_reason": str,
        }
    """
    if poison_types is None:
        poison_types = ["numeric_inversion", "source_swap", "null_injection"]

    results_by_type: dict[str, dict] = {}

    for ptype in poison_types:
        control_scores = []
        treatment_scores = []
        deltas = []

        for turn_data in turn_data_list:
            response_text = turn_data.get("response_text", "")
            gathered_data = turn_data.get("gathered_data", [])

            if not response_text or not gathered_data:
                continue

            poisoned = create_poisoned_gathered_data(gathered_data, ptype)
            effect = measure_injection_effect(
                response_text, gathered_data, poisoned, groundedness_fn,
            )

            c_ratio = effect["control"].get("grounded_claim_ratio", 0.0)
            t_ratio = effect["treatment"].get("grounded_claim_ratio", 0.0)

            control_scores.append(c_ratio)
            treatment_scores.append(t_ratio)
            deltas.append(effect["grounded_ratio_delta"])

        n = len(control_scores)
        mean_delta = sum(deltas) / n if n > 0 else 0.0
        # 하락한 턴: treatment 점수가 control보다 낮은 경우
        degradation_count = sum(1 for d in deltas if d < -0.05)
        degradation_rate = degradation_count / n if n > 0 else 0.0

        results_by_type[ptype] = {
            "control_scores": control_scores,
            "treatment_scores": treatment_scores,
            "deltas": deltas,
            "mean_delta": round(mean_delta, 4),
            "degradation_rate": round(degradation_rate, 4),
            "n": n,
        }

    # 종합 판정: null_injection에서 50% 이상 하락이면 PASS
    # (데이터를 완전히 제거했을 때 groundedness가 떨어져야 함)
    null_result = results_by_type.get("null_injection", {})
    null_degradation = null_result.get("degradation_rate", 0.0)

    # numeric_inversion에서도 30% 이상 하락이면 추가 확인
    numeric_result = results_by_type.get("numeric_inversion", {})
    numeric_degradation = numeric_result.get("degradation_rate", 0.0)

    if null_degradation >= 0.5:
        verdict = "PASS"
        reason = (
            f"null_injection 하락률 {null_degradation:.0%} >= 50% — "
            f"응답이 데이터에 근거함을 확인"
        )
    elif numeric_degradation >= 0.3 or null_degradation >= 0.3:
        verdict = "WEAK_PASS"
        reason = (
            f"null={null_degradation:.0%}, numeric={numeric_degradation:.0%} — "
            f"부분적 증거 있음, 데이터 추가 필요"
        )
    else:
        verdict = "FAIL"
        reason = (
            f"null={null_degradation:.0%}, numeric={numeric_degradation:.0%} — "
            f"오류 주입이 groundedness에 영향 없음 (데이터 부족 또는 응답이 데이터 무시)"
        )

    return {
        "n_turns": len(turn_data_list),
        "by_poison_type": results_by_type,
        "overall_verdict": verdict,
        "verdict_reason": reason,
    }
