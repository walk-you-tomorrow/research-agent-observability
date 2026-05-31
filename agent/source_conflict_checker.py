"""
agent/source_conflict_checker.py — 소스 간 정량 충돌 감지 (v3 REDEFINE 2026-04-29)

★ 4D 일관성(Consistency) — 패턴 B 보조 신호 ★

역할:
    gathered_data 항목 간 수치적 충돌을 정량적으로 감지한다.
    LLM 산출(self-referential)에 의존하지 않고 직접 측정.

데이터 흐름:
    입력: list[dict] (gathered_data, data_summary 필드 필요)
    출력: (detected: bool, summary: str) — 충돌 감지 여부 + 근거 텍스트

설계:
    - 각 항목의 data_summary에서 숫자 패턴(\\d+\\.?\\d*) 추출.
    - 페어별 항목 평균값 비교, |Δ| / max ≥ numeric_diff_threshold(yaml 기본 0.05)면 잠재 충돌.
    - 충돌 페어 비율이 conflict_detection_threshold(yaml 기본 0.1) 이상이면 detected=True.
    - 단위/맥락 무시는 한계 — LLM 산출과 OR 결합으로 보완 (호출측에서).

REDEFINE 동기 (analysis/33):
    AS-IS: LLM이 자기 분석 결과의 모순 여부를 self-evaluate (self-referential bias)
    TO-BE: 수치 비교로 정량 측정 (모델 독립)
"""
from __future__ import annotations

import re
from typing import Optional

_NUM_PATTERN = re.compile(r"\d+\.?\d*")


def detect_source_conflict(
    items: list[dict],
    numeric_diff_threshold: float = 0.05,
    conflict_detection_threshold: float = 0.1,
) -> tuple[bool, str]:
    """gathered_data 항목 간 수치 충돌을 감지한다.

    Args:
        items: gathered_data 리스트. 각 항목은 source + data_summary 필드 필요.
        numeric_diff_threshold: 페어 차이 임계 (기본 0.05 = 5%).
        conflict_detection_threshold: 충돌 페어 비율 임계 (기본 0.1 = 10%).

    Returns:
        (detected, summary): 충돌 감지 여부와 상위 3건 근거 텍스트.
        항목 < 2이거나 숫자 추출 < 2이면 (False, "").
    """
    if not items or len(items) < 2:
        return False, ""

    avgs: list[tuple[str, float]] = []
    for item in items:
        text = item.get("data_summary", "")
        if not text:
            continue
        nums = [float(m.group()) for m in _NUM_PATTERN.finditer(text)]
        if nums:
            avgs.append((item.get("source", "unknown"), sum(nums) / len(nums)))

    if len(avgs) < 2:
        return False, ""

    conflicts: list[str] = []
    total_pairs = 0
    for i in range(len(avgs)):
        for j in range(i + 1, len(avgs)):
            s1, v1 = avgs[i]
            s2, v2 = avgs[j]
            denom = max(abs(v1), abs(v2))
            if denom == 0:
                continue
            total_pairs += 1
            diff = abs(v1 - v2) / denom
            if diff > numeric_diff_threshold:
                conflicts.append(
                    f"{s1}({v1:.0f}) vs {s2}({v2:.0f}): {diff * 100:.1f}% 차이"
                )

    if total_pairs == 0:
        return False, ""

    conflict_rate = len(conflicts) / total_pairs
    detected = conflict_rate >= conflict_detection_threshold
    return detected, "; ".join(conflicts[:3])
