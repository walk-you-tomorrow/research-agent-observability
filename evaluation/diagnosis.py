"""
evaluation/diagnosis.py — Post-5: 4D 품질 진단 및 개선 제안

역할:
    4D 점수와 속성값을 분석하여 품질 저하 원인을 진단하고 개선 제안을 생성한다.
    모든 진단은 규칙 기반이며 LLM을 호출하지 않는다.

데이터 흐름:
    입력: scores (4D 점수), trace_data (metadata 포함)
    출력: 진단 결과 리스트 [{dimension, score, diagnosis, suggestion, related_attrs}]

진단 규칙:
    각 4D 차원별로 점수가 임계값 미만이면 관련 속성을 검사하여 원인을 특정한다.
    속성 조합에 따라 구체적인 개선 제안을 생성한다.
    FIX-1: 일관성 패턴 D(Groundedness) 규칙 2개 (hallucination_detected, grounded_claim_ratio < 0.6)
    FIX-2: 일관성 패턴 A(Iteration) 규칙 2개 (confidence_delta < 0, missing_info_resolved=False)
    FIX-3: 관련성 규칙 2개 (noise_ratio > 0.3, source.contribution 불균형)
    F12: rot 진단 규칙 3개 (rot_risk_high, rot_velocity_accelerating, rot_gate_active)는
    4D 점수와 무관하게 rot 지표만으로 트리거된다.
"""

from agent.monitoring_schema import (
    THRESHOLDS,
    get_contradicts_from_metadata,
    get_contradiction_resolved_from_metadata,
)


# --- v2/v3 trace 호환 헬퍼 ---
def _v3_contradicts(metadata: dict) -> bool | None:
    """v2(separate fields) / v3(conflict_tracking dict) 양쪽에서 모순 감지 여부."""
    return get_contradicts_from_metadata(metadata)


def _v3_contradiction_resolved(metadata: dict) -> bool | None:
    """v2/v3 trace에서 모순 해결 여부."""
    return get_contradiction_resolved_from_metadata(metadata)


# --- 진단 규칙 정의 ---
# 각 규칙: {dimension, condition(scores, metadata), diagnosis, suggestion, related_attrs}
_DIAGNOSIS_RULES = [
    # 완전성 (Completeness)
    {
        "dimension": "completeness",
        "check": lambda s, m: s.get("completeness", 1.0) < THRESHOLDS["completeness"] and m.get("gather.items_collected", 99) < 3,
        "diagnosis": f"수집 항목 부족 (completeness < {THRESHOLDS['completeness']} AND items_collected < 3)",
        "suggestion": "도구 계획(tool_plan)에 더 많은 소스를 포함하거나, 수집 전략을 확장하세요",
        "related_attrs": ["gather.items_collected", "gather.tools_called", "query.tool_plan"],
    },
    {
        "dimension": "completeness",
        "check": lambda s, m: s.get("completeness", 1.0) < THRESHOLDS["completeness"] and m.get("context.is_sufficient") is False,
        "diagnosis": f"충분성 판단 실패 (completeness < {THRESHOLDS['completeness']} AND is_sufficient=False)",
        "suggestion": "missing_info에 명시된 데이터를 추가 수집하세요",
        "related_attrs": ["context.is_sufficient", "context.missing_info"],
    },
    # 효율성 (Efficiency)
    {
        "dimension": "efficiency",
        "check": lambda s, m: s.get("efficiency", 1.0) < THRESHOLDS["efficiency"] and m.get("context.noise_ratio", 0) > 0.4,
        "diagnosis": f"이전 턴 데이터 과다 (efficiency < {THRESHOLDS['efficiency']} AND noise_ratio > 0.4)",
        "suggestion": "이전 턴 데이터를 요약하거나 불필요한 결론을 제거하세요",
        "related_attrs": ["context.noise_ratio", "context.source.previous_turns_tokens"],
    },
    {
        "dimension": "efficiency",
        "check": lambda s, m: s.get("efficiency", 1.0) < THRESHOLDS["efficiency"] and m.get("context.window_utilization", 0) > 0.8,
        "diagnosis": f"컨텍스트 윈도우 과다 사용 (efficiency < {THRESHOLDS['efficiency']} AND utilization > 80%)",
        "suggestion": "수집 데이터를 더 적극적으로 요약하거나 소스 수를 줄이세요",
        "related_attrs": ["context.window_utilization", "context.total_tokens"],
    },
    # 관련성 (Relevance)
    {
        "dimension": "relevance",
        "check": lambda s, m: s.get("relevance", 1.0) < THRESHOLDS["relevance"] and m.get("gather.items_excluded", 0) > 2,
        "diagnosis": f"제외 항목 과다 (relevance < {THRESHOLDS['relevance']} AND items_excluded > 2)",
        "suggestion": "도구 선택 전략을 검토하여 관련성 높은 도구만 호출하세요",
        "related_attrs": ["gather.items_excluded", "gather.exclusion_reasons"],
    },
    # FIX-3: 관련성 진단 규칙 보강 (analysis/48)
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            s.get("relevance", 1.0) < THRESHOLDS["relevance"]
            and m.get("context.noise_ratio", 0) > 0.3
        ),
        "diagnosis": f"노이즈 비율 과다 (relevance < {THRESHOLDS['relevance']} AND noise_ratio > 0.3)",
        "suggestion": "수집된 데이터에서 질문과 무관한 항목을 필터링하거나, 도구 쿼리를 더 구체화하세요",
        "related_attrs": ["context.noise_ratio", "context.effective_noise_ratio", "gather.exclusion_reasons"],
    },
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            s.get("relevance", 1.0) < THRESHOLDS["relevance"]
            and isinstance(m.get("source.contribution"), dict)
            and any(v < 0.1 for v in m.get("source.contribution", {}).values())
        ),
        "diagnosis": f"소스 기여도 불균형 — 일부 소스 기여도 10% 미만 (relevance < {THRESHOLDS['relevance']})",
        "suggestion": "기여도 낮은 소스의 도구 선택을 재검토하거나 해당 소스 쿼리 전략을 개선하세요",
        "related_attrs": ["source.contribution", "gather.tools_called", "query.tool_plan"],
    },
    # 일관성 (Consistency)
    {
        "dimension": "consistency",
        "check": lambda s, m: s.get("consistency", 1.0) < THRESHOLDS["consistency"] and m.get("context.fidelity_score", 1.0) < 0.5,
        "diagnosis": f"전달 충실도 저하 (consistency < {THRESHOLDS['consistency']} AND fidelity_score < 0.5)",
        "suggestion": "결론 압축 시 조건·뉘앙스를 보존하도록 압축 방식을 개선하세요",
        "related_attrs": ["context.fidelity_score", "response.conditions_preserved", "response.compression_ratio"],
    },
    {
        "dimension": "consistency",
        # v3 통합 (2026-04-29): contradicts_previous + contradiction_resolved → conflict_tracking dict
        "check": lambda s, m: (
            s.get("consistency", 1.0) < THRESHOLDS["consistency"]
            and _v3_contradicts(m) is True
            and not _v3_contradiction_resolved(m)
        ),
        "diagnosis": f"미해결 모순 (consistency < {THRESHOLDS['consistency']} AND contradicts=True AND resolved=False)",
        "suggestion": "이전 턴 결론과의 모순을 명시적으로 해결하는 설명을 추가하세요",
        "related_attrs": ["analysis.conflict_tracking"],
    },
    # FIX-1: 일관성 패턴 D (Groundedness) 규칙 2개 (analysis/48)
    {
        "dimension": "consistency",
        "check": lambda s, m: (
            s.get("consistency", 1.0) < THRESHOLDS["consistency"]
            and m.get("response.hallucination_detected") is True
        ),
        "diagnosis": f"Groundedness 실패 — 환각 감지 (consistency < {THRESHOLDS['consistency']} AND hallucination_detected=True)",
        "suggestion": "답변의 ungrounded_claims를 확인하고, 컨텍스트에 없는 정보로 주장을 생성하는 원인을 분석하세요",
        "related_attrs": ["response.hallucination_detected", "response.ungrounded_claims", "response.grounded_claim_ratio"],
    },
    {
        "dimension": "consistency",
        "check": lambda s, m: (
            s.get("consistency", 1.0) < THRESHOLDS["consistency"]
            and isinstance(m.get("response.grounded_claim_ratio"), (int, float))
            and m.get("response.grounded_claim_ratio", 1.0) < 0.6
        ),
        "diagnosis": f"Groundedness 저하 — 주장의 40% 이상이 컨텍스트에 근거 없음 (grounded_claim_ratio < 0.6)",
        "suggestion": "generate_analysis에서 컨텍스트 외 추론을 제한하는 프롬프트 제약을 강화하세요",
        "related_attrs": ["response.grounded_claim_ratio", "response.ungrounded_claims"],
    },
    # FIX-2: 일관성 패턴 A (Iteration 내) 규칙 2개 (analysis/48)
    {
        "dimension": "consistency",
        "check": lambda s, m: (
            s.get("consistency", 1.0) < THRESHOLDS["consistency"]
            and isinstance(m.get("context.confidence_delta"), (int, float))
            and m.get("context.confidence_delta", 0) < 0
        ),
        "diagnosis": f"재수집 후 신뢰도 하락 (consistency < {THRESHOLDS['consistency']} AND confidence_delta < 0)",
        "suggestion": "재수집이 오히려 컨텍스트 품질을 낮추는 경우입니다. gather_data의 supplemental 전략을 점검하세요",
        "related_attrs": ["context.confidence_delta", "context.missing_info_resolved", "context.is_sufficient"],
    },
    {
        "dimension": "consistency",
        "check": lambda s, m: (
            s.get("consistency", 1.0) < THRESHOLDS["consistency"]
            and m.get("context.missing_info_resolved") is False
        ),
        "diagnosis": f"재수집 후에도 missing_info 미해결 (consistency < {THRESHOLDS['consistency']} AND missing_info_resolved=False)",
        "suggestion": "missing_info에 명시된 데이터를 추가 수집 경로(다른 도구, 다른 소스)로 해결하세요",
        "related_attrs": ["context.missing_info_resolved", "context.missing_info", "context.is_sufficient"],
    },

    # ── Part 1: 신규 속성 3개 규칙 ─────────────────────────────────────────────
    # query.session_continuity → consistency (User Pivot 탐지)
    # - 독립 트리거: 점수 게이팅 없음.
    # - 세션 궤적에서 갑작스러운 주제 전환은 점수와 무관하게 즉각 알려야 한다.
    #   consistency 점수는 후행 지표(judge가 여러 입력을 집계한 결과)이므로
    #   pivot이 발생한 그 턴에 점수가 낮지 않을 수도 있다.
    # - 임계값 < 0.5: 0~1 스케일에서 0.5는 "어느 쪽도 아님"이 아니라
    #   "의미 있는 방향 전환"의 관찰 가능한 하한이다.
    #   < 0.7이면 정상적인 화제 전환(질문 좁히기 등)까지 포함해 false positive가 높다.
    #   < 0.5면 intent embedding cosine 또는 taxonomy jump가 명확한 경우만 포착한다.
    {
        "dimension": "consistency",
        "check": lambda s, m: (
            isinstance(m.get("query.session_continuity"), (int, float))
            and m.get("query.session_continuity", 1.0) < 0.5
        ),
        "diagnosis": "User Pivot 감지 — 세션 내 쿼리 방향이 급격히 전환되었습니다 (session_continuity < 0.5)",
        "suggestion": (
            "쿼리 의도(intent)가 이전 턴들의 궤적과 크게 다릅니다. "
            "사용자가 의도적으로 주제를 바꾼 경우 이전 turn_conclusions는 "
            "현재 턴에서 노이즈가 될 수 있습니다. "
            "evaluate_context의 conclusion_window를 축소하거나 "
            "이전 결론 참조에 pivot 이전/이후 필터를 적용하세요."
        ),
        "related_attrs": [
            "query.session_continuity",
            "query.intent",
            "context.contributing_turns",
            "context.conclusion_window_size",
        ],
    },
    # analysis.query_alignment → relevance (Agent Drift at analysis stage)
    # - 점수 게이팅 + 속성 독립 트리거 이중 조건.
    # - 설계 근거:
    #   (a) 점수 < 0.7 + alignment < 0.6: relevance 점수가 낮고 분석 단계 drift가
    #       동시에 관측되면 drift가 점수 하락의 원인임을 특정할 수 있다.
    #   (b) 점수 무관 + alignment < 0.4: 극심한 drift는 relevance 점수가 아직
    #       반영되지 않았더라도(judge 지연, 첫 턴 등) 즉각 경보가 필요하다.
    # - < 0.6 임계값: 분석 출력이 쿼리 의도와 60% 이하로 겹칠 때가 사용자가
    #   체감하는 "답이 다른 얘기"의 경험적 경계다.
    #   < 0.7이면 주제가 관련은 있지만 약간 벗어난 정상 케이스도 포함된다.
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            s.get("relevance", 1.0) < THRESHOLDS["relevance"]
            and isinstance(m.get("analysis.query_alignment"), (int, float))
            and m.get("analysis.query_alignment", 1.0) < 0.6
        ),
        "diagnosis": (
            f"Agent Drift (분석 단계) — 분석 내용이 쿼리 의도와 어긋납니다 "
            f"(relevance < {THRESHOLDS['relevance']} AND analysis.query_alignment < 0.6)"
        ),
        "suggestion": (
            "generate_analysis 출력의 주제/초점이 query_analysis.intent와 다릅니다. "
            "생성 프롬프트에 intent를 명시적으로 앵커로 포함하고, "
            "분석 중간에 'intent 체크' 단계를 추가하세요. "
            "gathered_data의 relevance 필터링이 부족한 경우도 점검하세요."
        ),
        "related_attrs": [
            "analysis.query_alignment",
            "query.intent",
            "context.noise_ratio",
            "context.effective_noise_ratio",
            "gather.items_excluded",
        ],
    },
    # analysis.query_alignment 독립 트리거 — 극심한 drift (점수 게이팅 없음)
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            isinstance(m.get("analysis.query_alignment"), (int, float))
            and m.get("analysis.query_alignment", 1.0) < 0.4
        ),
        "diagnosis": "Agent Drift 심각 (분석 단계) — 분석이 쿼리 의도와 매우 낮은 정렬도를 보입니다 (analysis.query_alignment < 0.4)",
        "suggestion": (
            "분석 결과가 사용자 쿼리와 거의 다른 주제를 다루고 있습니다. "
            "generate_analysis 프롬프트의 intent 앵커링을 점검하고, "
            "gathered_data 필터링 단계에서 노이즈 항목을 제거하세요."
        ),
        "related_attrs": [
            "analysis.query_alignment",
            "query.intent",
            "context.noise_ratio",
        ],
    },
    # response.query_alignment → relevance (Agent Drift at response stage)
    # - 점수 게이팅 + 속성 독립 트리거 이중 조건 (analysis 규칙과 동일 패턴).
    # - 응답 단계 drift는 최종 사용자에게 직접 노출되므로 분석 단계보다
    #   더 낮은 임계값(< 0.5)으로 독립 트리거를 설정한다.
    #   분석이 맞아도 응답 압축 과정에서 다른 내용이 나올 수 있다.
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            s.get("relevance", 1.0) < THRESHOLDS["relevance"]
            and isinstance(m.get("response.query_alignment"), (int, float))
            and m.get("response.query_alignment", 1.0) < 0.6
        ),
        "diagnosis": (
            f"Agent Drift (응답 단계) — 최종 응답이 쿼리 의도와 어긋납니다 "
            f"(relevance < {THRESHOLDS['relevance']} AND response.query_alignment < 0.6)"
        ),
        "suggestion": (
            "최종 응답 텍스트가 query_analysis.intent를 충분히 반영하지 않습니다. "
            "respond_to_user 프롬프트에 intent를 명시적으로 앵커로 포함하고, "
            "analysis 결과를 응답으로 압축할 때 핵심 intent 관련 주장이 보존되는지 확인하세요."
        ),
        "related_attrs": [
            "response.query_alignment",
            "query.intent",
            "analysis.query_alignment",
            "context.fidelity_score",
        ],
    },
    # response.query_alignment 독립 트리거 — 극심한 drift (점수 게이팅 없음)
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            isinstance(m.get("response.query_alignment"), (int, float))
            and m.get("response.query_alignment", 1.0) < 0.5
        ),
        "diagnosis": "Agent Drift 심각 (응답 단계) — 최종 응답이 쿼리 의도와 매우 낮은 정렬도를 보입니다 (response.query_alignment < 0.5)",
        "suggestion": (
            "최종 응답이 사용자가 요청하지 않은 내용을 주로 다루고 있습니다. "
            "respond_to_user 단계의 intent 앵커링을 강화하고, "
            "analysis.query_alignment도 함께 점검하세요."
        ),
        "related_attrs": [
            "response.query_alignment",
            "query.intent",
            "analysis.query_alignment",
        ],
    },

    # ── Part 2: 기존 커버리지 갭 Top 3 ──────────────────────────────────────────
    # 선정 기준:
    #   1. context.effective_noise_ratio (relevance, tier 2) — 관련성 커버리지 9%가
    #      가장 취약하며, effective_noise는 noise_ratio보다 정밀한 신호다.
    #      noise_ratio 기존 규칙과 조합하면 "허위 노이즈 경보" 분류가 가능해진다.
    #   2. verify.numeric_discrepancies (completeness, tier 1) — tier 1 속성이면서
    #      규칙이 없다. 수치 오류는 도메인(상권 분석)에서 사용자 신뢰도에 직결된다.
    #   3. analysis.conclusion_utilization (consistency, pattern B) — 이전 결론을
    #      참조했으나 실제로 활용하지 않는 패턴은 패턴 B 일관성의 조용한 실패다.
    #      contributing_turns와 source.conflict_detected는 지금 측정 코드가 완성되지
    #      않은 부분이 있어 즉시 발화 가능성이 낮으므로 3위에서 제외.

    # [Gap 1] context.effective_noise_ratio — 관련성 정밀 신호
    # - 독립 트리거: effective_noise는 인과 영향력 기반으로 필터링된 노이즈다.
    #   이 값이 높으면 noise_ratio가 낮아도 실제 저품질 데이터가 컨텍스트를 채우고 있다.
    # - < 0.4 임계값: noise_ratio의 기존 임계값(> 0.3)과 맞추되,
    #   effective_noise는 더 보수적 측정이므로 0.4에서 독립 경보를 설정한다.
    #   (uncalibrated 상태이므로 0.5보다 낮은 0.4가 안전한 시작점이다.)
    {
        "dimension": "relevance",
        "check": lambda s, m: (
            isinstance(m.get("context.effective_noise_ratio"), (int, float))
            and m.get("context.effective_noise_ratio", 0) > 0.4
        ),
        "diagnosis": "실효 노이즈 과다 — 인과 영향력 기반 저품질 토큰이 40% 초과 (effective_noise_ratio > 0.4)",
        "suggestion": (
            "context.noise_ratio보다 정밀한 effective_noise_ratio가 높습니다. "
            "causal_sources의 impact 분포를 점검하여 impact가 낮은 수집 항목을 "
            "gather_data 단계에서 조기 필터링하거나, "
            "evaluate_context의 context 조합 전략을 개선하세요."
        ),
        "related_attrs": [
            "context.effective_noise_ratio",
            "context.noise_ratio",
            "context.causal_sources",
            "gather.items_excluded",
        ],
    },

    # [Gap 2] verify.numeric_discrepancies — 수치 불일치 건수 (completeness, tier 1)
    # - 독립 트리거: 수치 불일치는 4D 점수와 무관하게 즉각 경보가 필요하다.
    #   verify_result는 점수를 생산하는 단계가 아니라 별도 체크포인트다.
    #   completeness 점수가 정상이어도 수치 오류는 발생할 수 있다.
    # - > 0 임계값: 단 1건의 수치 불일치도 사용자 신뢰 손상 리스크가 있다.
    #   > 1로 하면 단일 오류가 무시된다. 수치 분석 도메인에서는 0이 기준이다.
    {
        "dimension": "completeness",
        "check": lambda s, m: (
            isinstance(m.get("verify.numeric_discrepancies"), int)
            and m.get("verify.numeric_discrepancies", 0) > 0
        ),
        "diagnosis_fn": lambda m: (
            f"수치 불일치 감지 — {m.get('verify.numeric_discrepancies', 0)}건의 "
            f"분석 수치가 pandas 검증과 일치하지 않습니다 (numeric_discrepancies > 0)"
        ),
        "diagnosis": "수치 불일치 감지 — 분석 수치가 pandas 검증과 일치하지 않습니다 (numeric_discrepancies > 0)",
        "suggestion": (
            "verify.issues에서 불일치 항목을 확인하고, "
            "generate_analysis 단계에서 수치 인용 방식을 점검하세요. "
            "pandas_query 결과를 직접 텍스트에 삽입하는 방식을 권장합니다."
        ),
        "related_attrs": [
            "verify.numeric_discrepancies",
            "verify.numeric_check_passed",
            "verify.issues",
            "verify.overall_verdict",
        ],
    },

    # [Gap 3] analysis.conclusion_utilization — 패턴 B 조용한 실패 감지
    # - 점수 게이팅: consistency < 0.7 AND utilization이 낮을 때.
    #   referenced_turns > 0인데 utilization이 낮으면 에이전트가 이전 결론을
    #   "참조한 척"만 하고 실제로 활용하지 않는 패턴 B 실패다.
    # - nullable 속성: None인 경우(referenced_turns == 0)는 규칙을 발화하지 않는다.
    #   첫 턴이나 이전 결론이 없는 경우에는 유효한 진단 대상이 아니다.
    # - < 0.5 임계값: 참조 결론의 절반 이상을 활용하지 않으면 "일관성 유지 의도"가
    #   있어도 실제 연속성이 없다. < 0.7이면 정상 범위까지 포함한다.
    {
        "dimension": "consistency",
        "check": lambda s, m: (
            s.get("consistency", 1.0) < THRESHOLDS["consistency"]
            and isinstance(m.get("analysis.conclusion_utilization"), (int, float))
            and m.get("analysis.conclusion_utilization", 1.0) < 0.5
        ),
        "diagnosis": (
            f"패턴 B 실패 — 이전 결론을 참조했으나 절반 미만만 실제 활용 "
            f"(consistency < {THRESHOLDS['consistency']} AND conclusion_utilization < 0.5)"
        ),
        "suggestion": (
            "analysis.referenced_turns에 비해 analysis.utilized_conclusions가 적습니다. "
            "generate_analysis 프롬프트에서 '참조한 이전 결론을 어떻게 활용했는지' "
            "명시적으로 기술하도록 요구하거나, "
            "lookup_previous 도구 결과를 분석 프롬프트에 직접 포함하세요."
        ),
        "related_attrs": [
            "analysis.conclusion_utilization",
            "analysis.referenced_turns",
            "analysis.utilized_conclusions",
            "context.contributing_turns",
        ],
    },

    # F12: Rot 진단 규칙 3개 — 4D 점수와 무관하게 rot 지표만으로 진단
    {
        "dimension": "efficiency",
        "check": lambda s, m: m.get("context.rot_risk", 0) > 0.3,
        "diagnosis": "Context rot 위험: 이전 턴 데이터 비중이 높습니다. Rot Gate 활성화를 권장합니다.",
        "suggestion": "rot_risk > 0.3 — 이전 턴 결론을 pruning하거나 요약을 강화하세요",
        "related_attrs": ["context.rot_risk", "context.noise_ratio", "context.window_utilization"],
    },
    {
        "dimension": "efficiency",
        "check": lambda s, m: m.get("context.rot_velocity", 0) > 0.05,
        "diagnosis": "Rot 가속 중: rot_risk가 빠르게 증가하고 있습니다.",
        "suggestion": "rot_velocity > 0.05 — 연속 턴에서 rot이 악화되고 있으므로 조기 개입이 필요합니다",
        "related_attrs": ["context.rot_velocity", "context.rot_risk"],
    },
    {
        "dimension": "efficiency",
        "check": lambda s, m: m.get("context.rot_gate_triggered") is True,
        # v3 폐기 (2026-04-29): rot_gate_pruned_tokens는 derived (dead_weight_tokens × rot_gate_triggered).
        # diagnosis_fn은 v2 trace는 rot_gate_pruned_tokens, v3 trace는 dead_weight_tokens 사용.
        "diagnosis_fn": lambda m: (
            f"Rot Gate 활성화됨: "
            f"{m.get('context.rot_gate_pruned_tokens') or m.get('context.dead_weight_tokens', 0)} "
            f"토큰의 low-impact 결론이 pruning되었습니다."
        ),
        "diagnosis": "Rot Gate 활성화됨: low-impact 결론이 pruning되었습니다.",
        "suggestion": "Rot Gate가 트리거되었습니다. pruning된 결론의 품질 영향을 모니터링하세요",
        "related_attrs": ["context.rot_gate_triggered", "context.dead_weight_tokens"],
    },
]


def diagnose_quality(scores: dict[str, float], trace_data: dict) -> list[dict]:
    """4D 점수와 속성값을 분석하여 개선 제안을 생성한다. (LLM 호출 없음)

    Args:
        scores: 4D 점수 딕셔너리 {"completeness": 0.8, "efficiency": 0.5, ...}.
        trace_data: {"metadata": {...}} 구조의 트레이스 데이터.

    Returns:
        진단 결과 리스트. 각 항목:
        {dimension, score, diagnosis, suggestion, related_attrs}
        진단 대상이 없으면 빈 리스트.
    """
    metadata = trace_data.get("metadata", {})
    results = []

    for rule in _DIAGNOSIS_RULES:
        try:
            if rule["check"](scores, metadata):
                # diagnosis_fn이 있으면 metadata 기반 동적 메시지 생성, 없으면 정적 메시지 사용
                diagnosis_msg = (
                    rule["diagnosis_fn"](metadata)
                    if "diagnosis_fn" in rule
                    else rule["diagnosis"]
                )
                results.append({
                    "dimension": rule["dimension"],
                    "score": scores.get(rule["dimension"], 0),
                    "diagnosis": diagnosis_msg,
                    "suggestion": rule["suggestion"],
                    "related_attrs": rule["related_attrs"],
                })
        except (TypeError, KeyError):
            continue

    return results
