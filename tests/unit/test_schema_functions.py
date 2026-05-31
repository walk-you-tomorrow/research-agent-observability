"""
tests/unit/test_schema_functions.py — G4: 스키마 기반 Judge 로딩 단위 테스트

테스트 대상:
    - get_judge_attributes(): YAML judge_input 필드 기반 속성 조회
    - extract_judge_metadata(): metadata에서 judge별 속성 추출
    - attrs_for_producer(): 노드별 속성 조회
    - validate_metadata(): 스키마 검증

테스트 ID: H2-1 ~ H2-10
"""
import pytest

from agent.monitoring_schema import (
    ATTR_META,
    ATTRS,
    THRESHOLDS,
    attrs_for_producer,
    extract_judge_metadata,
    get_judge_attributes,
    validate_metadata,
)


class TestH2GetJudgeAttributes:
    """get_judge_attributes() 함수 테스트."""

    def test_completeness_has_attributes(self):
        """H2-1: completeness judge에 최소 7개 속성."""
        attrs = get_judge_attributes("completeness")
        assert len(attrs) >= 7
        assert "context.is_sufficient" in attrs
        assert "context.sufficiency_confidence" in attrs
        assert "query.intent" in attrs

    def test_efficiency_has_attributes(self):
        """H2-2: efficiency judge에 최소 7개 속성."""
        attrs = get_judge_attributes("efficiency")
        assert len(attrs) >= 7
        assert "context.total_tokens" in attrs
        assert "context.window_utilization" in attrs

    def test_relevance_has_attributes(self):
        """H2-3: relevance judge에 최소 7개 속성."""
        attrs = get_judge_attributes("relevance")
        assert len(attrs) >= 7
        assert "context.noise_ratio" in attrs
        assert "gather.tools_called" in attrs

    def test_consistency_has_attributes(self):
        """H2-4: consistency judge에 최소 6개 속성 (Pattern A+B+C+D).

        v3 통합 (2026-04-29):
          - contradicts_previous/contradiction_resolved/previous_conclusion → conflict_tracking dict
          - continuity_score 폐기 (fidelity_score와 의미 중복)
        """
        attrs = get_judge_attributes("consistency")
        assert len(attrs) >= 6
        # Pattern A
        assert "context.missing_info_resolved" in attrs
        assert "context.confidence_delta" in attrs
        # Pattern B (v3: conflict_tracking으로 통합)
        assert "analysis.conflict_tracking" in attrs or "analysis.contradicts_previous" in attrs
        assert "analysis.referenced_turns" in attrs
        # Pattern C (G1)
        assert "context.fidelity_score" in attrs

    def test_unknown_judge_returns_empty(self):
        """H2-5: 존재하지 않는 judge명은 빈 리스트."""
        attrs = get_judge_attributes("nonexistent_judge")
        assert attrs == []

    def test_multi_judge_attribute(self):
        """H2-6: query.intent은 completeness와 relevance 양쪽에 포함."""
        comp = get_judge_attributes("completeness")
        rel = get_judge_attributes("relevance")
        assert "query.intent" in comp
        assert "query.intent" in rel

    def test_new_g3_attributes_mapped(self):
        """H2-7: G3 새 속성이 적절한 judge에 매핑.

        2026-04-27 4D 재정의 (analysis/31):
          - effective_noise_ratio: relevance 전담 (efficiency에서 제거)

        v3 폐기 (2026-04-29): new_data_ratio 폐기 (derived from gathered_data_tokens / total_tokens).
        """
        eff = get_judge_attributes("efficiency")
        rel = get_judge_attributes("relevance")
        # noise 계열은 relevance 전담 (관련성 차원)
        assert "context.effective_noise_ratio" in rel
        assert "context.effective_noise_ratio" not in eff


class TestH2ExtractJudgeMetadata:
    """extract_judge_metadata() 함수 테스트."""

    def test_extract_completeness(self):
        """H2-8: completeness 속성만 추출."""
        metadata = {
            "context.is_sufficient": True,
            "context.total_tokens": 5000,  # efficiency 속성 — 포함되면 안 됨
            "context.noise_ratio": 0.1,    # relevance 속성 — 포함되면 안 됨
        }
        result = extract_judge_metadata("completeness", metadata)
        assert "context.is_sufficient" in result
        assert "context.total_tokens" not in result  # efficiency 전용
        assert "context.noise_ratio" not in result    # relevance 전용

    def test_missing_keys_excluded(self):
        """H2-9: metadata에 없는 속성은 결과에 미포함."""
        metadata = {"context.is_sufficient": True}  # 나머지 completeness 속성 없음
        result = extract_judge_metadata("completeness", metadata)
        assert "context.is_sufficient" in result
        assert len(result) == 1  # 존재하는 것만

    def test_empty_metadata(self):
        """H2-10: 빈 metadata → 빈 결과."""
        result = extract_judge_metadata("completeness", {})
        assert result == {}


class TestH2JudgeInputFieldConsistency:
    """YAML judge_input 필드의 일관성 검증."""

    def test_all_judge_input_values_valid(self):
        """H2-11: judge_input의 모든 값이 유효한 judge명."""
        valid_judges = {"completeness", "efficiency", "relevance", "consistency", "groundedness"}
        for attr_name, meta in ATTR_META.items():
            judge_input = meta.get("judge_input", [])
            for judge in judge_input:
                assert judge in valid_judges, (
                    f"{attr_name}의 judge_input에 잘못된 judge명: {judge}"
                )

    def test_no_orphan_judge_attributes(self):
        """H2-12: judge_input이 있는 속성은 해당 judge에서 실제로 조회 가능."""
        for attr_name, meta in ATTR_META.items():
            for judge in meta.get("judge_input", []):
                attrs = get_judge_attributes(judge)
                assert attr_name in attrs, (
                    f"{attr_name}이 judge_input에 {judge}를 포함하지만 "
                    f"get_judge_attributes('{judge}')에서 반환되지 않음"
                )


class TestH2AttrsForProducer:
    """attrs_for_producer() 함수 테스트."""

    def test_evaluate_context_produces_most(self):
        """H2-13: evaluate_context가 가장 많은 속성을 생산."""
        attrs = attrs_for_producer("evaluate_context")
        assert len(attrs) >= 15  # 원래 15개 + G1/G3/G5/Post-1/Post-3 추가

    def test_gather_data_produces(self):
        """H2-14: gather_data가 gather.* 속성 생산."""
        attrs = attrs_for_producer("gather_data")
        assert any("gather." in a for a in attrs)
        assert "gather.exclusion_reasons" in attrs  # G5

    def test_unknown_producer_empty(self):
        """H2-15: 존재하지 않는 생산자는 빈 리스트."""
        assert attrs_for_producer("nonexistent_node") == []


class TestH2ValidateMetadata:
    """validate_metadata() 함수 테스트."""

    def test_valid_metadata_passes(self):
        """H2-16: 스키마에 있는 키만 포함하면 통과."""
        metadata = {
            ATTRS["context.total_tokens"]: 5000,
            ATTRS["context.is_sufficient"]: True,
            "tags": ["test"],  # tags는 예외
        }
        validate_metadata(metadata)  # ValueError 없으면 통과

    def test_unknown_key_raises(self):
        """H2-17: 스키마에 없는 키가 있으면 ValueError."""
        metadata = {"unknown.attribute": 42}
        with pytest.raises(ValueError, match="스키마에 정의되지 않은"):
            validate_metadata(metadata)


class Test4DRedefinitionRegression:
    """4D 재정의 회귀 방지 테스트 (2026-04-27 — analysis/31).

    배경: efficiency↔relevance r=0.86 결함이 noise/rot 계열 4개 attribute를
    두 Judge가 공유한 것에서 발생. 향후 실수로 다시 합쳐지는 것을 방지.

    검증 항목:
        1. efficiency ↔ relevance Judge 입력 교집합 = 0 (구조적 차단)
        2. noise/rot 계열은 relevance 또는 efficiency 단독 (양쪽 X)
        3. groundedness 3개는 consistency 통합 (패턴 D)
        4. 신규 자원 attribute 3개 (latency/cost/tool_count)는 efficiency 전담
    """

    def test_efficiency_relevance_no_overlap(self):
        """4D-R1: efficiency ↔ relevance Judge 입력 교집합 = 0.

        이전 결함 (2026-04-08~04-09): noise_ratio, effective_noise_ratio,
        new_data_ratio, rot_risk가 양쪽 입력으로 들어가 r=0.86 발생.
        2026-04-27 재정의 후 구조적으로 차단됨. 회귀 시 이 테스트 실패.
        """
        eff = set(get_judge_attributes("efficiency"))
        rel = set(get_judge_attributes("relevance"))
        overlap = eff & rel
        assert overlap == set(), (
            f"efficiency ↔ relevance 교집합이 0이어야 함. "
            f"현재 {len(overlap)}개: {sorted(overlap)}. "
            f"r=0.86 결함 회귀 가능성. analysis/31 § 2.2/2.3 참조."
        )

    def test_noise_rot_in_relevance_only(self):
        """4D-R2: noise 계열 attribute는 relevance 전담 (efficiency에 없음)."""
        noise_attrs = ["context.noise_ratio", "context.effective_noise_ratio"]
        eff = set(get_judge_attributes("efficiency"))
        rel = set(get_judge_attributes("relevance"))
        for attr in noise_attrs:
            assert attr in rel, f"{attr}는 relevance Judge 입력이어야 함"
            assert attr not in eff, (
                f"{attr}는 efficiency에 있으면 안 됨. "
                f"noise는 relevance 전담 (analysis/31 § 2.3)"
            )

    def test_rot_in_efficiency_only(self):
        """4D-R3: rot_risk는 efficiency 전담 (자원 누적 관점)."""
        eff = set(get_judge_attributes("efficiency"))
        rel = set(get_judge_attributes("relevance"))
        # rot_risk는 자원 누적 관점 (Charter 의도)
        assert "context.rot_risk" in eff
        assert "context.rot_risk" not in rel

    def test_groundedness_in_consistency(self):
        """4D-R4: Groundedness 3개 attribute는 consistency 패턴 D로 통합.

        2026-04-27 재정의: groundedness를 framework 외부에서 일관성 패턴 D로
        통합. 답변↔컨텍스트 일관성은 일관성 차원에 자연스럽게 속함.
        """
        consistency = set(get_judge_attributes("consistency"))
        groundedness_attrs = [
            "response.grounded_claim_ratio",
            "response.hallucination_detected",
            "response.ungrounded_claims",
        ]
        for attr in groundedness_attrs:
            assert attr in consistency, (
                f"{attr}는 consistency Judge 입력이어야 함 (패턴 D, analysis/31 § 2.4)"
            )

    def test_new_resource_attributes_in_efficiency(self):
        """4D-R5: 신규 자원 attribute 3개는 efficiency 전담.

        2026-04-27 신규 추가:
          - turn.wall_time_ms (기존, judge_input 격상)
          - turn.tool_call_count (신규)
          - turn.total_cost_usd (신규)
        모두 자원(시간/호출/비용) 관점이므로 efficiency.
        """
        eff = set(get_judge_attributes("efficiency"))
        resource_attrs = [
            "turn.wall_time_ms",
            "turn.tool_call_count",
            "turn.total_cost_usd",
        ]
        for attr in resource_attrs:
            assert attr in eff, (
                f"{attr}는 efficiency Judge 입력이어야 함 "
                f"(자원 관점, analysis/31 § 4)"
            )

    def test_redundancy_in_efficiency_only(self):
        """4D-R6: redundancy_ratio는 efficiency 전담 (자원 낭비 관점)."""
        eff = set(get_judge_attributes("efficiency"))
        rel = set(get_judge_attributes("relevance"))
        assert "context.redundancy_ratio" in eff
        assert "context.redundancy_ratio" not in rel

    def test_4d_judges_have_orthogonal_inputs(self):
        """4D-R7: 4D Judge 6쌍 모두 교집합 ≤ 2개 (직교성 보장).

        목표: 어떤 두 Judge도 동일한 입력 attribute를 다수 공유하지 않음.
        H2 r=0.86 결함이 다른 차원 쌍에서도 발생하지 않도록 사전 차단.
        """
        import itertools
        judges = ["completeness", "efficiency", "relevance", "consistency"]
        sets = {j: set(get_judge_attributes(j)) for j in judges}
        for a, b in itertools.combinations(judges, 2):
            overlap = sets[a] & sets[b]
            assert len(overlap) <= 2, (
                f"{a} ↔ {b} 교집합이 {len(overlap)}개 (목표: ≤ 2). "
                f"교집합: {sorted(overlap)}. "
                f"4D 직교성 위반 가능성. analysis/31 § 3 참조."
            )
