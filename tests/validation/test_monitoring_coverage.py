"""
tests/validation/test_monitoring_coverage.py — 모니터링 속성 커버리지 검증

YAML SSOT(monitoring_schema.yaml)에 정의된 74개 속성이
코드에서 올바르게 참조되고, 노드별 생산자가 매핑되어 있는지 검증한다.

실제 Langfuse 트레이스를 분석하는 것이 아니라, 코드 수준에서
속성 정의와 사용 패턴의 정합성을 검증한다.

테스트 대상 (E1~E5 + F14):
    E1: 74개 속성이 ATTRS 딕셔너리에 등록되어 있는지
    E2: 4D 평가 임계값이 정의되어 있는지
    E3: 프로세스 5단계에 각각 속성이 있는지
    E4: 소스 기여도(source.contribution) 관련 속성이 정의되어 있는지
    E5: G4 judge_input 매핑 + G1~G5/Post-G5 속성이 등록되어 있는지

실행 방법:
    python -m pytest tests/validation/test_monitoring_coverage.py -v
"""
import os
import sys

import pytest
import yaml

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


# ═══════════════════════════════════════
# E1: 74개 속성이 ATTRS에 등록
# ═══════════════════════════════════════
class TestE1AttributeRegistry:
    """E1: 모든 속성이 ATTRS 딕셔너리에 등록되어 있는지 검증."""

    def test_attrs_count(self):
        """ATTRS에 최소 65개 이상의 속성이 등록되어 있다.

        v3 합리화 (2026-04-29, analysis/33): 88 → 65 활성 + 24 폐기.
        v2 호환: v2 yaml은 88개, v3 yaml은 65개. 양쪽 모두 65개 이상.
        """
        from agent.monitoring_schema import ATTRS
        assert len(ATTRS) >= 65, f"Expected >= 65 attrs, got {len(ATTRS)}"

    def test_attrs_meta_count(self):
        """ATTR_META에 모든 속성의 메타데이터가 있다."""
        from agent.monitoring_schema import ATTRS, ATTR_META
        assert len(ATTR_META) == len(ATTRS), \
            f"ATTR_META ({len(ATTR_META)}) != ATTRS ({len(ATTRS)})"

    def test_core_attributes_exist(self):
        """핵심 속성들이 존재하는지 확인."""
        from agent.monitoring_schema import ATTRS

        # v3 합리화 반영 (2026-04-29): 폐기된 attribute 제거.
        # contradicts_previous → conflict_tracking으로 통합 (v2 호환 유지).
        core_attrs = [
            "context.total_tokens",
            "context.window_utilization",
            "context.is_sufficient",
            "context.sufficiency_confidence",
            "context.noise_ratio",
            "gather.tools_called",
            "gather.items_collected",
            "verify.overall_verdict",
            "source.contribution",
        ]
        for attr in core_attrs:
            assert attr in ATTRS, f"Core attribute '{attr}' missing from ATTRS"

    def test_attrs_identity_mapping(self):
        """ATTRS[key] == key 패턴이 유지되는지 확인."""
        from agent.monitoring_schema import ATTRS

        for key, value in ATTRS.items():
            assert key == value, f"ATTRS['{key}'] = '{value}' (should be '{key}')"

    def test_yaml_schema_loadable(self):
        """monitoring_schema.yaml이 정상적으로 로드되는지 확인 (v2/v3 양립)."""
        schema_path = os.path.join("config", "monitoring_schema.yaml")
        assert os.path.exists(schema_path), f"Schema file not found: {schema_path}"

        with open(schema_path, encoding="utf-8") as f:
            schema = yaml.safe_load(f)

        # v2: attributes 평면 / v3: core_attributes + domain_attributes + partial_dependent_attributes
        assert "attributes" in schema or "core_attributes" in schema, \
            "Schema must have 'attributes' (v2) or 'core_attributes' (v3)"
        # thresholds: v2는 evaluation.thresholds, v3는 constants.thresholds
        assert "evaluation" in schema or "constants" in schema, \
            "Schema must have 'evaluation' (v2) or 'constants' (v3)"

    def test_each_attr_has_required_meta(self):
        """각 속성의 메타데이터에 필수 키(type, lifecycle, producer)가 있다."""
        from agent.monitoring_schema import ATTR_META

        required_keys = {"type", "lifecycle", "producer"}
        for attr_name, meta in ATTR_META.items():
            for key in required_keys:
                assert key in meta, \
                    f"Attribute '{attr_name}' missing meta key '{key}'"


# ═══════════════════════════════════════
# E2: 4D 평가 임계값 정의
# ═══════════════════════════════════════
class TestE2ThresholdsDefined:
    """E2: 4D 평가 임계값이 정의되고 합리적인 범위인지 검증."""

    def test_four_thresholds_exist(self):
        """completeness, efficiency, relevance, consistency 4개 임계값이 정의됨."""
        from agent.monitoring_schema import THRESHOLDS

        expected = ["completeness", "efficiency", "relevance", "consistency"]
        for dim in expected:
            assert dim in THRESHOLDS, f"Threshold '{dim}' missing"

    def test_thresholds_in_range(self):
        """임계값이 0.0~1.0 범위 내에 있다."""
        from agent.monitoring_schema import THRESHOLDS

        for dim, value in THRESHOLDS.items():
            assert 0.0 <= value <= 1.0, \
                f"Threshold '{dim}' = {value} out of [0.0, 1.0]"

    def test_thresholds_reasonable(self):
        """임계값이 합리적인 범위(0.5~0.9)에 있다."""
        from agent.monitoring_schema import THRESHOLDS

        for dim, value in THRESHOLDS.items():
            assert 0.5 <= value <= 0.9, \
                f"Threshold '{dim}' = {value} seems unreasonable"


# ═══════════════════════════════════════
# E3: 프로세스 5단계 커버리지
# ═══════════════════════════════════════
class TestE3LifecycleCoverage:
    """E3: 프로세스 5단계에 각각 최소 1개 이상의 속성이 있는지 검증."""

    def test_all_lifecycle_stages_have_attributes(self):
        """5개 프로세스 단계 모두에 속성이 정의되어 있다.

        v2 lifecycle: 1_plan, 2_collect, 3_organize, 4_generate, 5_memory (prefix 포함).
        v3 lifecycle: plan, collect, organize, generate, memory (prefix 없음).
        """
        from agent.monitoring_schema import ATTR_META, SCHEMA_VERSION

        if SCHEMA_VERSION == "v3":
            expected_stages = {"plan", "collect", "organize", "generate", "memory"}
        else:
            expected_stages = {"1_plan", "2_collect", "3_organize", "4_generate", "5_memory"}
        found_stages = set()

        for attr_name, meta in ATTR_META.items():
            lifecycle = meta.get("lifecycle", "")
            if lifecycle in expected_stages:
                found_stages.add(lifecycle)

        missing = expected_stages - found_stages
        assert not missing, f"Missing lifecycle stages: {missing}"

    def test_each_stage_has_multiple_attributes(self):
        """각 프로세스 단계에 최소 3개 이상의 속성이 있다 (v2/v3 lifecycle naming 양립)."""
        from agent.monitoring_schema import ATTR_META, SCHEMA_VERSION

        stage_counts = {}
        for attr_name, meta in ATTR_META.items():
            lifecycle = meta.get("lifecycle", "")
            stage_counts[lifecycle] = stage_counts.get(lifecycle, 0) + 1

        if SCHEMA_VERSION == "v3":
            stages = ["plan", "collect", "organize", "generate", "memory"]
        else:
            stages = ["1_plan", "2_collect", "3_organize", "4_generate", "5_memory"]

        for stage in stages:
            count = stage_counts.get(stage, 0)
            assert count >= 3, \
                f"Stage '{stage}' has only {count} attributes (expected >= 3)"


# ═══════════════════════════════════════
# E4: 소스 기여도 추적 속성
# ═══════════════════════════════════════
class TestE4SourceContribution:
    """E4: 소스 선택/기여도 관련 속성이 정의되어 있는지 검증."""

    def test_source_namespace_exists(self):
        """source.* 네임스페이스 속성이 정의되어 있다."""
        from agent.monitoring_schema import ATTRS

        source_attrs = [k for k in ATTRS if k.startswith("source.")]
        assert len(source_attrs) >= 3, \
            f"Expected >= 3 source.* attrs, got {len(source_attrs)}: {source_attrs}"

    def test_source_selection_reasoning_exists(self):
        """source.selection_reasoning 속성이 정의되어 있다.

        v3 폐기: source.types_selected (gather.tools_called에서 도출).
        """
        from agent.monitoring_schema import ATTRS
        assert "source.selection_reasoning" in ATTRS

    def test_source_contribution_exists(self):
        """source.contribution 속성이 정의되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "source.contribution" in ATTRS

    def test_source_conflict_attributes(self):
        """소스 간 충돌 관련 속성이 정의되어 있다.

        v3 통합: source.conflict_resolution → analysis.conflict_tracking.resolution.source_resolution.
        """
        from agent.monitoring_schema import ATTRS
        assert "source.conflict_detected" in ATTRS

    def test_web_namespace_exists(self):
        """web.* 네임스페이스 속성이 정의되어 있다.

        v3 폐기 (2026-04-29):
          - web.search_count (gather.iteration이 더 정확)
          - web.result_count (gather.items_collected에 흡수)
        """
        from agent.monitoring_schema import ATTRS

        web_attrs = [k for k in ATTRS if k.startswith("web.")]
        assert len(web_attrs) >= 2, \
            f"Expected >= 2 web.* attrs, got {len(web_attrs)}: {web_attrs}"

    def test_producer_coverage(self):
        """각 주요 노드(analyze_query, gather_data, evaluate_context, generate_analysis, verify_result)가
        최소 1개 이상의 속성을 생산한다."""
        from agent.monitoring_schema import attrs_for_producer

        producers = [
            "analyze_query",
            "gather_data",
            "evaluate_context",
            "generate_analysis",
            "verify_result",
        ]
        for producer in producers:
            attrs = attrs_for_producer(producer)
            assert len(attrs) >= 1, \
                f"Producer '{producer}' has no attributes"


# ═══════════════════════════════════════
# E5: G1~G4 + Post-G5 속성 및 judge_input 매핑
# ═══════════════════════════════════════
class TestE5ObservabilityEnhancements:
    """E5: G1~G4 + Post-G5 관측 체계 보완 속성 검증."""

    def test_g1_fidelity_attributes(self):
        """G1: 충실도(fidelity) 관련 속성이 등록되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "context.fidelity_score" in ATTRS

    def test_g5_exclusion_reason_attributes(self):
        """G5: 탈락 이유 속성이 등록되어 있다.

        v3 통합 (2026-04-29): truncation_reasons → truncated_items list[dict]에 흡수.
        """
        from agent.monitoring_schema import ATTRS
        assert "gather.exclusion_reasons" in ATTRS
        assert "context.truncated_items" in ATTRS

    def test_g3_evolution_attributes(self):
        """G3: 교차 턴 진화 속성이 등록되어 있다.

        v3 폐기 (2026-04-29):
          - new_data_ratio (derived: gathered_data_tokens / total_tokens)
          - token_delta (derived: total_tokens 비교)
          - sufficiency_by_source (Judge 활용 약함)
        """
        from agent.monitoring_schema import ATTRS
        # v3 잔존 G3 attribute
        assert "context.contributing_turns" in ATTRS
        # R12: inherited_ratio가 삭제되었는지 확인
        assert "context.inherited_ratio" not in ATTRS, \
            "R12: context.inherited_ratio should be removed"

    def test_g4_judge_input_exists(self):
        """G4: judge_input 필드가 있는 속성이 최소 25개 이상."""
        from agent.monitoring_schema import ATTR_META
        with_judge = [
            name for name, meta in ATTR_META.items()
            if meta.get("judge_input")
        ]
        assert len(with_judge) >= 25, \
            f"Expected >= 25 attrs with judge_input, got {len(with_judge)}"

    def test_g4_all_judges_have_inputs(self):
        """G4: 4개 Judge 모두 최소 6개 입력 속성을 가진다."""
        from agent.monitoring_schema import get_judge_attributes
        for judge in ["completeness", "efficiency", "relevance", "consistency"]:
            attrs = get_judge_attributes(judge)
            assert len(attrs) >= 6, \
                f"Judge '{judge}' has only {len(attrs)} input attrs (expected >= 6)"

    def test_post1_causal_attributes(self):
        """Post-1: 인과 전파 속성이 등록되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "context.causal_sources" in ATTRS

    def test_post2_contribution_attributes(self):
        """Post-2: 기여도 속성이 등록되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "analysis.conclusion_utilization" in ATTRS
        assert "analysis.utilized_conclusions" in ATTRS

    def test_post3_density_attributes(self):
        """Post-3: 정보 밀도 속성이 등록되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "context.information_density" in ATTRS
        assert "context.redundancy_ratio" in ATTRS

    def test_post5_diagnosis_attributes(self):
        """Post-5: 진단 속성이 등록되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "eval.diagnosis" in ATTRS
        assert "eval.improvement_applied" in ATTRS

    def test_eval_namespace_exists(self):
        """Post-5: eval 네임스페이스가 존재한다."""
        from agent.monitoring_schema import ATTRS
        eval_attrs = [k for k in ATTRS if k.startswith("eval.")]
        assert len(eval_attrs) >= 2, \
            f"Expected >= 2 eval.* attrs, got {len(eval_attrs)}"

    def test_total_namespaces(self):
        """네임스페이스 다양성 검증 — ATTRS 키의 prefix 기준 (v2/v3 양립)."""
        from agent.monitoring_schema import ATTRS
        # ATTRS 키에서 첫 점(.) 앞을 namespace로 추출
        namespaces = {k.split(".", 1)[0] for k in ATTRS}
        # v3 활성 namespace: turn, eval, verify, gather, context, analysis, response, query, source, web
        assert len(namespaces) >= 8, \
            f"Expected >= 8 namespaces, got {len(namespaces)}: {sorted(namespaces)}"

    def test_f1_effective_noise_ratio_exists(self):
        """F1: effective_noise_ratio가 등록되어 있다."""
        from agent.monitoring_schema import ATTRS
        assert "context.effective_noise_ratio" in ATTRS

    def test_a2_messages_tokens_deprecated(self):
        """A2: v3 폐기 (2026-04-29) — messages_tokens는 활용 0 ('관찰 후 결정' 미결).

        v2 호환을 위해 ATTRS 멤버십은 검증하지 않고, A2 검증의 의도("토큰 계측 attribute 존재")는
        context.total_tokens / context.window_utilization 등 자원 attribute로 대체된다.
        """
        from agent.monitoring_schema import ATTRS
        # A2 의도 보존: 토큰 계측 가능
        assert "context.total_tokens" in ATTRS

    def test_f7_response_conclusion_attrs_exist(self):
        """F7: response.conclusion_summary와 response.key_claims가 YAML에 등록."""
        from agent.monitoring_schema import ATTRS
        assert "response.conclusion_summary" in ATTRS, \
            "response.conclusion_summary should be in ATTRS (F7 SSOT)"
        assert "response.key_claims" in ATTRS, \
            "response.key_claims should be in ATTRS (F7 SSOT)"


# ═══════════════════════════════════════
# F14: SSOT 정합성 — 코드 참조 vs YAML 정의
# ═══════════════════════════════════════
class TestF14SSOTIntegrity:
    """F14: 코드에서 ATTRS[...]로 참조하는 키가 YAML에 모두 존재하는지,
    그리고 문자열 리터럴로 metadata에 기록하는 키가 없는지 검증."""

    def test_all_attrs_references_exist_in_yaml(self):
        """코드에서 ATTRS[...]로 참조하는 모든 키가 YAML에 존재한다.

        ATTRS 딕셔너리는 YAML에서 로드되므로, 존재하지 않는 키를 참조하면
        KeyError가 발생하여 이 테스트 없이도 런타임 에러가 발생한다.
        이 테스트는 YAML 파싱 후 ATTRS 딕셔너리의 무결성을 재확인한다.
        """
        from agent.monitoring_schema import ATTRS
        import re

        # 노드 파일에서 ATTRS["..."] 패턴 추출
        node_files = [
            os.path.join("agent", "nodes", f)
            for f in os.listdir(os.path.join("agent", "nodes"))
            if f.endswith(".py")
        ]
        node_files.append(os.path.join("main.py"))

        pattern = re.compile(r'ATTRS\["([^"]+)"\]')
        referenced_keys = set()

        for filepath in node_files:
            if not os.path.exists(filepath):
                continue
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
            matches = pattern.findall(content)
            referenced_keys.update(matches)

        missing = referenced_keys - set(ATTRS.keys())
        assert not missing, \
            f"코드에서 ATTRS[...]로 참조하지만 YAML에 없는 키: {missing}"

    def test_no_string_literal_metadata_keys(self):
        """F7 검증: 노드 코드에서 metadata에 문자열 리터럴로 직접 기록하는 키가 없다.

        허용 예외: 'tags' (Langfuse 예약어)
        """
        import re

        node_dir = os.path.join("agent", "nodes")
        # metadata={...} 블록 내에서 "key.name": value 패턴을 탐지한다
        # ATTRS["..."] 대신 문자열 리터럴을 직접 사용하는 경우를 잡는다
        literal_pattern = re.compile(
            r'^\s+"([a-z_]+\.[a-z_]+)":\s',
            re.MULTILINE,
        )
        allowed_literals = {"tags"}  # Langfuse 예약어

        violations = []
        for filename in os.listdir(node_dir):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(node_dir, filename)
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
            matches = literal_pattern.findall(content)
            for m in matches:
                if m not in allowed_literals:
                    violations.append(f"{filename}: \"{m}\"")

        assert not violations, \
            f"F7 위반 — ATTRS[] 대신 문자열 리터럴 사용: {violations}"
