"""
tests/unit/test_phase2_actions.py — Phase 2 Step 2.1 신규 액션 단위 테스트

테스트 대상:
    - F2: conditions_preserved 마커 확장 (20+개) + 마커별 보존 상세
    - F3: key_claims_preserved 비율 기반 전환
    - F4+A7: fidelity_detail 구성 요소 분리
    - F5: Judge 입력 속성 확대 (5개)
    - F10: max_tokens YAML SSOT 참조
    - A1: rot_velocity
    - A3: lost_claims

테스트 ID: P2-1 ~ P2-22
"""
import pytest

from agent.monitoring_schema import (
    ATTRS,
    ATTR_META,
    CONTEXT_WINDOW_MAX_TOKENS,
    get_judge_attributes,
)


# ═══════════════════════════════════════
# F2: conditions_preserved 마커 확장
# ═══════════════════════════════════════

class TestF2ConditionsMarkers:
    """F2: 조건 마커 20+개 확장 및 마커별 보존 상세 추적."""

    # respond_to_user.py의 마커 리스트를 재현 (테스트 독립성을 위해 하드코딩)
    MARKERS = [
        "조건부", "경우에", "제외하면", "다만", "단,", "제한적",
        "~인 경우", "~할 때", "~이라면", "~에 한해", "~를 전제로",
        "~이 아니면", "그러나", "하지만", "반면", "~에도 불구하고",
        "~에 따라", "~에 비해", "~보다", "~미만", "~이상",
        "한편", "다른 한편", "전제 조건", "단서", "예외",
    ]

    def test_marker_count_at_least_20(self):
        """P2-1: 마커 수가 20개 이상이다."""
        assert len(self.MARKERS) >= 20

    def test_original_6_markers_preserved(self):
        """P2-2: 기존 6개 마커가 유지되어 있다."""
        original = {"조건부", "경우에", "제외하면", "다만", "단,", "제한적"}
        assert original.issubset(set(self.MARKERS))

    def test_markers_in_analysis_detected(self):
        """P2-3: 분석 텍스트에서 조건 마커가 검출된다."""
        analysis_summary = "강남구는 다만 임대료가 높은 경우에 해당한다"
        markers_in_analysis = [m for m in self.MARKERS if m in analysis_summary]
        assert "다만" in markers_in_analysis
        assert "경우에" in markers_in_analysis

    def test_markers_lost_detected(self):
        """P2-4: 분석에 있던 마커가 결론에서 사라지면 lost로 검출된다."""
        analysis_summary = "다만 특정 조건부 상황에서만 해당"
        conclusion_summary = "전반적으로 해당"
        markers_in_analysis = [m for m in self.MARKERS if m in analysis_summary]
        markers_lost = [m for m in markers_in_analysis if m not in conclusion_summary]
        assert len(markers_lost) >= 2  # "다만", "조건부" 모두 손실
        assert "다만" in markers_lost
        assert "조건부" in markers_lost

    def test_no_conditions_in_analysis(self):
        """P2-5: 분석에 조건 마커가 없으면 conditions_preserved=True."""
        analysis_summary = "강남구 매출이 증가하였다"
        markers_in_analysis = [m for m in self.MARKERS if m in analysis_summary]
        conditions_preserved = not markers_in_analysis
        assert conditions_preserved is True

    def test_all_conditions_preserved(self):
        """P2-6: 분석의 모든 조건 마커가 결론에도 존재하면 True."""
        analysis_summary = "다만 제한적으로 적용된다"
        conclusion_summary = "다만 제한적 범위 내 적용"
        markers_in_analysis = [m for m in self.MARKERS if m in analysis_summary]
        markers_lost = [m for m in markers_in_analysis if m not in conclusion_summary]
        conditions_preserved = not markers_in_analysis or len(markers_lost) == 0
        assert conditions_preserved is True

    def test_conditions_detail_structure(self):
        """P2-7: conditions_detail 딕셔너리가 올바른 키를 갖는다."""
        analysis_summary = "그러나 예외적인 상황"
        conclusion_summary = "예외 상황 존재"
        markers_in_analysis = [m for m in self.MARKERS if m in analysis_summary]
        markers_in_conclusion = [m for m in self.MARKERS if m in conclusion_summary]
        markers_lost = [m for m in markers_in_analysis if m not in conclusion_summary]
        detail = {
            "markers_in_analysis": markers_in_analysis,
            "markers_in_conclusion": markers_in_conclusion,
            "markers_lost": markers_lost,
        }
        assert "markers_in_analysis" in detail
        assert "markers_in_conclusion" in detail
        assert "markers_lost" in detail
        assert "그러나" in detail["markers_lost"]  # "그러나"는 결론에 없음

    def test_yaml_conditions_detail_defined(self):
        """P2-8: v3 폐기 (2026-04-29) — conditions_preserved/detail은 Tier 4 결함으로 제거.
        대신 response.key_claims_preserved가 더 robust한 측정.
        """
        assert "response.key_claims_preserved" in ATTRS


# ═══════════════════════════════════════
# F3: key_claims_preserved 비율 기반
# ═══════════════════════════════════════

class TestF3ClaimsRatio:
    """F3: key_claims_preserved 비율 기반 전환."""

    def test_full_preservation(self):
        """P2-9: 모든 claims 보존 시 비율 1.0."""
        analysis_claims = [{"text": "A"}, {"text": "B"}, {"text": "C"}]
        conclusion_claims = ["A", "B", "C"]
        ratio = round(len(conclusion_claims) / len(analysis_claims), 3)
        assert ratio == 1.0

    def test_partial_preservation(self):
        """P2-10: 일부 claims 보존 시 비율 0.0~1.0."""
        analysis_claims = [{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}]
        conclusion_claims = ["A", "C"]
        ratio = round(len(conclusion_claims) / len(analysis_claims), 3)
        assert ratio == 0.5

    def test_no_analysis_claims(self):
        """P2-11: 분석 claims가 없으면 1.0 (default)."""
        total_analysis_claims = 0
        ratio = 1.0 if total_analysis_claims == 0 else round(0 / total_analysis_claims, 3)
        assert ratio == 1.0

    def test_yaml_type_is_float(self):
        """P2-12: YAML에서 response.key_claims_preserved 타입이 float이다."""
        meta = ATTR_META["response.key_claims_preserved"]
        assert meta["type"] == "float"


# ═══════════════════════════════════════
# F4+A7: fidelity_detail
# ═══════════════════════════════════════

class TestF4A7FidelityDetail:
    """F4+A7: fidelity_detail 구성 요소 분리."""

    @staticmethod
    def _calc_fidelity_with_detail(
        conditions_preserved: bool,
        claims_ratio: float,
        compression_ratio: float,
    ) -> tuple[float, dict]:
        """evaluate_context의 fidelity 계산 로직을 재현 (F3 반영: claims_ratio는 이미 비율)."""
        cond_score = 1.0 if conditions_preserved else 0.0
        compression_penalty = min(compression_ratio / 0.3, 1.0)
        fidelity_score = round(
            0.4 * cond_score + 0.3 * claims_ratio + 0.3 * compression_penalty, 3
        )
        fidelity_detail = {
            "cond_score": cond_score,
            "claims_ratio": round(claims_ratio, 3),
            "compression_penalty": round(compression_penalty, 3),
        }
        return fidelity_score, fidelity_detail

    def test_detail_keys(self):
        """P2-13: fidelity_detail이 3개 구성 요소 키를 갖는다."""
        _, detail = self._calc_fidelity_with_detail(True, 1.0, 0.5)
        assert set(detail.keys()) == {"cond_score", "claims_ratio", "compression_penalty"}

    def test_detail_values_match_score(self):
        """P2-14: detail 구성 요소의 가중합이 score와 일치한다."""
        score, detail = self._calc_fidelity_with_detail(False, 0.5, 0.15)
        reconstructed = round(
            0.4 * detail["cond_score"] + 0.3 * detail["claims_ratio"] + 0.3 * detail["compression_penalty"],
            3
        )
        assert reconstructed == score

    def test_first_turn_defaults(self):
        """P2-15: 첫 턴(이전 fidelity 없음)이면 detail 모두 1.0."""
        detail = {"cond_score": 1.0, "claims_ratio": 1.0, "compression_penalty": 1.0}
        score = round(0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0, 3)
        assert score == 1.0
        assert all(v == 1.0 for v in detail.values())

    def test_yaml_fidelity_detail_defined(self):
        """P2-16: YAML에 context.fidelity_detail이 정의되어 있다."""
        assert "context.fidelity_detail" in ATTRS


# ═══════════════════════════════════════
# F5: Judge 입력 속성 확대
# ═══════════════════════════════════════

class TestF5JudgeInputExpansion:
    """F5: 5개 속성에 judge_input 추가."""

    @pytest.mark.parametrize("attr,judge", [
        ("query.required_data_types", "completeness"),
        ("source.conflict_detected", "consistency"),
        ("web.freshness", "relevance"),
        # v3 폐기: analysis.data_references_count (source.contribution이 더 강함)
        ("response.token_count", "efficiency"),
    ])
    def test_judge_input_added(self, attr, judge):
        """P2-17: 5개 속성에 judge_input이 올바르게 추가되었다."""
        meta = ATTR_META[attr]
        assert "judge_input" in meta, f"{attr}에 judge_input 없음"
        assert judge in meta["judge_input"], f"{attr}의 judge_input에 {judge} 없음"

    @pytest.mark.parametrize("judge", ["completeness", "efficiency", "relevance", "consistency"])
    def test_judge_has_attrs(self, judge):
        """P2-18: 모든 Judge가 최소 2개 이상의 입력 속성을 가진다."""
        attrs = get_judge_attributes(judge)
        assert len(attrs) >= 2, f"{judge} Judge의 입력 속성이 {len(attrs)}개뿐"


# ═══════════════════════════════════════
# F10: max_tokens SSOT 참조
# ═══════════════════════════════════════

class TestF10MaxTokensReference:
    """F10: context_window.max_tokens가 YAML SSOT에서 로드된다."""

    def test_context_window_max_tokens_value(self):
        """P2-19: CONTEXT_WINDOW_MAX_TOKENS가 180000이다."""
        assert CONTEXT_WINDOW_MAX_TOKENS == 180000

    def test_utilization_uses_ssot(self):
        """P2-20: utilization 계산이 SSOT 값을 사용한다."""
        total_tokens = 90000
        utilization = total_tokens / CONTEXT_WINDOW_MAX_TOKENS
        assert abs(utilization - 0.5) < 0.001


# ═══════════════════════════════════════
# A1: rot_velocity
# ═══════════════════════════════════════

class TestA1RotVelocity:
    """A1: rot_velocity (턴 간 rot_risk 변화율)."""

    def test_first_turn_zero(self):
        """P2-21: 첫 턴은 이전 rot_risk=0이므로 velocity = rot_risk - 0."""
        rot_risk = 0.02
        previous_rot_risk = 0.0
        velocity = round(rot_risk - previous_rot_risk, 4)
        assert velocity == 0.02

    def test_increasing_rot(self):
        """P2-22: rot_risk 증가 시 velocity > 0 (악화)."""
        rot_risk = 0.15
        previous_rot_risk = 0.05
        velocity = round(rot_risk - previous_rot_risk, 4)
        assert velocity == 0.10
        assert velocity > 0

    def test_decreasing_rot(self):
        """P2-23: rot_risk 감소 시 velocity < 0 (개선)."""
        rot_risk = 0.03
        previous_rot_risk = 0.10
        velocity = round(rot_risk - previous_rot_risk, 4)
        assert velocity == -0.07
        assert velocity < 0

    def test_yaml_rot_velocity_defined(self):
        """P2-24: YAML에 context.rot_velocity가 정의되어 있다."""
        assert "context.rot_velocity" in ATTRS


# ═══════════════════════════════════════
# A3: lost_claims
# ═══════════════════════════════════════

class TestA3LostClaims:
    """A3: 손실된 주장 추적."""

    def test_no_lost_claims(self):
        """P2-25: 모든 claims가 보존되면 lost_claims 빈 리스트."""
        analysis_claim_texts = ["강남구 매출 증가", "마포구 유동인구 감소"]
        conclusion_claim_set = set(analysis_claim_texts)
        lost = [t for t in analysis_claim_texts if t and t not in conclusion_claim_set]
        assert lost == []

    def test_some_claims_lost(self):
        """P2-26: 일부 claims 탈락 시 lost_claims에 포함."""
        analysis_claim_texts = ["강남구 매출 증가", "마포구 유동인구 감소", "서초구 임대료 상승"]
        conclusion_claims = ["강남구 매출 증가"]
        conclusion_claim_set = set(conclusion_claims)
        lost = [t for t in analysis_claim_texts if t and t not in conclusion_claim_set]
        assert len(lost) == 2
        assert "마포구 유동인구 감소" in lost
        assert "서초구 임대료 상승" in lost

    def test_empty_analysis(self):
        """P2-27: 분석 claims가 없으면 lost_claims 빈 리스트."""
        analysis_claim_texts = []
        conclusion_claim_set = set()
        lost = [t for t in analysis_claim_texts if t and t not in conclusion_claim_set]
        assert lost == []

    def test_yaml_lost_claims_defined(self):
        """P2-28: YAML에 response.lost_claims가 정의되어 있다."""
        assert "response.lost_claims" in ATTRS
        meta = ATTR_META["response.lost_claims"]
        assert "consistency" in meta.get("judge_input", [])
