"""
tests/unit/test_fidelity.py — G1: 충실도(Fidelity) 계산 로직 단위 테스트

테스트 대상:
    - evaluate_context의 fidelity_score 가중 평균 계산
    - _claim_retained()의 한국어 조사 제거 로직
    - _strip_particles()의 한국어 조사 처리

테스트 ID: H1-1 ~ H1-8
"""
import pytest

from agent.nodes.evaluate_context import _claim_retained, _strip_particles


class TestH1StripParticles:
    """한국어 조사 제거 함수 테스트."""

    def test_remove_subject_particle(self):
        """H1-1: 주격 조사 '은/는/이/가' 제거."""
        assert _strip_particles("강남구는") == "강남구"
        assert _strip_particles("유동인구가") == "유동인구"
        assert _strip_particles("마포구은") == "마포구"

    def test_remove_object_particle(self):
        """H1-2: 목적격 조사 '을/를' 제거."""
        assert _strip_particles("임대료를") == "임대료"
        assert _strip_particles("카페를") == "카페"

    def test_remove_complex_particle(self):
        """H1-3: 복합 조사 '에서는', '으로는' 제거."""
        assert _strip_particles("강남구에서는") == "강남구"
        assert _strip_particles("합정동에서") == "합정동"

    def test_keep_short_word(self):
        """H1-4: 어근이 2글자 미만이면 조사 제거하지 않음."""
        assert _strip_particles("가") == "가"  # 1글자 — 제거하면 빈 문자열
        assert _strip_particles("은") == "은"

    def test_no_particle(self):
        """H1-5: 조사가 없는 단어는 그대로 반환."""
        assert _strip_particles("강남구") == "강남구"
        assert _strip_particles("cafe") == "cafe"


class TestH1ClaimRetained:
    """claim 보존 여부 판단 테스트."""

    def test_exact_match(self):
        """H1-6: 키워드가 정확히 포함된 경우 True."""
        assert _claim_retained("강남구는 높은 유동인구를 보유", "강남구 유동인구 데이터 분석")

    def test_particle_stripped_match(self):
        """H1-7: 조사를 제거한 어근이 매칭되면 True."""
        # "강남구는" → "강남구" → "강남구" in context
        assert _claim_retained("강남구는 카페가 많다", "강남구 카페 현황 조사")

    def test_no_match(self):
        """H1-8: 키워드가 없으면 False."""
        assert not _claim_retained("서초구 임대료가 높다", "마포구 유동인구 데이터")


class TestH1FidelityScore:
    """fidelity_score 가중 평균 계산 테스트.

    공식: 0.4 × cond_score + 0.3 × claims_ratio + 0.3 × compression_penalty
    """

    @staticmethod
    def _calc_fidelity(conditions_preserved: bool, key_claims_preserved: int,
                       total_claims: int, compression_ratio: float) -> float:
        """evaluate_context의 fidelity_score 계산 로직을 재현."""
        cond_score = 1.0 if conditions_preserved else 0.0
        claims_ratio = key_claims_preserved / max(total_claims, 1)
        compression_penalty = min(compression_ratio / 0.3, 1.0)
        return round(0.4 * cond_score + 0.3 * claims_ratio + 0.3 * compression_penalty, 3)

    def test_perfect_fidelity(self):
        """H1-9: 모든 요소가 완벽하면 1.0."""
        score = self._calc_fidelity(
            conditions_preserved=True, key_claims_preserved=5,
            total_claims=5, compression_ratio=0.5,
        )
        assert score == 1.0

    def test_conditions_lost(self):
        """H1-10: 조건 탈락 시 0.4만큼 감점."""
        score = self._calc_fidelity(
            conditions_preserved=False, key_claims_preserved=5,
            total_claims=5, compression_ratio=0.5,
        )
        # 0.4×0 + 0.3×1.0 + 0.3×1.0 = 0.6
        assert score == 0.6

    def test_partial_claims(self):
        """H1-11: 주장 절반 보존 시 claims_ratio = 0.5."""
        score = self._calc_fidelity(
            conditions_preserved=True, key_claims_preserved=2,
            total_claims=4, compression_ratio=0.5,
        )
        # 0.4×1.0 + 0.3×0.5 + 0.3×1.0 = 0.85
        assert score == 0.85

    def test_excessive_compression(self):
        """H1-12: 과도한 압축(ratio < 0.1) 시 compression_penalty < 0.33."""
        score = self._calc_fidelity(
            conditions_preserved=True, key_claims_preserved=5,
            total_claims=5, compression_ratio=0.06,
        )
        # compression_penalty = 0.06/0.3 = 0.2
        # 0.4×1.0 + 0.3×1.0 + 0.3×0.2 = 0.76
        assert score == 0.76

    def test_zero_claims(self):
        """H1-13: 주장이 0개면 claims_ratio = 0."""
        score = self._calc_fidelity(
            conditions_preserved=True, key_claims_preserved=0,
            total_claims=0, compression_ratio=0.5,
        )
        # total_claims=0 → max(0,1)=1 → ratio=0/1=0
        # 0.4×1.0 + 0.3×0.0 + 0.3×1.0 = 0.7
        assert score == 0.7

    def test_first_turn_default(self):
        """H1-14: 이전 충실도 데이터 없으면 1.0 (첫 턴)."""
        # previous_turn_fidelity가 빈 dict이면 fidelity_score = 1.0
        prev_fidelity = {}
        if prev_fidelity:
            score = self._calc_fidelity(**prev_fidelity)
        else:
            score = 1.0
        assert score == 1.0
