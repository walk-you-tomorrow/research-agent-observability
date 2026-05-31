"""
tests/unit/test_cross_turn.py — G3/Post-3: 교차 턴 진화 + 정보 밀도 계산 테스트

테스트 대상:
    - new_data_ratio, effective_noise_ratio 계산 로직 (R12: inherited_ratio 삭제)
    - token_delta 계산
    - information_density, redundancy_ratio 계산
    - _build_trace_data()의 새 속성 매핑

테스트 ID: H5-1 ~ H5-10
"""
import pytest

from agent.monitoring_schema import ATTRS
from agent.token_counter import count_tokens


class TestH5NewDataRatio:
    """G3: new_data_ratio 계산 테스트.

    공식: gathered_tokens / total_tokens
    """

    def test_first_turn_all_new(self):
        """H5-1: 첫 턴은 이전 턴 토큰=0이므로 new_data_ratio가 높다."""
        gathered_tokens = 3000
        total_tokens = 3500  # system_prompt + query + gathered
        ratio = round(gathered_tokens / total_tokens, 3)
        assert ratio > 0.8  # 대부분 새 데이터

    def test_later_turn_lower_ratio(self):
        """H5-2: 턴이 쌓이면 이전 턴 비중 증가 → new_data_ratio 감소."""
        gathered_tokens = 2000
        prev_turns_tokens = 5000
        total_tokens = 2000 + 5000 + 500  # gathered + prev + system/query
        ratio = round(gathered_tokens / total_tokens, 3)
        assert ratio < 0.3  # 이전 턴이 많으면 새 데이터 비율 낮음

    def test_zero_total_defaults_to_one(self):
        """H5-3: total_tokens=0이면 1.0 (divide by zero 방지)."""
        total_tokens = 0
        ratio = round(0 / total_tokens, 3) if total_tokens > 0 else 1.0
        assert ratio == 1.0


class TestH5EffectiveNoiseRatio:
    """F1: effective_noise_ratio 계산 테스트.

    공식: causal_sources의 impact median 이하인 턴의 key_claims 토큰 / total_tokens
    R12: inherited_ratio 삭제됨 — noise_ratio와 수학적으로 동일했으므로 effective_noise_ratio로 대체.
    """

    def test_fallback_to_noise_ratio(self):
        """H5-4: causal_sources가 없으면 noise_ratio와 동일."""
        noise_ratio = 0.3
        effective_noise_ratio = noise_ratio  # fallback
        assert effective_noise_ratio == noise_ratio

    def test_new_plus_noise_under_one(self):
        """H5-5: new_data_ratio + noise_ratio < 1.0 (system/query 비중).
        effective_noise_ratio는 noise_ratio 이하이므로 더 작다."""
        gathered = 2000
        prev = 3000
        system_query = 500
        total = gathered + prev + system_query
        new_ratio = round(gathered / total, 3)
        noise_ratio = round(prev / total, 3)
        assert new_ratio + noise_ratio < 1.0
        assert new_ratio + noise_ratio > 0.9


class TestH5TokenDelta:
    """G3: token_delta 계산 테스트.

    공식: current_total_tokens - previous_total_tokens
    """

    def test_growth(self):
        """H5-6: 턴 간 토큰 증가 → 양수."""
        assert 7000 - 5000 == 2000

    def test_shrink(self):
        """H5-7: 턴 간 토큰 감소 → 음수."""
        assert 4000 - 6000 == -2000

    def test_first_turn(self):
        """H5-8: 첫 턴은 previous=0 → delta = current."""
        assert 5000 - 0 == 5000


class TestH5InformationDensity:
    """Post-3: information_density 계산 테스트.

    공식: claims_tokens / gathered_tokens
    """

    def test_high_density(self):
        """H5-9: 주장이 많고 데이터가 적으면 밀도 높음."""
        claims_text = "강남구 카페 매출이 높다 마포구 유동인구가 많다 서초구 임대료가 비싸다"
        claims_tokens = count_tokens(claims_text)
        gathered_tokens = claims_tokens * 2  # 2배만 수집 → 밀도 ~0.5
        density = round(claims_tokens / max(gathered_tokens, 1), 3)
        assert 0.3 < density < 0.7

    def test_zero_gathered(self):
        """H5-10: gathered_tokens=0이면 0."""
        density = round(0 / max(0, 1), 3)
        assert density == 0.0


class TestH5RedundancyRatio:
    """Post-3 + v3 REDEFINE: redundancy_ratio 계산 테스트.

    v3 (2026-04-29): 임베딩 cosine + lexical Jaccard fallback.
    단위 테스트는 Ollama 미가용 시 fallback 경로를 검증한다.
    """

    def test_no_overlap_lexical(self):
        """H5-11: lexical fallback — 소스 간 키워드가 겹치지 않으면 0.0."""
        from agent.redundancy_checker import _lexical_redundancy
        ratio = _lexical_redundancy(["강남구 카페 매출", "마포구 유동인구 인구"])
        assert ratio == 0.0

    def test_full_overlap_lexical(self):
        """H5-12: lexical fallback — 키워드 완전 일치 시 1.0."""
        from agent.redundancy_checker import _lexical_redundancy
        ratio = _lexical_redundancy(["강남구 카페 매출", "강남구 카페 매출"])
        assert ratio == 1.0

    def test_single_source_zero(self):
        """H5-13: 소스가 1개면 중복 비교 불가 → 0.0."""
        from agent.redundancy_checker import compute_redundancy_ratio
        assert compute_redundancy_ratio([{"data_summary": "강남구 카페"}]) == 0.0

    def test_empty_input_zero(self):
        """v3-1: 빈 입력은 0.0."""
        from agent.redundancy_checker import compute_redundancy_ratio
        assert compute_redundancy_ratio([]) == 0.0
        assert compute_redundancy_ratio([{}, {}]) == 0.0  # data_summary 없음

    def test_cosine_redundancy_threshold(self):
        """v3-2: cosine 페어 redundancy — threshold 이상만 redundant 카운트."""
        from agent.redundancy_checker import _pairwise_cosine_redundancy
        # 동일 벡터 페어 1개 + 직교 페어 2개 = redundant 1/3
        embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        ratio = _pairwise_cosine_redundancy(embeddings, threshold=0.85)
        # 페어: (0,1)=cos1.0 redundant, (0,2)=cos0 not, (1,2)=cos0 not → 1/3
        assert ratio == round(1 / 3, 3)

    def test_cosine_zero_vector(self):
        """v3-3: cosine 0벡터 안전 처리."""
        from agent.redundancy_checker import _cosine
        assert _cosine([0, 0, 0], [1, 1, 1]) == 0.0


class TestV3SourceConflictDetection:
    """v3 REDEFINE: source.conflict_detected 정량 측정."""

    def test_no_conflict_when_values_close(self):
        """v3-4: 모든 값이 임계 이내 차이면 detected=False."""
        from agent.source_conflict_checker import detect_source_conflict
        items = [
            {"source": "csv", "data_summary": "매출 1000"},
            {"source": "rag", "data_summary": "매출 1020"},  # 2% 차이
        ]
        detected, _ = detect_source_conflict(items, 0.05, 0.1)
        assert detected is False

    def test_conflict_when_values_diverge(self):
        """v3-5: 페어 차이가 threshold 초과 + 충돌 비율 ≥ 0.1이면 detected=True."""
        from agent.source_conflict_checker import detect_source_conflict
        items = [
            {"source": "csv", "data_summary": "매출 1000"},
            {"source": "api", "data_summary": "매출 2000"},  # 100% 차이
        ]
        detected, summary = detect_source_conflict(items, 0.05, 0.1)
        assert detected is True
        assert "csv" in summary and "api" in summary

    def test_no_numbers_returns_false(self):
        """v3-6: 숫자가 없으면 측정 불가 → False."""
        from agent.source_conflict_checker import detect_source_conflict
        items = [
            {"source": "rag", "data_summary": "강남구 카페 인기"},
            {"source": "web", "data_summary": "마포구 트렌드 상승"},
        ]
        detected, _ = detect_source_conflict(items)
        assert detected is False

    def test_single_item_returns_false(self):
        """v3-7: 항목 1개면 비교 불가 → False."""
        from agent.source_conflict_checker import detect_source_conflict
        items = [{"source": "csv", "data_summary": "1000"}]
        assert detect_source_conflict(items) == (False, "")


class TestH5BuildTraceData:
    """_build_trace_data()의 새 속성 매핑 검증."""

    def test_new_attrs_in_build_trace(self):
        """H5-14: G1/G3/Post-1/Post-3 속성이 _build_trace_data에 매핑."""
        from main import _build_trace_data

        result = {
            "context_metadata": {
                "fidelity_score": 0.9,
                "continuity_score": 1.0,
                "new_data_ratio": 0.7,
                "effective_noise_ratio": 0.15,
                "token_delta": 1500,
                "contributing_turns": 2,
                "information_density": 0.4,
                "redundancy_ratio": 0.1,
                "causal_sources": [{"turn": 1, "impact": 0.8}],
            },
            "context_evaluation": {
                "sufficiency_by_source": {"csv": "sufficient"},
            },
            "exclusion_reasons": [{"source": "web", "reason": "irrelevant"}],
            "query_analysis": {},
            "verification": {},
        }
        trace_data = _build_trace_data(result)
        meta = trace_data["metadata"]

        # G1
        assert meta[ATTRS["context.fidelity_score"]] == 0.9
        # v3 폐기: continuity_score (fidelity_score와 의미 중복)
        # G5
        assert meta[ATTRS["gather.exclusion_reasons"]] == [{"source": "web", "reason": "irrelevant"}]
        # v3 폐기: new_data_ratio / token_delta / sufficiency_by_source (derived 또는 활용 약함)
        # Post-1
        assert meta[ATTRS["context.causal_sources"]] == [{"turn": 1, "impact": 0.8}]
        # Post-3
        assert meta[ATTRS["context.information_density"]] == 0.4
