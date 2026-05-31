"""
tests/unit/test_anomaly.py — Post-4: 이상 패턴 감지 단위 테스트

테스트 대상:
    - detect_anomaly_patterns(): 4가지 규칙 기반 이상 패턴 감지
    - compute_attribute_trends(): 3턴 이상 연속 하락/상승 감지

테스트 ID: H4-1 ~ H4-12
"""
import pytest

from dashboard.analysis import compute_attribute_trends, detect_anomaly_patterns


# --- 테스트 픽스처 ---

def _turn(turn_num: int, **meta) -> dict:
    """테스트용 턴 데이터 생성."""
    metadata = {"turn.number": turn_num}
    metadata.update(meta)
    return {"turn_number": turn_num, "metadata": metadata}


class TestH4ContextRot:
    """이상 패턴: context rot (noise + rot_risk)."""

    def test_context_rot_detected(self):
        """H4-1: noise_ratio > 0.5 AND rot_risk > 0.3 → context rot 경고."""
        turns = [_turn(3, **{"context.noise_ratio": 0.6, "context.rot_risk": 0.4})]
        anomalies = detect_anomaly_patterns(turns)
        rot = [a for a in anomalies if a["name"] == "context_rot"]
        assert len(rot) == 1
        assert rot[0]["severity"] == "warning"

    def test_no_context_rot_low_noise(self):
        """H4-2: noise_ratio < 0.5이면 감지하지 않음."""
        turns = [_turn(3, **{"context.noise_ratio": 0.3, "context.rot_risk": 0.4})]
        anomalies = detect_anomaly_patterns(turns)
        rot = [a for a in anomalies if a["name"] == "context_rot"]
        assert len(rot) == 0


class TestH4SemanticLoss:
    """이상 패턴: 의미 손실 (키워드 보존되나 의미 왜곡)."""

    def test_semantic_loss_detected_v2(self):
        """H4-3 (v2 trace): fidelity < 0.5 AND continuity > 0.8 → 의미 손실."""
        turns = [_turn(2, **{
            "context.fidelity_score": 0.3,
            "context.continuity_score": 0.9,
        })]
        anomalies = detect_anomaly_patterns(turns)
        sl = [a for a in anomalies if a["name"] == "semantic_loss"]
        assert len(sl) == 1

    def test_semantic_loss_detected_v3(self):
        """v3 trace: continuity_score 폐기 → fidelity_score 단독으로 의미 손실 감지."""
        turns = [_turn(2, **{"context.fidelity_score": 0.3})]  # continuity_score 키 없음
        anomalies = detect_anomaly_patterns(turns)
        sl = [a for a in anomalies if a["name"] == "semantic_loss"]
        assert len(sl) == 1

    def test_no_semantic_loss_high_fidelity(self):
        """H4-4: fidelity > 0.5이면 감지하지 않음 (v2/v3 모두)."""
        turns = [_turn(2, **{
            "context.fidelity_score": 0.8,
            "context.continuity_score": 0.9,
        })]
        anomalies = detect_anomaly_patterns(turns)
        sl = [a for a in anomalies if a["name"] == "semantic_loss"]
        assert len(sl) == 0


class TestH4TokenBloat:
    """이상 패턴: 토큰 팽창 (utilization 높고 density 낮음)."""

    def test_token_bloat_detected(self):
        """H4-5: utilization > 0.8 AND density < 0.2 → 토큰 팽창."""
        turns = [_turn(5, **{
            "context.window_utilization": 0.85,
            "context.information_density": 0.1,
        })]
        anomalies = detect_anomaly_patterns(turns)
        tb = [a for a in anomalies if a["name"] == "token_bloat"]
        assert len(tb) == 1


class TestH4ExcessiveCompression:
    """이상 패턴: 과도한 압축."""

    def test_excessive_compression_detected_v2(self):
        """H4-6 (v2 trace): compression_ratio < 0.1 AND conditions_preserved=False → 과도한 압축."""
        turns = [_turn(4, **{
            "response.compression_ratio": 0.05,
            "response.conditions_preserved": False,
        })]
        anomalies = detect_anomaly_patterns(turns)
        ec = [a for a in anomalies if a["name"] == "excessive_compression"]
        assert len(ec) == 1
        assert ec[0]["severity"] == "error"

    def test_excessive_compression_detected_v3(self):
        """v3 trace: fidelity_detail.compression_appropriateness < 0.3 → 과도한 압축."""
        turns = [_turn(4, **{
            "context.fidelity_detail": {"compression_appropriateness": 0.2},
        })]
        anomalies = detect_anomaly_patterns(turns)
        ec = [a for a in anomalies if a["name"] == "excessive_compression"]
        assert len(ec) == 1


class TestH4MultipleAnomalies:
    """복수 이상 패턴 동시 감지."""

    def test_multiple_patterns_in_one_turn(self):
        """H4-7: 한 턴에서 여러 패턴 동시 감지."""
        turns = [_turn(5, **{
            "context.noise_ratio": 0.7,
            "context.rot_risk": 0.5,
            "context.window_utilization": 0.9,
            "context.information_density": 0.05,
        })]
        anomalies = detect_anomaly_patterns(turns)
        names = {a["name"] for a in anomalies}
        assert "context_rot" in names
        assert "token_bloat" in names

    def test_empty_turns(self):
        """H4-8: 빈 턴 리스트에서도 에러 없음."""
        assert detect_anomaly_patterns([]) == []

    def test_missing_metadata(self):
        """H4-9: metadata에 해당 키가 없으면 규칙 건너뜀."""
        turns = [_turn(1)]  # 빈 metadata
        anomalies = detect_anomaly_patterns(turns)
        assert anomalies == []


class TestH4AttributeTrends:
    """속성 추세 감지 테스트."""

    def test_downtrend_detected(self):
        """H4-10: 3턴 연속 하락 → downtrend 감지."""
        turns = [
            _turn(1, **{"context.noise_ratio": 0.1}),
            _turn(2, **{"context.noise_ratio": 0.2}),
            _turn(3, **{"context.noise_ratio": 0.3}),
            _turn(4, **{"context.noise_ratio": 0.4}),
        ]
        trends = compute_attribute_trends(turns)
        noise_trends = [t for t in trends if t["attribute"] == "context.noise_ratio"]
        assert len(noise_trends) >= 1
        assert noise_trends[0]["direction"] == "up"
        assert noise_trends[0]["consecutive_turns"] >= 3

    def test_no_trend_with_two_turns(self):
        """H4-11: 2턴만 있으면 추세 감지 불가."""
        turns = [
            _turn(1, **{"context.noise_ratio": 0.1}),
            _turn(2, **{"context.noise_ratio": 0.2}),
        ]
        trends = compute_attribute_trends(turns)
        assert trends == []

    def test_no_trend_with_fluctuation(self):
        """H4-12: 오르락내리락이면 추세 없음."""
        turns = [
            _turn(1, **{"context.noise_ratio": 0.1}),
            _turn(2, **{"context.noise_ratio": 0.3}),
            _turn(3, **{"context.noise_ratio": 0.2}),
            _turn(4, **{"context.noise_ratio": 0.4}),
        ]
        trends = compute_attribute_trends(turns)
        noise_trends = [t for t in trends if t["attribute"] == "context.noise_ratio"]
        # 연속 3턴 같은 방향이 없으므로 비어야 함
        assert len(noise_trends) == 0
