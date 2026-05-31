"""
tests/unit/test_diagnosis.py — Post-5: 품질 진단 모듈 단위 테스트

테스트 대상:
    - diagnose_quality(): 6개 기본 진단 규칙 + F12 rot 진단 규칙 3개
    - 정상 점수일 때 빈 결과 반환
    - 복합 조건 (여러 규칙 동시 트리거)

테스트 ID: H3-1 ~ H3-10, F12-1 ~ F12-4
"""
import pytest

from evaluation.diagnosis import diagnose_quality


# --- 테스트 픽스처 ---

def _scores(**overrides) -> dict:
    """기본 4D 점수 (모두 양호)."""
    base = {"completeness": 0.9, "efficiency": 0.8, "relevance": 0.9, "consistency": 0.9}
    base.update(overrides)
    return base


def _trace(**meta_overrides) -> dict:
    """기본 trace_data (모두 정상)."""
    base_meta = {
        "gather.items_collected": 5,
        "context.is_sufficient": True,
        "context.noise_ratio": 0.1,
        "context.window_utilization": 0.3,
        "gather.items_excluded": 0,
        "context.fidelity_score": 0.9,
        # v3 통합 (2026-04-29): contradicts_previous + contradiction_resolved → conflict_tracking dict
        "analysis.conflict_tracking": {
            "detected": False,
            "resolution": {"has_explanation": False, "conflict_summary": "", "source_resolution": ""},
        },
    }
    base_meta.update(meta_overrides)
    return {"metadata": base_meta}


class TestH3NoDiagnosis:
    """정상 점수일 때 진단 없음."""

    def test_all_pass_no_diagnosis(self):
        """H3-1: 모든 점수가 임계값 이상이면 빈 리스트."""
        result = diagnose_quality(_scores(), _trace())
        assert result == []

    def test_empty_metadata(self):
        """H3-2: 빈 metadata에서도 에러 없이 동작."""
        result = diagnose_quality(_scores(), {"metadata": {}})
        assert isinstance(result, list)


class TestH3CompletenessRules:
    """완전성 진단 규칙 테스트."""

    def test_low_items_collected(self):
        """H3-3: completeness < 0.7 AND items_collected < 3 → 수집 부족 진단."""
        result = diagnose_quality(
            _scores(completeness=0.5),
            _trace(**{"gather.items_collected": 2}),
        )
        diag = [d for d in result if d["dimension"] == "completeness"]
        assert len(diag) >= 1
        assert "수집" in diag[0]["diagnosis"] or "gathered" in diag[0]["diagnosis"]

    def test_insufficient(self):
        """H3-4: completeness < 0.7 AND is_sufficient=False → 충분성 실패 진단."""
        result = diagnose_quality(
            _scores(completeness=0.5),
            _trace(**{"context.is_sufficient": False}),
        )
        diag = [d for d in result if "충분성" in d.get("diagnosis", "")]
        assert len(diag) >= 1


class TestH3EfficiencyRules:
    """효율성 진단 규칙 테스트."""

    def test_high_noise(self):
        """H3-5: efficiency < 0.6 AND noise_ratio > 0.4 → 노이즈 과다 진단."""
        result = diagnose_quality(
            _scores(efficiency=0.4),
            _trace(**{"context.noise_ratio": 0.6}),
        )
        diag = [d for d in result if d["dimension"] == "efficiency"]
        assert len(diag) >= 1
        assert "noise" in diag[0]["diagnosis"].lower() or "노이즈" in diag[0]["diagnosis"]

    def test_high_utilization(self):
        """H3-6: efficiency < 0.6 AND utilization > 0.8 → 윈도우 과다 진단."""
        result = diagnose_quality(
            _scores(efficiency=0.4),
            _trace(**{"context.window_utilization": 0.9}),
        )
        diag = [d for d in result if "utilization" in d.get("diagnosis", "").lower() or "윈도우" in d.get("diagnosis", "")]
        assert len(diag) >= 1


class TestH3RelevanceRule:
    """관련성 진단 규칙 테스트."""

    def test_high_exclusion(self):
        """H3-7: relevance < 0.7 AND items_excluded > 2 → 제외 과다 진단."""
        result = diagnose_quality(
            _scores(relevance=0.5),
            _trace(**{"gather.items_excluded": 4}),
        )
        diag = [d for d in result if d["dimension"] == "relevance"]
        assert len(diag) >= 1


class TestH3ConsistencyRules:
    """일관성 진단 규칙 테스트."""

    def test_low_fidelity(self):
        """H3-8: consistency < 0.7 AND fidelity_score < 0.5 → 충실도 저하 진단."""
        result = diagnose_quality(
            _scores(consistency=0.5),
            _trace(**{"context.fidelity_score": 0.3}),
        )
        diag = [d for d in result if "충실도" in d.get("diagnosis", "") or "fidelity" in d.get("diagnosis", "").lower()]
        assert len(diag) >= 1

    def test_unresolved_contradiction(self):
        """H3-9: consistency < 0.7 AND contradicts=True, resolved=False → 미해결 모순.

        v3 통합: contradicts_previous/contradiction_resolved → conflict_tracking dict.
        """
        result = diagnose_quality(
            _scores(consistency=0.5),
            _trace(**{
                "analysis.conflict_tracking": {
                    "detected": True,
                    "resolution": {
                        "has_explanation": False,
                        "conflict_summary": "",
                        "source_resolution": "",
                    },
                },
            }),
        )
        diag = [d for d in result if "모순" in d.get("diagnosis", "")]
        assert len(diag) >= 1


class TestH3MultipleRules:
    """복합 조건 테스트."""

    def test_multiple_diagnoses(self):
        """H3-10: 여러 차원이 동시에 낮으면 복수 진단."""
        result = diagnose_quality(
            _scores(completeness=0.5, efficiency=0.4),
            _trace(**{
                "gather.items_collected": 1,
                "context.noise_ratio": 0.6,
            }),
        )
        dimensions = {d["dimension"] for d in result}
        assert "completeness" in dimensions
        assert "efficiency" in dimensions

    def test_result_structure(self):
        """H3-11: 진단 결과의 구조 검증."""
        result = diagnose_quality(
            _scores(completeness=0.5),
            _trace(**{"gather.items_collected": 1}),
        )
        assert len(result) >= 1
        d = result[0]
        assert "dimension" in d
        assert "score" in d
        assert "diagnosis" in d
        assert "suggestion" in d
        assert "related_attrs" in d
        assert isinstance(d["related_attrs"], list)


class TestF12RotDiagnosisRules:
    """F12: Rot 진단 규칙 테스트 (4D 점수와 무관하게 rot 지표만으로 트리거)."""

    def test_rot_risk_high(self):
        """F12-1: rot_risk > 0.3 → rot 위험 진단."""
        result = diagnose_quality(
            _scores(),  # 모든 점수 양호
            _trace(**{"context.rot_risk": 0.35}),
        )
        diag = [d for d in result if "rot" in d["diagnosis"].lower() and "위험" in d["diagnosis"]]
        assert len(diag) >= 1
        assert "context.rot_risk" in diag[0]["related_attrs"]

    def test_rot_velocity_accelerating(self):
        """F12-2: rot_velocity > 0.05 → rot 가속 진단."""
        result = diagnose_quality(
            _scores(),
            _trace(**{"context.rot_velocity": 0.08}),
        )
        diag = [d for d in result if "가속" in d["diagnosis"]]
        assert len(diag) >= 1
        assert "context.rot_velocity" in diag[0]["related_attrs"]

    def test_rot_gate_active(self):
        """F12-3: rot_gate_triggered=True → rot gate 활성화 진단.

        v3 폐기: rot_gate_pruned_tokens (derived: dead_weight_tokens × rot_gate_triggered).
        대신 dead_weight_tokens를 직접 mock한다.
        """
        result = diagnose_quality(
            _scores(),
            _trace(**{
                "context.rot_gate_triggered": True,
                "context.dead_weight_tokens": 1500,
            }),
        )
        diag = [d for d in result if "Rot Gate" in d["diagnosis"]]
        assert len(diag) >= 1
        # 동적 메시지에 pruned_tokens 값이 포함되어야 한다
        assert "1500" in diag[0]["diagnosis"]

    def test_rot_rules_not_triggered_when_low(self):
        """F12-4: rot 지표가 낮으면 rot 진단이 트리거되지 않는다."""
        result = diagnose_quality(
            _scores(),
            _trace(**{
                "context.rot_risk": 0.1,
                "context.rot_velocity": 0.02,
                "context.rot_gate_triggered": False,
            }),
        )
        rot_diags = [d for d in result if "rot" in d["diagnosis"].lower() or "Rot" in d["diagnosis"]]
        assert len(rot_diags) == 0
