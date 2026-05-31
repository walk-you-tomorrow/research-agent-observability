"""
tests/unit/test_detail_log_classify.py — classify_llm_calls의 Judge generation 분류 검증

방법 A 변경(2026-06-01) 회귀 방지:
    4D/정렬 Judge generation은 노드 SPAN 하위가 아니라 턴 trace 루트에 직접
    부착되므로, parent-chain 해석이 아닌 generation 이름 접두사로 식별되어야 한다.
    이 테스트는 Judge가 'evaluation' 카테고리로 분류되고 'unknown'이 되지 않음을 보장한다.
"""
import pytest

from dashboard.analysis import JUDGE_NAME_PREFIXES, classify_llm_calls


def _gen(gen_id: str, name: str, parent_id: str | None, start: str) -> dict:
    """합성 GENERATION observation을 만든다."""
    return {
        "id": gen_id,
        "name": name,
        "type": "GENERATION",
        "parent_observation_id": parent_id,
        "start_time": start,
        "end_time": None,
        "usage": {"input": 10, "output": 5},
        "model": "claude-haiku-4-5",
        "input": "x",
        "output": "y",
    }


@pytest.mark.unit
def test_judge_4d_generation_classified_as_evaluation():
    """trace 루트 직속 judge_4d.* generation은 evaluation으로 분류된다."""
    obs = [
        _gen("g1", "judge_4d.completeness_score", None, "2026-06-01T00:00:01"),
    ]
    calls = classify_llm_calls(obs)
    assert len(calls) == 1
    assert calls[0]["call_category"] == "evaluation"
    assert calls[0]["parent_node"] == "judge_4d.completeness_score"
    assert calls[0]["purpose_label"] == JUDGE_NAME_PREFIXES["judge_4d."]


@pytest.mark.unit
def test_alignment_judge_generation_classified_as_evaluation():
    """judge_alignment.* generation도 evaluation으로 분류된다."""
    obs = [
        _gen("g1", "judge_alignment.analysis.query_alignment", None, "2026-06-01T00:00:01"),
    ]
    calls = classify_llm_calls(obs)
    assert calls[0]["call_category"] == "evaluation"
    assert calls[0]["purpose_label"] == JUDGE_NAME_PREFIXES["judge_alignment."]


@pytest.mark.unit
def test_judge_not_misclassified_as_unknown():
    """부모 SPAN이 없어도 Judge는 unknown이 되지 않는다."""
    obs = [
        _gen("g1", "judge_4d.efficiency_score", None, "2026-06-01T00:00:01"),
        _gen("g2", "judge_alignment.response.query_alignment", None, "2026-06-01T00:00:02"),
    ]
    calls = classify_llm_calls(obs)
    assert all(c["parent_node"] != "unknown" for c in calls)
    assert all(c["call_category"] == "evaluation" for c in calls)


@pytest.mark.unit
def test_node_generation_still_classified_by_parent_chain():
    """일반 노드 generation은 기존대로 parent-chain으로 분류된다 (회귀 방지)."""
    obs = [
        {
            "id": "span1", "name": "analyze_query", "type": "SPAN",
            "parent_observation_id": None, "start_time": "2026-06-01T00:00:00",
            "end_time": None, "usage": {}, "model": None,
        },
        _gen("g1", "analyze_query.intent_plan", "span1", "2026-06-01T00:00:01"),
    ]
    calls = classify_llm_calls(obs)
    node_call = next(c for c in calls if c["parent_node"] == "analyze_query")
    assert node_call["call_category"] == "user_task"


@pytest.mark.unit
def test_mixed_node_and_judge_calls():
    """노드 호출 + Judge 호출이 섞여도 각각 올바른 카테고리로 분류된다."""
    obs = [
        {
            "id": "span1", "name": "analyze_query", "type": "SPAN",
            "parent_observation_id": None, "start_time": "2026-06-01T00:00:00",
            "end_time": None, "usage": {}, "model": None,
        },
        _gen("g1", "analyze_query.intent_plan", "span1", "2026-06-01T00:00:01"),
        _gen("g2", "judge_4d.completeness_score", None, "2026-06-01T00:00:10"),
        _gen("g3", "judge_4d.relevance_score", None, "2026-06-01T00:00:11"),
        _gen("g4", "judge_alignment.analysis.query_alignment", None, "2026-06-01T00:00:12"),
    ]
    calls = classify_llm_calls(obs)
    cats = [c["call_category"] for c in calls]
    assert cats.count("evaluation") == 3
    assert cats.count("user_task") == 1
    assert "unknown" not in [c["parent_node"] for c in calls]
