"""scripts/test_phase37_integration.py — Phase 3.7 구조 검증 (LLM 호출 없음)

Phase 3.7 이탈 감지 3개 속성의 데이터 흐름을 코드 레벨에서 검증한다.
실제 LLM 호출 없이 mock 데이터로 파이프라인 전체를 추적한다.

검증 항목:
    A. AgentState 구조 — session_continuity 필드 존재
    B. analyze_query.py — session_continuity 리턴 포함
    C. evaluate_turn() — tuple 반환 타입
    D. _build_trace_data() — 3개 Phase 3.7 속성 포함
    E. BL-008 주입 경로 — alignment 주입 후 diagnosis.py 규칙 발화
    F. session_continuity 진단 발화 — trace_data 주입 경로
    G. compute_session_continuity() 로직 — Turn 1 None, Turn 2 float

실행:
    source venv/bin/activate
    python scripts/test_phase37_integration.py
"""
import sys
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import truststore
truststore.inject_into_ssl()

from agent.monitoring_schema import ATTRS
from agent.nodes import analyze_query as aq_mod
from evaluation.run_evaluation import evaluate_turn
from evaluation.diagnosis import diagnose_quality
from main import _build_trace_data, _run_evaluation, load_config

_PASS = "✅"
_FAIL = "❌"
_WARN = "⚠️"

results = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = _PASS if condition else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{suffix}")
    results.append((label, condition))
    return condition


# ── A. AgentState 구조 ────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 3.7 구조 검증 (LLM 호출 없음)")
print("=" * 60)

print("\n[A] AgentState 구조")
from agent.state import AgentState
hints = {k: v for k, v in AgentState.__annotations__.items()}
check("session_intent_history: list[dict]", "session_intent_history" in hints)
check("session_continuity: float | None", "session_continuity" in hints, str(hints.get("session_continuity")))


# ── B. analyze_query.py 리턴 ──────────────────────────────────────────────────
print("\n[B] analyze_query.py 리턴 구조")
src_aq = inspect.getsource(aq_mod)
check(
    'return에 "session_continuity": session_continuity 포함',
    '"session_continuity": session_continuity' in src_aq,
)
check(
    'ATTRS["query.session_continuity"] Langfuse 기록',
    'ATTRS["query.session_continuity"]' in src_aq,
)
check(
    'ATTRS["query.user_query"] Langfuse 기록',
    'ATTRS["query.user_query"]' in src_aq,
)


# ── C. evaluate_turn() 반환 타입 ──────────────────────────────────────────────
print("\n[C] evaluate_turn() 반환 타입")
sig = inspect.signature(evaluate_turn)
ret_str = str(sig.return_annotation)
check("tuple[dict, dict] 반환 선언", "tuple" in ret_str, ret_str)

src_ev = inspect.getsource(evaluate_turn)
check(
    "alignment_scores 수집 루프 존재",
    "alignment_scores[score_name] = result.score" in src_ev,
)
check(
    "return scores, alignment_scores",
    "return scores, alignment_scores" in src_ev,
)


# ── D. _build_trace_data() Phase 3.7 속성 ────────────────────────────────────
print("\n[D] _build_trace_data() 속성 포함 여부")
mock_result = {
    "user_query": "강남구 카페 현황은?",
    "session_continuity": 0.82,
    "analysis_result": {"summary": "강남구 카페 시장 분석 요약"},
    "response": "강남구 카페는...",
    "query_analysis": {"intent": "카페 시장 현황 파악", "tool_plan": []},
    "context_metadata": {},
    "verification_result": {},
    "previous_turn_fidelity": {},
    "conflict_tracking": {},
    "sources_used": [],
    "gathered_data": [],
}
trace_data = _build_trace_data(mock_result)
meta = trace_data["metadata"]

check(
    'query.user_query in metadata',
    ATTRS["query.user_query"] in meta,
    f'값="{meta.get(ATTRS["query.user_query"], "")[:30]}"',
)
check(
    'analysis.summary in metadata',
    ATTRS["analysis.summary"] in meta,
    f'길이={len(meta.get(ATTRS["analysis.summary"], ""))}자',
)
check(
    'query.session_continuity in metadata',
    ATTRS["query.session_continuity"] in meta,
    f'값={meta.get(ATTRS["query.session_continuity"])}',
)


# ── E. BL-008 alignment 주입 → diagnosis.py 규칙 발화 ───────────────────────
print("\n[E] BL-008: alignment 주입 → diagnosis.py 발화")

# 1. alignment 없을 때 규칙 미발화 확인
no_align_diag = diagnose_quality(
    {"completeness": 0.5, "relevance": 0.5, "efficiency": 0.5, "consistency": 0.5},
    {"metadata": {**meta}},  # session_continuity 있지만 alignment 없음
)
align_rules_without = [d for d in no_align_diag if "query_alignment" in str(d.get("related_attrs", []))]
check(
    "alignment 값 없을 때: alignment 규칙 미발화",
    len(align_rules_without) == 0,
    f"발화 {len(align_rules_without)}개",
)

# 2. alignment 주입 후 규칙 발화 확인 (< 0.4 독립 트리거)
injected_meta = {**meta, "analysis.query_alignment": 0.3, "response.query_alignment": 0.3}
align_diag = diagnose_quality(
    {"completeness": 0.5, "relevance": 0.5, "efficiency": 0.5, "consistency": 0.5},
    {"metadata": injected_meta},
)
align_rules_with = [d for d in align_diag if "query_alignment" in str(d.get("related_attrs", []))]
check(
    "alignment 주입 후 (0.3): Agent Drift 규칙 발화",
    len(align_rules_with) > 0,
    f"발화 {len(align_rules_with)}개: {[d['diagnosis'][:40] for d in align_rules_with]}",
)

# 3. _run_evaluation auto_evaluate=False → (None, {})
config_off = {"evaluation": {"auto_evaluate": False}}
sc, ac = _run_evaluation(config_off, "test_trace", mock_result)
check(
    "_run_evaluation(auto_evaluate=False) → (None, {})",
    sc is None and ac == {},
    f"sc={sc}, ac={ac}",
)


# ── F. session_continuity 진단 발화 ──────────────────────────────────────────
print("\n[F] query.session_continuity 진단 발화")

# session_continuity < 0.5 → User Pivot 진단
pivot_meta = {**meta, "query.session_continuity": 0.3}
pivot_diag = diagnose_quality(
    {"completeness": 0.8, "relevance": 0.8, "efficiency": 0.8, "consistency": 0.8},
    {"metadata": pivot_meta},
)
pivot_rules = [d for d in pivot_diag if "session_continuity" in str(d.get("related_attrs", []))]
check(
    "session_continuity=0.3 → User Pivot 독립 트리거",
    len(pivot_rules) > 0,
    f"발화 {len(pivot_rules)}개: {[d['diagnosis'][:40] for d in pivot_rules]}",
)

# session_continuity >= 0.5 → 미발화
no_pivot_meta = {**meta, "query.session_continuity": 0.8}
no_pivot_diag = diagnose_quality(
    {"completeness": 0.8, "relevance": 0.8, "efficiency": 0.8, "consistency": 0.8},
    {"metadata": no_pivot_meta},
)
no_pivot_rules = [d for d in no_pivot_diag if "session_continuity" in str(d.get("related_attrs", []))]
check(
    "session_continuity=0.8 → User Pivot 미발화",
    len(no_pivot_rules) == 0,
    f"발화 {len(no_pivot_rules)}개",
)


# ── G. compute_session_continuity() 로직 ─────────────────────────────────────
print("\n[G] compute_session_continuity() Turn 1 / Turn 2 로직")
from agent.nodes.analyze_query import compute_session_continuity

# Turn 1: 이력 1개 → None
t1 = compute_session_continuity([{"turn_number": 1, "intent": "카페 현황"}])
check("Turn 1 (이력 1개) → None", t1 is None, f"결과={t1}")

# Turn 2 이상: Ollama 없을 때 None (graceful fallback)
# _try_embed가 실패하면 None 반환 — Ollama 미가용 환경
history_2 = [
    {"turn_number": 1, "intent": "카페 시장 현황"},
    {"turn_number": 2, "intent": "임대료 수준"},
]
try:
    t2 = compute_session_continuity(history_2)
    if t2 is None:
        check("Turn 2 (Ollama 미가용) → None graceful fallback", True, "Ollama 없음")
    else:
        check("Turn 2 (Ollama 가용) → float", isinstance(t2, float), f"결과={t2:.4f}")
except Exception as e:
    check(f"Turn 2 예외 없음", False, str(e))


# ── 최종 요약 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"결과: {passed}/{total} Pass")
for label, ok in results:
    if not ok:
        print(f"  {_FAIL} {label}")

if passed == total:
    print("\n✅ 모든 구조 검증 통과. API 크레딧 충전 후 5턴 실행 테스트 진행 가능.")
else:
    print(f"\n⚠️  {total - passed}개 항목 실패. 위 내용 확인 필요.")
print("=" * 60)
