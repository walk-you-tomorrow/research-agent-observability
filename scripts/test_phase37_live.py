"""scripts/test_phase37_live.py — Phase 3.7 라이브 통합 테스트 (실제 LLM 호출)

5턴 세션을 실제로 실행하고 Phase 3.7 이탈 감지 속성이
올바르게 기록·전달되는지 검증한다.

검증 항목:
    1. session_continuity: Turn 1=None, Turn 2+=float
    2. Turn 3 Pivot (미용실로 주제 전환) → continuity 하락
    3. alignment_scores 비어 있지 않음 (evaluate_turn 정상 실행)
    4. diagnose_quality에 alignment 발화 여부 (Phase 3.8 BL-008)
    5. 에러 없이 5턴 완주

실행:
    source venv/bin/activate
    python scripts/test_phase37_live.py
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import truststore
truststore.inject_into_ssl()

from main import load_config, _build_trace_data, _run_evaluation
from agent.monitoring_schema import ATTRS
from evaluation.diagnosis import diagnose_quality

_PASS = "✅"
_FAIL = "❌"
_WARN = "⚠️"

# Turn 3에서 의도적 주제 전환 (카페 → 미용실) → session_continuity 하락 기대
QUERIES = [
    "강남구 카페 시장 현황은 어떤가요?",           # T1: 카페, continuity=None
    "강남구 카페의 임대료 수준은 어떻게 되나요?",   # T2: 카페 (high continuity 기대)
    "강남구 미용실 창업 비용은 얼마나 드나요?",     # T3: Pivot (low continuity 기대)
    "마포구 미용실과 비교해주세요.",                # T4: 미용실 계속
    "위 분석을 종합해서 추천 입지를 알려주세요.",   # T5: 종합
]

results = []

def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = _PASS if condition else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{suffix}")
    results.append((label, condition))
    return condition


def main():
    print("=" * 60)
    print("Phase 3.7 라이브 통합 테스트")
    print("=" * 60)

    config = load_config()

    # 원본 _run_evaluation을 래핑해서 alignment_scores를 캡처한다
    captured = {"alignment_scores_per_turn": [], "diagnosis_per_turn": []}
    original_run_eval = None

    import main as main_mod

    original_run_eval = main_mod._run_evaluation

    def patched_run_eval(cfg, trace_id, result):
        sc, ac = original_run_eval(cfg, trace_id, result)
        captured["alignment_scores_per_turn"].append(ac)
        # diagnose 결과도 캡처
        if sc:
            td = _build_trace_data(result)
            td["metadata"].update(ac)
            diag = diagnose_quality(sc, td)
            captured["diagnosis_per_turn"].append(diag)
        else:
            captured["diagnosis_per_turn"].append([])
        return sc, ac

    main_mod._run_evaluation = patched_run_eval

    # ── 세션 실행 ──
    print(f"\n쿼리 {len(QUERIES)}턴 실행 중 (중간 출력 최소화)...")
    start = time.time()
    try:
        from main import run_session
        session = run_session(QUERIES, config)
    except Exception as e:
        print(f"  {_FAIL} 세션 실행 중 예외: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        main_mod._run_evaluation = original_run_eval  # 복원

    elapsed = time.time() - start
    print(f"  완료 ({elapsed:.0f}초)\n")

    turn_results = session.get("turn_results", [])

    # ── 검증 1: 턴 수 ──
    print("[1] 세션 완주")
    check(f"{len(QUERIES)}턴 모두 완료", len(turn_results) == len(QUERIES), f"{len(turn_results)}턴")

    # ── 검증 2: session_continuity 흐름 ──
    print("\n[2] session_continuity 흐름")
    conts = [r.get("session_continuity") for r in turn_results]
    print(f"  턴별 값: {conts}")

    check("Turn 1 = None", conts[0] is None, f"값={conts[0]}")
    if len(conts) > 1:
        check("Turn 2 = float", isinstance(conts[1], float), f"값={conts[1]}")
    if len(conts) > 2:
        t2_str = f"{conts[1]:.4f}" if conts[1] is not None else "None"
        t3_str = f"{conts[2]:.4f}" if conts[2] is not None else "None"
        check(
            "Turn 3 Pivot → Turn 2보다 낮음 (카페→미용실)",
            conts[2] is not None and conts[1] is not None and conts[2] < conts[1],
            f"T2={t2_str}, T3={t3_str}",
        )

    # ── 검증 3: session_intent_history 누적 ──
    print("\n[3] session_intent_history 누적")
    history = session.get("session_intent_history", [])
    check(f"intent 이력 {len(QUERIES)}개", len(history) == len(QUERIES), f"{len(history)}개")
    if history:
        print(f"  T1 intent: {history[0].get('intent', '')[:50]}")
        if len(history) > 2:
            print(f"  T3 intent: {history[2].get('intent', '')[:50]}")

    # ── 검증 4: alignment_scores ──
    print("\n[4] alignment_scores (evaluate_turn 반환)")
    for i, ac in enumerate(captured["alignment_scores_per_turn"]):
        t = i + 1
        has_analysis = "analysis.query_alignment" in ac
        has_response = "response.query_alignment" in ac
        a_val = ac.get("analysis.query_alignment")
        r_val = ac.get("response.query_alignment")
        check(
            f"Turn {t}: analysis.query_alignment + response.query_alignment 존재",
            has_analysis and has_response,
            f"analysis={a_val:.3f if a_val else None}, response={r_val:.3f if r_val else None}",
        )

    # ── 검증 5: BL-008 — diagnosis에 alignment 발화 여부 ──
    print("\n[5] BL-008: diagnose_quality alignment 규칙 발화")
    any_drift = False
    for i, diag in enumerate(captured["diagnosis_per_turn"]):
        align_rules = [d for d in diag if "query_alignment" in str(d.get("related_attrs", []))]
        if align_rules:
            any_drift = True
            print(f"  Turn {i+1} Agent Drift 발화: {align_rules[0]['diagnosis'][:60]}")

    if not any_drift:
        print("  (이번 세션에서 Agent Drift 미발화 — 정상 케이스일 수 있음)")
    check(
        "BL-008 주입 경로 작동 (발화 여부 무관, 규칙 평가 자체 실행)",
        len(captured["alignment_scores_per_turn"]) == len(QUERIES),
        f"{len(captured['alignment_scores_per_turn'])}턴 평가됨",
    )

    # ── 검증 6: _build_trace_data에 session_continuity 포함 ──
    print("\n[6] _build_trace_data session_continuity 포함")
    if turn_results:
        last = turn_results[-1]
        td = _build_trace_data(last)
        sc_in_meta = ATTRS["query.session_continuity"] in td["metadata"]
        sc_val = td["metadata"].get(ATTRS["query.session_continuity"])
        check("마지막 턴 trace_data에 session_continuity 포함", sc_in_meta, f"값={sc_val}")

    # ── 최종 요약 ──
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"결과: {passed}/{total} Pass")
    failed = [(l, ok) for l, ok in results if not ok]
    if failed:
        print("실패 항목:")
        for l, _ in failed:
            print(f"  {_FAIL} {l}")
    else:
        print("✅ Phase 3.7 라이브 통합 검증 완료.")
        print("   Langfuse에서 해당 세션 trace를 열어 다음 항목 확인:")
        print("   - query.session_continuity (Turn 2+)")
        print("   - analysis.query_alignment / response.query_alignment (Scores 탭)")
        print("   - eval.diagnosis (metadata)")
    print("=" * 60)


if __name__ == "__main__":
    main()
