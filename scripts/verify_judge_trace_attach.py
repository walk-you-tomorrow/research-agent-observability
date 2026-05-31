"""
scripts/verify_judge_trace_attach.py — 4D Judge generation의 턴 trace 부착 검증 (일회성)

목적:
    방법 A 변경(invoke_with_retry(trace_id=...))이 실제 Langfuse에서
    Judge generation을 턴 trace의 자식으로 부착하는지, 그리고 대시보드의
    load_observations + classify_llm_calls가 이를 'evaluation' 카테고리로
    인식하는지 end-to-end로 검증한다.

    evaluate_turn과 동일 경로:
      ① @observe() 턴 trace 생성 → trace_id 확보 → 컨텍스트 닫힘
      ② 닫힌 뒤(밖에서) invoke_with_retry(..., trace_id=trace_id)로 judge 호출
      ③ flush + 짧은 대기 후 load_observations(trace_id)로 읽어서 분류 확인

실행:
    .venv/bin/python scripts/verify_judge_trace_attach.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse import get_client, observe

load_dotenv()

from agent.llm import create_llm, invoke_with_retry  # noqa: E402
from dashboard.analysis import classify_llm_calls  # noqa: E402
from dashboard.data_loader import load_observations  # noqa: E402

SESSION_ID = "verify_judge_attach"


@observe(name="turn_verify")
def _make_turn_trace() -> str:
    """턴 trace를 생성하고 trace_id를 반환한다 (컨텍스트는 반환 후 닫힘)."""
    client = get_client()
    client.update_current_trace(session_id=SESSION_ID, tags=["verify"])
    # 턴 trace 안에서 일어나는 노드 LLM 호출 1건 시뮬레이션 (컨텍스트 내부 → trace_id 불필요)
    llm = create_llm(purpose="agent")
    invoke_with_retry(
        llm,
        [HumanMessage(content="강남구 상권을 한 문장으로 요약해줘.")],
        generation_name="analyze_query.intent_plan",
    )
    return client.get_current_trace_id()


def main() -> None:
    print("① 턴 trace 생성 + 노드 LLM 1건 (컨텍스트 내부)...")
    trace_id = _make_turn_trace()
    print(f"   trace_id = {trace_id}")

    print("② 턴 trace 닫힌 뒤 Judge LLM 2건 호출 (trace_id 명시 부착)...")
    judge_llm = create_llm(purpose="evaluation")
    for gen_name in ("judge_4d.completeness_score", "judge_alignment.analysis.query_alignment"):
        invoke_with_retry(
            judge_llm,
            [HumanMessage(content='0.0~1.0 점수 하나만 JSON {"score":0.8,"reasoning":"ok"} 형식으로.')],
            generation_name=gen_name,
            trace_id=trace_id,
        )
        print(f"   부착: {gen_name}")

    get_client().flush()

    print("③ Langfuse 수집 대기 + 재시도 조회 (최대 ~50초)...")
    observations = None
    for attempt in range(5):
        time.sleep(10)
        observations = load_observations(trace_id)
        if observations:
            print(f"   조회 성공 (시도 {attempt + 1}): observation {len(observations)}건")
            break
        print(f"   시도 {attempt + 1}: 아직 수집 안 됨")
    if not observations:
        print("   ✘ observation 조회 실패 (수집 지연 — 잠시 후 재실행 권장)")
        sys.exit(1)

    print("④ classify_llm_calls 분류...")

    calls = classify_llm_calls(observations)
    print(f"\n   분류된 LLM 호출 {len(calls)}건:")
    for c in calls:
        print(f"     {c['call_category']:12} | {c['parent_node']:42} | {c['purpose_label']}")

    judge_calls = [c for c in calls if c["call_category"] == "evaluation"]
    print()
    if len(judge_calls) >= 2:
        print(f"   ✅ PASS — Judge generation {len(judge_calls)}건이 턴 trace에 부착되어")
        print("            'evaluation' 카테고리로 상세 탭에 노출됨.")
    else:
        print(f"   ✘ FAIL — Judge generation이 {len(judge_calls)}건만 인식됨 (기대 ≥ 2).")
        sys.exit(1)


if __name__ == "__main__":
    main()
