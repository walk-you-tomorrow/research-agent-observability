"""
evaluation/run_evaluation.py — 4D 평가 실행기 + 이탈 감지 (Phase 3.7)

이 모듈은 Context Monitoring의 4차원 품질 평가를 실행하고,
결과를 Langfuse trace에 Score로 부착한다.

4차원 평가:
    1. 완전성(Completeness): 필요한 데이터가 모두 수집되었는가?
    2. 효율성(Efficiency): 컨텍스트 토큰이 효율적으로 배분되었는가?
    3. 관련성(Relevance): 수집 데이터가 질문과 관련 있는가?
    4. 일관성(Consistency): 이전 턴과 판단이 일관적인가?

이탈 감지 (Phase 3.7):
    5. 분석 쿼리 정렬(analysis.query_alignment): 분석 결과가 사용자 쿼리에 정렬되었는가?
    6. 응답 쿼리 정렬(response.query_alignment): 최종 응답이 사용자 쿼리에 정렬되었는가?
    ※ 이탈 감지 점수는 4D 기준선 집계에 포함하지 않는다 (기준선 오염 방지).

각 차원은 별도의 Judge 모듈(judge_*.py)이 프롬프트를 구성하고,
LLM(Claude)이 0.0~1.0 점수와 근거를 반환한다.

사용 방법:
    scores = evaluate_turn(trace_id="trace-id", trace_data={...})
    # 반환: {"completeness_score": 0.85, "efficiency_score": 0.9, ...}

    results = evaluate_session(session_traces=[...])
    # 반환: [{"trace_id": "...", "scores": {...}}, ...]

Langfuse Score 부착:
    각 평가 결과는 langfuse.score()로 해당 trace에 직접 부착된다.
    Langfuse 대시보드의 Scores 탭에서 확인할 수 있다.
"""
import json

import truststore

truststore.inject_into_ssl()

from langfuse import Langfuse

from agent.llm import create_llm, invoke_with_retry
from agent.monitoring_schema import CROSS_MODEL_EVALUATORS
from agent.parser import parse_llm_json
from evaluation.judge_completeness import build_completeness_input, PASS_THRESHOLD as COMPLETENESS_THRESHOLD
from evaluation.judge_consistency import build_consistency_input, PASS_THRESHOLD as CONSISTENCY_THRESHOLD
from evaluation.judge_efficiency import build_efficiency_input, PASS_THRESHOLD as EFFICIENCY_THRESHOLD
from evaluation.judge_relevance import build_relevance_input, PASS_THRESHOLD as RELEVANCE_THRESHOLD
from evaluation.judge_query_alignment import (
    build_analysis_alignment_input,
    build_response_alignment_input,
    ALIGNMENT_PASS_THRESHOLD,
)
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class JudgeScore(BaseModel):
    """LLM Judge의 평가 결과 모델.

    Attributes:
        score: 0.0~1.0 범위의 평가 점수. 높을수록 품질이 좋다.
        reasoning: 점수의 근거 설명 (한 문장).
    """
    score: float
    reasoning: str = ""


# --- 4개 Judge 매핑 ---
# Score 이름 → (프롬프트 구성 함수, Pass/Fail 임계값).
# Literature Review #47 (Hamel Husain): numeric score(추세 분석) + binary pass/fail(단일 판단) hybrid 접근.
JUDGES = {
    "completeness_score": (build_completeness_input, COMPLETENESS_THRESHOLD),   # 완전성 Judge
    "efficiency_score": (build_efficiency_input, EFFICIENCY_THRESHOLD),          # 효율성 Judge
    "relevance_score": (build_relevance_input, RELEVANCE_THRESHOLD),            # 관련성 Judge
    "consistency_score": (build_consistency_input, CONSISTENCY_THRESHOLD),       # 일관성 Judge
}


def evaluate_turn(trace_id: str, trace_data: dict) -> tuple[dict[str, float], dict[str, float]]:
    """한 턴(trace)의 4차원 평가를 실행하고 Langfuse에 Score를 부착한다.

    Args:
        trace_id: Langfuse trace ID. Langfuse 대시보드에서 확인 가능.
        trace_data: 트레이스 메타데이터. metadata 키 아래에 Layer 2 attribute를 포함.
                    예: {"metadata": {"context.is_sufficient": True, ...}}

    Returns:
        (scores_4d, alignment_scores) 튜플.
        scores_4d: 4D 점수 딕셔너리. 예: {"completeness_score": 0.85, ...}
        alignment_scores: 이탈 감지 점수 딕셔너리. 예: {"analysis.query_alignment": 0.9, ...}
                          BL-008: diagnose_quality()가 Pattern I/II/III 규칙을 발화하려면
                          alignment_scores를 trace_data["metadata"]에 주입해야 한다.

    처리 과정:
        1. 4개 Judge를 순차 실행
        2. 각 Judge가 구성한 프롬프트를 LLM에 전달
        3. LLM의 JSON 응답을 JudgeScore 모델로 파싱
        4. langfuse.score()로 해당 trace에 Score 부착
        5. 파싱 실패 시 기본 점수 0.5로 폴백
    """
    langfuse = Langfuse()
    llm = create_llm(purpose="evaluation")
    scores = {}

    for score_name, (build_input_fn, threshold) in JUDGES.items():
        # Judge별 평가 프롬프트 구성
        prompt = build_input_fn(trace_data)

        # LLM에 평가 프롬프트를 전달하여 점수와 근거를 받아옴.
        # generation_name으로 Tab 7에서 4D Judge LLM이 차원별로 구분되어 보이도록.
        # trace_id 전달: 이 호출은 _execute_turn의 @observe() 컨텍스트가 닫힌 뒤(밖에서)
        # 실행되므로, trace_id를 명시하지 않으면 generation이 별개 root trace로 떨어져
        # 대시보드 Tab 7(턴 trace의 observation만 조회)에서 보이지 않는다.
        response = invoke_with_retry(
            llm,
            [HumanMessage(content=prompt)],
            generation_name=f"judge_4d.{score_name}",
            trace_id=trace_id,
        )
        try:
            result = parse_llm_json(response.content, JudgeScore)
        except ValueError:
            # JSON 파싱 실패 시: 기본 점수 0.5 (판단 불가)
            result = JudgeScore(score=0.5, reasoning="평가 파싱 실패")

        # Pass/Fail 판정: numeric score는 추세 분석, binary verdict는 단일 판단 신뢰성
        verdict = "PASS" if result.score >= threshold else "FAIL"
        comment = f"[{verdict}, threshold={threshold}] {result.reasoning}"

        # Langfuse trace에 Score 부착
        # 대시보드에서 해당 trace를 열면 Scores 탭에서 확인 가능
        # Langfuse SDK v3: score() → create_score() 변경

        # ① Numeric score: 0.0~1.0 연속값 (추세 분석용)
        langfuse.create_score(
            trace_id=trace_id,
            name=score_name,          # 예: "completeness_score"
            data_type="NUMERIC",      # Langfuse 대시보드에서 히스토그램/시계열 차트 활성화
            value=result.score,        # 예: 0.85
            comment=comment,           # 예: "[PASS, threshold=0.7] 필요한 데이터가 대부분 수집됨"
        )

        # ② Categorical score: PASS/FAIL (필터링/집계용)
        # Langfuse 대시보드에서 "4d_verdict" 필터로 실패 trace만 빠르게 조회 가능
        dimension_label = score_name.replace("_score", "")  # 예: "completeness"
        langfuse.create_score(
            trace_id=trace_id,
            name=f"{dimension_label}_verdict",   # 예: "completeness_verdict"
            data_type="CATEGORICAL",
            value=verdict,                        # "PASS" 또는 "FAIL"
            comment=f"threshold={threshold}, score={result.score:.2f}",
        )
        scores[score_name] = result.score

    # --- 이탈 감지 (Phase 3.7) ---
    # 4D 기준선 집계에 포함하지 않는다 — 기준선 오염 방지.
    # cross_model: monitoring_schema.yaml cross_model_evaluators.alignment_judge 모델 사용.
    # Agent(haiku)와 다른 모델로 self-referential 회피 (independence: cross_model 보장).
    _alignment_model = CROSS_MODEL_EVALUATORS.get("alignment_judge")
    alignment_llm = create_llm(purpose="evaluation", model_override=_alignment_model)

    # score_name은 monitoring_schema.yaml attribute 이름(점 표기) 그대로 사용.
    ALIGNMENT_JUDGES = {
        "analysis.query_alignment": build_analysis_alignment_input,
        "response.query_alignment": build_response_alignment_input,
    }

    # BL-008: alignment 점수를 별도 dict에 수집하여 반환.
    # diagnose_quality()가 Pattern I/II/III 규칙을 발화하려면
    # caller(main.py)가 이 dict를 trace_data["metadata"]에 주입해야 한다.
    alignment_scores: dict[str, float] = {}

    for score_name, build_input_fn in ALIGNMENT_JUDGES.items():
        # 이탈 감지 평가 프롬프트 구성
        prompt = build_input_fn(trace_data)

        # alignment_llm(cross_model)에 평가 프롬프트 전달 — generation_name으로 Langfuse Tab 7에서 구분
        # trace_id 전달: 4D Judge와 동일 이유 (컨텍스트 밖 호출 → 턴 trace 명시 부착)
        response = invoke_with_retry(
            alignment_llm,
            [HumanMessage(content=prompt)],
            generation_name=f"judge_alignment.{score_name}",
            trace_id=trace_id,
        )
        try:
            result = parse_llm_json(response.content, JudgeScore)
        except ValueError:
            # JSON 파싱 실패 시: 기본 점수 0.5 (판단 불가)
            result = JudgeScore(score=0.5, reasoning="평가 파싱 실패")

        # Pass/Fail 판정 (ALIGNMENT_PASS_THRESHOLD 공유 — Phase 3.8에서 속성별 분리 예정)
        verdict = "PASS" if result.score >= ALIGNMENT_PASS_THRESHOLD else "FAIL"
        comment = f"[{verdict}, threshold={ALIGNMENT_PASS_THRESHOLD}] {result.reasoning}"

        # ① Numeric score: 0.0~1.0 연속값 (추세 분석용)
        langfuse.create_score(
            trace_id=trace_id,
            name=score_name,          # 예: "analysis.query_alignment"
            data_type="NUMERIC",
            value=result.score,
            comment=comment,
        )

        # ② Categorical score: PASS/FAIL (필터링/집계용)
        # score_name의 점(.)을 밑줄(_)로 변환하여 verdict 이름 생성
        verdict_name = score_name.replace(".", "_") + "_verdict"
        langfuse.create_score(
            trace_id=trace_id,
            name=verdict_name,        # 예: "analysis_query_alignment_verdict"
            data_type="CATEGORICAL",
            value=verdict,
            comment=f"threshold={ALIGNMENT_PASS_THRESHOLD}, score={result.score:.2f}",
        )

        # BL-008: alignment 점수 수집 (4D 기준선은 오염하지 않음 — scores에는 미포함)
        alignment_scores[score_name] = result.score

    # 버퍼링된 Score를 Langfuse 서버에 전송
    langfuse.flush()
    return scores, alignment_scores


def evaluate_session(session_traces: list[dict]) -> list[dict]:
    """세션 내 모든 턴을 순차 평가한다.

    Args:
        session_traces: 트레이스 리스트. 각 항목은 {"trace_id": str, "metadata": dict} 구조.

    Returns:
        평가 결과 리스트. 각 항목: {"trace_id": str, "scores": dict, "alignment_scores": dict}
    """
    results = []
    for trace in session_traces:
        trace_id = trace["trace_id"]
        scores_4d, alignment_scores = evaluate_turn(trace_id, trace)
        results.append({"trace_id": trace_id, "scores": scores_4d, "alignment_scores": alignment_scores})
    return results
