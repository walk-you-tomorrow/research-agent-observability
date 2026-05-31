"""
agent/models.py — Pydantic 모델 정의 (LLM 출력 파싱용)

이 모듈은 LLM이 JSON으로 반환하는 구조화된 출력을 파싱하기 위한
Pydantic 모델들을 정의한다. 각 모델은 특정 노드에서 사용된다.

사용 흐름:
    LLM 응답(텍스트) → parser.py의 parse_llm_json() → Pydantic 모델 인스턴스
    → model.model_dump()으로 dict 변환 → AgentState에 저장

모델-노드 매핑:
    QueryAnalysis       → analyze_query 노드
    ContextEvaluation   → evaluate_context 노드
    AnalysisResult      → generate_analysis 노드
    InterpretationCheck → verify_result 노드 (STEP 2)
    VerificationResult  → verify_result 노드 (최종 결과)
"""
from typing import Any, Union

from pydantic import BaseModel, Field, field_validator


class QueryAnalysis(BaseModel):
    """사용자 질의 분석 결과. analyze_query 노드에서 LLM이 생성한다.

    LLM은 사용자 질문을 분석하여 의도(intent), 필요한 데이터 유형,
    호출할 도구 계획, 이전 턴 참조 여부를 JSON으로 반환한다.

    Attributes:
        intent: 사용자 의도 (예: "compare_districts", "cafe_location_recommendation")
        required_data: 필요한 CSV 데이터 키 리스트 (예: ["foot_traffic", "rent"])
        required_docs: 필요한 마크다운 문서 키 리스트 (예: ["mapo_profile"])
        tool_plan: 호출할 도구 이름 리스트 (예: ["data_query", "doc_read"])
        references_previous: 이전 턴을 참조하는 질문인지 여부
        referenced_turns: 참조하는 이전 턴 번호 리스트 (예: [2, 3])
        api_params: api_query 도구 호출 시 파라미터 (예: {"api": "commercial_change", "params": {"STDR_YYQU_CD": "20244"}})
    """
    intent: str
    required_data: list[str] = []
    required_docs: list[str] = []
    tool_plan: list[str] = []
    references_previous: bool = False
    referenced_turns: list[int] = []
    api_params: dict | None = None
    source_types: list[str] = []          # ["csv", "rag", "web", "api"]
    source_reasoning: str = ""            # 소스 선택 근거


class ContextEvaluation(BaseModel):
    """컨텍스트 충분성 평가 결과. evaluate_context 노드에서 LLM이 생성한다.

    LLM은 수집된 데이터가 사용자 질문에 답하기에 충분한지 판단하여
    JSON으로 반환한다. 이 결과는 should_continue_gather 분기에서 사용된다.

    Attributes:
        is_sufficient: 충분 여부 (True=분석 진행, False=재수집 필요)
        missing_info: 부족한 정보 항목 목록 (충분하면 빈 리스트)
        confidence_score: 판단 신뢰도 (0.0~1.0). 0.7 이상이면 "충분"으로 진행.
    """
    is_sufficient: bool
    missing_info: list[str] = []
    confidence_score: float = 0.5
    sufficiency_by_source: dict = {}   # G3: 소스별 충분성 {"csv": "sufficient", ...}

    @field_validator("missing_info", mode="before")
    @classmethod
    def _coerce_missing_info(cls, v):
        # LLM이 구버전 string 포맷으로 반환한 경우 list로 변환 (하위 호환)
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class AnalysisResult(BaseModel):
    """데이터 기반 분석 결과. generate_analysis 노드에서 LLM이 생성한다.

    LLM은 수집된 데이터를 기반으로 분석 결론, 근거 주장, 데이터 참조,
    주의사항, 모순 감지 결과를 JSON으로 반환한다.

    Attributes:
        summary: 분석 결론 한 문장 (예: "마포구 합정동이 카페 창업에 유리합니다")
        claims: 근거 주장 리스트. 각 항목은 {text, source, value} 구조.
        data_references: 분석에 사용한 데이터 소스 목록 (예: ["foot_traffic.csv"])
        caveats: 주의사항/한계 리스트 (예: ["2024년 데이터 기준"])
        contradicts_previous: 이전 턴 결론과 모순 여부 (LLM 출력 — node에서 conflict_tracking dict로 통합됨)
        contradiction_explanation: 모순 설명 (모순 시 "이전에는 X, 이제는 Y" 형식)
        referenced_turns: 참조한 이전 턴 번호 리스트

    Note: v3 통합 (2026-04-29) — LLM 출력 단계의 contradicts_previous, contradiction_explanation,
          source_conflict_resolution은 generate_analysis 노드에서 단일 conflict_tracking dict로
          묶여 state에 저장된다. 본 모델은 LLM 출력 파싱 전용 (node 산출 후 변환).
    """
    summary: str
    claims: list[dict] = []
    data_references: list[str] = []
    caveats: list[str] = []
    contradicts_previous: bool = False       # 일관성 패턴 B: 이전 턴과 모순 여부 (LLM raw)
    contradiction_explanation: str = ""      # 모순 설명 (LLM raw)
    referenced_turns: list[int] = []         # 참조한 이전 턴 번호
    source_conflict: bool = False            # 소스 간 충돌 여부
    source_conflict_resolution: str = ""     # 충돌 해결 설명 (LLM raw)
    utilized_previous: list[dict] = []       # Post-2: 실제 활용한 이전 결론 [{turn, claim, used_in}]


class InterpretationCheck(BaseModel):
    """LLM 해석 검증 결과. verify_result 노드의 STEP 2에서 LLM이 반환한다.

    검증 LLM(LLM-as-a-Judge)이 분석 결과가 수집된 데이터에 기반한
    정확한 해석인지 판단하여 점수와 문제점을 반환한다.

    Attributes:
        score: 해석 충실도 점수 (0.0~1.0). 0.6 미만이면 "fail_interpretation" 판정.
        issues: 발견된 문제점 리스트 (예: ["유동인구 해석이 부정확"])
    """
    score: float = 1.0
    issues: list[str] = []

    @field_validator("issues", mode="before")
    @classmethod
    def normalize_issues(cls, v: Any) -> list[str]:
        """LLM이 문자열 리스트 또는 딕셔너리 리스트로 반환하는 경우 모두 처리한다."""
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # {"severity": "high", "description": "..."} → description 추출
                result.append(item.get("description", str(item)))
            else:
                result.append(str(item))
        return result


class VerificationResult(BaseModel):
    """최종 검증 결과. verify_result 노드가 조합하여 state에 저장한다.

    수치 검증(STEP 1)과 해석 검증(STEP 2)의 결과를 통합한 최종 판정.
    route_after_verify 분기에서 이 결과를 읽어 다음 동작을 결정한다.

    Attributes:
        numeric_check: 수치 검증 결과 {passed: bool, discrepancies: list}
        interpretation_check: 해석 검증 결과 {score: float, issues: list}
        overall_verdict: 최종 판정 ("pass", "fail_numeric", "fail_interpretation", "error")
        issues: 모든 문제점 통합 리스트
    """
    numeric_check: dict = Field(default_factory=lambda: {"passed": True, "discrepancies": []})
    interpretation_check: dict = Field(default_factory=lambda: {"score": 0.0, "issues": []})
    overall_verdict: str = "pass"
    issues: list[str] = []
