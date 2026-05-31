"""
agent/nodes/respond_to_user.py — 노드 6: 최종 응답 생성 + 턴 결론 저장

역할:
    1. 분석 결과를 사용자에게 친절하게 전달하는 응답을 생성한다
    2. 모순이 감지된 경우, 이전 결론과의 차이점을 명시적으로 설명한다
    3. 이번 턴의 결론(turn_conclusion)을 저장하여 다음 턴에서 참조할 수 있게 한다
    4. 결론 압축 과정의 충실도를 관측한다 (G2 보완)

프로세스 단계: ⑤ Memory
품질 차원: 효율성 (압축률), 일관성 패턴 B (주장·조건 보존)

데이터 흐름:
    입력: analysis_result, contradicts_previous, previous_conclusion, user_query
    출력: response (최종 응답), turn_conclusions (누적), current_turn

turn_conclusions 구조:
    각 턴 결론은 다음 필드를 포함한다:
    - turn_number: 턴 번호
    - conclusion_summary: 결론 요약 (analysis_result.summary)
    - key_claims: 핵심 주장 리스트 (claims의 text만 추출)
    - data_sources_used: 사용된 데이터 소스 리스트
    - (모순 시) contradicts_turn: 모순된 이전 턴 번호
    - (모순 시) resolution: 해결 설명

Langfuse 기록:
    response.token_count: 응답 토큰 수
    response.conclusion_token_count: 결론 토큰 수
    response.compression_ratio: 분석 → 결론 압축률
    response.key_claims_preserved: 보존된 주장 수
    response.conditions_preserved: 조건/뉘앙스 보존 여부
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import get_client, observe

from agent.config_loader import get_config
from agent.llm import create_llm, invoke_with_retry
from agent.monitoring_schema import ATTRS
from agent.token_counter import count_tokens

# --- 응답 생성 프롬프트 ---
# LLM에게 "서울 상권 분석 Assistant" 역할을 부여한다.
# 핵심: 모순이 감지된 경우, 이전 결론을 어떻게 수정했는지 명시적으로 설명하도록 지시한다.
# 이를 통해 사용자가 분석 결론의 변경 이유를 투명하게 이해할 수 있다.
RESPONSE_PROMPT = """당신은 서울 상권 분석 Assistant입니다.
분석 결과를 사용자에게 친절하게 전달하세요.
모순이 감지된 경우, 이전 결론을 어떻게 수정했는지 명시적으로 설명하세요.
"""


def _build_template_response(
    analysis: dict, contradicts: bool, prev_conclusion: str
) -> str:
    """분석 결과를 템플릿으로 포맷팅한다 (LLM 미사용).

    generate_analysis가 이미 생성한 summary, claims, caveats를 구조화하여
    사용자에게 전달할 응답 텍스트를 구성한다.

    Args:
        analysis: generate_analysis 결과 (summary, claims, caveats, data_references).
        contradicts: 이전 턴과 모순 여부.
        prev_conclusion: 모순된 이전 결론 원문.

    Returns:
        사용자에게 표시할 응답 텍스트.
    """
    parts = []

    # 모순 설명 (있으면 먼저)
    if contradicts and prev_conclusion:
        parts.append(
            f"이전 분석에서는 \"{prev_conclusion[:100]}\"이라고 했으나, "
            f"새로운 데이터를 반영하여 결론이 변경되었습니다.\n"
        )

    # 요약
    parts.append(analysis.get("summary", "분석 결과를 생성하지 못했습니다."))

    # 근거 (claims, 최대 5개)
    claims = analysis.get("claims", [])
    if claims:
        parts.append("\n**근거:**")
        for c in claims[:5]:
            text = c.get("text", "")
            source = c.get("source", "")
            value = c.get("value", "")
            if value:
                parts.append(f"- {text} ({value}, 출처: {source})")
            elif source:
                parts.append(f"- {text} (출처: {source})")
            else:
                parts.append(f"- {text}")

    # 주의사항 (최대 3개)
    caveats = analysis.get("caveats", [])
    if caveats:
        parts.append("\n**주의사항:**")
        for caveat in caveats[:3]:
            parts.append(f"- {caveat}")

    return "\n".join(parts)


@observe(name="respond_to_user")
def respond_to_user(state: dict) -> dict:
    """최종 응답을 생성하고 turn_conclusions를 저장한다.

    Args:
        state: 현재 AgentState. analysis_result, contradicts_previous,
               previous_conclusion, user_query, turn_conclusions를 참조.

    Returns:
        {
            "response": str,                  # 사용자에게 전달할 응답 텍스트
            "turn_conclusions": list[dict],   # 이번 턴 결론을 포함한 전체 결론 리스트
            "current_turn": int,              # 현재 턴 번호
        }

    처리 과정:
        1. LLM에 분석 결과 + 모순 정보를 전달하여 응답 생성
        2. 이번 턴의 결론을 구조화 (summary, claims, data_sources)
        3. 모순 정보가 있으면 결론에 추가
        4. 기존 turn_conclusions에 이번 결론을 append
        5. Langfuse에 응답 메타데이터 기록
    """
    analysis = state.get("analysis_result", {})
    # v3 통합 (2026-04-29): conflict_tracking dict에서 모순 정보 추출.
    conflict_tracking = state.get("conflict_tracking") or {}
    contradicts = bool(conflict_tracking.get("detected", False))
    prev_conclusion = (conflict_tracking.get("resolution") or {}).get("conflict_summary", "")

    # --- 바이패스 여부 확인 ---
    # optimization.bypass_respond_llm=true이면 템플릿 기반 응답 생성 (LLM 미호출).
    # generate_analysis가 이미 Claude로 생성한 summary/claims/caveats를 재활용하므로
    # 이중 LLM 호출을 방지하여 토큰을 절감한다.
    _opt = get_config().get("optimization", {})
    bypass = _opt.get("bypass_respond_llm", False)

    if bypass:
        # 템플릿 기반 응답 생성 (LLM 미호출)
        response_text = _build_template_response(analysis, contradicts, prev_conclusion)
    else:
        # 기존 LLM 호출 (AS-IS)
        llm = create_llm()
        response_input = json.dumps({
            "analysis": analysis,
            "contradicts_previous": contradicts,
            "previous_conclusion": prev_conclusion,
            "user_query": state.get("user_query", ""),
        }, ensure_ascii=False)

        response = invoke_with_retry(
            llm,
            [
                SystemMessage(content=RESPONSE_PROMPT),
                HumanMessage(content=response_input),
            ],
            generation_name="respond_to_user.compose",
        )
        response_text = response.content

    # --- 턴 결론 구조화 및 저장 ---
    # 이번 턴의 결론을 구조화하여 turn_conclusions에 추가한다.
    # 이 데이터는 다음 턴의 analyze_query, generate_analysis에서
    # 이전 결론을 참조하는 데 사용된다 (일관성 패턴 B).
    turn_number = state.get("current_turn", 0)

    new_conclusion = {
        "turn_number": turn_number,
        "conclusion_summary": analysis.get("summary", ""),
        "key_claims": [c["text"] for c in analysis.get("claims", [])],
        "data_sources_used": analysis.get("data_references", []),
    }

    # 모순이 감지된 경우: 어떤 턴과 모순되었는지, 어떻게 해결했는지 기록
    if contradicts:
        new_conclusion["contradicts_turn"] = turn_number - 1  # 바로 이전 턴과 모순
        # caveats의 첫 번째 항목을 해결 설명으로 사용 (없으면 빈 문자열)
        new_conclusion["resolution"] = analysis.get("caveats", [""])[0] if analysis.get("caveats") else ""

    # 기존 turn_conclusions에 이번 결론을 append
    # list()로 복사하여 원본 state를 변경하지 않는다 (LangGraph 불변성 규칙)
    existing_conclusions = list(state.get("turn_conclusions", []))
    updated_conclusions = existing_conclusions + [new_conclusion]

    # ═══ G2: 결론 압축 충실도 관측 ═══
    # 분석 결과(analysis)에서 결론(new_conclusion)으로 압축될 때 무엇이 보존되고 손실되는지 관측한다.
    # 이 데이터는 다음 턴의 evaluate_context에서 충실도(축 2) 측정의 기반이 된다.

    # 분석 전체 토큰 수 (분모)
    analysis_text = json.dumps(analysis, ensure_ascii=False) if analysis else ""
    analysis_token_count = count_tokens(analysis_text)

    # 결론 토큰 수 (분자)
    conclusion_text = json.dumps(new_conclusion, ensure_ascii=False)
    conclusion_token_count = count_tokens(conclusion_text)

    # 압축률: 결론 토큰 / 분석 토큰 (1.0이면 무압축, 낮을수록 많이 압축)
    compression_ratio = (
        round(conclusion_token_count / analysis_token_count, 3)
        if analysis_token_count > 0 else 1.0
    )

    # F3: 주장 보존을 비율 기반으로 전환 (절대 수 → 0.0~1.0)
    analysis_claims = analysis.get("claims", [])
    conclusion_claims = new_conclusion.get("key_claims", [])
    total_analysis_claims = len(analysis_claims)
    preserved_claims_ratio = (
        round(len(conclusion_claims) / total_analysis_claims, 3)
        if total_analysis_claims > 0 else 1.0
    )

    # A3: 손실된 주장 추적 — 분석에는 있으나 결론에서 탈락한 주장
    analysis_claim_texts = [c.get("text", "") for c in analysis_claims]
    conclusion_claim_set = set(conclusion_claims)
    lost_claims = [t for t in analysis_claim_texts if t and t not in conclusion_claim_set]

    # 조건·뉘앙스 보존 여부: summary에 조건 표현이 포함되어 있는지 확인
    # "조건부", "~인 경우", "~을 제외하면", "다만" 등 조건 표현이 분석에 있었는데
    # 결론에서 사라졌다면 conditions_preserved=False
    _condition_markers = [
        # 기존 6개
        "조건부", "경우에", "제외하면", "다만", "단,", "제한적",
        # F2 확장: 조건·전제 표현 (14+개 추가)
        "~인 경우", "~할 때", "~이라면", "~에 한해", "~를 전제로",
        "~이 아니면", "그러나", "하지만", "반면", "~에도 불구하고",
        "~에 따라", "~에 비해", "~보다", "~미만", "~이상",
        "한편", "다른 한편", "전제 조건", "단서", "예외",
    ]
    analysis_summary = analysis.get("summary", "")
    conclusion_summary = new_conclusion.get("conclusion_summary", "")

    # F2: 마커별 보존 상세 추적
    markers_in_analysis = [m for m in _condition_markers if m in analysis_summary]
    markers_in_conclusion = [m for m in _condition_markers if m in conclusion_summary]
    markers_lost = [m for m in markers_in_analysis if m not in conclusion_summary]
    conditions_preserved = (
        not markers_in_analysis  # 분석에 조건이 없었으면 True
        or len(markers_lost) == 0  # 분석의 모든 조건 마커가 결론에도 존재하면 True
    )
    conditions_detail = {
        "markers_in_analysis": markers_in_analysis,
        "markers_in_conclusion": markers_in_conclusion,
        "markers_lost": markers_lost,
    }

    # P0' 보강 (2026-05-08): Groundedness Checker hook 부착.
    # evaluation/groundedness_checker는 결정론적(LLM 미사용)이므로 노드 내부에서 호출 가능.
    # 메인 trace에 직접 metadata 부착하여 Tab 4 ④ Groundedness 영역에 표시.
    grounded_meta: dict = {}
    try:
        from evaluation.groundedness_checker import check_groundedness
        gc_result = check_groundedness(
            response_text=response_text,
            gathered_data=state.get("gathered_data", []),
            analysis_result=analysis,
        )
        if isinstance(gc_result, dict):
            grounded_meta = {
                ATTRS["response.grounded_claim_ratio"]: gc_result.get("grounded_claim_ratio"),
                ATTRS["response.hallucination_detected"]: gc_result.get("hallucination_detected"),
                ATTRS["response.ungrounded_claims"]: gc_result.get("ungrounded_claims", []),
            }
    except Exception as e:
        print(f"    ⚠ groundedness_checker 호출 실패: {type(e).__name__}: {str(e)[:120]}")

    # --- Langfuse 메타데이터 기록 ---
    get_client().update_current_span(
        metadata={
            ATTRS["response.token_count"]: count_tokens(response_text),  # 응답 토큰 수
            # v3 폐기: conclusion_token_count / compression_ratio (context.fidelity_detail에 흡수)
            # v3 폐기: conditions_preserved / conditions_detail (Tier 4 결함 — 한국어 마커 휴리스틱)
            ATTRS["response.key_claims_preserved"]: preserved_claims_ratio,  # 주장 보존 비율
            ATTRS["response.lost_claims"]: lost_claims,  # A3: 손실된 주장 목록
            # Dashboard Tab 3(구성)에서 턴별 결론 내용을 표시하기 위한 데이터
            ATTRS["response.conclusion_summary"]: new_conclusion.get("conclusion_summary", ""),
            ATTRS["response.key_claims"]: new_conclusion.get("key_claims", []),
            # H0 재설계: groundedness 평가를 위한 최종 응답 텍스트 (최대 2000자)
            ATTRS["response.final_text"]: response_text[:2000],
            # P0' (2026-05-08): Groundedness Checker 결과 부착
            **grounded_meta,
        }
    )

    return {
        "response": response_text,                 # 사용자에게 표시할 응답
        "turn_conclusions": updated_conclusions,    # 누적된 턴 결론
        "current_turn": turn_number,                # 턴 번호 유지
        # G1: 다음 턴의 evaluate_context에서 충실도(fidelity_score) 계산에 사용
        "previous_turn_fidelity": {
            "compression_ratio": compression_ratio,
            "conditions_preserved": conditions_preserved,
            "conditions_detail": conditions_detail,
            "key_claims_preserved": preserved_claims_ratio,
            "total_claims": len(analysis_claims),
            "lost_claims": lost_claims,
        },
        # 턴 간 LLM 대화 히스토리 누적: AgentState의 add_messages 리듀서가
        # 기존 messages에 이번 턴의 사용자 질문 + 에이전트 응답을 append한다
        "messages": [
            HumanMessage(content=state["user_query"]),
            AIMessage(content=response_text),
        ],
    }
