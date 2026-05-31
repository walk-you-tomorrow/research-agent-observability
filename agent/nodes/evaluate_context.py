"""
agent/nodes/evaluate_context.py — 노드 3: 컨텍스트 충분성 평가

★ Context Monitoring 핵심 노드 ★

역할:
    수집된 컨텍스트의 충분성을 평가하고, 모니터링 메타데이터를 Langfuse에 기록한다.
    이 노드는 4개의 STEP으로 구성되며, AI Agent Execution Process의 ③ Organize 단계 + 충분성 평가 checkpoint를 담당한다.

프로세스 단계: ③ Organize + 충분성 평가 checkpoint
품질 차원: 완전성, 효율성, 관련성, 일관성(A) — 4차원 모두 관여

4개 STEP:
    STEP 1: 컨텍스트 조립 + 토큰 카운팅 (③ Organize)
    STEP 2: LLM 충분성 판단 (충분성 평가 checkpoint)
    STEP 3: 일관성 패턴 A 계산 (이전 iteration과 비교)
    STEP 4: Langfuse에 Layer 2 attribute 기록

데이터 흐름:
    입력: gathered_data, query_analysis, turn_conclusions, previous_missing_info, previous_confidence
    출력: context_evaluation, context_metadata, previous_missing_info, previous_confidence

Langfuse 기록 (총 27개 attribute):
    효율성: context.total_tokens, context.window_utilization, context.source.*,
            context.messages_tokens, context.information_density, context.redundancy_ratio
    관련성: context.noise_ratio, context.effective_noise_ratio, context.new_data_ratio
    완전성: context.is_sufficient, context.missing_info, context.sufficiency_confidence,
            context.sufficiency_by_source (v3 폐기)
    일관성(A): context.missing_info_resolved, context.confidence_delta
    일관성(B): context.continuity_score
    일관성(C): context.fidelity_score
    교차 턴: context.token_delta, context.contributing_turns, context.causal_sources
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, observe

from agent.llm import create_llm, invoke_with_retry
from agent.models import ContextEvaluation
from agent.monitoring_schema import ATTRS, CONTEXT_WINDOW_MAX_TOKENS, FIDELITY_SCORE_WEIGHTS, ROT_GATE_THRESHOLD, is_rot_gate_enabled
from agent.parser import parse_llm_json
from agent.token_counter import count_tokens

# --- 충분성 판단 프롬프트 ---
# LLM에게 "충분성 평가 모듈" 역할을 부여한다.
# is_sufficient, missing_info, confidence_score 3가지를 JSON으로 반환하도록 요청한다.
# confidence_score는 should_continue_gather 분기의 임계값(0.7)과 비교된다.
SUFFICIENCY_PROMPT = """당신은 Research Agent의 데이터 충분성 평가 모듈입니다.
수집된 데이터가 사용자 질문에 답하기에 충분한지 판단하세요.

JSON으로만 응답:
{"is_sufficient": false, "missing_info": ["강남구 평균 영업기간 데이터", "월별 매출 추이"], "confidence_score": 0.55,
 "sufficiency_by_source": {"csv": "partial", "rag": "sufficient"}}

규칙:
- is_sufficient=true: 정확한 분석이 가능
- is_sufficient=false: 핵심 데이터 부족, 추가 수집 필요
- missing_info: 부족한 정보 항목을 문자열 배열로 나열 (충분하면 빈 배열 [])
  각 항목은 "어떤 데이터가 왜 필요한지"를 구체적으로 작성 (예: "강남구 월별 유동인구 2024년")
- confidence_score: 0.0~1.0
- sufficiency_by_source: 사용된 각 소스 유형별 충분성 ("sufficient", "partial", "insufficient", "not_used")
  - 소스 키는 반드시 csv, rag, web, api 4종 중에서만 사용하세요 (lightrag_local 등 변형 금지)"""

# --- sufficiency_by_source 키 정규화 ---
# LLM이 "api_query", "database", "web_search" 등 비표준 키를 생성할 수 있다.
# 표준 4종(csv, rag, web, api)으로 매핑하여 대시보드 표시를 일관되게 한다.
_SOURCE_KEY_MAP = {
    "api_query": "api",
    "api_search": "api",
    "seoul_api": "api",
    "database": "csv",          # database는 pandas CSV 쿼리를 의미
    "pandas": "csv",
    "web_search": "web",
    "lightrag": "rag",
    "lightrag_local": "rag",
    "lightrag_global": "rag",
    "lightrag_mix": "rag",
    "lightrag_hybrid": "rag",
}


def _normalize_source_keys(sbs: dict) -> dict:
    """sufficiency_by_source의 키를 표준 4종(csv, rag, web, api)으로 정규화한다.

    Args:
        sbs: LLM이 반환한 소스별 충분성 dict.

    Returns:
        정규화된 dict. 비표준 키는 표준 키로 매핑.
    """
    if not isinstance(sbs, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in sbs.items():
        std_key = _SOURCE_KEY_MAP.get(key, key)
        # 같은 표준 키가 이미 있으면 더 나쁜 상태를 유지
        if std_key in normalized:
            _priority = {"insufficient": 0, "partial": 1, "sufficient": 2, "not_used": 3}
            existing = _priority.get(normalized[std_key], 3)
            new = _priority.get(value, 3)
            if new < existing:
                normalized[std_key] = value
        else:
            normalized[std_key] = value
    return normalized


# --- 한국어 조사(particle) 제거 ---
# 한국어는 어근에 조사가 붙어서 "강남구는", "유동인구가" 같은 형태가 된다.
# 조사를 제거해야 "강남구"가 다른 텍스트의 "강남구"와 매칭될 수 있다.
_KOREAN_PARTICLES = [
    "에서는", "으로는", "이랑은",                      # 3글자 복합 조사
    "에서", "으로", "이랑", "하고", "보다", "까지",     # 2글자 조사
    "부터", "처럼", "마저", "조차", "에게",
    "은", "는", "이", "가", "을", "를",                 # 1글자 조사
    "에", "와", "과", "의", "도", "로", "만",
]


def _strip_particles(word: str) -> str:
    """한국어 단어에서 trailing 조사를 제거하여 어근을 반환한다."""
    for p in _KOREAN_PARTICLES:
        # 어근이 최소 2글자 남아야 조사 제거 (예: "가"는 조사 제거하면 빈 문자열)
        if word.endswith(p) and len(word) - len(p) >= 2:
            return word[:-len(p)]
    return word


def _claim_retained(claim: str, context_text: str) -> bool:
    """하나의 claim 문장이 컨텍스트 텍스트에 보존되어 있는지 확인한다.

    조사를 제거한 어근 기준으로 매칭하므로 "강남구는"과 "강남구를"이 같은 어근 "강남구"로 인식된다.

    Args:
        claim: key_claim 텍스트 (예: "강남구는 높은 유동인구를 보유")
        context_text: 현재 컨텍스트 전체 텍스트 (소문자 변환 완료)

    Returns:
        claim의 핵심 키워드 중 하나 이상이 context_text에 존재하면 True
    """
    for word in claim.lower().split():
        stem = _strip_particles(word)
        if len(stem) > 1 and stem in context_text:
            return True
    return False


@observe(name="evaluate_context")
def evaluate_context(state: dict) -> dict:
    """수집된 컨텍스트의 충분성을 평가하고 모니터링 메타데이터를 기록한다.

    Args:
        state: 현재 AgentState. gathered_data, query_analysis, turn_conclusions,
               previous_missing_info, previous_confidence를 참조.

    Returns:
        {
            "context_evaluation": dict,       # {is_sufficient, missing_info, confidence_score}
            "context_metadata": dict,         # 토큰 통계 (관측 전용)
            "previous_missing_info": str,     # 일관성 패턴 A용: 이번 missing_info 저장
            "previous_confidence": float,     # 일관성 패턴 A용: 이번 confidence 저장
        }
    """
    gathered_data = state.get("gathered_data", [])

    # ═══════════════════════════════════════
    # STEP 1: 컨텍스트 조립 + 토큰 카운팅 (프로세스 단계 ③ Organize)
    # ═══════════════════════════════════════
    # 컨텍스트를 구성하는 각 소스의 토큰 수를 정확히 계산한다.
    # 이 데이터는 효율성(Efficiency) 차원의 핵심 지표가 된다.

    # 시스템 프롬프트 토큰 수 (고정 비용)
    system_tokens = count_tokens(SUFFICIENCY_PROMPT)

    # 질의 분석 결과 토큰 수
    query_tokens = count_tokens(json.dumps(state.get("query_analysis", {})))

    # 수집된 데이터의 총 토큰 수 (각 항목의 token_count 합산)
    gathered_tokens = sum(item.get("token_count", 0) for item in gathered_data)

    # 이전 턴 결론 전체의 토큰 수 (다중 턴 시 누적되어 증가)
    prev_turns_tokens = count_tokens(
        json.dumps(state.get("turn_conclusions", []))
    )

    # 턴 결론 요약만의 토큰 수 (conclusion_summary 필드만 추출)
    conclusions_tokens = count_tokens(
        json.dumps([
            c.get("conclusion_summary", "")
            for c in state.get("turn_conclusions", [])
        ])
    )

    # 총 토큰 수 = 시스템 프롬프트 + 질의 분석 + 수집 데이터 + 이전 턴
    total_tokens = system_tokens + query_tokens + gathered_tokens + prev_turns_tokens

    # F10: 컨텍스트 윈도우 사용률 — YAML SSOT에서 max_tokens를 참조
    # 0.05 = 5% 사용, 0.7 = 70% 사용 (목표). 1.0 이상이면 윈도우 초과 위험.
    utilization = total_tokens / CONTEXT_WINDOW_MAX_TOKENS if CONTEXT_WINDOW_MAX_TOKENS > 0 else 0

    # v3 REDEFINE (2026-04-29): truncated_items 구조 변경 (str → dict).
    # AS-IS: list[str] (소스명만) + 별도 truncation_reasons (사유) — 2 attribute
    # TO-BE: list[dict[source, reason]] — 1 attribute로 통합, Judge에서 사유 직접 활용 가능
    truncated_items = [
        {"source": item["source"], "reason": "token_budget"}
        for item in gathered_data
        if item.get("data_summary", "").endswith("[truncated]")
    ]

    # --- 소스 유형별 토큰 기여율 ---
    # 각 도구를 소스 유형(csv, rag, web, api)으로 매핑하여 소스별 토큰 기여 비율을 계산한다.
    # 이 지표는 소스 선택 결정의 품질을 평가하는 데 사용된다.
    source_tokens = {"csv": 0, "rag": 0, "web": 0, "api": 0}
    for item in gathered_data:
        tool = item.get("tool_used", "")
        tokens = item.get("token_count", 0)
        if tool in ("pandas_query", "calculate"):
            source_tokens["csv"] += tokens
        elif tool.startswith("rag_"):
            source_tokens["rag"] += tokens
        elif tool == "web_search":
            source_tokens["web"] += tokens
        elif tool == "api_query":
            source_tokens["api"] += tokens

    total_source = sum(source_tokens.values())
    source_contribution = {
        k: round(v / total_source, 3) for k, v in source_tokens.items() if v > 0
    } if total_source > 0 else {}

    # 노이즈 비율: 이전 턴 토큰 / 전체 토큰
    # 높을수록 현재 턴의 "새로운" 데이터 비중이 낮다는 뜻이다.
    # 다중 턴이 진행될수록 이 값이 증가하여 관련성(Relevance) 저하를 의미한다.
    noise_ratio = round(
        prev_turns_tokens / total_tokens, 3
    ) if total_tokens > 0 else 0

    # 실효 노이즈 비율 (effective_noise_ratio): 인과 영향력 기반 noise 재분류
    # F-002 발견: impact=0인 causal_source가 실질적으로 발생하지 않으므로
    # 이진 분류(impact>0 vs =0) 대신 상대 순위(median 이하)를 사용한다.
    # impact가 중간값(median) 이하인 턴의 토큰만 noise로 간주한다.
    # NOTE: rot_risk 신공식이 dead_weight_tokens에 의존하므로 dead_weight 계산을 먼저 수행한다.
    effective_noise_ratio = noise_ratio  # fallback: causal_sources 없으면 기존 noise_ratio 사용
    # A4: dead_weight_tokens 초기화 — Rot Gate(A5)의 입력으로, 아래 블록에서 갱신
    dead_weight_tokens = 0
    _tc_causal_pairs = []
    turn_conclusions = state.get("turn_conclusions", [])
    if turn_conclusions and total_tokens > 0:
        # causal_sources는 아래 Post-1 블록에서 계산되므로 여기서 선행 계산한다
        _pre_causal = []
        _pre_ctx_parts = []
        for item in gathered_data:
            _pre_ctx_parts.append(item.get("source", ""))
            _pre_ctx_parts.append(item.get("data_summary", ""))
        _pre_ctx_parts.append(state.get("user_query", ""))
        _pre_ctx_text = " ".join(_pre_ctx_parts).lower()
        # (tc, causal_entry) 쌍을 만들어야 zip 정렬 유지
        _tc_causal_pairs = []
        for tc in turn_conclusions:
            claims = tc.get("key_claims", [])
            if claims:
                retained = sum(1 for c in claims if _claim_retained(c, _pre_ctx_text))
                _tc_causal_pairs.append((tc, {
                    "turn": tc.get("turn_number", 0),
                    "claims_total": len(claims),
                    "claims_retained": retained,
                    "impact": round(retained / len(claims), 3),
                }))
        if _tc_causal_pairs:
            impacts = [cs["impact"] for _, cs in _tc_causal_pairs]
            median_impact = sorted(impacts)[len(impacts) // 2]
            low_impact_tokens = sum(
                count_tokens(json.dumps(tc.get("key_claims", [])))
                for tc, cs in _tc_causal_pairs
                if cs["impact"] <= median_impact
            )
            effective_noise_ratio = round(low_impact_tokens / total_tokens, 3)
            dead_weight_tokens = low_impact_tokens  # A4: Rot Gate 입력으로 전달

    # A2: messages 토큰 계측 — total_tokens에 현재 미포함되어 window_utilization 과소계측
    # 현재는 관측 전용으로 기록하고, total_tokens에는 합산하지 않는다.
    # 이유: total_tokens 공식 변경 시 기존 데이터와의 비교가 불가능해지므로,
    # 먼저 messages_tokens를 독립적으로 관측하여 규모를 파악한 후 합산 여부를 결정한다.
    messages = state.get("messages", [])
    messages_text = " ".join(
        msg.content if hasattr(msg, "content") else str(msg)
        for msg in messages
        if msg  # None 방어
    )
    messages_tokens = count_tokens(messages_text) if messages_text else 0

    # ── rot_risk: 컨텍스트 rot 복합 지표 (가중합) ──
    # 구공식(2026-04-23 이전): util × noise_ratio = prev_turns/180K — 두 인자가 약분되어 단일 신호로 붕괴.
    # 실측 결과(H4b, 2026-04-24): 8턴 누적해도 max 0.04 — 임계값 0.3에 도달 불가.
    # 신공식(2026-04-24): 두 신호의 가중합 — 약분되지 않고 0~1로 자연 분산.
    #   - stale_share: 컨텍스트에서 과거 결론이 차지하는 비중 (= noise_ratio)
    #   - waste_share: 과거 결론 중 dead weight(low-impact) 비중
    #   둘 다 높을 때만 rot_risk가 0.5+ 영역으로 진입하여 Gate 발동.
    stale_share = round(prev_turns_tokens / total_tokens, 4) if total_tokens > 0 else 0.0
    waste_share = (
        round(dead_weight_tokens / prev_turns_tokens, 4) if prev_turns_tokens > 0 else 0.0
    )
    rot_risk = round(0.5 * stale_share + 0.5 * waste_share, 4)

    # A1: rot_velocity — 턴 간 rot_risk 변화율 (H4a 검증용)
    previous_rot_risk = state.get("previous_rot_risk", 0.0)
    rot_velocity = round(rot_risk - previous_rot_risk, 4)

    # ── A5: Rot Gate — rot_risk가 임계값을 초과하면 low-impact 결론을 pruning ──
    # H4a 검증 PASS (rho=-0.6443, 구공식 기준): rot_risk 증가 시 4D 품질 저하 경향 확인됨.
    # Rot Gate는 state의 turn_conclusions를 직접 수정하므로 비가역적 — 주의 필요.
    # H4b 실험용: ROT_GATE_ENABLED=0으로 비활성화하면 pruning 없이 관측 지표만 기록된다.
    rot_gate_triggered = (
        is_rot_gate_enabled() and rot_risk > ROT_GATE_THRESHOLD and dead_weight_tokens > 0
    )
    rot_gate_pruned_tokens = 0
    if rot_gate_triggered:
        rot_gate_pruned_tokens = dead_weight_tokens
        # low-impact 턴 결론을 turn_conclusions에서 제거 (비가역적)
        if _tc_causal_pairs:
            impacts = [cs["impact"] for _, cs in _tc_causal_pairs]
            median_impact = sorted(impacts)[len(impacts) // 2]
            pruned_turn_numbers = {
                tc.get("turn_number", -1) for tc, cs in _tc_causal_pairs
                if cs["impact"] <= median_impact
            }
            turn_conclusions = [
                tc for tc in turn_conclusions
                if tc.get("turn_number", -1) not in pruned_turn_numbers
            ]

    # continuity_score: 이전 턴 결론의 key_claims가 현재 컨텍스트에 보존되었는가
    # "현재 컨텍스트" = 수집 데이터 소스 + 사용자 질문 + 대화 히스토리 + scope 내 결론
    # 1.0 = 완전 보존, 0.0 = 완전 손실
    turn_conclusions = state.get("turn_conclusions", [])
    if turn_conclusions:
        all_claims = []
        for tc in turn_conclusions:
            all_claims.extend(tc.get("key_claims", []))
        if all_claims:
            # 현재 턴에서 접근 가능한 전체 컨텍스트로 검색 대상 구성
            # data_summary만으로는 "검색 결과: 3개 문서" 같은 메타 정보뿐이므로,
            # 실제 도메인 키워드가 포함된 소스명, 질문, 대화, scope 내 결론을 포함한다
            context_parts = []
            for item in gathered_data:
                context_parts.append(item.get("source", ""))
                context_parts.append(item.get("data_summary", ""))
            context_parts.append(state.get("user_query", ""))
            for msg in state.get("messages", []):
                context_parts.append(str(getattr(msg, "content", ""))[:500])
            # generate_analysis가 참조하는 scope 내 결론 (최근 5개)
            for tc_scope in turn_conclusions[-5:]:
                context_parts.append(tc_scope.get("conclusion_summary", ""))
            gathered_text = " ".join(context_parts).lower()

            retained = sum(
                1 for claim in all_claims
                if _claim_retained(claim, gathered_text)
            )
            continuity_score = round(retained / len(all_claims), 3) if all_claims else 1.0
        else:
            continuity_score = 1.0  # claim이 없으면 보존 문제 없음
    else:
        continuity_score = 1.0  # 첫 턴은 이전 결론이 없으므로 1.0

    # ── G3: 교차 턴 진화 지표 (Cross-Turn Evolution) ──
    # 턴이 진행되면서 컨텍스트가 어떻게 변하는지 추적한다.
    # new_data_ratio: 이번 턴에서 새로 수집한 데이터의 비율
    new_data_ratio = round(
        gathered_tokens / total_tokens, 3
    ) if total_tokens > 0 else 1.0
    # R12: inherited_ratio 삭제 — noise_ratio와 수학적으로 100% 동일한 공식이었으므로
    # effective_noise_ratio로 대체하여 이중 평가를 해소한다.
    # token_delta: 이전 턴 대비 총 토큰 변화량
    prev_total_tokens = state.get("previous_total_tokens", 0)
    token_delta = total_tokens - prev_total_tokens
    # v3 REDEFINE (2026-04-29): 방향 정정.
    # AS-IS: 모든 이전 턴 결론 수 (단순 카운트, 측정 의미 약함)
    # TO-BE: 현재 답변이 참조하는 이전 턴 수 (analyze_query.referenced_turns 길이)
    # state.get("query_analysis", {})로 변수가 정의되기 전에 참조하던 NameError 수정 (2026-05-08)
    _qa = state.get("query_analysis") or {}
    referenced_turns = state.get("referenced_turns") or _qa.get("referenced_turns") or []
    # 중복 제거 후 카운트 (LLM이 중복 추출하더라도 unique 턴 수)
    contributing_turns_count = len(set(referenced_turns))

    # ── G1: 충실도 점수 (일관성 패턴 C — Memory → Organize) ──
    # respond_to_user가 기록한 G2 metrics(compression_ratio, conditions_preserved,
    # key_claims_preserved)를 활용하여 이전 턴 결론의 의미적 보존도를 측정한다.
    # continuity_score는 키워드 기반, fidelity_score는 의미적 보존도.
    prev_fidelity = state.get("previous_turn_fidelity", {})
    if prev_fidelity:
        # 3요소 가중 평균: 조건 보존(0.4) + 주장 보존 비율(0.3) + 압축 적절성(0.3)
        cond_score = 1.0 if prev_fidelity.get("conditions_preserved", True) else 0.0
        # F3: key_claims_preserved가 이미 비율(0.0~1.0)로 전달됨 (절대 수 → 비율 전환)
        claims_ratio = prev_fidelity.get("key_claims_preserved", 1.0)
        comp_ratio = prev_fidelity.get("compression_ratio", 1.0)
        compression_penalty = min(comp_ratio / 0.3, 1.0)  # 0.3 이상이면 1.0 (적절한 압축)
        _fw = FIDELITY_SCORE_WEIGHTS
        fidelity_score = round(
            _fw["cond_score"] * cond_score
            + _fw["claims_ratio"] * claims_ratio
            + _fw["compression_penalty"] * compression_penalty,
            3,
        )
        # F4+A7: fidelity_detail — 충실도 구성 요소 분리 (진단 드릴다운용)
        fidelity_detail = {
            "cond_score": cond_score,
            "claims_ratio": round(claims_ratio, 3),
            "compression_penalty": round(compression_penalty, 3),
        }
    else:
        fidelity_score = 1.0  # 첫 턴이거나 이전 충실도 데이터 없으면 1.0
        fidelity_detail = {"cond_score": 1.0, "claims_ratio": 1.0, "compression_penalty": 1.0}

    # ── Post-1: 인과 전파 (Causal Propagation) ──
    # 이전 턴의 결론이 현재 컨텍스트에 실제로 영향을 주는지 턴별로 측정한다.
    # continuity_score 블록에서 구성한 context_parts를 재활용한다.
    causal_sources = []
    if turn_conclusions:
        # continuity_score 계산 시 gathered_text가 생성된 경우 재활용
        # (all_claims가 비어있으면 gathered_text가 생성되지 않으므로 직접 구성)
        causal_ctx = ""
        try:
            causal_ctx = gathered_text  # continuity_score 블록에서 생성된 변수
        except NameError:
            # all_claims가 비어있어 gathered_text가 없는 경우
            parts = [item.get("data_summary", "") for item in gathered_data]
            parts.append(state.get("user_query", ""))
            causal_ctx = " ".join(parts).lower()
        for tc in turn_conclusions:
            turn_num = tc.get("turn_number", 0)
            claims = tc.get("key_claims", [])
            if claims:
                retained = sum(1 for c in claims if _claim_retained(c, causal_ctx))
                causal_sources.append({
                    "turn": turn_num,
                    "claims_total": len(claims),
                    "claims_retained": retained,
                    "impact": round(retained / len(claims), 3),
                })

    # ── Post-3: 의미적 정보 밀도 (Semantic Information Density) ──
    # 토큰 수가 아닌 "유용한 정보 비율" — key_claims 토큰 / gathered_data 토큰
    claims_text = " ".join(
        claim for tc in turn_conclusions for claim in tc.get("key_claims", [])
    )
    claims_tokens = count_tokens(claims_text) if claims_text else 0
    information_density = round(
        claims_tokens / max(gathered_tokens, 1), 3
    )

    # v3 REDEFINE (2026-04-29): 의미적 중복 측정.
    # AS-IS: Jaccard 단어 set 교집합 — 동의어/번역/표현 차이를 무시 (의미 중복 놓침)
    # TO-BE: nomic-embed-text 임베딩 + pairwise cosine, threshold 0.85 (yaml customizable)
    # Ollama 미가용 시 lexical fallback 자동 적용 (회귀 안전망).
    from agent.redundancy_checker import compute_redundancy_ratio
    from agent.monitoring_schema import get_customizable_threshold
    cosine_threshold = get_customizable_threshold("redundancy_cosine_threshold", 0.85) or 0.85
    redundancy_ratio = compute_redundancy_ratio(gathered_data, cosine_threshold=cosine_threshold)

    # 컨텍스트 메타데이터를 딕셔너리로 구성 (관측 전용, 분기 로직에는 사용되지 않음)
    context_metadata = {
        "total_tokens": total_tokens,
        "source_breakdown": {
            "system_prompt": system_tokens,         # 시스템 프롬프트 (고정)
            "query_analysis": query_tokens,          # 질의 분석 결과
            "gathered_data": gathered_tokens,         # 수집 데이터 (가변)
            "previous_turns": prev_turns_tokens,     # 이전 턴 (턴 수에 비례)
            "turn_conclusions": conclusions_tokens,   # 턴 결론 요약
        },
        "gathered_count": len(gathered_data),         # 수집 항목 수
        "truncated_items_count": len(truncated_items), # 잘린 항목 수
        "truncated_items": truncated_items,            # 잘린 항목 소스명
        "context_window_utilization": round(utilization, 3),  # 윈도우 사용률
        "noise_ratio": noise_ratio,                    # 노이즈 비율 (raw)
        "effective_noise_ratio": effective_noise_ratio, # 실효 노이즈 비율 (인과 기반)
        "rot_risk": rot_risk,                          # context rot 복합 지표
        "rot_velocity": rot_velocity,                  # A1: 턴 간 rot_risk 변화율
        "continuity_score": continuity_score,          # 컨텍스트 보존도
        "messages_tokens": messages_tokens,                # A2: 메시지 토큰 수 (관측 전용)
        # A4+A5: Rot Gate
        "dead_weight_tokens": dead_weight_tokens,
        "rot_gate_triggered": rot_gate_triggered,
        "rot_gate_pruned_tokens": rot_gate_pruned_tokens,
        # G1: 충실도 점수
        "fidelity_score": fidelity_score,
        # F4+A7: 충실도 구성 요소 분리
        "fidelity_detail": fidelity_detail,
        # G3: 교차 턴 진화 지표
        "new_data_ratio": new_data_ratio,
        "token_delta": token_delta,
        "contributing_turns": contributing_turns_count,
        # Post-3: 의미적 정보 밀도
        "information_density": information_density,
        "redundancy_ratio": redundancy_ratio,
    }

    # ═══════════════════════════════════════
    # STEP 2: LLM 충분성 판단 (충분성 평가 checkpoint)
    # ═══════════════════════════════════════
    # LLM에게 수집된 데이터의 충분성을 판단하도록 요청한다.
    # LLM은 사용자 질문과 수집된 데이터를 비교하여
    # is_sufficient, missing_info, confidence_score를 반환한다.
    llm = create_llm()

    # LLM에 전달할 평가 입력 구성: 질의 분석(핵심만) + 수집 항목 요약 + 사용자 질문
    # 충분성 판단에 필요한 최소 정보만 전달하여 토큰을 절감한다.
    # query_analysis: intent와 tool_plan만 (나머지는 충분성 판단에 불필요)
    # summary: 250자로 축소 (1000자 → 250자, ~60% 절감)
    qa = state.get("query_analysis", {})
    eval_input = json.dumps({
        "query_analysis": {
            "intent": qa.get("intent", ""),
            "tool_plan": qa.get("tool_plan", []),
        },
        "gathered_items": [
            {
                "source": d["source"],
                "summary": d["data_summary"][:250],  # 250자 제한 (토큰 절감)
                "tokens": d["token_count"],
            }
            for d in gathered_data
        ],
        "user_query": state.get("user_query", ""),
    }, ensure_ascii=False)

    response = invoke_with_retry(
        llm,
        [
            SystemMessage(content=SUFFICIENCY_PROMPT),
            HumanMessage(content=eval_input),
        ],
        generation_name="evaluate_context.sufficiency",
    )
    evaluation = parse_llm_json(response.content, ContextEvaluation)
    eval_dict = evaluation.model_dump()

    # ═══════════════════════════════════════
    # STEP 3: 일관성 패턴 A (이전 iteration과 비교)
    # ═══════════════════════════════════════
    # 같은 턴 내에서 evaluate_context가 여러 번 호출될 수 있다 (재수집 루프).
    # 이전 iteration의 missing_info와 confidence를 비교하여 "개선되었는가?"를 추적한다.
    #
    # missing_info_resolved: 이전에 부족했던 정보가 이번에 해결되었는가?
    #   - True: 이전 missing_info가 있었는데 이번에는 빈 문자열 → 해결됨
    #   - False: 아직 미해결이거나, 이전에 부족한 정보가 없었음
    #
    # confidence_delta: 이전 대비 신뢰도 변화량
    #   - 양수: 재수집으로 신뢰도 상승 (좋은 신호)
    #   - 음수: 오히려 신뢰도 하락 (나쁜 신호)
    #   - 0: 변화 없음

    prev_missing = state.get("previous_missing_info", [])
    # 하위 호환: 이전 state에 string이 저장된 경우 list로 변환
    if isinstance(prev_missing, str):
        prev_missing = [prev_missing] if prev_missing.strip() else []
    prev_confidence = state.get("previous_confidence", 0.0)

    # 이전에 missing_info가 있었고, 이번에는 없다면 → 해결됨
    missing_info_resolved = bool(prev_missing) and not eval_dict.get("missing_info", [])

    # 신뢰도 변화량 계산
    confidence_delta = round(
        eval_dict.get("confidence_score", 0) - prev_confidence, 3
    )

    # 일관성 패턴 A 결과를 context_metadata에 추가 (터미널 출력용)
    context_metadata["confidence_delta"] = confidence_delta
    context_metadata["missing_info_resolved"] = missing_info_resolved

    # ═══════════════════════════════════════
    # STEP 4: Langfuse에 Layer 2 attribute 기록
    # ═══════════════════════════════════════
    # 이 스팬에 15개의 Context Monitoring attribute를 메타데이터로 첨부한다.
    # 이 데이터는 Langfuse 대시보드에서 효율성, 완전성, 관련성, 일관성을 분석하는 데 사용된다.
    get_client().update_current_span(
        metadata={
            # 효율성 (Efficiency) 지표
            ATTRS["context.total_tokens"]: context_metadata["total_tokens"],
            ATTRS["context.window_utilization"]: context_metadata["context_window_utilization"],
            ATTRS["context.source.system_prompt_tokens"]: system_tokens,
            ATTRS["context.source.query_analysis_tokens"]: query_tokens,
            ATTRS["context.source.gathered_data_tokens"]: gathered_tokens,
            ATTRS["context.source.previous_turns_tokens"]: prev_turns_tokens + conclusions_tokens,
            # v3 폐기: turn_conclusions_tokens → previous_turns_tokens에 통합
            # v3 폐기: messages_tokens (활용 0, 관측 전용)

            # 관련성 (Relevance) 지표
            ATTRS["context.noise_ratio"]: noise_ratio,
            ATTRS["context.effective_noise_ratio"]: effective_noise_ratio,

            # 완전성 (Completeness) 지표
            ATTRS["context.is_sufficient"]: eval_dict.get("is_sufficient"),
            ATTRS["context.missing_info"]: eval_dict.get("missing_info", ""),
            ATTRS["context.sufficiency_confidence"]: eval_dict.get("confidence_score", 0),

            # 일관성 패턴 A (Consistency - Pattern A) 지표
            ATTRS["context.missing_info_resolved"]: missing_info_resolved,
            ATTRS["context.confidence_delta"]: confidence_delta,

            # v3 폐기: continuity_score (fidelity_score와 의미 중복)

            # Literature Review #26 (Anthropic): context rot 복합 지표
            ATTRS["context.rot_risk"]: rot_risk,
            # A1: 턴 간 rot_risk 변화율 (H4a 검증용)
            ATTRS["context.rot_velocity"]: rot_velocity,

            # 소스 선택 관측 (Source Selection)
            ATTRS["source.contribution"]: source_contribution,

            # G1: 충실도 (일관성 패턴 C)
            ATTRS["context.fidelity_score"]: fidelity_score,
            # F4+A7: 충실도 구성 요소 분리 (진단 드릴다운용)
            ATTRS["context.fidelity_detail"]: fidelity_detail,

            # v3 통합: truncated_items list[dict[source, reason]]
            ATTRS["context.truncated_items"]: truncated_items,

            # G3: 교차 턴 진화
            # v3 폐기: new_data_ratio (derived from gathered_data_tokens / total_tokens)
            # v3 폐기: token_delta (derived from total_tokens 비교)
            ATTRS["context.contributing_turns"]: contributing_turns_count,

            # A4+A5: Rot Gate
            ATTRS["context.dead_weight_tokens"]: dead_weight_tokens,
            ATTRS["context.rot_gate_triggered"]: rot_gate_triggered,
            # v3 폐기: rot_gate_pruned_tokens (derived: dead_weight_tokens × rot_gate_triggered)

            # Post-1: 인과 전파
            ATTRS["context.causal_sources"]: causal_sources,

            # Post-3: 의미적 정보 밀도
            ATTRS["context.information_density"]: information_density,
            ATTRS["context.redundancy_ratio"]: redundancy_ratio,

            # v3 폐기: sufficiency_by_source (Judge 활용 약함, is_sufficient + missing_info로 충분)

            # 태그 (v3에서는 metadata에 포함)
            "tags": [
                "context_monitoring",
                "sufficient" if eval_dict.get("is_sufficient") else "insufficient",
            ],
        },
    )

    result = {
        "context_evaluation": eval_dict,              # should_continue_gather 분기에서 사용
        "context_metadata": context_metadata,          # 관측 전용 (노드 간 전달용)
        "previous_missing_info": eval_dict.get("missing_info", []),   # 일관성 A: 다음 iteration용
        "previous_confidence": eval_dict.get("confidence_score", 0),  # 일관성 A: 다음 iteration용
        "previous_total_tokens": total_tokens,         # G3: 다음 턴 token_delta 계산용
        "previous_rot_risk": rot_risk,                 # A1: 다음 턴 rot_velocity 계산용
    }
    # A5: Rot Gate가 turn_conclusions를 pruning한 경우에만 state 업데이트
    if rot_gate_triggered and rot_gate_pruned_tokens > 0:
        result["turn_conclusions"] = turn_conclusions
    return result
