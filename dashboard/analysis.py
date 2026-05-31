"""
dashboard/analysis.py — 순수 통계 분석 (LLM 호출 없음)

역할:
    Dashboard에서 사용하는 분석 함수를 제공한다.
    모든 분석은 pandas/scipy 기반이며 LLM 호출을 하지 않는다.

    1. compute_session_aggregates: 세션 레벨 집계 (Tab 1)
    2. classify_llm_calls: observation을 노드별 LLM 호출로 분류
    3. extract_node_contributions: 노드별 생산 속성 추출
    4. compute_attribute_correlations: attribute-4D 상관 분석 (Step 6)

데이터 흐름:
    입력: observations (list[dict]) — data_loader.load_observations()의 반환값
    출력: 분류/분석 결과 (list[dict], pd.DataFrame)
"""
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from agent.monitoring_schema import ATTR_META, ATTRS, THRESHOLDS

# --- 노드별 용도 라벨 ---
NODE_PURPOSE_LABELS = {
    "analyze_query":     "사용자 질의 분석",
    "gather_data":       "데이터 수집 (도구 호출)",
    "evaluate_context":  "충분성 판단 (체크포인트)",
    "generate_analysis": "분석 생성",
    "verify_result":     "수치/해석 검증 (체크포인트)",
    "respond_to_user":   "응답 생성",
}

# --- 4D/이탈 Judge generation 이름 접두사 → 용도 라벨 ---
# Judge LLM은 노드 SPAN 하위가 아니라 턴 trace 루트에 직접 부착되므로
# parent_node 체인이 아닌 generation 자체의 name(접두사)으로 식별한다.
JUDGE_NAME_PREFIXES = {
    "judge_4d.":        "4D 품질 평가 (Judge)",
    "judge_alignment.": "쿼리 정렬 평가 (Judge)",
}

# --- LLM 호출 카테고리 분류 ---
USER_TASK_NODES = {"analyze_query", "generate_analysis", "respond_to_user"}
MONITORING_CHECKPOINT_NODES = {"evaluate_context", "verify_result"}  # 모니터링 체크포인트: 충분성 판단 + 검증

# --- 노드 실행 순서 (정렬용) ---
NODE_ORDER = {
    "analyze_query": 1,
    "gather_data": 2,
    "evaluate_context": 3,
    "generate_analysis": 4,
    "verify_result": 5,
    "respond_to_user": 6,
}


# ═══════════════════════════════════════
# 세션 레벨 집계 (Tab 1: 컨텍스트 여정)
# ═══════════════════════════════════════

# 도구명 → 소스 유형 매핑
_TOOL_SOURCE_MAP = {
    "pandas_query": "csv",
    "calculate": "csv",
    "rag_search": "rag",
    "rag_deep_read": "rag",
    "rag_global_summary": "rag",
    "rag_compare": "rag",
    "web_search": "web",
    "api_query": "api",
}


def compute_session_aggregates(turns: list[dict]) -> dict:
    """세션 전체의 집계 지표를 계산한다. (LLM 호출 없음)

    Tab 1 "한눈에" 탭의 컨텍스트 여정 내러티브에 필요한 세션 레벨 집계를 반환한다.
    파생 가능한 값(items_delivered, unique_source_types 등)은 뷰에서 계산한다.

    Args:
        turns: enriched 턴별 데이터 리스트 (load_enriched_session_data의 반환값).

    Returns:
        세션 집계 딕셔너리. 키 설명은 계획 문서 참조.
    """
    total_turns = len(turns)

    # --- 소스 기여 집계 ---
    # source.contribution이 있으면 사용, 없으면 tools_called에서 추론
    source_contrib_sums: dict[str, float] = {}
    source_contrib_counts: dict[str, int] = {}

    # --- 도구 빈도 ---
    tool_freq: dict[str, int] = {}

    # --- 수집/제외/절단 ---
    items_collected_total = 0
    items_excluded_total = 0
    truncated_items_total = 0

    # --- 윈도우/토큰 ---
    window_utils: list[float] = []
    total_tokens_list: list[float] = []

    # 토큰 소스별 합계 (나중에 평균 계산)
    token_source_keys = {
        "system": ATTRS["context.source.system_prompt_tokens"],
        "gathered": ATTRS["context.source.gathered_data_tokens"],
        "previous": ATTRS["context.source.previous_turns_tokens"],
        "conclusions": ATTRS.get("context.source.turn_conclusions_tokens", "context.source.turn_conclusions_tokens"),
        "query": ATTRS["context.source.query_analysis_tokens"],
    }
    token_source_sums: dict[str, float] = {k: 0.0 for k in token_source_keys}
    token_source_counts: dict[str, int] = {k: 0 for k in token_source_keys}

    # --- 이벤트 카운터 ---
    regather_count = 0
    sufficiency_fail_count = 0
    contradiction_count = 0
    unresolved_contradictions = 0
    verify_fail_count = 0
    verify_total_count = 0

    for turn in turns:
        meta = turn.get("metadata", {})

        # 소스 기여: source.contribution (dict: source_type → ratio)
        contrib = meta.get(ATTRS.get("source.contribution", "source.contribution"), {})
        if isinstance(contrib, dict) and contrib:
            for src_type, ratio in contrib.items():
                if isinstance(ratio, (int, float)):
                    source_contrib_sums[src_type] = source_contrib_sums.get(src_type, 0.0) + ratio
                    source_contrib_counts[src_type] = source_contrib_counts.get(src_type, 0) + 1

        # 도구 호출 빈도
        tools = meta.get(ATTRS.get("gather.tools_called", "gather.tools_called"), [])
        if isinstance(tools, list):
            for tool in tools:
                tool_freq[tool] = tool_freq.get(tool, 0) + 1
                # source.contribution이 없으면 도구에서 소스 타입 추론
                if not contrib:
                    src = _TOOL_SOURCE_MAP.get(tool)
                    if src:
                        source_contrib_sums[src] = source_contrib_sums.get(src, 0.0) + 1.0
                        source_contrib_counts[src] = source_contrib_counts.get(src, 0) + 1

        # 수집/제외/절단
        collected = meta.get(ATTRS.get("gather.items_collected", "gather.items_collected"), 0)
        if isinstance(collected, (int, float)):
            items_collected_total += int(collected)

        excluded = meta.get(ATTRS.get("gather.items_excluded", "gather.items_excluded"), 0)
        if isinstance(excluded, (int, float)):
            items_excluded_total += int(excluded)

        truncated = meta.get(ATTRS.get("context.truncated_items_count", "context.truncated_items_count"), 0)
        if isinstance(truncated, (int, float)):
            truncated_items_total += int(truncated)

        # 윈도우 사용률
        win_util = meta.get(ATTRS["context.window_utilization"])
        if isinstance(win_util, (int, float)):
            window_utils.append(float(win_util))

        # 총 토큰
        total_tok = meta.get(ATTRS["context.total_tokens"])
        if isinstance(total_tok, (int, float)):
            total_tokens_list.append(float(total_tok))

        # 토큰 소스별 합계
        for label, attr_key in token_source_keys.items():
            val = meta.get(attr_key)
            if isinstance(val, (int, float)):
                token_source_sums[label] += float(val)
                token_source_counts[label] += 1

        # 재수집: gather.iteration > 1
        iteration = meta.get(ATTRS.get("gather.iteration", "gather.iteration"), 1)
        if isinstance(iteration, (int, float)) and iteration > 1:
            regather_count += 1

        # 충분성 미달
        is_sufficient = meta.get(ATTRS["context.is_sufficient"])
        if is_sufficient is False:
            sufficiency_fail_count += 1

        # 모순 (v2/v3 trace 양립 helper 사용)
        from agent.monitoring_schema import (
            get_contradicts_from_metadata,
            get_contradiction_resolved_from_metadata,
        )
        contradicts = get_contradicts_from_metadata(meta)
        if contradicts is True:
            contradiction_count += 1
            resolved = get_contradiction_resolved_from_metadata(meta)
            if resolved is not True:
                unresolved_contradictions += 1

        # 검증
        verdict = meta.get(ATTRS["verify.overall_verdict"])
        if verdict is not None and verdict != "":
            verify_total_count += 1
            if verdict != "pass":
                verify_fail_count += 1

    # --- 평균 계산 ---
    # 소스 기여: 합산 후 정규화 (비율 합이 1이 되도록)
    source_contribution_agg: dict[str, float] = {}
    if source_contrib_sums:
        total_contrib = sum(source_contrib_sums.values())
        if total_contrib > 0:
            source_contribution_agg = {
                k: v / total_contrib for k, v in source_contrib_sums.items()
            }

    # 토큰 소스 평균
    token_sources_avg: dict[str, float] = {}
    for label in token_source_keys:
        cnt = token_source_counts[label]
        if cnt > 0:
            token_sources_avg[label] = token_source_sums[label] / cnt

    return {
        "total_turns": total_turns,
        "source_contribution_agg": source_contribution_agg,
        "tool_frequency": tool_freq,
        "items_collected_total": items_collected_total,
        "items_excluded_total": items_excluded_total,
        "truncated_items_total": truncated_items_total,
        "window_utilization_avg": (
            sum(window_utils) / len(window_utils) if window_utils else 0.0
        ),
        "token_sources_avg": token_sources_avg,
        "total_tokens_avg": (
            sum(total_tokens_list) / len(total_tokens_list) if total_tokens_list else 0.0
        ),
        "regather_count": regather_count,
        "sufficiency_fail_count": sufficiency_fail_count,
        "contradiction_count": contradiction_count,
        "unresolved_contradictions": unresolved_contradictions,
        "verify_fail_count": verify_fail_count,
        "verify_total_count": verify_total_count,
    }


# ═══════════════════════════════════════
# LLM 호출 분류 (Tab 4)
# ═══════════════════════════════════════

def classify_llm_calls(observations: list[dict]) -> list[dict]:
    """Observation 목록에서 LLM 호출(GENERATION)을 추출하고 용도별로 분류한다.

    분류 전략: GENERATION의 parent_observation_id를 따라가서
    부모(CHAIN 또는 SPAN)의 name을 노드명으로 사용한다.

    Args:
        observations: load_observations()의 반환값.

    Returns:
        분류된 LLM 호출 리스트. 각 항목:
        {parent_node, purpose_label, input_text, output_text,
         tokens_in, tokens_out, latency_ms, model, start_time, iteration}
    """
    obs_by_id = {o["id"]: o for o in observations}

    # GENERATION 타입만 추출
    generations = [o for o in observations if o["type"] == "GENERATION"]

    # 같은 노드의 호출 횟수 추적 (evaluate_context iteration 등)
    node_call_counts: dict[str, int] = {}

    results = []
    for gen in sorted(generations, key=lambda g: g.get("start_time") or ""):
        gen_name = gen.get("name") or ""

        # Judge generation 우선 식별 — 노드 SPAN 하위가 아니라 trace 루트에 직접 부착되므로
        # parent_node 체인 대신 generation 이름 접두사로 분류한다.
        judge_label = next(
            (label for prefix, label in JUDGE_NAME_PREFIXES.items() if gen_name.startswith(prefix)),
            None,
        )
        if judge_label:
            # parent_node에는 차원명까지 보이도록 generation 이름을 그대로 사용
            # (예: "judge_4d.completeness_score")
            parent_node = gen_name
            purpose_label = judge_label
            call_category = "evaluation"
        else:
            # 부모 체인을 따라가서 노드명 찾기
            parent_node = _resolve_parent_node(gen, obs_by_id)
            purpose_label = NODE_PURPOSE_LABELS.get(parent_node, f"기타 ({parent_node})")
            # 호출 카테고리: 사용자 작업 vs 평가 노드
            if parent_node in USER_TASK_NODES:
                call_category = "user_task"
            elif parent_node in MONITORING_CHECKPOINT_NODES:
                call_category = "monitoring_checkpoint"
            else:
                call_category = "other"

        # iteration 번호 (같은 노드가 여러 번 호출된 경우)
        node_call_counts[parent_node] = node_call_counts.get(parent_node, 0) + 1
        iteration = node_call_counts[parent_node]

        # 토큰 및 지연시간
        usage = gen.get("usage", {})
        tokens_in = usage.get("input")
        tokens_out = usage.get("output")
        latency_ms = _calc_latency_ms(gen.get("start_time"), gen.get("end_time"))

        results.append({
            "parent_node": parent_node,
            "purpose_label": purpose_label,
            "call_category": call_category,
            "iteration": iteration,
            "input_text": gen.get("input"),
            "output_text": gen.get("output"),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "model": gen.get("model") or "N/A",
            "start_time": gen.get("start_time"),
        })

    return results


def _resolve_parent_node(gen: dict, obs_by_id: dict) -> str:
    """GENERATION의 부모 체인을 따라가서 노드명을 찾는다.

    구조: GENERATION → CHAIN:{node_name} → CHAIN:LangGraph → trace root
    또는: GENERATION → SPAN:{node_name} → ...

    Args:
        gen: GENERATION observation dict.
        obs_by_id: id → observation 매핑.

    Returns:
        노드명 (예: "analyze_query"). 못 찾으면 "unknown".
    """
    parent_id = gen.get("parent_observation_id")
    visited = set()

    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent = obs_by_id.get(parent_id)
        if parent is None:
            break
        parent_name = parent.get("name") or ""
        # LangGraph나 _execute_turn은 컨테이너 — 건너뛴다
        if parent_name in ("LangGraph", "_execute_turn", ""):
            parent_id = parent.get("parent_observation_id")
            continue
        # 분기 함수도 건너뛴다
        if parent_name.startswith("should_") or parent_name.startswith("route_"):
            parent_id = parent.get("parent_observation_id")
            continue
        # 노드명 발견
        return parent_name

    return "unknown"


def _calc_latency_ms(start: str | None, end: str | None) -> int | None:
    """start_time과 end_time 문자열에서 지연시간(ms)을 계산한다.

    Args:
        start: ISO 형식 시작 시간 문자열.
        end: ISO 형식 종료 시간 문자열.

    Returns:
        지연시간(ms). 계산 불가 시 None.
    """
    if not start or not end:
        return None
    try:
        # "2026-03-10 10:28:55.123456+00:00" 같은 형식 처리
        s = datetime.fromisoformat(str(start))
        e = datetime.fromisoformat(str(end))
        return round((e - s).total_seconds() * 1000)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════
# 노드별 Contribution (Tab 2 실행 흐름)
# ═══════════════════════════════════════

def extract_node_contributions(observations: list[dict]) -> list[dict]:
    """SPAN observation에서 각 노드가 생산한 모니터링 속성을 추출한다.

    각 노드는 서로 다른 namespace의 속성을 기록하므로 "diff"가 아니라
    "각 노드가 이번 턴에서 생산한 것"을 보여준다.
    evaluate_context가 여러 번 실행된 경우 iteration 간 비교도 제공한다.

    Args:
        observations: load_observations()의 반환값.

    Returns:
        노드별 contribution 리스트. 각 항목:
        {node_name, order, start_time, end_time, iteration,
         attributes: [{key, value, type, quality_dimension, description}]}
    """
    # SPAN 타입만 추출 (노드 실행 단위)
    spans = [o for o in observations if o["type"] == "SPAN"]
    # 컨테이너 SPAN 제외
    spans = [s for s in spans if s.get("name") not in ("_execute_turn", None, "")]

    # start_time 기준 정렬
    spans.sort(key=lambda s: s.get("start_time") or "")

    # 같은 노드의 호출 횟수 추적
    node_counts: dict[str, int] = {}
    results = []

    for span in spans:
        name = span.get("name", "unknown")
        node_counts[name] = node_counts.get(name, 0) + 1
        iteration = node_counts[name]

        metadata = span.get("metadata", {})
        # resourceAttributes, scope 등 Langfuse 내부 키 제외
        skip_keys = {"resourceAttributes", "scope", "tags"}
        filtered_meta = {
            k: v for k, v in metadata.items()
            if k not in skip_keys
        }

        # ATTR_META에서 속성 상세 정보 추가
        attributes = []
        for key, value in sorted(filtered_meta.items()):
            meta_info = ATTR_META.get(key, {})
            attributes.append({
                "key": key,
                "value": value,
                "type": meta_info.get("type", "unknown"),
                "quality_dimension": meta_info.get("quality", "—"),
                "description": meta_info.get("description", ""),
                "producer": meta_info.get("producer", "—"),
                "otel_mapping": meta_info.get("otel_mapping", "—"),
            })

        results.append({
            "node_name": name,
            "order": NODE_ORDER.get(name, 99),
            "start_time": span.get("start_time"),
            "end_time": span.get("end_time"),
            "iteration": iteration,
            "attributes": attributes,
        })

    # 실행 순서대로 정렬
    results.sort(key=lambda r: (r.get("start_time") or "", r["order"]))
    return results


# ═══════════════════════════════════════
# Attribute → 4D Impact 상관 분석 (Tab 3)
# ═══════════════════════════════════════

# 4D 차원명 → Langfuse score key
SCORE_KEYS = {
    "completeness": "completeness_score",
    "efficiency": "efficiency_score",
    "relevance": "relevance_score",
    "consistency": "consistency_score",
}

# 상관 분석 최소 표본 수
MIN_SAMPLES = 5


def compute_attribute_correlations(
    turns: list[dict],
    filter_related_only: bool = False,
) -> pd.DataFrame:
    """수치/boolean 속성과 4D 점수 간의 Spearman 상관 분석을 수행한다.

    Benjamini-Hochberg FDR 보정을 적용하여 다중 비교 문제를 방지한다.

    Args:
        turns: 턴별 데이터 리스트 (load_session_data 또는 load_multi_session_data의 반환값).
        filter_related_only: True면 YAML quality 필드에서 관련된 차원만 테스트.

    Returns:
        DataFrame with columns:
        [attribute, dimension, correlation, p_value, q_value, n_samples, is_significant, is_related]
        데이터 부족 시 빈 DataFrame.
    """
    if len(turns) < MIN_SAMPLES:
        return pd.DataFrame()

    # 수치/boolean 속성만 필터링
    numeric_attrs = _get_numeric_attributes()
    if not numeric_attrs:
        return pd.DataFrame()

    # 턴별 속성값 + 점수 추출
    rows = []
    for turn in turns:
        metadata = turn.get("metadata", {})
        scores = turn.get("scores", {})

        # 4D 점수가 하나라도 없으면 건너뛴다
        if not any(scores.get(SCORE_KEYS[d]) is not None for d in SCORE_KEYS):
            continue

        row = {}
        # 속성값 추출
        for attr_key in numeric_attrs:
            val = metadata.get(attr_key)
            if val is not None:
                if isinstance(val, bool):
                    row[attr_key] = 1.0 if val else 0.0
                elif isinstance(val, (int, float)):
                    row[attr_key] = float(val)
        # 점수 추출
        for dim, score_key in SCORE_KEYS.items():
            val = scores.get(score_key)
            if val is not None:
                row[f"_score_{dim}"] = float(val)

        if row:
            rows.append(row)

    if len(rows) < MIN_SAMPLES:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 상관계수 계산
    results = []
    for attr_key in numeric_attrs:
        if attr_key not in df.columns:
            continue

        attr_meta = ATTR_META.get(attr_key, {})
        related_quality = attr_meta.get("quality")

        for dim in SCORE_KEYS:
            score_col = f"_score_{dim}"
            if score_col not in df.columns:
                continue

            # 관련 차원만 필터링 옵션
            is_related = _is_quality_related(related_quality, dim)
            if filter_related_only and not is_related:
                continue

            # 둘 다 non-null인 행만
            valid = df[[attr_key, score_col]].dropna()
            n = len(valid)
            if n < MIN_SAMPLES:
                continue

            # 분산이 0이면 상관 계산 불가
            if valid[attr_key].std() == 0 or valid[score_col].std() == 0:
                continue

            corr, p_value = scipy_stats.spearmanr(valid[attr_key], valid[score_col])

            results.append({
                "attribute": attr_key,
                "dimension": dim,
                "correlation": corr,
                "p_value": p_value,
                "n_samples": n,
                "is_related": is_related,
            })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR 보정
    p_values = result_df["p_value"].values
    q_values = _benjamini_hochberg(p_values)
    result_df["q_value"] = q_values
    result_df["is_significant"] = result_df["q_value"] < 0.05

    return result_df.sort_values("q_value")


def _get_numeric_attributes() -> list[str]:
    """ATTR_META에서 수치/boolean 타입 속성만 추출한다.

    Returns:
        속성 키 리스트.
    """
    numeric_types = {"int", "float", "bool"}
    return [
        key for key, meta in ATTR_META.items()
        if meta.get("type") in numeric_types
    ]


def _is_quality_related(attr_quality: str | None, dimension: str) -> bool:
    """속성의 quality 필드가 해당 차원과 관련 있는지 확인한다.

    Args:
        attr_quality: ATTR_META의 quality 값 (예: "completeness", "consistency_a").
        dimension: 4D 차원명 (예: "completeness", "consistency").

    Returns:
        관련 있으면 True.
    """
    if not attr_quality:
        return False
    # consistency_a, consistency_b → consistency에 매핑
    normalized = attr_quality.split("_")[0] if "_" in attr_quality else attr_quality
    return normalized == dimension


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 보정을 적용한다.

    Args:
        p_values: raw p-value 배열.

    Returns:
        FDR-adjusted q-value 배열 (같은 크기).
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    # p-value 오름차순 정렬 인덱스
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH 보정: q_i = p_i * n / rank_i (단조 증가 보장)
    ranks = np.arange(1, n + 1)
    q_values = sorted_p * n / ranks

    # 뒤에서부터 누적 최솟값으로 단조 증가 보장
    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])

    # 1.0 상한
    q_values = np.minimum(q_values, 1.0)

    # 원래 순서로 복원
    result = np.empty(n)
    result[sorted_idx] = q_values
    return result


# ═══════════════════════════════════════
# Post-4: 이상 패턴 감지 (Anomaly Detection)
# ═══════════════════════════════════════

# 이상 패턴 규칙: {조건 → 경고 메시지}
_ANOMALY_RULES = [
    {
        "name": "context_rot",
        "check": lambda m: m.get("context.noise_ratio", 0) > 0.5 and m.get("context.rot_risk", 0) > 0.3,
        "severity": "warning",
        "message": "Context Rot 경고: noise_ratio > 0.5 이고 rot_risk > 0.3 — 이전 턴 데이터 정리 필요",
    },
    {
        "name": "semantic_loss",
        # v2/v3 trace 양립: v2는 fidelity + continuity 둘 다, v3는 continuity 폐기로 fidelity 단독 검증.
        "check": lambda m: (
            m.get("context.fidelity_score", 1.0) < 0.5
            and (
                m.get("context.continuity_score", 0) > 0.8
                if "context.continuity_score" in m
                else True  # v3 trace: continuity 없으면 fidelity 단독으로 의미 손실 감지
            )
        ),
        "severity": "warning",
        "message": "의미 손실 경고: 의미적 충실도가 낮음 (fidelity < 0.5)",
    },
    {
        "name": "token_bloat",
        "check": lambda m: m.get("context.window_utilization", 0) > 0.8 and m.get("context.information_density", 1.0) < 0.2,
        "severity": "warning",
        "message": "토큰 팽창 경고: 윈도우 사용률 > 80% 이지만 정보 밀도 < 20%",
    },
    {
        "name": "excessive_compression",
        # v2/v3 trace 양립: v2는 compression_ratio + conditions_preserved, v3는 fidelity_detail에 흡수.
        "check": lambda m: (
            (m.get("response.compression_ratio", 1.0) < 0.1 and not m.get("response.conditions_preserved", True))
            if "response.compression_ratio" in m
            else (
                isinstance(m.get("context.fidelity_detail"), dict)
                and (m["context.fidelity_detail"].get("compression_appropriateness", 1.0) < 0.3)
            )
        ),
        "severity": "error",
        "message": "과도한 압축: 압축률 과다 또는 fidelity_detail의 compression_appropriateness 낮음",
    },
]


def detect_anomaly_patterns(turns: list[dict]) -> list[dict]:
    """속성값 조합에서 이상 패턴을 감지한다. (LLM 호출 없음)

    규칙 기반 감지로, 관측 데이터에서 품질 저하를 시사하는 속성 조합을 찾는다.

    Args:
        turns: 턴별 데이터 리스트 (metadata를 포함).

    Returns:
        감지된 이상 패턴 리스트. 각 항목:
        {turn_number, name, severity, message}
    """
    anomalies = []
    for turn in turns:
        metadata = turn.get("metadata", {})
        turn_num = metadata.get("turn.number", turn.get("turn_number", "?"))
        for rule in _ANOMALY_RULES:
            try:
                if rule["check"](metadata):
                    anomalies.append({
                        "turn_number": turn_num,
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "message": rule["message"],
                    })
            except (TypeError, KeyError):
                continue
    return anomalies


def compute_attribute_trends(turns: list[dict]) -> list[dict]:
    """3턴 이상 연속 하락/상승하는 속성 패턴을 감지한다. (LLM 호출 없음)

    Args:
        turns: 턴별 데이터 리스트.

    Returns:
        감지된 추세 리스트. 각 항목:
        {attribute, direction, consecutive_turns, start_turn, end_turn}
    """
    if len(turns) < 3:
        return []

    # 수치 속성 추출
    numeric_attrs = _get_numeric_attributes()
    trends = []

    for attr in numeric_attrs:
        values = []
        for turn in turns:
            val = turn.get("metadata", {}).get(attr)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(None)

        # None이 아닌 연속 구간에서 방향 감지
        streak = 0
        direction = None
        start_idx = 0
        for i in range(1, len(values)):
            if values[i] is None or values[i - 1] is None:
                if streak >= 3:
                    trends.append({
                        "attribute": attr,
                        "direction": direction,
                        "consecutive_turns": streak,
                        "start_turn": start_idx + 1,
                        "end_turn": i,
                    })
                streak = 0
                direction = None
                start_idx = i
                continue

            curr_dir = "up" if values[i] > values[i - 1] else ("down" if values[i] < values[i - 1] else None)
            if curr_dir is None:
                continue

            if curr_dir == direction:
                streak += 1
            else:
                if streak >= 3:
                    trends.append({
                        "attribute": attr,
                        "direction": direction,
                        "consecutive_turns": streak,
                        "start_turn": start_idx + 1,
                        "end_turn": i,
                    })
                direction = curr_dir
                streak = 1
                start_idx = i - 1

        if streak >= 3:
            trends.append({
                "attribute": attr,
                "direction": direction,
                "consecutive_turns": streak,
                "start_turn": start_idx + 1,
                "end_turn": len(values),
            })

    return trends


# ═══════════════════════════════════════
# Tab 1: 3축 관측 + 여정 타임라인
# ═══════════════════════════════════════

# --- Health Score 가중치 (session_overview.py에서도 사용) ---
HEALTH_WEIGHTS = {
    "completeness": 0.25,
    "efficiency": 0.25,
    "relevance": 0.25,
    "consistency": 0.25,
}


def _compute_turn_health(scores: dict) -> float | None:
    """단일 턴의 4D 가중 평균 Health Score를 계산한다.

    Args:
        scores: 턴의 scores dict (score_key → value).

    Returns:
        Health Score (0~1). 점수가 없으면 None.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, weight in HEALTH_WEIGHTS.items():
        val = scores.get(SCORE_KEYS[dim])
        if val is not None:
            weighted_sum += val * weight
            total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None



def extract_journey_timeline(turns: list[dict]) -> list[dict]:
    """턴별 3축 + 품질 여정 데이터를 추출한다.

    컨텍스트 여정 테이블 렌더링에 사용. 각 턴의 구성/변형/전달/품질
    항목을 metadata에서 추출하여 구조화한다.

    Args:
        turns: enriched 턴별 데이터 리스트.

    Returns:
        턴별 여정 데이터 리스트.
    """
    timeline = []
    for turn in turns:
        meta = turn.get("metadata", {})
        scores = turn.get("scores", {})

        # 구성 — v3 폐기 attribute 정정:
        #   source.types_selected → gather.tools_called에서 _TOOL_SOURCE_MAP으로 도출
        #   gather.tools_count    → len(gather.tools_called)
        tools_called = meta.get("gather.tools_called", []) or []
        if not isinstance(tools_called, list):
            tools_called = []
        sources = sorted({
            _TOOL_SOURCE_MAP.get(t, t) for t in tools_called if t
        })
        tools_count = len(tools_called)
        items_collected = meta.get("gather.items_collected", 0) or 0

        # 변형
        new_data_ratio = meta.get("context.new_data_ratio")
        fidelity = meta.get("context.fidelity_score")
        # 이벤트 아이콘 축약
        events = []
        if meta.get("analysis.contradicts_previous"):
            resolved = meta.get("analysis.contradiction_resolved", False)
            events.append("⚠모순(해결)" if resolved else "⚠모순")
        if meta.get("source.conflict_detected"):
            events.append("⚠충돌")
        verdict = meta.get("verify.overall_verdict")
        if verdict and verdict != "pass":
            label = "수치" if verdict == "fail_numeric" else "해석"
            events.append(f"✗검증({label})")
        iteration = meta.get("gather.iteration", 1) or 1
        if iteration > 1:
            events.append(f"↻재수집×{iteration - 1}")

        # 전달
        window_util = meta.get("context.window_utilization")
        density = meta.get("context.information_density")
        noise = meta.get("context.noise_ratio")

        # 품질
        health = _compute_turn_health(scores)

        timeline.append({
            "turn_number": turn.get("turn_number", 0),
            "composition": {
                "sources": sources,
                "tools_count": tools_count,
                "items_collected": items_collected,
            },
            "transformation": {
                "new_data_ratio": new_data_ratio,
                "fidelity": fidelity,
                "events": events,
            },
            "delivery": {
                "window_util": window_util,
                "density": density,
                "noise": noise,
            },
            "quality": {
                "C": scores.get(SCORE_KEYS["completeness"]),
                "E": scores.get(SCORE_KEYS["efficiency"]),
                "R": scores.get(SCORE_KEYS["relevance"]),
                "S": scores.get(SCORE_KEYS["consistency"]),
                "H": health,
            },
        })

    return timeline


# --- 주의 이벤트 → 탭 참조 매핑 ---
_EVENT_TAB_MAP = {
    "regather": "🔍① 구성",
    "contradiction": "🔍② 변형",
    "source_conflict": "🔍② 변형",
    "verify_fail": "📊 측정&진단",
}


def detect_attention_events(turns: list[dict]) -> list[dict]:
    """주의가 필요한 이벤트를 감지한다.

    각 턴의 metadata와 iterations에서 프로세스 이상 이벤트를 추출하고,
    관측자가 어떤 탭을 참조해야 하는지 안내한다.

    Args:
        turns: enriched 턴별 데이터 리스트.

    Returns:
        이벤트 리스트. 각 항목:
        {turn_number, event_type, description, tab_reference}
    """
    events = []
    for turn in turns:
        meta = turn.get("metadata", {})
        turn_num = turn.get("turn_number", "?")
        iterations = turn.get("iterations", {})

        # 재수집 (gather.iteration > 1)
        iteration = meta.get("gather.iteration", 1) or 1
        if iteration > 1:
            events.append({
                "turn_number": turn_num,
                "event_type": "regather",
                "description": f"충분성 미달 → 재수집 {iteration - 1}회",
                "tab_reference": _EVENT_TAB_MAP["regather"],
            })

        # 모순 감지
        if meta.get("analysis.contradicts_previous"):
            resolved = meta.get("analysis.contradiction_resolved", False)
            status = "해결됨" if resolved else "미해결"
            events.append({
                "turn_number": turn_num,
                "event_type": "contradiction",
                "description": f"이전 턴과 모순 감지 ({status})",
                "tab_reference": _EVENT_TAB_MAP["contradiction"],
            })

        # 소스 충돌
        if meta.get("source.conflict_detected"):
            events.append({
                "turn_number": turn_num,
                "event_type": "source_conflict",
                "description": "소스 간 충돌 감지",
                "tab_reference": _EVENT_TAB_MAP["source_conflict"],
            })

        # 검증 실패 — 최종 verdict 확인
        verdict = meta.get("verify.overall_verdict")
        if verdict and verdict != "pass":
            label = "수치 불일치" if verdict == "fail_numeric" else "해석 오류"
            events.append({
                "turn_number": turn_num,
                "event_type": "verify_fail",
                "description": f"검증 실패 ({label})",
                "tab_reference": _EVENT_TAB_MAP["verify_fail"],
            })

        # 중간 검증 실패 (최종 pass여도 iterations에서 중간 fail 감지)
        if verdict == "pass" and "verify_result" in iterations:
            for iter_record in iterations["verify_result"]:
                iter_verdict = iter_record.get("verify.overall_verdict")
                if iter_verdict and iter_verdict != "pass":
                    label = "수치 불일치" if iter_verdict == "fail_numeric" else "해석 오류"
                    events.append({
                        "turn_number": turn_num,
                        "event_type": "verify_fail",
                        "description": f"검증 실패 ({label}) → 재시도 후 통과",
                        "tab_reference": _EVENT_TAB_MAP["verify_fail"],
                    })
                    break  # 한 번만 기록

    return events
