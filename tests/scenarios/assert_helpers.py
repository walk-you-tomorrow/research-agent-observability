"""
tests/scenarios/assert_helpers.py — 시나리오 테스트용 Soft/Hard Assertion 헬퍼

역할:
    시나리오 테스트에서 turn_results를 활용한 세밀한 검증을 지원한다.
    LLM 비결정성을 고려하여 soft(경고만)/hard(실패) 2단계 정책을 제공한다.
    ScenarioTestLog 클래스로 자기 설명적 구조화 로그를 생성한다.

데이터 흐름:
    입력: session_state["turn_results"] (main.py run_session()이 축적한 per-turn 결과)
    출력: assertion 결과 (pass/fail/warn) + 구조화된 로그 (stdout → TeeWriter → .log)

Soft vs Hard 정책:
    - Hard (assert): 세션 완료, 결론 수, 턴 번호 시퀀스, is_sufficient 등 구조적 사실
    - Soft (warn): 도구 호출, 소스 선택, 모순 감지, 참조 턴 등 LLM 비결정적 판단
"""


def get_turn_result(session_state: dict, turn_index: int) -> dict:
    """turn_results에서 특정 턴의 결과를 추출한다.

    Args:
        session_state: run_session()의 반환값.
        turn_index: 0-based 턴 인덱스.

    Returns:
        해당 턴의 graph.invoke() 결과 딕셔너리.
    """
    turn_results = session_state.get("turn_results", [])
    assert len(turn_results) > turn_index, (
        f"turn_results에 {turn_index}번째 턴이 없음 (총 {len(turn_results)}개)"
    )
    return turn_results[turn_index]


def warn_if_false(condition: bool, message: str) -> bool:
    """조건이 거짓이면 WARNING을 출력하지만 테스트를 실패시키지 않는다.

    LLM 비결정성으로 인해 항상 보장할 수 없는 검증에 사용한다.
    stdout으로 출력하여 TeeWriter가 로그 파일에 기록한다.

    Args:
        condition: 검증할 조건.
        message: 거짓일 때 출력할 메시지.

    Returns:
        condition 값을 그대로 반환.
    """
    if not condition:
        print(f"  ⚠ [SOFT FAIL] {message}")
    return condition


# --- 도구 호출 검증 ---

def assert_tool_called(result: dict, tool_name: str, *, soft: bool = True) -> bool:
    """특정 도구가 호출되었는지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        tool_name: 검증할 도구 이름.
        soft: True면 경고만, False면 assert.

    Returns:
        도구가 호출되었으면 True.
    """
    tools_called = result.get("tools_called", [])
    found = tool_name in tools_called
    msg = f"'{tool_name}' 미호출 (tools_called={tools_called})"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_tool_not_called(result: dict, tool_name: str, *, soft: bool = True) -> bool:
    """특정 도구가 호출되지 않았는지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        tool_name: 검증할 도구 이름.
        soft: True면 경고만, False면 assert.

    Returns:
        도구가 호출되지 않았으면 True.
    """
    tools_called = result.get("tools_called", [])
    not_found = tool_name not in tools_called
    msg = f"'{tool_name}' 이 호출됨 (tools_called={tools_called}) — 미호출 기대"
    if soft:
        return warn_if_false(not_found, msg)
    assert not_found, msg
    return True


def assert_any_tool_called(result: dict, tool_names: list[str], *, soft: bool = True) -> bool:
    """주어진 도구 목록 중 하나 이상이 호출되었는지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        tool_names: 검증할 도구 이름 목록.
        soft: True면 경고만, False면 assert.

    Returns:
        도구 목록 중 하나라도 호출되었으면 True.
    """
    tools_called = result.get("tools_called", [])
    found = any(t in tools_called for t in tool_names)
    msg = f"{tool_names} 중 아무것도 미호출 (tools_called={tools_called})"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


# --- 소스 선택 검증 ---

def assert_source_type_selected(result: dict, source_type: str, *, soft: bool = True) -> bool:
    """query_analysis.source_types에 특정 소스가 포함되었는지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        source_type: 검증할 소스 유형 (csv, rag, web, api).
        soft: True면 경고만, False면 assert.

    Returns:
        해당 소스가 선택되었으면 True.
    """
    qa = result.get("query_analysis", {})
    source_types = qa.get("source_types", [])
    found = source_type in source_types
    msg = f"source_types에 '{source_type}' 없음 (source_types={source_types})"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_multi_source_used(result: dict, min_count: int = 2, *, soft: bool = True) -> bool:
    """gathered_data에서 다중 소스 유형이 사용되었는지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        min_count: 최소 소스 유형 수.
        soft: True면 경고만, False면 assert.

    Returns:
        다중 소스가 사용되었으면 True.
    """
    gathered = result.get("gathered_data", [])
    # gathered_data 항목의 tool_used에서 소스 유형 추론
    source_types = set()
    for item in gathered:
        tool = item.get("tool_used", "")
        if tool in ("pandas_query", "calculate"):
            source_types.add("csv")
        elif tool.startswith("rag_"):
            source_types.add("rag")
        elif tool == "web_search":
            source_types.add("web")
        elif tool == "api_query":
            source_types.add("api")
    found = len(source_types) >= min_count
    msg = f"소스 유형 {len(source_types)}개 < 최소 {min_count}개 (types={source_types})"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


# --- 충분성/컨텍스트 검증 ---

def assert_is_sufficient(result: dict) -> bool:
    """컨텍스트 충분성 판단이 True인지 hard assert한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.

    Returns:
        True (assert 통과 시).
    """
    ctx_eval = result.get("context_evaluation", {})
    is_sufficient = ctx_eval.get("is_sufficient", False)
    assert is_sufficient, (
        f"is_sufficient=False "
        f"(confidence={ctx_eval.get('confidence_score')}, "
        f"missing={ctx_eval.get('missing_info', '')})"
    )
    return True


def assert_gather_iteration_gt(result: dict, min_iter: int, *, soft: bool = True) -> bool:
    """gather_iteration이 특정 값보다 큰지 검증한다 (재수집 발생 검증).

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        min_iter: 최소 iteration 값 (초과해야 통과).
        soft: True면 경고만, False면 assert.

    Returns:
        gather_iteration > min_iter이면 True.
    """
    iteration = result.get("gather_iteration", 0)
    found = iteration > min_iter
    msg = f"gather_iteration={iteration} <= {min_iter} (재수집 미발생)"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


# --- 일관성 검증 ---

def assert_contradicts_previous(result: dict, expected: bool = True, *, soft: bool = True) -> bool:
    """모순 감지 결과를 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        expected: 기대하는 contradicts_previous 값.
        soft: True면 경고만, False면 assert.

    Returns:
        contradicts_previous가 expected와 같으면 True.
    """
    actual = result.get("contradicts_previous", False)
    found = actual == expected
    msg = f"contradicts_previous={actual}, expected={expected}"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_referenced_turns_populated(result: dict, *, soft: bool = True) -> bool:
    """referenced_turns가 비어있지 않은지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        referenced_turns가 비어있지 않으면 True.
    """
    refs = result.get("referenced_turns", [])
    found = len(refs) > 0
    msg = f"referenced_turns 비어있음 (이전 턴 참조 없음)"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


# --- G1~G4 + Post-G5 관측 체계 검증 ---

def assert_fidelity_score_valid(result: dict, *, soft: bool = True) -> bool:
    """G1: fidelity_score가 0~1 범위인지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        fidelity_score가 유효하면 True.
    """
    ctx_meta = result.get("context_metadata", {})
    score = ctx_meta.get("fidelity_score")
    if score is None:
        return warn_if_false(False, "fidelity_score 없음") if soft else False
    valid = 0.0 <= score <= 1.0
    msg = f"fidelity_score={score} (범위 [0,1] 벗어남)" if not valid else ""
    if soft:
        return warn_if_false(valid, msg)
    assert valid, msg
    return True


def assert_first_turn_defaults(result: dict, *, soft: bool = True) -> bool:
    """G1/G3: 첫 턴의 기본값이 올바른지 검증한다.

    첫 턴: fidelity=1.0, inherited_ratio=0.0, contributing_turns=0

    Args:
        result: 첫 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        기본값이 올바르면 True.
    """
    ctx_meta = result.get("context_metadata", {})
    checks = []

    fidelity = ctx_meta.get("fidelity_score", 1.0)
    checks.append(fidelity == 1.0)

    inherited = ctx_meta.get("inherited_ratio", 0.0)
    checks.append(inherited == 0.0)

    contributing = ctx_meta.get("contributing_turns", 0)
    checks.append(contributing == 0)

    all_ok = all(checks)
    detail = (f"fidelity={fidelity}, inherited={inherited}, contributing={contributing}")
    msg = f"첫 턴 기본값 오류: {detail}"
    if soft:
        return warn_if_false(all_ok, msg)
    assert all_ok, msg
    return True


def assert_evolution_metrics_present(result: dict, *, soft: bool = True) -> bool:
    """G3: 교차 턴 진화 지표가 존재하는지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        진화 지표가 모두 존재하면 True.
    """
    ctx_meta = result.get("context_metadata", {})
    keys = ["new_data_ratio", "inherited_ratio", "token_delta", "contributing_turns"]
    missing = [k for k in keys if k not in ctx_meta]
    found = len(missing) == 0
    msg = f"G3 진화 지표 누락: {missing}"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_inherited_ratio_increases(results: list[dict], *, soft: bool = True) -> bool:
    """G3: 멀티턴에서 inherited_ratio가 증가 추세인지 검증한다.

    Args:
        results: 턴별 graph.invoke() 결과 리스트.
        soft: True면 경고만, False면 assert.

    Returns:
        inherited_ratio가 비감소 추세이면 True.
    """
    ratios = []
    for r in results:
        ctx_meta = r.get("context_metadata", {})
        ir = ctx_meta.get("inherited_ratio", 0.0)
        ratios.append(ir)

    # 마지막 턴의 inherited_ratio가 첫 턴보다 크거나 같아야 함
    trend_ok = len(ratios) < 2 or ratios[-1] >= ratios[0]
    msg = f"inherited_ratio 추세 이상: {ratios}"
    if soft:
        return warn_if_false(trend_ok, msg)
    assert trend_ok, msg
    return True


def assert_exclusion_reasons_structure(result: dict, *, soft: bool = True) -> bool:
    """G5: exclusion_reasons의 구조가 올바른지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        exclusion_reasons가 올바른 구조이면 True.
    """
    reasons = result.get("exclusion_reasons", [])
    if not reasons:
        # 제외 항목 없으면 검증 대상 없음 — 통과
        return True

    all_valid = all(
        isinstance(r, dict) and "source" in r and "reason" in r
        for r in reasons
    )
    msg = f"exclusion_reasons 구조 오류: {reasons}"
    if soft:
        return warn_if_false(all_valid, msg)
    assert all_valid, msg
    return True


def assert_causal_sources_present(result: dict, *, soft: bool = True) -> bool:
    """Post-1: causal_sources가 멀티턴에서 존재하는지 검증한다.

    Args:
        result: 2턴 이후의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        causal_sources가 비어있지 않으면 True.
    """
    ctx_meta = result.get("context_metadata", {})
    causal = ctx_meta.get("causal_sources", [])
    # context_metadata 외에도 직접 결과에 없을 수 있음
    found = len(causal) > 0
    msg = f"causal_sources 비어있음 (이전 턴 영향 추적 없음)"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_information_density_positive(result: dict, *, soft: bool = True) -> bool:
    """Post-3: information_density가 존재하고 유효한지 검증한다.

    첫 턴은 이전 key_claims가 없어 density=0이 정상이므로,
    contributing_turns > 0인 경우에만 density > 0을 기대한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        information_density가 유효하면 True.
    """
    ctx_meta = result.get("context_metadata", {})
    density = ctx_meta.get("information_density")
    contributing = ctx_meta.get("contributing_turns", 0)

    # density 필드 자체가 존재하는지 확인
    if density is None:
        msg = "information_density 필드 없음"
        return warn_if_false(False, msg) if soft else False

    # 첫 턴(이전 key_claims 없음)이면 density=0.0도 정상
    if contributing == 0:
        return True

    # 2턴 이후: 이전 결론이 있으므로 density > 0 기대
    found = density > 0
    msg = f"information_density={density} (이전 결론 존재하나 밀도 0)"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_sufficiency_by_source_present(result: dict, *, soft: bool = True) -> bool:
    """G3: sufficiency_by_source가 존재하고 비어있지 않은지 검증한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        sufficiency_by_source가 비어있지 않으면 True.
    """
    ctx_eval = result.get("context_evaluation", {})
    sbs = ctx_eval.get("sufficiency_by_source", {})
    found = len(sbs) > 0
    msg = f"sufficiency_by_source 비어있음: {sbs}"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


# --- 검증(verify) 관련 ---

def assert_verification_verdict(result: dict, verdict: str, *, soft: bool = True) -> bool:
    """검증 판정(verdict)을 확인한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        verdict: 기대하는 overall_verdict 값 (pass, fail_numeric, fail_interpretation).
        soft: True면 경고만, False면 assert.

    Returns:
        verdict가 일치하면 True.
    """
    verification = result.get("verification", {})
    actual = verification.get("overall_verdict", "N/A")
    found = actual == verdict
    msg = f"verification verdict='{actual}', expected='{verdict}'"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


def assert_verify_retry_occurred(result: dict, *, soft: bool = True) -> bool:
    """검증 재시도가 발생했는지 확인한다.

    Args:
        result: 특정 턴의 graph.invoke() 결과.
        soft: True면 경고만, False면 assert.

    Returns:
        verify_retry_count >= 1이면 True.
    """
    retry_count = result.get("verify_retry_count", 0)
    found = retry_count >= 1
    msg = f"verify_retry_count={retry_count} (검증 재시도 미발생)"
    if soft:
        return warn_if_false(found, msg)
    assert found, msg
    return True


# ═══════════════════════════════════════════════════════════════
# ScenarioTestLog — 자기 설명적 구조화 테스트 로그
# ═══════════════════════════════════════════════════════════════

class ScenarioTestLog:
    """시나리오 테스트의 구조화된 로그를 생성한다.

    3단계 로그 구조:
        1. Header: 시나리오 배경, 질문, 검증 포인트 (run_session 전)
        2. Assertion Log: 각 검증의 pass/warn/fail 결과 (검증 직후)
        3. Summary: 턴별 요약 + 전체 pass/fail 집계 (모든 검증 후)

    Args:
        scenario_id: 시나리오 식별자 (예: "A1", "D2").
        name: 시나리오 이름 (예: "CSV-Only Query").
        description: 시나리오 목적 및 배경 설명 (한국어).
        queries: 테스트 질문 리스트.
        key_checks: 주요 검증 포인트 목록 (예: "[HARD] 결론 존재").
    """

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str,
        queries: list[str],
        key_checks: list[str],
    ):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.queries = queries
        self.key_checks = key_checks
        # 검증 결과 축적: {type: "hard"|"soft", name: str, passed: bool, detail: str}
        self.assertions: list[dict] = []

    def print_header(self):
        """시나리오 시작 시 배경 정보를 출력한다. run_session 전에 호출."""
        title = f"Scenario {self.scenario_id}: {self.name}"
        print()
        print(f"{'=' * 64}")
        print(f"  {title}")
        print(f"{'=' * 64}")
        print()
        print("Background:")
        for line in self.description.strip().splitlines():
            print(f"  {line}")
        print()
        print("Queries:")
        for i, q in enumerate(self.queries, 1):
            print(f"  [Turn {i}] {q}")
        print()
        print("Expected Checks:")
        for check in self.key_checks:
            print(f"  [ ] {check}")
        print()

    def check_hard(self, condition: bool, name: str, detail: str = "") -> bool:
        """Hard assertion: 기록 + assert. 실패 시 테스트 중단.

        Args:
            condition: 검증할 조건.
            name: 검증 항목 이름.
            detail: 부가 정보 (값, 상태 등).

        Returns:
            condition 값 (True만 반환 — 실패 시 assert로 중단).
        """
        self.assertions.append({
            "type": "hard",
            "name": name,
            "passed": condition,
            "detail": detail,
        })
        if condition:
            suffix = f" — {detail}" if detail else ""
            print(f"  \u2713 [HARD] {name}{suffix}")
        else:
            suffix = f" — {detail}" if detail else ""
            print(f"  \u2717 [HARD] {name}{suffix}")
            assert condition, f"[HARD FAIL] {name}: {detail}"
        return condition

    def check_soft(self, condition: bool, name: str, detail: str = "") -> bool:
        """Soft assertion: 기록 + 경고만. 실패해도 테스트 계속.

        Args:
            condition: 검증할 조건.
            name: 검증 항목 이름.
            detail: 부가 정보 (값, 상태 등).

        Returns:
            condition 값을 그대로 반환.
        """
        self.assertions.append({
            "type": "soft",
            "name": name,
            "passed": condition,
            "detail": detail,
        })
        if condition:
            suffix = f" — {detail}" if detail else ""
            print(f"  \u2713 [SOFT] {name}{suffix}")
        else:
            suffix = f" — {detail}" if detail else ""
            print(f"  \u26a0 [SOFT] {name} — WARN: {detail}")
        return condition

    def print_turn_details(self, session_state: dict):
        """턴별 핵심 정보를 요약 출력한다.

        Args:
            session_state: run_session()의 반환값.
        """
        turn_results = session_state.get("turn_results", [])
        conclusions = session_state.get("turn_conclusions", [])

        print()
        print("Turn Results:")
        for i, r in enumerate(turn_results):
            turn_num = i + 1
            query = self.queries[i] if i < len(self.queries) else "N/A"

            # 응답 텍스트 (200자 요약)
            response = r.get("response", "")
            response_preview = response[:200] + "..." if len(response) > 200 else response

            # 도구, 소스
            tools = r.get("tools_called", [])
            qa = r.get("query_analysis", {})
            source_types = qa.get("source_types", [])

            # 충분성
            ctx_eval = r.get("context_evaluation", {})
            is_sufficient = ctx_eval.get("is_sufficient", "N/A")
            confidence = ctx_eval.get("confidence_score", "N/A")

            # verdict
            verification = r.get("verification", {})
            verdict = verification.get("overall_verdict", "N/A")

            print(f"  Turn {turn_num}:")
            # G1/G3/Post-3 관측 지표
            ctx_meta = r.get("context_metadata", {})
            fidelity = ctx_meta.get("fidelity_score", "N/A")
            new_data = ctx_meta.get("new_data_ratio", "N/A")
            inherited = ctx_meta.get("inherited_ratio", "N/A")
            token_delta = ctx_meta.get("token_delta", "N/A")
            density = ctx_meta.get("information_density", "N/A")
            contributing = ctx_meta.get("contributing_turns", "N/A")

            # G5 탈락 이유
            exclusion_reasons = r.get("exclusion_reasons", [])

            # G3 소스별 충분성
            sbs = r.get("context_evaluation", {}).get("sufficiency_by_source", {})

            print(f"    Query:      {query}")
            print(f"    Response:   {response_preview}")
            print(f"    Tools:      {tools}")
            print(f"    Sources:    {source_types}")
            print(f"    Sufficient: {is_sufficient} (confidence={confidence})")
            print(f"    Verdict:    {verdict}")
            print(f"    --- G1~G3 관측 ---")
            print(f"    Fidelity:   {fidelity}")
            print(f"    NewData:    {new_data}  Inherited: {inherited}  Delta: {token_delta}")
            print(f"    Density:    {density}  Contributing: {contributing}")
            if exclusion_reasons:
                print(f"    Excluded:   {exclusion_reasons}")
            if sbs:
                print(f"    SuffBySrc:  {sbs}")

    def print_summary(self):
        """최종 pass/fail 집계 + 전체 assertion 목록을 출력한다."""
        pass_count = sum(1 for a in self.assertions if a["passed"])
        warn_count = sum(
            1 for a in self.assertions if not a["passed"] and a["type"] == "soft"
        )
        fail_count = sum(
            1 for a in self.assertions if not a["passed"] and a["type"] == "hard"
        )

        title = f"Test Summary — {self.scenario_id}: {self.name}"
        print()
        print(f"{'=' * 64}")
        print(f"  {title}")
        print(f"{'=' * 64}")
        print()
        print(f"Assertion Results: {pass_count} PASS / {warn_count} WARN / {fail_count} FAIL")
        for a in self.assertions:
            if a["passed"]:
                print(f"  \u2713 {a['name']}")
            elif a["type"] == "soft":
                print(f"  \u26a0 {a['name']} (WARN)")
            else:
                print(f"  \u2717 {a['name']} (FAIL)")

        # 전체 판정: hard fail이 하나라도 있으면 FAIL
        if fail_count > 0:
            overall = "FAIL"
        elif warn_count > 0:
            overall = "PASS (with warnings)"
        else:
            overall = "PASS"
        print(f"\nOverall: {overall}")
        print()
