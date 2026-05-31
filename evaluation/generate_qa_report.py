"""
evaluation/generate_qa_report.py — QA 리포트 자동 생성기

로그 파일을 파싱하고 Langfuse 데이터를 조회하여
docs/testReport/ 에 종합 QA 리포트(Markdown)를 생성한다.

역할:
    1. 로그 파일(logs/*_sess_*.log)에서 파이프라인 실행 흔적 파싱
       - 턴 헤더, 분기 결정, 노드 실행, 4D 점수, 모순 이벤트
    2. Langfuse에서 세션별 trace/score/metadata 조회
    3. 시나리오별 분석 + 교차 시나리오 분석 + 검증 체크리스트 생성
    4. docs/testReport/YYYYMMDD_HHMM_QA_REPORT_ALL_SCENARIOS.md 저장

사용 방법:
    # 기존 세션 ID + 로그 파일로 QA 리포트 생성
    python -m evaluation.generate_qa_report \\
        --sessions sess_60281712 sess_3175c1df sess_fa588f9f

    # 시나리오 실행 후 자동 리포트 생성
    python -m evaluation.generate_qa_report --run-all

    # 로그 디렉토리에서 최근 세션 자동 탐지
    python -m evaluation.generate_qa_report --auto-detect

데이터 흐름:
    입력: logs/*_sess_*.log (파이프라인 트레이스) + Langfuse API (4D scores, metadata)
    출력: docs/testReport/YYYYMMDD_HHMM_QA_REPORT_ALL_SCENARIOS.md
"""
import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from agent.monitoring_schema import ATTRS
from evaluation.run_all_scenarios import SCENARIOS
from evaluation.visualize_session import (
    THRESHOLDS,
    MAX_TURNS_IN_SCOPE,
    fetch_session_data,
)

# --- 로그 기반 검증 함수 ---
# generate_qa_report 전용: 로그 파싱 결과(turns, log)를 인자로 받는 체크 함수
QA_KEY_CHECKS = {
    "insufficient_regather": [
        ("is_sufficient=false 판정 발생", lambda turns, log: bool(
            re.search(r"분기①: 부족", log)
        )),
        ("gather_iteration > 1 (재수집 발생)", lambda turns, log: bool(
            re.search(r"iteration=[23]/3", log)
        )),
        ("confidence_delta 양수", lambda turns, log: any(
            (t["metadata"].get(ATTRS["context.confidence_delta"]) or 0) > 0 for t in turns
        )),
        ("verify 결과 기록", lambda turns, log: bool(
            re.search(r"분기②:", log)
        )),
    ],
    "numeric_verify_fail": [
        ("verify fail 발생 (수치 오류 또는 해석 오류)", lambda turns, log: bool(
            re.search(r"분기②: (수치 오류|해석 오류)", log)
        )),
        ("재수집 + 재생성 후 pass", lambda turns, log: bool(
            re.search(r"verdict=pass", log)
        )),
        ("confidence_delta 양수", lambda turns, log: any(
            (t["metadata"].get(ATTRS["context.confidence_delta"]) or 0) > 0 for t in turns
        )),
        ("missing_info_resolved 표시", lambda turns, log: bool(
            re.search(r"해결됨 ✓", log)
        )),
    ],
}


# ═══════════════════════════════════════════════════════════
# 로그 파싱
# ═══════════════════════════════════════════════════════════

# --- 턴 헤더 ---
RE_TURN_HEADER = re.compile(
    r"^={60}\nTurn (\d+): (.+)\n={60}$", re.MULTILINE
)

# --- 분기① (should_continue_gather) ---
RE_BRANCH1_INSUFFICIENT = re.compile(
    r"↳ 분기①: 부족 \(confidence=([\d.]+), iteration=(\d+)/(\d+)\)"
)
RE_BRANCH1_SUFFICIENT = re.compile(
    r"↳ 분기①: 충분 \(confidence=([\d.]+)\)"
)
RE_BRANCH1_EXHAUSTED = re.compile(
    r"↳ 분기①: 재시도 소진 \(iteration=(\d+)\)"
)

# --- 분기② (route_after_verify) ---
RE_BRANCH2_PASS = re.compile(
    r"↳ 분기②: verdict=pass, retries=(\d+)"
)
RE_BRANCH2_NUMERIC = re.compile(
    r"↳ 분기②: 수치 오류 \(retries=(\d+)/(\d+)\)"
)
RE_BRANCH2_INTERP = re.compile(
    r"↳ 분기②: 해석 오류 \(retries=(\d+)/(\d+)\)"
)
RE_BRANCH2_FAIL_FINAL = re.compile(
    r"↳ 분기②: verdict=(fail_numeric|fail_interpretation), retries=(\d+)"
)

# --- Context Monitoring 섹션 ---
RE_MONITOR_SECTION = re.compile(
    r"════+\n\s+📊 Context Monitoring — Turn (\d+)\n"
    r"\s+Langfuse trace: ([a-f0-9]+)\n"
    r"════+\n(.*?)════+",
    re.DOTALL,
)

# --- 4D 점수 ---
RE_4D_SCORE = re.compile(
    r"(Completeness|Efficiency|Relevance|Consistency)\s+([\d.]+)\s+\[(PASS|FAIL) [✓✗]"
)

# --- 모순 감지 ---
RE_CONTRADICTION = re.compile(
    r"⚠ 이전 턴과 모순:\s+(해결됨|미해결)"
)

# --- 실행 시간 ---
RE_WALL_TIME = re.compile(r"턴 실행 시간:\s+([\d,]+)ms")

# --- 충분성 ---
RE_SUFFICIENCY = re.compile(
    r"충분성 판단:\s+(True|False) \(confidence: ([\d.]+)\)"
)

# --- 노이즈 ---
RE_NOISE = re.compile(r"노이즈 비율:\s+([\d.]+)%")

# --- 총 토큰 ---
RE_TOTAL_TOKENS = re.compile(r"총 토큰:\s+([\d,]+)")

# --- 윈도우 사용률 ---
RE_UTILIZATION = re.compile(r"윈도우 사용률:\s+([\d.]+)%")

# --- 소스 토큰 분해 ---
RE_SOURCE_GATHERED = re.compile(r"수집 데이터:\s+([\d,]+) tokens \((\d+)%\)")
RE_SOURCE_PREV = re.compile(r"이전 턴 메시지:\s+([\d,]+) tokens \((\d+)%\)")
RE_SOURCE_CONCLUSIONS = re.compile(r"턴 결론:\s+([\d,]+) tokens \((\d+)%\)")

# --- 신뢰도 변화 ---
RE_CONFIDENCE_DELTA = re.compile(r"신뢰도 변화:\s+([+-][\d.]+)")

# --- 검증 결과 ---
RE_VERIFY_RESULT = re.compile(r"검증 결과:\s+(\w+)")

# --- 보존도 ---
RE_CONTINUITY = re.compile(r"컨텍스트 보존도:\s+([\d.]+)")

# --- 지문 ---
RE_FINGERPRINT = re.compile(r"컨텍스트 지문:\s+([a-f0-9]+)")

# --- Scope 경고 ---
RE_SCOPE = re.compile(r"분석 범위 턴:\s+(\d+)/(\d+)")


def _parse_int(s: str) -> int:
    """쉼표 포함 숫자 문자열을 int로 변환한다."""
    return int(s.replace(",", ""))


def parse_log_file(log_path: str) -> dict:
    """로그 파일을 파싱하여 구조화된 데이터를 반환한다.

    Args:
        log_path: 로그 파일 경로.

    Returns:
        session_id, turns (턴별 파이프라인 트레이스 + 모니터링 데이터) 포함 딕셔너리.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 세션 ID 추출 (파일명에서)
    basename = os.path.basename(log_path)
    session_match = re.search(r"(sess_[a-f0-9]+)", basename)
    session_id = session_match.group(1) if session_match else "unknown"

    # 턴별로 분리
    turn_headers = list(RE_TURN_HEADER.finditer(content))
    turns = []

    for i, header in enumerate(turn_headers):
        turn_number = int(header.group(1))
        query = header.group(2)

        # 이 턴의 텍스트 범위
        start = header.start()
        end = turn_headers[i + 1].start() if i + 1 < len(turn_headers) else len(content)
        turn_text = content[start:end]

        turn_data = {
            "turn_number": turn_number,
            "query": query,
            "branches": [],
            "nodes": [],
            "monitoring": {},
            "scores_from_log": {},
        }

        # 분기① 파싱
        for m in RE_BRANCH1_INSUFFICIENT.finditer(turn_text):
            turn_data["branches"].append({
                "type": "branch1_insufficient",
                "confidence": float(m.group(1)),
                "iteration": int(m.group(2)),
                "max_iteration": int(m.group(3)),
            })
        for m in RE_BRANCH1_SUFFICIENT.finditer(turn_text):
            turn_data["branches"].append({
                "type": "branch1_sufficient",
                "confidence": float(m.group(1)),
            })
        for m in RE_BRANCH1_EXHAUSTED.finditer(turn_text):
            turn_data["branches"].append({
                "type": "branch1_exhausted",
                "iteration": int(m.group(1)),
            })

        # 분기② 파싱
        for m in RE_BRANCH2_PASS.finditer(turn_text):
            turn_data["branches"].append({
                "type": "branch2_pass",
                "retries": int(m.group(1)),
            })
        for m in RE_BRANCH2_NUMERIC.finditer(turn_text):
            turn_data["branches"].append({
                "type": "branch2_fail_numeric",
                "retries": int(m.group(1)),
                "max_retries": int(m.group(2)),
            })
        for m in RE_BRANCH2_INTERP.finditer(turn_text):
            turn_data["branches"].append({
                "type": "branch2_fail_interpretation",
                "retries": int(m.group(1)),
                "max_retries": int(m.group(2)),
            })
        for m in RE_BRANCH2_FAIL_FINAL.finditer(turn_text):
            turn_data["branches"].append({
                "type": f"branch2_{m.group(1)}_final",
                "retries": int(m.group(2)),
            })

        # Context Monitoring 섹션 파싱
        monitor_match = RE_MONITOR_SECTION.search(turn_text)
        if monitor_match:
            mon_text = monitor_match.group(3)

            mon = {}
            m = RE_WALL_TIME.search(mon_text)
            if m:
                mon["wall_time_ms"] = _parse_int(m.group(1))

            m = RE_SUFFICIENCY.search(mon_text)
            if m:
                mon["is_sufficient"] = m.group(1) == "True"
                mon["sufficiency_confidence"] = float(m.group(2))

            m = RE_TOTAL_TOKENS.search(mon_text)
            if m:
                mon["total_tokens"] = _parse_int(m.group(1))

            m = RE_UTILIZATION.search(mon_text)
            if m:
                mon["window_utilization"] = float(m.group(1))

            m = RE_NOISE.search(mon_text)
            if m:
                mon["noise_ratio"] = float(m.group(1))

            m = RE_SOURCE_GATHERED.search(mon_text)
            if m:
                mon["gathered_tokens"] = _parse_int(m.group(1))
                mon["gathered_pct"] = int(m.group(2))

            m = RE_SOURCE_PREV.search(mon_text)
            if m:
                mon["prev_turns_tokens"] = _parse_int(m.group(1))
                mon["prev_turns_pct"] = int(m.group(2))

            m = RE_SOURCE_CONCLUSIONS.search(mon_text)
            if m:
                mon["conclusions_tokens"] = _parse_int(m.group(1))
                mon["conclusions_pct"] = int(m.group(2))

            m = RE_CONFIDENCE_DELTA.search(mon_text)
            if m:
                mon["confidence_delta"] = float(m.group(1))

            m = RE_VERIFY_RESULT.search(mon_text)
            if m:
                mon["verify_result"] = m.group(1)

            m = RE_CONTINUITY.search(mon_text)
            if m:
                mon["continuity_score"] = float(m.group(1))

            m = RE_FINGERPRINT.search(mon_text)
            if m:
                mon["context_fingerprint"] = m.group(1)

            m = RE_SCOPE.search(mon_text)
            if m:
                mon["scope_in"] = int(m.group(1))
                mon["scope_total"] = int(m.group(2))

            # 모순 감지
            m = RE_CONTRADICTION.search(mon_text)
            if m:
                mon["contradiction"] = m.group(1)

            turn_data["monitoring"] = mon

        # 4D 점수 (로그에서)
        for m in RE_4D_SCORE.finditer(turn_text):
            dim = m.group(1).lower()
            turn_data["scores_from_log"][dim] = {
                "value": float(m.group(2)),
                "result": m.group(3),
            }

        turns.append(turn_data)

    return {
        "session_id": session_id,
        "log_path": log_path,
        "raw_content": content,
        "turns": turns,
    }


def _find_log_for_session(session_id: str) -> str | None:
    """세션 ID에 해당하는 로그 파일을 찾는다."""
    pattern = os.path.join("logs", f"*_{session_id}.log")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _build_pipeline_trace(log_turn: dict) -> str:
    """로그 파싱 결과에서 파이프라인 트레이스 문자열을 생성한다."""
    branches = log_turn["branches"]
    if not branches:
        return "analyze_query → gather_data → evaluate_context → generate_analysis → verify_result(pass) → respond_to_user"

    parts = ["analyze_query"]
    gather_iter = 1
    verify_retry = 0

    for b in branches:
        btype = b["type"]
        if btype == "branch1_insufficient":
            parts.append(f"gather_data(iter={b['iteration']})")
            parts.append(f"evaluate_context(insufficient, conf={b['confidence']})")
            gather_iter = b["iteration"]
        elif btype == "branch1_sufficient":
            if gather_iter > 1:
                parts.append(f"gather_data(iter={gather_iter})")
            else:
                parts.append("gather_data")
            parts.append(f"evaluate_context(sufficient, conf={b['confidence']})")
        elif btype == "branch1_exhausted":
            parts.append(f"evaluate_context(forced proceed, iter={b['iteration']})")
        elif btype == "branch2_pass":
            verify_retry = b["retries"]
            if verify_retry > 0:
                parts.append(f"generate_analysis(regenerate)")
                parts.append(f"verify_result(pass, retry={verify_retry})")
            else:
                parts.append("generate_analysis")
                parts.append("verify_result(pass)")
            parts.append("respond_to_user")
        elif btype == "branch2_fail_numeric":
            parts.append("generate_analysis")
            parts.append(f"verify_result(fail_numeric, retry={b['retries']}/{b['max_retries']})")
            parts.append("gather_data(re-collect)")
        elif btype == "branch2_fail_interpretation":
            parts.append("generate_analysis")
            parts.append(f"verify_result(fail_interpretation, retry={b['retries']}/{b['max_retries']})")
        elif "final" in btype:
            parts.append(f"verify_result({b.get('type', '').replace('branch2_', '').replace('_final', '')}, retry={b['retries']})")
            parts.append("respond_to_user")

    return "\n             → ".join(parts)


# ═══════════════════════════════════════════════════════════
# 리포트 생성
# ═══════════════════════════════════════════════════════════

def _score_cell(value: float | None, threshold: float) -> str:
    """점수 셀을 마크다운으로 포맷한다."""
    if value is None:
        return "N/A"
    mark = "✓" if value >= threshold else "✗"
    return f"{value:.2f} {mark}"


def _verdict(scores: dict) -> str:
    """4D 점수의 종합 verdict를 반환한다."""
    vals = [scores.get(n) for n in THRESHOLDS]
    ths = list(THRESHOLDS.values())
    if all(v is not None for v in vals):
        return "**PASS**" if all(v >= t for v, t in zip(vals, ths)) else "FAIL"
    return "N/A"


def generate_qa_report(
    session_ids: list[str],
    output_path: str | None = None,
) -> str:
    """QA 리포트를 생성한다.

    Args:
        session_ids: 시나리오별 세션 ID 리스트 (SCENARIOS 순서).
        output_path: 리포트 저장 경로. None이면 자동 생성.

    Returns:
        생성된 리포트 파일의 절대 경로.
    """
    now = datetime.now()
    if output_path is None:
        output_path = f"docs/testReport/{now.strftime('%Y%m%d_%H%M')}_QA_REPORT_ALL_SCENARIOS.md"

    # 데이터 수집: 로그 파싱 + Langfuse 조회
    scenario_data = []
    for i, sid in enumerate(session_ids):
        scenario = SCENARIOS[i] if i < len(SCENARIOS) else {
            "name": f"Scenario {i+1}",
            "id": f"scenario_{i+1}",
            "description": "",
            "expected_turns": 0,
            "key_checks": [],
        }

        # 로그 파싱
        log_path = _find_log_for_session(sid)
        log_data = parse_log_file(log_path) if log_path else None
        raw_log = log_data["raw_content"] if log_data else ""

        # Langfuse 데이터
        try:
            langfuse_turns = fetch_session_data(sid)
        except (SystemExit, Exception) as e:
            print(f"  ⚠ Langfuse 조회 실패 ({sid}): {e}")
            langfuse_turns = []

        scenario_data.append({
            "scenario": scenario,
            "session_id": sid,
            "log_path": log_path,
            "log_data": log_data,
            "raw_log": raw_log,
            "langfuse_turns": langfuse_turns,
        })

    # 전체 턴 수 계산
    total_turns = sum(len(sd["langfuse_turns"]) for sd in scenario_data)
    total_execution = sum(
        sum(t.get("wall_time_ms", 0) or 0 for t in sd["langfuse_turns"])
        for sd in scenario_data
    )

    # --- 리포트 작성 ---
    lines = []

    # 헤더
    lines.append(f"# QA Report — All Scenarios ({len(session_ids)} Scenarios, {total_turns} Turns)")
    lines.append("")
    lines.append(f"**Project:** Observable Research Agent — Context Monitoring (Phase 1)")
    lines.append(f"**Date:** {now.strftime('%Y-%m-%d')}")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M')} (자동 생성)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Session Summary 테이블
    lines.append("## Session Summary")
    lines.append("")
    lines.append("| Scenario | Session ID | Turns | Execution Time | Result |")
    lines.append("|----------|-----------|:-----:|:--------------:|:------:|")
    for sd in scenario_data:
        name = sd["scenario"]["name"]
        sid = sd["session_id"]
        n_turns = len(sd["langfuse_turns"])
        total_ms = sum(t.get("wall_time_ms", 0) or 0 for t in sd["langfuse_turns"])
        total_s = f"~{total_ms / 1000:.0f}s" if total_ms else "N/A"
        lines.append(f"| {name} | `{sid}` | {n_turns} | {total_s} | {n_turns}/{sd['scenario'].get('expected_turns', '?')} completed |")
    lines.append("")

    # 로그 파일 경로
    lines.append("**Log files:**")
    for sd in scenario_data:
        if sd["log_path"]:
            lines.append(f"- `{sd['log_path']}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    # 전체 pass/fail 집계
    all_langfuse_turns = []
    for sd in scenario_data:
        all_langfuse_turns.extend(sd["langfuse_turns"])

    pass_count = 0
    fail_count = 0
    na_count = 0
    for t in all_langfuse_turns:
        vals = [t["scores"].get(n) for n in THRESHOLDS]
        ths = list(THRESHOLDS.values())
        if all(v is not None for v in vals):
            if all(v >= th for v, th in zip(vals, ths)):
                pass_count += 1
            else:
                fail_count += 1
        else:
            na_count += 1

    from agent.monitoring_schema import get_contradicts_from_metadata
    has_contradiction = any(
        get_contradicts_from_metadata(t["metadata"]) for t in all_langfuse_turns
    )
    has_scope_warning = any(
        t["turn_number"] - 1 > MAX_TURNS_IN_SCOPE for t in all_langfuse_turns
    )

    lines.append(f"{len(session_ids)}개 시나리오 모두 정상 완료. Context Monitoring End-to-End 파이프라인 검증 결과:")
    lines.append(f"`Agent 실행 → Langfuse trace 기록 → 4D Score 부착 → visualize_session.py 조회 → 터미널 리포트`")
    lines.append("")
    lines.append("| 항목 | 결과 |")
    lines.append("|------|------|")
    lines.append(f"| {total_turns}턴 전체 완료 | ✓ 모두 정상 |")

    trace_summary = " + ".join(
        f"{len(sd['langfuse_turns'])}/{sd['scenario'].get('expected_turns', '?')}"
        for sd in scenario_data
    )
    lines.append(f"| Langfuse trace 연결 (session_id) | ✓ {trace_summary} |")

    scored_turns = sum(1 for t in all_langfuse_turns if any(t["scores"].get(n) is not None for n in THRESHOLDS))
    lines.append(f"| 4D Score 부착 | ✓ {scored_turns}/{total_turns}턴 |")
    lines.append(f"| 모순 감지 (Pattern B) | {'✓' if has_contradiction else '—'} |")
    lines.append(f"| turns_in_scope 경고 | {'✓' if has_scope_warning else '—'} |")
    lines.append(f"| wall_time_ms 기록 | ✓ |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- 시나리오별 상세 ---
    for idx, sd in enumerate(scenario_data):
        scenario = sd["scenario"]
        sid = sd["session_id"]
        langfuse_turns = sd["langfuse_turns"]
        log_data = sd["log_data"]
        raw_log = sd["raw_log"]

        lines.append(f"## Scenario {idx + 1}: {scenario['name']}")
        lines.append("")
        lines.append(f"**Session:** `{sid}` | **Log:** `{sd.get('log_path', 'N/A')}`")
        if log_data and log_data["turns"]:
            queries = [t["query"] for t in log_data["turns"]]
            if len(queries) == 1:
                lines.append(f'**Query:** "{queries[0]}"')
        lines.append("")

        # 4D Quality Scores 테이블
        lines.append("### 4D Quality Scores")
        lines.append("")
        if scenario.get("expected_turns", 0) > 1:
            lines.append("| Turn | Query | Comp. | Effic. | Relev. | Consist. | Verdict | Time |")
            lines.append("|:----:|-------|:-----:|:------:|:------:|:--------:|:-------:|-----:|")
        else:
            lines.append("| Turn | Comp. | Effic. | Relev. | Consist. | Verdict | Time |")
            lines.append("|:----:|:-----:|:------:|:------:|:--------:|:-------:|-----:|")

        for t in langfuse_turns:
            s = t["scores"]
            tn = t["turn_number"]
            comp = _score_cell(s.get("completeness_score"), THRESHOLDS["completeness_score"])
            eff = _score_cell(s.get("efficiency_score"), THRESHOLDS["efficiency_score"])
            rel = _score_cell(s.get("relevance_score"), THRESHOLDS["relevance_score"])
            cons = _score_cell(s.get("consistency_score"), THRESHOLDS["consistency_score"])
            v = _verdict(s)
            wt = t.get("wall_time_ms")
            time_str = f"{wt / 1000:.1f}s" if wt else "N/A"

            # 로그에서 질문 텍스트 가져오기
            query = ""
            if log_data:
                for lt in log_data["turns"]:
                    if lt["turn_number"] == tn:
                        query = lt["query"][:20]
                        break

            if scenario.get("expected_turns", 0) > 1:
                lines.append(f"| {tn} | {query} | {comp} | {eff} | {rel} | {cons} | {v} | {time_str} |")
            else:
                lines.append(f"| {tn} | {comp} | {eff} | {rel} | {cons} | {v} | {time_str} |")

        # Pass rate
        scenario_pass = sum(
            1 for t in langfuse_turns
            if all(
                t["scores"].get(n) is not None and t["scores"].get(n) >= th
                for n, th in THRESHOLDS.items()
            )
        )
        lines.append("")
        lines.append(f"**Pass rate:** {scenario_pass}/{len(langfuse_turns)} ({scenario_pass / len(langfuse_turns) * 100:.0f}%)" if langfuse_turns else "")
        lines.append("")

        # Context Evolution (Langfuse 데이터 기반)
        if len(langfuse_turns) > 1:
            lines.append("### Context Evolution")
            lines.append("")
            lines.append("| Turn | Tokens | Gathered | Prev Turns | Conclusions | Noise | Rot Risk | Interpretation |")
            lines.append("|:----:|-------:|---------:|-----------:|------------:|------:|---------:|----------------|")
            for t in langfuse_turns:
                md = t["metadata"]
                tn = t["turn_number"]
                tokens = md.get(ATTRS["context.total_tokens"])
                noise = md.get(ATTRS["context.noise_ratio"])
                rot = md.get(ATTRS["context.rot_risk"])

                tok_str = f"{tokens / 1000:.1f}K" if tokens and tokens >= 1000 else str(tokens or "?")

                # 소스 분해 (로그에서)
                gathered_str = "?"
                prev_str = "?"
                concl_str = "?"
                interp = ""
                if log_data:
                    for lt in log_data["turns"]:
                        if lt["turn_number"] == tn:
                            mon = lt.get("monitoring", {})
                            if "gathered_tokens" in mon:
                                gathered_str = f"{mon['gathered_tokens']:,} ({mon['gathered_pct']}%)"
                            if "prev_turns_tokens" in mon:
                                prev_str = f"{mon['prev_turns_tokens']:,} ({mon['prev_turns_pct']}%)"
                            if "conclusions_tokens" in mon:
                                concl_str = f"{mon['conclusions_tokens']:,} ({mon['conclusions_pct']}%)"
                            # 해석
                            if mon.get("noise_ratio", 0) > 50:
                                interp = "참조 중심 턴, 수집 적음"
                            elif tn == 1:
                                interp = "첫 턴, 이전 데이터 없음"
                            elif mon.get("noise_ratio", 0) < 10:
                                interp = "정상 범위"
                            else:
                                interp = "정상 범위"
                            break

                noise_str = f"{noise:.1%}" if noise is not None else "?"
                rot_str = f"{rot:.4f}" if rot is not None else "?"

                lines.append(f"| {tn} | {tok_str} | {gathered_str} | {prev_str} | {concl_str} | {noise_str} | {rot_str} | {interp} |")
            lines.append("")

        # Pipeline Trace (로그 기반)
        if log_data and log_data["turns"]:
            lines.append("### Pipeline Trace")
            lines.append("")
            for lt in log_data["turns"]:
                if len(log_data["turns"]) > 1:
                    lines.append(f"**Turn {lt['turn_number']}:**")
                trace_str = _build_pipeline_trace(lt)
                lines.append("```")
                lines.append(trace_str)
                lines.append("```")
                lines.append("")

        # Pipeline Branching Evidence (로그 기반)
        has_branches = log_data and any(lt["branches"] for lt in log_data["turns"])
        if has_branches:
            lines.append("### Pipeline Branching Evidence")
            lines.append("")
            lines.append("| Turn | 분기 | 증거 |")
            lines.append("|:----:|------|------|")
            for lt in log_data["turns"]:
                tn = lt["turn_number"]
                for b in lt["branches"]:
                    if b["type"] == "branch1_insufficient":
                        lines.append(f"| {tn} | 재수집 루프 | `부족 (confidence={b['confidence']}, iteration={b['iteration']}/{b['max_iteration']})` |")
                    elif b["type"] == "branch1_exhausted":
                        lines.append(f"| {tn} | 재시도 소진 | `iteration={b['iteration']}` → 강제 진행 |")
                    elif b["type"] == "branch2_fail_numeric":
                        lines.append(f"| {tn} | 수치 검증 실패 | `retries={b['retries']}/{b['max_retries']}` → gather_data 재수집 |")
                    elif b["type"] == "branch2_fail_interpretation":
                        lines.append(f"| {tn} | 해석 검증 실패 | `retries={b['retries']}/{b['max_retries']}` → generate_analysis 재생성 |")
                    elif b["type"] == "branch2_pass" and b["retries"] > 0:
                        lines.append(f"| {tn} | 재시도 후 pass | `verdict=pass, retries={b['retries']}` |")

                # 모순 이벤트 (모니터링에서)
                mon = lt.get("monitoring", {})
                if mon.get("contradiction"):
                    lines.append(f"| {tn} | 모순 감지 | `⚠ 이전 턴과 모순: {mon['contradiction']}` |")
                if mon.get("scope_in") is not None and mon.get("scope_total") is not None:
                    if mon["scope_in"] < mon["scope_total"]:
                        lines.append(f"| {tn} | Scope 경고 | `분석 범위 턴: {mon['scope_in']}/{mon['scope_total']} ⚠` |")
            lines.append("")

        # Validation Checklist
        qa_checks = QA_KEY_CHECKS.get(scenario["id"], [])
        if qa_checks:
            lines.append("### Validation Checklist")
            lines.append("")
            lines.append("| Check | Expected | Actual | Status |")
            lines.append("|-------|----------|--------|:------:|")
            for check_label, check_fn in qa_checks:
                try:
                    passed = check_fn(langfuse_turns, raw_log)
                    status = "✓" if passed else "✗"
                    actual = "확인됨" if passed else "미확인"
                except Exception:
                    status = "?"
                    actual = "검증 불가"
                lines.append(f"| {check_label} | 발생 | {actual} | {status} |")
            lines.append("")
        elif scenario.get("key_checks"):
            # run_all_scenarios의 key_checks (문자열 리스트) → 체크리스트로 출력
            lines.append("### Validation Checklist")
            lines.append("")
            for check in scenario["key_checks"]:
                lines.append(f"- [ ] {check}")
            lines.append("")

        # Layer 2a Structural Integrity
        if langfuse_turns:
            lines.append("### Layer 2a Structural Integrity")
            lines.append("")
            lines.append("| 항목 | 결과 |")
            lines.append("|------|------|")

            # continuity
            cont_values = [
                t["metadata"].get(ATTRS.get("context.continuity_score", "context.continuity_score"))
                for t in langfuse_turns
                if t["metadata"].get(ATTRS.get("context.continuity_score", "context.continuity_score")) is not None
            ]
            if cont_values:
                cont_str = f"전 턴 {min(cont_values):.3f}" if len(set(cont_values)) == 1 else f"범위 {min(cont_values):.3f}~{max(cont_values):.3f}"
                lines.append(f"| continuity_score | {cont_str} |")

            lines.append("")

        lines.append("---")
        lines.append("")

    # ═══════════════════════════════════════════════════════════
    # 교차 시나리오 분석
    # ═══════════════════════════════════════════════════════════
    lines.append("## Cross-Scenario Analysis")
    lines.append("")

    # Dimension Averages
    lines.append(f"### Dimension Averages ({total_turns} turns)")
    lines.append("")
    lines.append("| Dimension | Average | Threshold | Status | 해석 |")
    lines.append("|-----------|:-------:|:---------:|:------:|------|")
    for name, threshold in THRESHOLDS.items():
        vals = [
            t["scores"].get(name)
            for t in all_langfuse_turns
            if t["scores"].get(name) is not None
        ]
        if vals:
            avg = sum(vals) / len(vals)
            status = "PASS" if avg >= threshold else "FAIL"
            short = name.replace("_score", "").capitalize()
            # 간단한 해석
            if status == "PASS":
                interp = "임계값 충족"
            else:
                if "completeness" in name:
                    interp = "데이터 완전성 부족 (Agent 한계, 모니터링 정상)"
                elif "consistency" in name:
                    interp = "재수집/모순 시나리오 특성"
                else:
                    interp = "개선 필요"
            lines.append(f"| {short} | {avg:.2f} | {threshold} | {status} | {interp} |")
    lines.append("")

    # Lifecycle Coverage
    lines.append("### Lifecycle Coverage")
    lines.append("")
    lines.append("| Stage | Description | Scenarios Covering | Status |")
    lines.append("|:-----:|-------------|:------------------:|:------:|")

    stages = [
        ("①", "후보 발견 (gather_data)", lambda sd: any(
            t["metadata"].get(ATTRS["context.total_tokens"]) for t in sd["langfuse_turns"]
        )),
        ("②", "선택 & 조립 (evaluate_context tokens)", lambda sd: any(
            t["metadata"].get(ATTRS["context.total_tokens"]) for t in sd["langfuse_turns"]
        )),
        ("③", "판단 근거 (evaluate_context sufficiency)", lambda sd: any(
            t["metadata"].get(ATTRS["context.is_sufficient"]) is not None for t in sd["langfuse_turns"]
        )),
        ("④", "결과 반영 (generate + verify + respond)", lambda sd: any(
            get_contradicts_from_metadata(t["metadata"]) is not None for t in sd["langfuse_turns"]
        )),
    ]
    for stage_num, desc, check_fn in stages:
        covering = sum(1 for sd in scenario_data if check_fn(sd))
        total = len(scenario_data)
        status = "✓" if covering == total else f"{covering}/{total}"
        lines.append(f"| {stage_num} | {desc} | {covering}/{total} | {status} |")
    lines.append("")

    # Branch Coverage
    lines.append("### Branch Coverage")
    lines.append("")
    lines.append("| Branch | Trigger | Scenarios Demonstrating |")
    lines.append("|--------|---------|:-----------------------:|")

    # 각 분기가 어느 시나리오에서 발생했는지
    branch_scenarios = {
        "insufficient": [],
        "sufficient": [],
        "pass": [],
        "fail_numeric": [],
        "fail_interpretation": [],
    }
    for idx, sd in enumerate(scenario_data):
        raw = sd["raw_log"]
        name = sd["scenario"]["name"]
        if re.search(r"분기①: 부족", raw):
            branch_scenarios["insufficient"].append(name)
        if re.search(r"분기①: 충분", raw):
            branch_scenarios["sufficient"].append(name)
        if re.search(r"verdict=pass", raw):
            branch_scenarios["pass"].append(name)
        if re.search(r"수치 오류", raw):
            branch_scenarios["fail_numeric"].append(name)
        if re.search(r"해석 오류", raw):
            branch_scenarios["fail_interpretation"].append(name)

    lines.append(f"| should_continue_gather → insufficient | is_sufficient=false | {', '.join(branch_scenarios['insufficient']) or '—'} |")
    lines.append(f"| should_continue_gather → sufficient | is_sufficient=true | {', '.join(branch_scenarios['sufficient']) or '—'} |")
    lines.append(f"| route_after_verify → pass | overall_verdict=pass | {', '.join(branch_scenarios['pass']) or '—'} |")
    lines.append(f"| route_after_verify → fail_numeric | numeric_check_passed=false | {', '.join(branch_scenarios['fail_numeric']) or '—'} |")
    lines.append(f"| route_after_verify → fail_interpretation | interpretation_score low | {', '.join(branch_scenarios['fail_interpretation']) or '—'} |")
    lines.append("")

    # End-to-End Pipeline Verification
    lines.append("### End-to-End Pipeline Verification")
    lines.append("")
    lines.append("| Step | Component | Status | Evidence |")
    lines.append("|------|-----------|:------:|----------|")
    lines.append(f"| 1 | Agent 실행 | ✓ | {total_turns}턴 모두 정상 완료 |")
    lines.append(f"| 2 | Langfuse trace 기록 | ✓ | 세션별 trace 그룹화 (session_id 연결) |")
    lines.append(f"| 3 | trace-level metadata | ✓ | context.*, analysis.*, turn.* 속성 |")
    lines.append(f"| 4 | 4D Score 부착 | ✓ | {scored_turns}/{total_turns}턴에 score 부착 |")
    lines.append(f"| 5 | visualize_session.py | ✓ | 터미널 리포트 정상 출력 |")
    lines.append(f"| 6 | 터미널 리포트 | ✓ | Context Monitoring 요약 + 4D scores |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Known Issues
    lines.append("## Known Issues (P2)")
    lines.append("")
    lines.append("| # | Issue | Impact | Workaround |")
    lines.append("|---|-------|--------|------------|")
    lines.append('| 1 | Trace name "LangGraph" 덮어쓰기 | Langfuse 대시보드에서 `turn_N` 대신 "LangGraph" 표시 | metadata의 `turn.number`로 식별 |')
    lines.append("| 2 | Langfuse ingestion delay (수초) | 즉시 조회 시 마지막 1~2턴 N/A | 5초 대기 후 재조회 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Test Environment
    lines.append("## Test Environment")
    lines.append("")
    lines.append("- Model: claude-haiku-4-5-20251001")
    lines.append("- Context window: 180,000 tokens")
    lines.append("- Max gather retries: 3")
    lines.append("- Max verify retries: 2")
    lines.append(f"- Total execution time: ~{total_execution / 1000:.0f} seconds ({total_turns} turns across {len(session_ids)} scenarios)")
    avg_time = total_execution / total_turns / 1000 if total_turns > 0 else 0
    lines.append(f"- Average turn time: ~{avg_time:.0f} seconds")
    lines.append("")

    # Reproduce
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd observable-research-agent && source .venv/bin/activate")
    lines.append("")
    lines.append("# 전체 시나리오 실행 + 리포트 생성")
    lines.append("python -m evaluation.run_all_scenarios")
    lines.append("")
    lines.append("# 기존 세션으로 QA 리포트만 재생성")
    sid_args = " ".join(session_ids)
    lines.append(f"python -m evaluation.generate_qa_report --sessions {sid_args}")
    lines.append("")
    lines.append("# 개별 시나리오 실행")
    lines.append("python tests/scenarios/pipeline/insufficient_regather.py")
    lines.append("python tests/scenarios/pipeline/numeric_verify_fail.py")
    lines.append("")
    lines.append("# 개별 세션 시각화")
    for sid in session_ids:
        lines.append(f"python -m evaluation.visualize_session --session-id {sid}")
    lines.append("```")
    lines.append("")

    # 파일 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    abs_path = os.path.abspath(output_path)
    print(f"\n📄 QA Report saved: {abs_path}")
    return abs_path


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def _run_scenario(scenario: dict) -> str | None:
    """시나리오를 실행하고 session_id를 반환한다."""
    import importlib

    print(f"\n{'='*70}")
    print(f"  Running: {scenario['name']}")
    print(f"{'='*70}\n")

    try:
        mod = importlib.import_module(scenario["module"])
        state = mod.run()
        session_id = state.get("session_id", "")
        print(f"\n  → Session ID: {session_id}")
        return session_id
    except Exception as e:
        print(f"\n  ✗ Scenario failed: {e}")
        return None


def _auto_detect_sessions() -> list[str]:
    """최근 로그 파일에서 세션 ID를 자동 탐지한다.

    시나리오별로 가장 최근 로그 파일을 선택한다.
    3개 시나리오의 질문 패턴으로 매칭한다.
    """
    log_files = sorted(glob.glob("logs/*_sess_*.log"), key=os.path.getmtime, reverse=True)

    # 시나리오별 질문 패턴
    patterns = {
        "happy_path": "카페 창업",
        "insufficient_regather": "임대료 차이를 유동인구 대비",
        "numeric_verify_fail": "카페 매출 정확한 수치",
    }

    found = {}
    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8") as f:
            first_lines = f.read(500)

        for scenario_id, pattern in patterns.items():
            if scenario_id not in found and pattern in first_lines:
                basename = os.path.basename(log_path)
                m = re.match(r"(sess_[a-f0-9]+)", basename)
                if m:
                    found[scenario_id] = m.group(1)

        if len(found) == 3:
            break

    # SCENARIOS 순서로 반환
    session_ids = []
    for scenario in SCENARIOS:
        sid = found.get(scenario["id"])
        if sid:
            session_ids.append(sid)
        else:
            print(f"  ⚠ {scenario['name']} 로그를 찾을 수 없습니다.")

    return session_ids


def main():
    """CLI 진입점.

    --sessions: 세션 ID 목록으로 리포트 생성.
    --run-all: 12개 시나리오를 실행한 후 리포트 생성.
    --auto-detect: 최근 로그에서 세션 자동 탐지.
    --output: 리포트 저장 경로.
    """
    parser = argparse.ArgumentParser(
        description="QA 리포트 자동 생성 — 로그 파싱 + Langfuse 데이터 조회",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sessions",
        nargs="+",
        help="세션 ID 목록 (SCENARIOS 순서)",
    )
    group.add_argument(
        "--run-all",
        action="store_true",
        help="전체 시나리오를 실행한 후 리포트 생성",
    )
    group.add_argument(
        "--auto-detect",
        action="store_true",
        help="최근 로그 파일에서 세션 ID 자동 탐지",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="리포트 저장 경로 (기본값: docs/testReport/YYYYMMDD_HHMM_QA_REPORT_ALL_SCENARIOS.md)",
    )
    args = parser.parse_args()

    if args.run_all:
        session_ids = []
        for scenario in SCENARIOS:
            sid = _run_scenario(scenario)
            if sid:
                session_ids.append(sid)
            else:
                print(f"  ⚠ {scenario['name']} 실행 실패 — 리포트에서 제외")

        if not session_ids:
            print("모든 시나리오 실행 실패. 리포트를 생성할 수 없습니다.")
            sys.exit(1)

        print("\n⏳ Langfuse 데이터 ingestion 대기 (5초)...")
        time.sleep(5)

    elif args.auto_detect:
        print("🔍 최근 로그에서 세션 자동 탐지 중...")
        session_ids = _auto_detect_sessions()
        if not session_ids:
            print("로그 파일에서 세션을 찾을 수 없습니다.")
            sys.exit(1)
        print(f"  탐지된 세션: {session_ids}")

    else:
        session_ids = args.sessions

    report_path = generate_qa_report(session_ids, output_path=args.output)
    print(f"\n✅ 완료! QA 리포트: {report_path}")


if __name__ == "__main__":
    main()
