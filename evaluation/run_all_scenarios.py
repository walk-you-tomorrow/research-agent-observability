"""
evaluation/run_all_scenarios.py — 전체 시나리오 실행 + 분석 리포트 생성

12개 검증 시나리오를 순차 실행하고,
Langfuse에서 각 세션의 데이터를 조회하여 종합 분석 리포트(Markdown)를 생성한다.

역할:
    1. 12개 시나리오를 순차 실행 (2 Pipeline + 6 Multi-Source + 4 Multi-Turn)
    2. Langfuse에서 각 세션의 trace/score/metadata를 수집
    3. 시나리오별 분석 + 전체 종합 분석을 Markdown 리포트로 생성
    4. docs/testReport/YYYYMMDD_HHMM_TEST_REPORT.md에 저장

사용 방법:
    # 시나리오를 실행하고 리포트 생성
    python -m evaluation.run_all_scenarios

    # 이미 실행된 세션 ID를 지정하여 리포트만 생성
    python -m evaluation.run_all_scenarios --report-only \\
        --sessions sess_abc123 sess_def456

데이터 흐름:
    입력: 12개 시나리오 스크립트 (tests/scenarios/**/*.py)
    출력: docs/testReport/YYYYMMDD_HHMM_TEST_REPORT.md
"""
import argparse
import os
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
from evaluation.visualize_session import (
    THRESHOLDS,
    MAX_TURNS_IN_SCOPE,
    fetch_session_data,
)

# --- 시나리오 정의 ---
# 2 Pipeline + 6 Multi-Source + 4 Multi-Turn = 총 12개 시나리오
SCENARIOS = [
    # ── Category P: Pipeline (분기 경로 검증) ──
    {
        "name": "P1: Insufficient → Re-gather",
        "id": "insufficient_regather",
        "module": "tests.scenarios.pipeline.insufficient_regather",
        "description": "복잡한 교차 분석 질문으로 데이터 부족 → 재수집 루프를 유도.",
        "expected_turns": 1,
        "key_checks": [
            "is_sufficient=false 판정 발생",
            "gather_iteration > 1 (재수집 발생)",
            "missing_info_resolved=true",
            "confidence_delta 양수 (재수집 후 개선)",
        ],
    },
    {
        "name": "P2: Numeric Verify Fail → Re-generate",
        "id": "numeric_verify_fail",
        "module": "tests.scenarios.pipeline.numeric_verify_fail",
        "description": "수치 정확성 강조 질문으로 검증 실패 → 재생성을 유도.",
        "expected_turns": 1,
        "key_checks": [
            "verify.numeric_check_passed=false 기록",
            "verify.overall_verdict=fail_numeric 발생",
            "재수집 + 재생성 후 pass",
            "verify_retry_count 증가",
        ],
    },
    # ── Category A: Multi-Source Selection ──
    {
        "name": "A1: CSV-Only Query",
        "id": "csv_only",
        "module": "tests.scenarios.multi_source.test_a1_csv_only",
        "description": "CSV 전용 수치 질문. pandas_query만 호출되는지 검증.",
        "expected_turns": 1,
        "key_checks": [
            "pandas_query 도구가 호출됨",
            "정확한 수치가 응답에 포함",
        ],
    },
    {
        "name": "A2: RAG-Only Query",
        "id": "rag_only",
        "module": "tests.scenarios.multi_source.test_a2_rag_only",
        "description": "RAG 전용 트렌드 질문. LightRAG 시맨틱 검색 위주.",
        "expected_turns": 1,
        "key_checks": [
            "rag_search 또는 rag_deep_read 호출됨",
            "KG 기반 엔티티 정보가 응답에 포함",
        ],
    },
    {
        "name": "A3: API-Triggered Query",
        "id": "api_trigger",
        "module": "tests.scenarios.multi_source.test_a3_api_trigger",
        "description": "상권변화지표 질문으로 api_query 호출 유도.",
        "expected_turns": 1,
        "key_checks": [
            "api_query 도구가 호출됨",
            "gather.api_called에 API 키 기록",
            "API 데이터가 gathered_data에 포함",
        ],
    },
    {
        "name": "A4: Web Search Triggered",
        "id": "web_trigger",
        "module": "tests.scenarios.multi_source.test_a4_web_trigger",
        "description": "최신 트렌드 질문으로 web_search 호출 유도.",
        "expected_turns": 1,
        "key_checks": [
            "web_search 도구가 호출됨",
            "web.search_count > 0",
            "web.result_count > 0",
        ],
    },
    {
        "name": "A5: Multi-Source Fusion",
        "id": "multi_source_fusion",
        "module": "tests.scenarios.multi_source.test_a5_fusion",
        "description": "종합 분석 질문. 3개 이상 소스 동시 사용.",
        "expected_turns": 1,
        "key_checks": [
            "len(source.types_selected) >= 3 (3개+ 소스 유형)",
            "복수 소스 데이터가 응답에 혼합",
        ],
    },
    {
        "name": "A6: Source Conflict Detection",
        "id": "source_conflict",
        "module": "tests.scenarios.multi_source.test_a6_source_conflict",
        "description": "소스 간 데이터 차이 감지 및 해결.",
        "expected_turns": 1,
        "key_checks": [
            "복수 소스에서 데이터 수집",
            "source.conflict_detected 기록",
        ],
    },
    # ── Category D: Multi-Turn Consistency ──
    {
        "name": "D1: Progressive Refinement",
        "id": "progressive",
        "module": "tests.scenarios.multi_turn.test_d1_progressive",
        "description": "구 → 동 → 동+업종으로 점진적 세밀화. 3턴.",
        "expected_turns": 3,
        "key_checks": [
            "3개 턴 모두 정상 완료",
            "후속 턴에서 이전 턴 결론 참조",
        ],
    },
    {
        "name": "D2: Contradiction & Resolution",
        "id": "contradiction",
        "module": "tests.scenarios.multi_turn.test_d2_contradiction",
        "description": "Turn 1에서 추천 후 Turn 2에서 반론 제기. 모순 감지.",
        "expected_turns": 2,
        "key_checks": [
            "contradicts_previous=true 기록",
            "contradiction_resolved=true",
        ],
    },
    {
        "name": "D3: Turn Reference Accuracy",
        "id": "turn_reference",
        "module": "tests.scenarios.multi_turn.test_d3_reference",
        "description": "2개 구 분석 후 3번째 턴에서 종합. lookup_previous 정확성.",
        "expected_turns": 3,
        "key_checks": [
            "3개 턴 모두 정상 완료",
            "Turn 3에서 이전 2턴 참조",
        ],
    },
    {
        "name": "D4: Session Memory Persistence",
        "id": "memory",
        "module": "tests.scenarios.multi_turn.test_d4_memory",
        "description": "3턴 세션에서 turn_conclusions 누적 및 상태 전달 검증.",
        "expected_turns": 3,
        "key_checks": [
            "turn_conclusions 리스트 길이 == 3",
            "turn_number가 1~3으로 연속",
            "messages가 턴 간 누적",
        ],
    },
]


# --- 시나리오 실행 ---
def _run_scenario(scenario: dict) -> str | None:
    """시나리오를 실행하고 session_id를 반환한다.

    Args:
        scenario: SCENARIOS 리스트의 항목.

    Returns:
        세션 ID (예: "sess_abc123") 또는 실행 실패 시 None.
    """
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


# --- 리포트 생성 ---
def _score_summary(scores: dict) -> str:
    """4D 점수를 한 줄 요약으로 반환한다."""
    parts = []
    for name, threshold in THRESHOLDS.items():
        val = scores.get(name)
        if val is not None:
            mark = "PASS" if val >= threshold else "FAIL"
            short = name.replace("_score", "").capitalize()
            parts.append(f"{short}={val:.2f}({mark})")
        else:
            short = name.replace("_score", "").capitalize()
            parts.append(f"{short}=N/A")
    return " | ".join(parts)


def _turn_score_table(turns: list[dict]) -> str:
    """턴별 점수 테이블을 Markdown으로 반환한다."""
    lines = [
        "| Turn | Completeness | Efficiency | Relevance | Consistency | Verdict | Time(ms) |",
        "|-----:|:------------:|:----------:|:---------:|:-----------:|:-------:|:--------:|",
    ]
    for t in turns:
        s = t["scores"]
        row = [f"{t['turn_number']}"]
        for name, threshold in THRESHOLDS.items():
            val = s.get(name)
            if val is not None:
                mark = "✓" if val >= threshold else "✗"
                row.append(f"{val:.2f} {mark}")
            else:
                row.append("N/A")

        # Verdict
        all_scores = [s.get(n) for n in THRESHOLDS]
        all_thresholds = list(THRESHOLDS.values())
        if all(v is not None for v in all_scores):
            verdict = "PASS" if all(
                v >= th for v, th in zip(all_scores, all_thresholds)
            ) else "FAIL"
        else:
            verdict = "N/A"
        row.append(verdict)

        wt = t.get("wall_time_ms")
        row.append(f"{wt:,}" if wt else "N/A")

        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _context_evolution_table(turns: list[dict]) -> str:
    """컨텍스트 진화 추이를 Markdown 테이블로 반환한다."""
    lines = [
        "| Turn | Tokens | Noise | Continuity | Rot Risk | Turns in Scope |",
        "|-----:|-------:|------:|-----------:|---------:|:--------------:|",
    ]
    for t in turns:
        md = t["metadata"]
        tn = t["turn_number"]
        tokens = md.get(ATTRS["context.total_tokens"])
        noise = md.get(ATTRS["context.noise_ratio"])
        cont = md.get(ATTRS.get("context.continuity_score", "context.continuity_score"))
        rot = md.get(ATTRS["context.rot_risk"])

        prior = tn - 1
        if prior <= 0:
            scope = "0/0"
        else:
            in_scope = min(prior, MAX_TURNS_IN_SCOPE)
            scope = f"{in_scope}/{prior}"
            if in_scope < prior:
                scope += " ⚠"

        tok_str = f"{tokens / 1000:.1f}K" if tokens and tokens >= 1000 else str(tokens or "?")
        noise_str = f"{noise:.1%}" if noise is not None else "?"
        cont_str = f"{cont:.2f}" if cont is not None else "?"
        rot_str = f"{rot:.4f}" if rot is not None else "?"

        lines.append(f"| {tn} | {tok_str} | {noise_str} | {cont_str} | {rot_str} | {scope} |")
    return "\n".join(lines)


def _events_section(turns: list[dict]) -> str:
    """이벤트 섹션을 Markdown으로 반환한다."""
    events = []
    for t in turns:
        for ev in t.get("events", []):
            events.append(f"- **Turn {t['turn_number']}**: {ev}")
        tn = t["turn_number"]
        prior = tn - 1
        if prior > MAX_TURNS_IN_SCOPE:
            out = prior - MAX_TURNS_IN_SCOPE
            events.append(
                f"- **Turn {tn}**: 이전 턴 {prior}개 중 {MAX_TURNS_IN_SCOPE}개만 scope 내 "
                f"(turn 1~{out} out of scope)"
            )
    return "\n".join(events) if events else "_이벤트 없음_"


def _coverage_summary(turns: list[dict]) -> dict:
    """전체 커버리지 요약을 딕셔너리로 반환한다."""
    total = len(turns)
    has_scores = sum(1 for t in turns if any(t["scores"].get(n) is not None for n in THRESHOLDS))
    all_pass = sum(
        1 for t in turns
        if all(
            t["scores"].get(n) is not None and t["scores"].get(n) >= th
            for n, th in THRESHOLDS.items()
        )
    )
    from agent.monitoring_schema import get_contradicts_from_metadata
    has_contradiction = any(
        get_contradicts_from_metadata(t["metadata"]) for t in turns
    )
    has_scope_warning = any(
        t["turn_number"] - 1 > MAX_TURNS_IN_SCOPE for t in turns
    )

    # 컨텍스트 메트릭 커버리지
    ctx_attrs = [
        ATTRS["context.total_tokens"], ATTRS["context.noise_ratio"],
        ATTRS.get("context.continuity_score", "context.continuity_score"), ATTRS["context.rot_risk"],
        ATTRS["context.is_sufficient"], ATTRS["context.sufficiency_confidence"],
    ]
    attr_coverage = {}
    for attr in ctx_attrs:
        count = sum(1 for t in turns if t["metadata"].get(attr) is not None)
        attr_coverage[attr] = f"{count}/{total}"

    return {
        "total_turns": total,
        "turns_with_scores": has_scores,
        "turns_all_pass": all_pass,
        "has_contradiction": has_contradiction,
        "has_scope_warning": has_scope_warning,
        "attr_coverage": attr_coverage,
    }


def generate_report(
    session_ids: list[str],
    scenario_names: list[str] | None = None,
    output_path: str | None = None,
) -> str:
    """Langfuse에서 데이터를 수집하여 종합 분석 리포트를 생성한다.

    Args:
        session_ids: 시나리오별 세션 ID 리스트.
        scenario_names: 시나리오 이름 리스트 (session_ids와 동일 순서). None이면 자동 부여.
        output_path: 리포트 저장 경로. None이면 docs/testReport/YYYYMMDD_HHMM_TEST_REPORT.md.

    Returns:
        생성된 리포트 파일의 절대 경로.
    """
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")
    if output_path is None:
        output_path = f"docs/testReport/{now.strftime('%Y%m%d_%H%M')}_TEST_REPORT.md"
    report = []
    report.append(f"# Test Report — {now_str}")
    report.append(f"\n> **Generated**: {now_str}")
    report.append(f"> **Scenarios**: {len(session_ids)}개 실행")
    report.append("")

    all_turns_all_scenarios = []
    total_pass = 0
    total_fail = 0
    total_na = 0

    for i, sid in enumerate(session_ids):
        scenario = SCENARIOS[i] if i < len(SCENARIOS) else None
        name = scenario["name"] if scenario else (scenario_names[i] if scenario_names else f"Scenario {i+1}")
        desc = scenario["description"] if scenario else ""
        key_checks = scenario["key_checks"] if scenario else []

        report.append(f"---\n")
        report.append(f"## Scenario {i+1}: {name}")
        report.append(f"\n**Session ID**: `{sid}`")
        if desc:
            report.append(f"\n**Description**: {desc}")
        report.append("")

        # Langfuse에서 데이터 조회
        try:
            turns = fetch_session_data(sid)
        except SystemExit:
            report.append(f"⚠ Langfuse에서 세션 데이터를 조회할 수 없습니다.\n")
            continue

        all_turns_all_scenarios.extend(turns)

        # 4D Score 테이블
        report.append("### 4D Quality Scores\n")
        report.append(_turn_score_table(turns))
        report.append("")

        # 턴별 verdict 집계
        for t in turns:
            all_scores = [t["scores"].get(n) for n in THRESHOLDS]
            all_th = list(THRESHOLDS.values())
            if all(v is not None for v in all_scores):
                if all(v >= th for v, th in zip(all_scores, all_th)):
                    total_pass += 1
                else:
                    total_fail += 1
            else:
                total_na += 1

        # Context Evolution
        report.append("### Context Evolution\n")
        report.append(_context_evolution_table(turns))
        report.append("")

        # Events
        events_md = _events_section(turns)
        if events_md != "_이벤트 없음_":
            report.append("### Events\n")
            report.append(events_md)
            report.append("")

        # 커버리지 요약
        cov = _coverage_summary(turns)
        report.append("### Coverage Summary\n")
        report.append(f"- **Turns**: {cov['total_turns']}")
        report.append(f"- **Turns with 4D Scores**: {cov['turns_with_scores']}/{cov['total_turns']}")
        report.append(f"- **All-PASS turns**: {cov['turns_all_pass']}/{cov['total_turns']}")
        report.append(f"- **Contradiction detected**: {'Yes' if cov['has_contradiction'] else 'No'}")
        report.append(f"- **Scope warning**: {'Yes' if cov['has_scope_warning'] else 'No'}")
        report.append("")

        # Attribute 커버리지
        report.append("**Attribute Coverage**:\n")
        report.append("| Attribute | Turns with data |")
        report.append("|-----------|:---------------:|")
        for attr, count in cov["attr_coverage"].items():
            report.append(f"| `{attr}` | {count} |")
        report.append("")

        # 체크리스트
        if key_checks:
            report.append("### Validation Checklist\n")
            for check in key_checks:
                report.append(f"- [ ] {check}")
            report.append("")

    # --- 전체 종합 ---
    report.append("---\n")
    report.append("## Overall Summary\n")

    total_turns = len(all_turns_all_scenarios)
    report.append(f"| Metric | Value |")
    report.append(f"|--------|------:|")
    report.append(f"| Total scenarios | {len(session_ids)} |")
    report.append(f"| Total turns | {total_turns} |")
    report.append(f"| PASS turns | {total_pass} |")
    report.append(f"| FAIL turns | {total_fail} |")
    report.append(f"| N/A turns | {total_na} |")
    report.append(f"| Pass rate | {total_pass / total_turns * 100:.0f}% |" if total_turns > 0 else "| Pass rate | N/A |")
    report.append("")

    # 4D 차원별 평균 점수
    report.append("### Dimension Averages\n")
    report.append("| Dimension | Average | Threshold | Status |")
    report.append("|-----------|:-------:|:---------:|:------:|")
    for name, threshold in THRESHOLDS.items():
        vals = [
            t["scores"].get(name)
            for t in all_turns_all_scenarios
            if t["scores"].get(name) is not None
        ]
        if vals:
            avg = sum(vals) / len(vals)
            status = "PASS" if avg >= threshold else "FAIL"
            short = name.replace("_score", "").capitalize()
            report.append(f"| {short} | {avg:.2f} | {threshold} | {status} |")
    report.append("")

    # 프로세스 단계 커버리지
    report.append("### Process Stage Coverage\n")
    report.append("| Stage | Description | Status |")
    report.append("|:-----:|-------------|:------:|")
    has_gather = any(t["metadata"].get(ATTRS["context.total_tokens"]) for t in all_turns_all_scenarios)
    has_eval = any(t["metadata"].get(ATTRS["context.is_sufficient"]) is not None for t in all_turns_all_scenarios)
    has_analysis = any(get_contradicts_from_metadata(t["metadata"]) is not None for t in all_turns_all_scenarios)
    report.append(f"| ① | Plan (analyze_query) | {'✅' if has_gather else '❌'} |")
    report.append(f"| ② | Collect (gather_data) | {'✅' if has_gather else '❌'} |")
    report.append(f"| ③ | Organize (evaluate_context) | {'✅' if has_eval else '❌'} |")
    report.append(f"| ④ | Generate (generate_analysis + verify) | {'✅' if has_analysis else '❌'} |")
    report.append(f"| ⑤ | Memory (respond_to_user) | {'✅' if has_analysis else '❌'} |")
    report.append("")

    # 세션 ID 목록
    report.append("### Session IDs\n")
    report.append("```")
    for i, sid in enumerate(session_ids):
        scenario = SCENARIOS[i] if i < len(SCENARIOS) else None
        name = scenario["name"] if scenario else f"Scenario {i+1}"
        report.append(f"{name}: {sid}")
    report.append("```\n")

    # 재현 명령어
    report.append("### Reproduce\n")
    report.append("```bash")
    report.append("cd observable-research-agent && source .venv/bin/activate")
    report.append("")
    report.append("# 전체 시나리오 실행 + 리포트 생성")
    report.append("python -m evaluation.run_all_scenarios")
    report.append("")
    report.append("# 기존 세션으로 리포트만 재생성")
    sid_args = " ".join(session_ids)
    report.append(f"python -m evaluation.run_all_scenarios --report-only --sessions {sid_args}")
    report.append("")
    report.append("# 개별 시나리오 실행")
    report.append("python tests/scenarios/pipeline/insufficient_regather.py")
    report.append("python tests/scenarios/pipeline/numeric_verify_fail.py")
    report.append("")
    report.append("# 개별 세션 시각화")
    for sid in session_ids:
        report.append(f"python -m evaluation.visualize_session --session-id {sid}")
    report.append("```\n")

    # 파일 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = "\n".join(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    abs_path = os.path.abspath(output_path)
    print(f"\n📄 Report saved: {abs_path}")
    return abs_path


def main():
    """CLI 진입점.

    --report-only: 시나리오 실행 없이 기존 세션 ID로 리포트만 생성.
    --sessions: 세션 ID 목록 (--report-only와 함께 사용).
    --output: 리포트 저장 경로 (기본값: docs/testReport/YYYYMMDD_HHMM_TEST_REPORT.md).
    """
    parser = argparse.ArgumentParser(
        description="전체 시나리오 실행 + Context Monitoring 분석 리포트 생성",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="시나리오 실행 없이 기존 세션 ID로 리포트만 생성",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="세션 ID 목록 (--report-only와 함께 사용)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="리포트 저장 경로 (기본값: docs/testReport/YYYYMMDD_HHMM_TEST_REPORT.md)",
    )
    args = parser.parse_args()

    if args.report_only:
        if not args.sessions:
            parser.error("--report-only 사용 시 --sessions 필수")
        session_ids = args.sessions
    else:
        # 12개 시나리오 순차 실행
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

        # Langfuse 데이터 ingestion 대기
        print("\n⏳ Langfuse 데이터 ingestion 대기 (5초)...")
        time.sleep(5)

    report_path = generate_report(session_ids, output_path=args.output)
    print(f"\n✅ 완료! 리포트: {report_path}")


if __name__ == "__main__":
    main()
