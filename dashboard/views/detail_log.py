"""
dashboard/views/detail_log.py — Tab 7: 상세 로그

관측자의 질문: "LLM은 뭘 했고, 각 노드는 뭘 생산했나?"

역할:
    LLM 호출 로그 (사용자 작업 vs 체크포인트 노드 구분)와
    노드별 Contribution을 통합 표시한다.

데이터 흐름:
    입력: turns (list[dict]) — enriched session data (observations 포함)
    출력: Streamlit UI
"""
import pandas as pd
import streamlit as st

from agent.monitoring_schema import ATTR_META
from dashboard.analysis import classify_llm_calls, extract_node_contributions

# --- 노드별 아이콘 ---
NODE_ICONS = {
    "analyze_query": "🔍",
    "gather_data": "📥",
    "evaluate_context": "⚖️",
    "generate_analysis": "📝",
    "verify_result": "✅",
    "respond_to_user": "💬",
}

# --- 카테고리 라벨 + 색상 ---
CATEGORY_LABELS = {
    "user_task": "🔵 사용자 작업",
    "monitoring_checkpoint": "🟠 체크포인트 노드",
    "evaluation": "🟣 4D/정렬 Judge",
    "other": "⚪ 기타",
}


def _format_attr_value(value) -> str:
    """속성 값을 표시용 문자열로 변환한다."""
    if isinstance(value, bool):
        return "✓ True" if value else "✗ False"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "[]"
    return str(value)


def _render_message_content(content) -> str:
    """LLM 메시지 content를 안전하게 문자열로 변환한다."""
    if content is None:
        return "(없음)"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", content.get("content", str(content)))
    return str(content)


def _format_tokens(n: int | None) -> str:
    if n is None:
        return "N/A"
    return f"{n:,}" if n >= 1000 else str(n)


_PREVIEW_LEN = 500  # 이 길이 초과 시 미리보기 + 전체 보기 expander 표시


def _render_message_block(role: str, content: str) -> None:
    """역할 레이블 + 내용 블록 렌더링 — 긴 내용은 미리보기 + 전체 보기로 분리한다."""
    st.markdown(f"**`[{role}]`**")
    if len(content) > _PREVIEW_LEN:
        st.code(content[:_PREVIEW_LEN] + "\n...", language=None)
        with st.expander(f"전체 보기 ({len(content):,} chars)"):
            st.code(content, language=None)
    else:
        st.code(content, language=None)


def render(turns: list[dict]) -> None:
    """상세 로그 탭을 렌더링한다."""

    st.subheader("상세 로그")
    st.caption(
        "**관측 질문: \"LLM은 뭘 했고, 각 노드는 뭘 생산했나?\"** "
        "LLM 호출 로그(입력 메시지·출력·토큰)와 노드별 기록 Contribution을 통합 표시합니다. "
        "속성 정의 → Tab 9 속성 카탈로그."
    )

    # ═══ 턴 선택 ═══
    turn_options = {t["turn_number"]: f"Turn {t['turn_number']}" for t in turns}
    selected_turn_num = st.selectbox(
        "턴 선택", options=list(turn_options.keys()),
        format_func=lambda x: turn_options[x],
        key="detail_log_turn_select",
    )
    selected_turn = next(t for t in turns if t["turn_number"] == selected_turn_num)
    observations = selected_turn.get("observations", [])

    if not observations:
        st.warning("이 턴의 observation 데이터가 없습니다.")
        return

    # ═══ 서브탭 (세션 운영 데이터만 — 속성 카탈로그는 Tab 8 참조) ═══
    sub1, sub2 = st.tabs(["🤖 LLM 호출 로그", "🧩 노드 Contribution"])

    # --- LLM 호출 로그 ---
    with sub1:
        all_calls = classify_llm_calls(observations)
        # 8.1~8.3 피드백: trace에 부재한 LLM 호출 명시
        # 6노드 중 어느 노드의 LLM이 trace에 잡히는지 자동 진단.
        # bypass 설정에 의해 의도적으로 LLM 미호출하는 노드는 "bypass" 카테고리로 분리.
        ALWAYS_LLM_NODES = {"analyze_query", "generate_analysis"}
        BYPASS_CONFIGURABLE_NODES = {
            "evaluate_context": "evaluate_context (충분성 LLM 호출)",
            "verify_result": "verify_result (`bypass_verify_interp` config로 미호출 가능)",
            "respond_to_user": "respond_to_user (`bypass_respond_llm` config로 미호출 가능)",
        }
        captured_nodes = {c["parent_node"] for c in all_calls}
        missing_always = ALWAYS_LLM_NODES - captured_nodes
        missing_bypass = set(BYPASS_CONFIGURABLE_NODES) - captured_nodes

        msg_parts = ["**이 탭은 메인 trace의 GENERATION만 표시합니다.**\n"]
        if missing_always:
            msg_parts.append(
                f"⚠ **미부착 노드 (LLM 호출 의도, trace 누락)**: "
                f"`{'`, `'.join(sorted(missing_always))}`. P0 보강 hook 점검 필요.\n"
            )
        else:
            msg_parts.append("✓ 항상 LLM 호출 노드(analyze_query, generate_analysis) 모두 부착됨\n")
        if missing_bypass:
            bypass_list = "\n".join(
                f"  - {BYPASS_CONFIGURABLE_NODES[n]}"
                for n in sorted(missing_bypass)
            )
            msg_parts.append(
                f"ℹ **bypass 가능 노드 — LLM 미호출 (정상)**:\n{bypass_list}\n"
                f"  config/agent_config.yaml의 `optimization.*` 설정에 따라 의도적 미호출일 수 있음.\n"
            )
        judge_count = sum(1 for c in all_calls if c["call_category"] == "evaluation")
        msg_parts.append(
            f"ℹ **4D/정렬 Judge LLM** — 이 턴 trace에 부착된 Judge generation "
            f"{judge_count}건 (`judge_4d.*` 4개 + `judge_alignment.*` 2개 예상). "
            "🟣 카테고리로 표시되며, 턴 trace 컨텍스트 밖에서 실행되지만 `trace_id`로 "
            "이 trace에 명시 부착된다.\n"
            "ℹ **Groundedness Checker** — `evaluation/groundedness_checker.py`는 결정론적 (LLM 미사용)"
        )
        st.info("\n".join(msg_parts))
        if not all_calls:
            generation_count = sum(1 for o in observations if o.get("type") == "GENERATION")
            st.warning(
                f"이 턴 trace의 GENERATION = {generation_count}건. "
                f"agent/nodes/* 코드의 LLM 호출이 GENERATION으로 부착되지 않은 상태."
            )
        else:
            # 진단: 카테고리별/부모 노드별 분포 — 모든 호출이 다 잡혔는지 확인
            with st.expander(
                f"🔎 진단 정보 — 전체 LLM 호출 분류 ({len(all_calls)}건)",
                expanded=False,
            ):
                from collections import Counter
                cat_counts = Counter(c["call_category"] for c in all_calls)
                node_counts = Counter(c["parent_node"] for c in all_calls)
                st.markdown(
                    f"- **카테고리별**: "
                    f"🔵 사용자 작업 {cat_counts.get('user_task', 0)}건 / "
                    f"🟠 체크포인트 노드 {cat_counts.get('monitoring_checkpoint', 0)}건 / "
                    f"🟣 4D/정렬 Judge {cat_counts.get('evaluation', 0)}건 / "
                    f"⚪ 기타 {cat_counts.get('other', 0)}건"
                )
                node_str = " / ".join(f"`{n}` ×{c}" for n, c in node_counts.most_common())
                st.markdown(f"- **부모 노드별**: {node_str}")
                if cat_counts.get("monitoring_checkpoint", 0) == 0:
                    st.warning(
                        "⚠ 체크포인트 노드(evaluate_context, verify_result)의 LLM 호출이 0건입니다. "
                        "원인 후보: ① 해당 노드가 이번 턴에 LLM을 호출하지 않음 "
                        "(예: regather/reverify 분기 미발생), "
                        "② parent_observation_id 체인이 LangGraph 컨테이너를 못 넘어가 "
                        "`unknown`으로 분류됨. "
                        "→ 위 '부모 노드별'의 unknown 카운트를 확인하세요."
                    )
                if "unknown" in node_counts:
                    st.error(
                        f"⚠ `unknown` 노드로 분류된 호출 {node_counts['unknown']}건이 있습니다. "
                        f"`_resolve_parent_node`가 부모 체인을 끝까지 못 올라간 케이스입니다."
                    )

            # 필터
            filter_option = st.radio(
                "표시 범위",
                ["전체", "🔵 사용자 작업만", "🟠 체크포인트 노드만", "🟣 4D/정렬 Judge만", "⚪ 기타만"],
                horizontal=True,
                key="llm_filter",
            )
            if filter_option == "🔵 사용자 작업만":
                calls = [c for c in all_calls if c["call_category"] == "user_task"]
            elif filter_option == "🟠 체크포인트 노드만":
                calls = [c for c in all_calls if c["call_category"] == "monitoring_checkpoint"]
            elif filter_option == "🟣 4D/정렬 Judge만":
                calls = [c for c in all_calls if c["call_category"] == "evaluation"]
            elif filter_option == "⚪ 기타만":
                calls = [c for c in all_calls if c["call_category"] == "other"]
            else:
                calls = all_calls

            # 요약
            total_in = sum(c["tokens_in"] or 0 for c in calls)
            total_out = sum(c["tokens_out"] or 0 for c in calls)
            col1, col2, col3 = st.columns(3)
            col1.metric("LLM 호출 수", len(calls))
            col2.metric("총 입력 토큰", _format_tokens(total_in))
            col3.metric("총 출력 토큰", _format_tokens(total_out))

            # 호출 목록
            for i, call in enumerate(calls, 1):
                cat_label = CATEGORY_LABELS.get(call["call_category"], "")
                iter_suffix = f" (iteration {call['iteration']})" if call["iteration"] > 1 else ""
                header = f"[{i}] {cat_label} **{call['parent_node']}** — {call['purpose_label']}{iter_suffix}"
                summary = f"model: {call['model']} | in: {_format_tokens(call['tokens_in'])} | out: {_format_tokens(call['tokens_out'])} | {call['latency_ms'] or '?'}ms"

                with st.expander(f"{header}\n\n`{summary}`", expanded=False):
                    # 입력
                    st.markdown("##### 입력")
                    input_text = call.get("input_text")
                    if input_text:
                        if isinstance(input_text, list):
                            for msg in input_text:
                                if isinstance(msg, dict):
                                    role = msg.get("role") or msg.get("type") or "unknown"
                                    content = _render_message_content(msg.get("content", msg.get("text", msg)))
                                    _render_message_block(role, content)
                                else:
                                    _render_message_block("unknown", str(msg))
                        else:
                            _render_message_block("input", str(input_text))
                    else:
                        st.caption("(입력 없음)")

                    # 출력
                    st.markdown("##### 출력")
                    output_text = call.get("output_text")
                    if output_text:
                        out_str = (
                            _render_message_content(output_text)
                            if isinstance(output_text, dict)
                            else str(output_text)
                        )
                        _render_message_block("assistant", out_str)
                    else:
                        st.caption("(출력 없음)")

    # --- 노드 Contribution ---
    with sub2:
        contributions = extract_node_contributions(observations)
        if not contributions:
            st.info("이 턴에서 노드 실행 데이터가 감지되지 않았습니다.")
        else:
            # 노드명으로 그룹핑 (반복 노드를 하나의 상위 expander로 묶기)
            from collections import OrderedDict
            node_groups: OrderedDict[str, list] = OrderedDict()
            for contrib in contributions:
                node = contrib["node_name"]
                if node not in node_groups:
                    node_groups[node] = []
                node_groups[node].append(contrib)

            st.caption(
                "각 노드가 기록한 모니터링 속성. 속성의 정의(설명·타입·품질차원·OTel매핑)는 "
                "**속성 카탈로그 탭** (마지막 탭)에서 확인하세요."
            )

            for node, group in node_groups.items():
                icon = NODE_ICONS.get(node, "⬜")
                total_attrs = sum(len(c["attributes"]) for c in group)

                if len(group) == 1:
                    # 단일 실행: 기존과 동일하게 flat expander
                    attrs = group[0]["attributes"]
                    with st.expander(
                        f"{icon} **{node}** — {len(attrs)}개 속성",
                        expanded=False,
                    ):
                        if not attrs:
                            st.caption("기록된 속성 없음")
                        else:
                            rows = []
                            for a in attrs:
                                rows.append({
                                    "속성": a["key"],
                                    "값": _format_attr_value(a["value"]),
                                    "단계": ATTR_META.get(a["key"], {}).get("lifecycle") or "—",
                                    "품질 차원": a["quality_dimension"] or "—",
                                    "생산자": a.get("producer", "—"),
                                    "OTel매핑": a.get("otel_mapping", "—"),
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    # 반복 실행: 상위 expander 안에 iteration별 하위 expander
                    with st.expander(
                        f"{icon} **{node}** — {len(group)}회 실행 (총 {total_attrs}개 속성)",
                        expanded=False,
                    ):
                        # 이전 iteration의 값을 저장하여 변경 하이라이트
                        prev_values: dict[str, str] = {}
                        for i, contrib in enumerate(group):
                            iter_num = contrib["iteration"]
                            attrs = contrib["attributes"]
                            with st.expander(
                                f"iteration {iter_num}: {len(attrs)}개 속성",
                                expanded=(i == 0),
                            ):
                                if not attrs:
                                    st.caption("기록된 속성 없음")
                                else:
                                    rows = []
                                    for a in attrs:
                                        formatted = _format_attr_value(a["value"])
                                        # iteration 2+ 에서 이전과 달라진 값 표시
                                        changed = ""
                                        if i > 0 and a["key"] in prev_values and prev_values[a["key"]] != formatted:
                                            changed = " 🔄"
                                        rows.append({
                                            "속성": a["key"],
                                            "값": formatted + changed,
                                            "단계": ATTR_META.get(a["key"], {}).get("lifecycle") or "—",
                                            "품질 차원": a["quality_dimension"] or "—",
                                            "생산자": a.get("producer", "—"),
                                            "OTel매핑": a.get("otel_mapping", "—"),
                                        })
                                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            # 현재 iteration 값을 저장
                            for a in attrs:
                                prev_values[a["key"]] = _format_attr_value(a["value"])
