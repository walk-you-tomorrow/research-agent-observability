"""
dashboard/views/schema_browser.py — Tab 8: 속성 카탈로그

관측 질문: "어떤 속성을 모니터링하고 있는가?"

역할:
    ATTR_META 기반 YAML 속성 정의 카탈로그 (세션 독립).
    70개 활성 모니터링 속성(v3: core 49 + domain 20 + partial 1)을 네임스페이스별로 탐색한다.
    Turn 데이터 불필요 — 세션 선택 없이 항상 접근 가능.

데이터 흐름:
    입력: 없음 (ATTR_META 상수 직접 참조)
    출력: Streamlit UI
"""
import pandas as pd
import streamlit as st

from agent.monitoring_schema import ATTR_META

# 네임스페이스 → 프로세스 단계 레이블 (프로세스 단계 순)
_NS_STAGE_LABELS = {
    "query":    "① Plan",
    "gather":   "② Collect",
    "source":   "소스 관측",
    "web":      "웹 검색",
    "context":  "③ Organize + checkpoint",
    "analysis": "④ Generate",
    "verify":   "검증 checkpoint",
    "response": "⑤ Memory",
    "turn":     "메타",
    "eval":     "진단",
}


def render() -> None:
    """속성 카탈로그 탭을 렌더링한다 — 세션 독립, YAML SSOT 기반."""
    st.subheader("속성 카탈로그")
    st.caption(
        f"v3 기준 활성 모니터링 속성 **{len(ATTR_META)}개** — `config/monitoring_schema.yaml` SSOT. "
        "세션 실측값은 Tab 7(상세)에서 확인."
    )

    # ATTR_META 실제 네임스페이스 기반 동적 생성
    _active_ns = {key.split(".")[0] for key in ATTR_META}
    namespaces = [
        (ns, f"{ns}.*", label)
        for ns, label in _NS_STAGE_LABELS.items()
        if ns in _active_ns
    ]

    for ns_key, ns_label, stage_label in namespaces:
        prefix = ns_label.replace("*", "")
        ns_attrs = [
            (key, meta) for key, meta in ATTR_META.items()
            if key.startswith(prefix)
        ]
        if not ns_attrs:
            continue

        with st.expander(
            f"**{ns_label}** — {stage_label} ({len(ns_attrs)}개)",
            expanded=False,
        ):
            rows = []
            for attr_key, attr_meta in sorted(ns_attrs, key=lambda x: x[0]):
                rows.append({
                    "속성": attr_key,
                    "설명": attr_meta.get("description", ""),
                    "타입": attr_meta.get("type", "—"),
                    "품질 차원": attr_meta.get("quality") or attr_meta.get("quality_dimension") or "—",
                    "생애 단계": attr_meta.get("lifecycle", "—"),
                    "Tier": attr_meta.get("tier", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
