"""
dashboard/widgets/cross_tab_link.py — Streamlit 탭 점프 헬퍼

역할:
    st.warning/error 대용 — HTML 배너 안에 다른 탭으로 점프하는 클릭 링크 포함.
    Tab 3 (composition) / Tab 5 (impact) 등 여러 탭에서 공유 사용.

데이터 흐름:
    입력: severity, body_html, tab_key, link_label
    출력: Streamlit components.html (iframe + JS)

Streamlit 제약 회피:
    - st.warning/error는 HTML 미허용
    - st.markdown(unsafe_allow_html=True)는 onclick 속성 strip
    → components.html(iframe) 안에 banner + JS를 같이 렌더하고
      JS는 window.parent.document에서 탭 버튼을 클릭한다.

주의:
    TAB_IDX는 app.py 9탭 순서와 동기화 필요.
"""
import streamlit.components.v1 as components


# --- 탭 인덱스 (app.py 9탭 순서) — JS 클릭 네비게이션용 ---
# 변경 시 app.py의 st.tabs(...) 순서와 동기화 필요.
# 2026-05-31: Tab 8 'trajectory' 폐기 → schema 인덱스 7로 당김.
# 2026-05-31: Tab 9 'analytics' 폐기 (사용자 요청).
TAB_IDX: dict[str, int] = {
    "overview":    0,
    "process":     1,
    "composition": 2,
    "fidelity":    3,
    "impact":      4,
    "diagnose":    5,
    "detail":      6,
    "schema":      7,
}


# --- severity별 색·배경·아이콘 매핑 ---
_PALETTE: dict[str, tuple[str, str, str]] = {
    "warn":  ("#f39c12", "rgba(243,156,18,0.15)", "⚠"),
    "error": ("#e74c3c", "rgba(231,76,60,0.15)", "🚨"),
    "info":  ("#3498db", "rgba(52,152,219,0.15)", "ℹ"),
}


def alert_with_tab_link(
    severity: str,
    body_html: str,
    tab_key: str,
    link_label: str,
    height: int = 58,
) -> None:
    """탭 점프 링크가 포함된 alert 배너를 렌더한다.

    Args:
        severity: "warn" / "error" / "info".
        body_html: 배너 본문 HTML (링크 앞까지).
        tab_key: TAB_IDX 의 키 (예: "impact", "fidelity").
        link_label: 클릭 가능한 링크에 표시할 텍스트 (예: "③ Rot 위험").
        height: iframe 높이 (기본 58 — 1줄용. 2줄 본문은 84 권장).
    """
    if severity not in _PALETTE:
        severity = "warn"
    color, bg, icon = _PALETTE[severity]
    tab_idx = TAB_IDX.get(tab_key)
    if tab_idx is None:
        components.html(f"""
        <div style="background:{bg};border-left:4px solid {color};padding:10px 14px;
                    border-radius:4px;color:#eee;
                    font-family:-apple-system,BlinkMacSystemFont,sans-serif;
                    font-size:14px;line-height:1.55;">
          <span style="color:{color};margin-right:8px;" aria-hidden="true">{icon}</span>{body_html}
        </div>
        """, height=height)
        return

    components.html(f"""
    <div style="background:{bg};border-left:4px solid {color};padding:10px 14px;
                border-radius:4px;color:#eee;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                font-size:14px;line-height:1.55;">
      <span style="color:{color};margin-right:8px;" aria-hidden="true">{icon}</span>{body_html}
      → <a href="#" onclick="cmGoToTab({tab_idx});return false;"
           style="color:{color};font-weight:600;text-decoration:underline;cursor:pointer;"
           aria-label="{link_label}로 이동">
        {link_label}</a>
    </div>
    <script>
    function cmGoToTab(idx) {{
      const doc = window.parent.document;
      const selectors = ['button[role="tab"]', '[data-baseweb="tab"]'];
      for (const sel of selectors) {{
        const tabs = doc.querySelectorAll(sel);
        if (tabs.length > idx) {{
          tabs[idx].click();
          tabs[idx].scrollIntoView({{behavior:'smooth', block:'start'}});
          return;
        }}
      }}
    }}
    </script>
    """, height=height)
