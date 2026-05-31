#!/usr/bin/env python3
"""
scripts/daily_research_digest.py — 매일 연구 동향 무료 수집기

역할:
    Context Monitoring 프로젝트와 관련된 최신 소식을 RSS/arXiv/HN에서 수집하여
    docs/research/daily-YYYY-MM-DD.md 파일로 저장한다.
    LLM 호출 없음 — 완전 무료.

데이터 소스:
    - RSS/Atom: 업계 블로그 (LangChain, Langfuse, OpenAI, DeepLearning.ai 등)
    - arXiv: cs.CL/cs.AI 카테고리에서 키워드 검색
    - Hacker News: Algolia API로 키워드 검색

사용:
    python -m scripts.daily_research_digest
    python -m scripts.daily_research_digest --hours 48     # 지난 48시간
    python -m scripts.daily_research_digest --output X.md  # 출력 경로 지정

스케줄링 (macOS cron, 매일 9시):
    crontab -e
    0 9 * * * cd /Users/sung-a.park/workspace/ObservabilityPower/ObservabilityPowers/observable-research-agent \\
        && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m scripts.daily_research_digest

데이터 흐름:
    입력: 환경 + (선택적) ~/.config/research-digest/sources.json
    출력: docs/research/daily-YYYY-MM-DD.md (raw dump, LLM 요약 없음)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# 회사 프록시 SSL 인증서 호환 — macOS 시스템 키체인 사용 (다른 스크립트와 동일 패턴)
# truststore가 없으면 무시하고 계속 진행 (개인 환경에서도 작동하도록)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# ════════════════════════════════════════════════════════════════
# 설정 — 프로젝트 관련성에 맞춰 조정 (수정 시 코멘트로 이유 명시)
# ════════════════════════════════════════════════════════════════

# RSS/Atom 피드 목록 — Context Monitoring 프로젝트 관련 업계/학계 채널
# 동작 안 하는 피드는 자동 스킵하므로 의심되면 그대로 두거나 주석 처리.
# 검증일: 2026-04-29 (LangChain/OpenAI/DeepMind/HuggingFace/Simon Willison 응답 확인)
RSS_FEEDS = [
    # 업계 블로그 (검증됨)
    ("LangChain Blog", "https://blog.langchain.dev/rss/"),
    ("OpenAI Blog", "https://openai.com/blog/rss.xml"),
    ("Google DeepMind", "https://deepmind.google/blog/rss.xml"),
    ("HuggingFace Blog", "https://huggingface.co/blog/feed.xml"),
    ("Simon Willison", "https://simonwillison.net/atom/everything/"),
    # 시도해볼 피드 (URL 변경 가능성 있음 — 404 시 콘솔 경고만)
    # ("Anthropic News", "https://www.anthropic.com/news/rss.xml"),       # 404 (2026-04-29)
    # ("Langfuse Changelog", "https://langfuse.com/changelog/rss.xml"),   # 404 (2026-04-29)
    # ("DeepLearning.ai", "https://www.deeplearning.ai/the-batch/feed/"), # 404 (2026-04-29)
    # ("Pinecone Learn", "https://www.pinecone.io/blog/rss.xml"),         # 404 (2026-04-29)
]

# arXiv 검색 쿼리 — 카테고리 + 키워드. (cat:cs.CL OR cat:cs.AI) AND (keyword)
# 키워드 추가/삭제로 관심사 조정
ARXIV_QUERIES = [
    'all:"context rot"',
    'all:"context engineering"',
    'all:"multi-turn" AND all:"agent"',
    'all:"LLM observability"',
    'all:"LLM-as-judge"',
    'all:"groundedness" AND all:"LLM"',
    'all:"agent evaluation"',
    'all:"context window" AND all:"degradation"',
]
ARXIV_CATEGORIES = "cat:cs.CL OR cat:cs.AI OR cat:cs.LG"

# Hacker News 검색 키워드 — Algolia API
HN_KEYWORDS = [
    "context rot",
    "LLM observability",
    "LangGraph",
    "Langfuse",
    "context engineering",
    "agent evaluation",
    "multi-turn LLM",
]

# 기본 출력 경로
DEFAULT_OUTPUT_DIR = "docs/research"


# ════════════════════════════════════════════════════════════════
# HTTP 헬퍼 (stdlib only)
# ════════════════════════════════════════════════════════════════

REQUEST_TIMEOUT = 15  # 초 — 응답 없는 피드는 스킵


def _fetch(url: str, accept: str = "*/*") -> bytes | None:
    """URL을 가져온다. 실패 시 None."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "research-digest/1.0 (+https://github.com/anthropics/claude-code)",
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read()
    except Exception as e:
        print(f"  ⚠ {url} → {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ════════════════════════════════════════════════════════════════
# RSS / Atom 파서 (stdlib)
# ════════════════════════════════════════════════════════════════

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def _parse_date(s: str) -> datetime | None:
    """RFC822(RSS) 또는 ISO8601(Atom) 날짜 문자열을 파싱한다."""
    if not s:
        return None
    s = s.strip()
    # ISO8601 (Atom)
    try:
        # python 3.11+ supports trailing Z
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    # RFC822 (RSS)
    try:
        return parsedate_to_datetime(s)
    except (TypeError, ValueError):
        return None


def parse_feed(content: bytes) -> list[dict]:
    """RSS 2.0 또는 Atom 피드를 파싱하여 [{title, link, published, summary}, ...] 반환."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    items: list[dict] = []

    # RSS 2.0: <rss><channel><item>...
    for it in root.iterfind(".//item"):
        items.append({
            "title": (it.findtext("title") or "").strip(),
            "link": (it.findtext("link") or "").strip(),
            "published_raw": (it.findtext("pubDate") or "").strip(),
            "summary": (it.findtext("description") or "").strip(),
        })

    if items:
        return items

    # Atom: <feed><entry>...
    for e in root.iterfind("a:entry", ATOM_NS):
        link_el = e.find("a:link", ATOM_NS)
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        published = (
            e.findtext("a:updated", default="", namespaces=ATOM_NS)
            or e.findtext("a:published", default="", namespaces=ATOM_NS)
        )
        summary = (
            e.findtext("a:summary", default="", namespaces=ATOM_NS)
            or e.findtext("a:content", default="", namespaces=ATOM_NS)
        )
        items.append({
            "title": (e.findtext("a:title", default="", namespaces=ATOM_NS)).strip(),
            "link": link.strip(),
            "published_raw": published.strip(),
            "summary": (summary or "").strip(),
        })

    return items


def fetch_rss_feeds(feeds: list[tuple[str, str]], cutoff: datetime) -> dict[str, list[dict]]:
    """모든 피드를 가져와 cutoff 이후 항목만 필터링하여 반환한다.

    Args:
        feeds: [(label, url), ...]
        cutoff: 이 시간 이후 발행된 항목만 포함.

    Returns:
        {label: [item, ...]} (빈 결과 채널은 제외)
    """
    result: dict[str, list[dict]] = {}
    for label, url in feeds:
        print(f"  📡 {label}", file=sys.stderr)
        content = _fetch(url, accept="application/rss+xml, application/atom+xml, application/xml, text/xml")
        if not content:
            continue
        entries = parse_feed(content)
        recent: list[dict] = []
        for e in entries:
            dt = _parse_date(e.get("published_raw", ""))
            if dt is None:
                # 날짜를 못 읽으면 cutoff 적용 불가 → 보수적으로 제외
                continue
            # tz-naive 보정
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                e["published"] = dt
                recent.append(e)
        if recent:
            recent.sort(key=lambda x: x["published"], reverse=True)
            result[label] = recent
    return result


# ════════════════════════════════════════════════════════════════
# arXiv 검색
# ════════════════════════════════════════════════════════════════

ARXIV_API = "http://export.arxiv.org/api/query"


def fetch_arxiv(query: str, max_results: int = 15) -> list[dict]:
    """arXiv API에서 query로 검색하여 최근 항목을 반환한다."""
    params = {
        "search_query": f"({ARXIV_CATEGORIES}) AND ({query})",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    content = _fetch(url, accept="application/atom+xml")
    if not content:
        return []
    return parse_feed(content)


def fetch_arxiv_all(queries: list[str], cutoff: datetime) -> dict[str, list[dict]]:
    """여러 arXiv 쿼리를 실행하고 cutoff 이후 항목만 dedupe하여 반환한다.

    같은 논문이 여러 쿼리에 매칭되면 첫 매칭에만 포함 (link 기준 dedupe).
    """
    seen_links: set[str] = set()
    result: dict[str, list[dict]] = {}
    for q in queries:
        print(f"  🔬 arXiv: {q}", file=sys.stderr)
        items = fetch_arxiv(q)
        recent: list[dict] = []
        for e in items:
            dt = _parse_date(e.get("published_raw", ""))
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                continue
            link = e.get("link", "")
            if link in seen_links:
                continue
            seen_links.add(link)
            e["published"] = dt
            recent.append(e)
        if recent:
            recent.sort(key=lambda x: x["published"], reverse=True)
            result[q] = recent
    return result


# ════════════════════════════════════════════════════════════════
# Hacker News 검색 (Algolia)
# ════════════════════════════════════════════════════════════════

HN_API = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_hn(keyword: str, cutoff: datetime, max_results: int = 10) -> list[dict]:
    """HN에서 키워드를 검색한다. cutoff 이후 항목만 반환."""
    params = {
        "query": keyword,
        "tags": "story",
        "numericFilters": f"created_at_i>{int(cutoff.timestamp())}",
        "hitsPerPage": str(max_results),
    }
    url = f"{HN_API}?{urllib.parse.urlencode(params)}"
    content = _fetch(url, accept="application/json")
    if not content:
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    items: list[dict] = []
    for hit in data.get("hits", []):
        items.append({
            "title": hit.get("title", "").strip(),
            "link": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "points": hit.get("points", 0),
            "comments": hit.get("num_comments", 0),
            "hn_url": f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "published": datetime.fromtimestamp(hit.get("created_at_i", 0), tz=timezone.utc),
        })
    return items


def fetch_hn_all(keywords: list[str], cutoff: datetime) -> dict[str, list[dict]]:
    """여러 키워드 결과를 dedupe하여 반환한다."""
    seen_links: set[str] = set()
    result: dict[str, list[dict]] = {}
    for kw in keywords:
        print(f"  🔥 HN: {kw}", file=sys.stderr)
        items = fetch_hn(kw, cutoff)
        new_items: list[dict] = []
        for it in items:
            link = it["link"]
            if link in seen_links:
                continue
            seen_links.add(link)
            new_items.append(it)
        if new_items:
            new_items.sort(key=lambda x: x["points"], reverse=True)
            result[kw] = new_items
    return result


# ════════════════════════════════════════════════════════════════
# 마크다운 렌더러
# ════════════════════════════════════════════════════════════════


def _truncate(s: str, n: int = 280) -> str:
    s = " ".join(s.split())  # 공백 정리
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _strip_html(s: str) -> str:
    """간단한 HTML 태그 제거 (의존성 없이)."""
    out: list[str] = []
    in_tag = False
    for c in s:
        if c == "<":
            in_tag = True
        elif c == ">":
            in_tag = False
        elif not in_tag:
            out.append(c)
    return "".join(out)


def render_markdown(
    cutoff: datetime,
    rss: dict[str, list[dict]],
    arxiv: dict[str, list[dict]],
    hn: dict[str, list[dict]],
) -> str:
    """수집 결과를 raw dump 마크다운으로 렌더링한다."""
    now = datetime.now()
    lines: list[str] = []
    lines.append(f"# Daily Research Digest — {now.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"> Generated: {now.strftime('%Y-%m-%d %H:%M %Z').strip()}")
    lines.append(f"> Cutoff: {cutoff.strftime('%Y-%m-%d %H:%M UTC')} 이후 항목만")
    lines.append("> 무료 raw dump — LLM 요약 없음. 본문은 직접 클릭해서 확인.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # RSS 섹션
    lines.append("## 🆕 업계/블로그 (RSS)")
    lines.append("")
    if not rss:
        lines.append("_지난 기간 새 글 없음 (또는 모든 피드 응답 실패)_")
        lines.append("")
    else:
        for label, items in rss.items():
            lines.append(f"### {label} ({len(items)}건)")
            lines.append("")
            for it in items:
                pub = it["published"].strftime("%Y-%m-%d %H:%M")
                title = it["title"] or "(제목 없음)"
                link = it["link"]
                summary = _truncate(_strip_html(it.get("summary", "")), 220)
                lines.append(f"- **[{title}]({link})** — _{pub}_")
                if summary:
                    lines.append(f"  - {summary}")
            lines.append("")

    # arXiv 섹션
    lines.append("## 📜 arXiv")
    lines.append("")
    if not arxiv:
        lines.append("_지난 기간 매칭 논문 없음_")
        lines.append("")
    else:
        for query, items in arxiv.items():
            lines.append(f"### `{query}` ({len(items)}건)")
            lines.append("")
            for it in items:
                pub = it["published"].strftime("%Y-%m-%d")
                title = it["title"] or "(제목 없음)"
                link = it["link"]
                summary = _truncate(_strip_html(it.get("summary", "")), 280)
                lines.append(f"- **[{title}]({link})** — _{pub}_")
                if summary:
                    lines.append(f"  - {summary}")
            lines.append("")

    # HN 섹션
    lines.append("## 🔥 Hacker News")
    lines.append("")
    if not hn:
        lines.append("_지난 기간 키워드 매칭 게시물 없음_")
        lines.append("")
    else:
        for kw, items in hn.items():
            lines.append(f"### `{kw}` ({len(items)}건)")
            lines.append("")
            for it in items:
                pub = it["published"].strftime("%Y-%m-%d")
                title = it["title"] or "(제목 없음)"
                lines.append(
                    f"- **[{title}]({it['link']})** — {it['points']}점, "
                    f"💬 {it['comments']} ([HN]({it['hn_url']})) · _{pub}_"
                )
            lines.append("")

    # 푸터
    lines.append("---")
    lines.append("")
    lines.append("## 🔧 설정")
    lines.append("")
    lines.append(f"- 피드 {len(RSS_FEEDS)}개 / arXiv 쿼리 {len(ARXIV_QUERIES)}개 / HN 키워드 {len(HN_KEYWORDS)}개")
    lines.append(f"- 소스 편집: `scripts/daily_research_digest.py` 상단 RSS_FEEDS/ARXIV_QUERIES/HN_KEYWORDS")
    lines.append("- 키워드 추가 시 최신 트렌드(예: 새 도구명, 새 논문 주제)를 반영")
    lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="매일 연구 동향 무료 수집기 (LLM 호출 없음)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hours", type=int, default=24, help="얼마나 거슬러 올라갈지 (시간 단위)")
    parser.add_argument("--output", default=None, help="출력 마크다운 경로 (미지정 시 docs/research/daily-YYYY-MM-DD.md)")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["rss", "arxiv", "hn"],
        help="이 소스는 스킵",
    )
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    print(f"⏱  Cutoff: {cutoff.isoformat()} ({args.hours}h ago)", file=sys.stderr)

    rss = fetch_rss_feeds(RSS_FEEDS, cutoff) if "rss" not in args.skip else {}
    arxiv = fetch_arxiv_all(ARXIV_QUERIES, cutoff) if "arxiv" not in args.skip else {}
    hn = fetch_hn_all(HN_KEYWORDS, cutoff) if "hn" not in args.skip else {}

    output = render_markdown(cutoff, rss, arxiv, hn)

    # 출력 경로 결정
    if args.output:
        path = args.output
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(DEFAULT_OUTPUT_DIR, f"daily-{today}.md")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(output)

    # 요약 통계
    total_rss = sum(len(v) for v in rss.values())
    total_arxiv = sum(len(v) for v in arxiv.values())
    total_hn = sum(len(v) for v in hn.values())
    print(
        f"\n✅ 저장: {os.path.abspath(path)}\n"
        f"   RSS {total_rss}건 ({len(rss)} 피드) / "
        f"arXiv {total_arxiv}건 ({len(arxiv)} 쿼리) / "
        f"HN {total_hn}건 ({len(hn)} 키워드)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
