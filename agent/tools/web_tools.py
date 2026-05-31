"""
agent/tools/web_tools.py — 웹 검색 도구

Claude의 built-in web search를 래핑하여 서울 상권 관련 실시간 정보를 검색한다.
pandas(과거 CSV), LightRAG(도메인 지식), API(공공데이터)에 없는 최신 뉴스/트렌드를 보충한다.

인터페이스:
    tool_fn(query_analysis: dict, state: dict) → dict

반환값 구조:
    {
        "source": "web_search",
        "summary": str,                  # 검색 결과 요약
        "data": list[dict],              # [{title, url, page_age, cited_text}, ...]
        "relevance": str,                # "relevant" 또는 "irrelevant"
        "relevance_reason": str,         # 관련성 판단 사유
        "web_meta": {                    # 모니터링용 메타데이터
            "search_count": int,
            "result_count": int,
            "source_domains": list[str],
            "freshness": list[str],
        },
    }
"""
import logging
import os
import re
from urllib.parse import urlparse

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def web_search(query_analysis: dict, state: dict) -> dict:
    """Claude의 built-in web search로 서울 상권 관련 최신 정보를 검색한다.

    query_analysis의 intent와 키워드를 기반으로 검색 프롬프트를 구성하고,
    Claude의 server-side web_search_20250305 도구를 사용하여 검색한다.

    Args:
        query_analysis: 질의 분석 결과. intent를 참조하여 검색 쿼리를 구성한다.
        state: 현재 AgentState. user_query를 참조한다.

    Returns:
        표준 도구 반환 형식. web_meta에 모니터링용 메타데이터 포함.
    """
    user_query = state.get("user_query", "")
    intent = query_analysis.get("intent", "")

    # --- 검색 프롬프트 구성 ---
    # 사용자 질문을 바탕으로 웹 검색할 내용을 지시한다.
    search_prompt = (
        f"서울 상권과 관련된 다음 질문에 대해 최신 뉴스, 트렌드, 정책 정보를 검색하세요.\n\n"
        f"질문: {user_query}\n\n"
        f"검색 결과를 한국어로 요약해 주세요. "
        f"각 정보의 출처와 시점을 명시해 주세요."
    )

    try:
        # --- Claude web search 호출 ---
        # Haiku를 사용하여 비용을 절감한다 (web search는 래퍼 역할만 함).
        # 목적별 API 키 분리: ANTHROPIC_API_KEY_WEBSEARCH → 기본 ANTHROPIC_API_KEY 폴백
        api_key = os.environ.get("ANTHROPIC_API_KEY_WEBSEARCH") or os.environ.get("ANTHROPIC_API_KEY")
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=2048,
            timeout=60,
            api_key=api_key,
        )

        # Claude의 built-in web search 도구를 바인딩한다
        llm_with_search = llm.bind_tools([
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            }
        ])

        response = llm_with_search.invoke([HumanMessage(content=search_prompt)])

        # --- 응답 파싱 ---
        citations = []
        summary_text = ""
        search_count = 0

        # response.content는 content block 리스트
        content_blocks = response.content if isinstance(response.content, list) else []

        for block in content_blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "")

                if block_type == "text":
                    summary_text += block.get("text", "")

                elif block_type == "web_search_tool_result":
                    search_count += 1
                    # web search 결과에서 검색 결과 추출
                    search_content = block.get("content", [])
                    for item in search_content:
                        if isinstance(item, dict) and item.get("type") == "web_search_result":
                            citations.append({
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "page_age": item.get("page_age", ""),
                                "cited_text": item.get("encrypted_content", "")[:200],
                            })
            elif isinstance(block, str):
                summary_text += block

        # --- 메타데이터 추출 ---
        source_domains = list(set(
            urlparse(c["url"]).netloc
            for c in citations
            if c.get("url")
        ))
        freshness = [c["page_age"] for c in citations if c.get("page_age")]

        web_meta = {
            "search_count": search_count,
            "result_count": len(citations),
            "source_domains": source_domains,
            "freshness": freshness,
        }

        if not summary_text.strip():
            summary_text = f"웹 검색 완료: {len(citations)}개 결과"

        return {
            "source": "web_search",
            "summary": summary_text[:2000],
            "data": citations,
            "relevance": "relevant" if citations else "irrelevant",
            "relevance_reason": f"웹 검색: {len(citations)}개 결과, {len(source_domains)}개 도메인",
            "web_meta": web_meta,
        }

    except Exception as e:
        logger.error("웹 검색 실패: %s", e)
        return {
            "source": "web_search",
            "summary": f"웹 검색 실패: {str(e)[:200]}",
            "data": [],
            "relevance": "irrelevant",
            "relevance_reason": f"웹 검색 에러: {str(e)[:100]}",
            "web_meta": {
                "search_count": 0,
                "result_count": 0,
                "source_domains": [],
                "freshness": [],
            },
        }
