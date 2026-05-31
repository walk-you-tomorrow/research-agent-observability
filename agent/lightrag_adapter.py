"""
agent/lightrag_adapter.py — LightRAG 어댑터 (싱글턴)

역할:
    LightRAG 인스턴스의 생성과 쿼리를 관리한다.
    모든 RAG 도구는 이 모듈의 query_knowledge()를 통해 검색한다.

설정:
    LLM: Anthropic Claude (agent_config.yaml의 lightrag.llm_model)
    Embedding: Ollama nomic-embed-text (로컬, 768차원)

데이터 흐름:
    입력: 검색 쿼리 문자열 + 검색 모드
    출력: 검색 결과 딕셔너리 (source, summary, data, relevance)

Event Loop 관리:
    LightRAG 내부의 asyncio Lock/PriorityQueue는 특정 event loop에 바인딩된다.
    asyncio.run()은 매 호출마다 새 event loop를 생성하므로 두 번째 호출부터
    'bound to a different event loop' 에러가 발생한다.
    이를 해결하기 위해 전용 백그라운드 스레드에서 단일 event loop를 유지한다.
"""

import asyncio
import logging
import os
import threading
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from langfuse import get_client, observe

from anthropic import AsyncAnthropic
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

from agent.config_loader import get_token_budget

logger = logging.getLogger(__name__)

# --- 회사 네트워크 SSL 인증서 호환 ---
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# --- 프로젝트 경로 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "agent_config.yaml"


# --- 싱글턴 인스턴스 ---
_rag_instance: LightRAG | None = None
_storages_initialized: bool = False

# --- 전용 event loop (백그라운드 스레드) ---
# LightRAG의 asyncio 내부 객체(Lock, PriorityQueue)가 특정 event loop에 바인딩되므로,
# 모든 비동기 호출이 동일한 event loop에서 실행되어야 한다.
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """전용 event loop를 반환한다. 없으면 백그라운드 스레드에서 생성한다."""
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    _loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _loop_thread = threading.Thread(target=_run_loop, daemon=True)
    _loop_thread.start()
    return _loop


def _run_async(coro) -> Any:
    """코루틴을 전용 event loop에서 실행하고 결과를 동기적으로 반환한다.

    asyncio.run() 대신 이 함수를 사용하여 event loop 재생성 문제를 방지한다.

    Args:
        coro: 실행할 코루틴.

    Returns:
        코루틴의 반환값.
    """
    loop = _get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)  # 5분 타임아웃


def _load_lightrag_config() -> dict:
    """agent_config.yaml에서 LightRAG 설정을 로드한다.

    Returns:
        LightRAG 설정 딕셔너리.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("lightrag", {})


def _build_embedding_func(config: dict) -> EmbeddingFunc:
    """Ollama 기반 embedding 함수를 생성한다.

    Args:
        config: LightRAG 설정 딕셔너리.

    Returns:
        EmbeddingFunc 인스턴스 (nomic-embed-text, 768차원).
    """
    embed_config = config.get("embedding", {})
    embed_model = embed_config.get("model", "nomic-embed-text:latest")
    embed_dim = embed_config.get("dim", 768)
    ollama_host = embed_config.get("host", "http://localhost:11434")

    # ollama_embed는 이미 @wrap_embedding_func_with_attrs로 래핑되어 있으므로
    # .func으로 원본 함수에 접근해야 이중 래핑을 방지한다
    return EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=8192,
        func=partial(
            ollama_embed.func,
            embed_model=embed_model,
            host=ollama_host,
        ),
    )


async def _anthropic_complete_no_stream(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """Anthropic Claude를 호출하여 문자열 응답을 반환한다.

    LightRAG 내장 anthropic_complete는 항상 stream=True로 async_generator를
    반환하여 엔티티 추출 시 오류가 발생한다. 이 함수는 stream=False로
    호출하여 문자열을 직접 반환한다.

    Args:
        prompt: 사용자 프롬프트.
        system_prompt: 시스템 프롬프트.
        history_messages: 이전 대화 메시지.
        **kwargs: 추가 파라미터 (hashing_kv 등 LightRAG 내부용 제거됨).

    Returns:
        LLM 응답 텍스트.
    """
    if history_messages is None:
        history_messages = []

    # LightRAG 내부 파라미터에서 모델명 추출 후 제거
    hashing_kv = kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    timeout = kwargs.pop("timeout", None)

    # 목적별 API 키 분리: ANTHROPIC_API_KEY_LIGHTRAG → 기본 ANTHROPIC_API_KEY 폴백
    api_key = os.environ.get("ANTHROPIC_API_KEY_LIGHTRAG") or os.environ.get("ANTHROPIC_API_KEY")
    # LightRAG는 hashing_kv.global_config에 llm_model_name을 저장한다
    model = "claude-haiku-4-5-20251001"
    if hashing_kv and hasattr(hashing_kv, "global_config"):
        model = hashing_kv.global_config.get("llm_model_name", model)

    client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    messages = list(history_messages)
    messages.append({"role": "user", "content": prompt})

    create_params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,           # 핵심: 스트리밍 비활성화
        "max_tokens": kwargs.pop("max_tokens", 4096),
        **kwargs,
    }
    if system_prompt:
        create_params["system"] = system_prompt

    response = await client.messages.create(**create_params)
    text = response.content[0].text if response.content else ""
    logger.debug(f"Anthropic response: {len(text)} chars")
    return text


def get_rag() -> LightRAG:
    """LightRAG 싱글턴 인스턴스를 반환한다.

    최초 호출 시 인스턴스를 생성하고, 이후 동일 인스턴스를 재사용한다.
    LLM은 Anthropic Claude, Embedding은 Ollama nomic-embed-text를 사용한다.

    Returns:
        LightRAG 인스턴스.
    """
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    config = _load_lightrag_config()
    working_dir = str(PROJECT_ROOT / config.get("working_dir", "lightrag_storage"))

    # working_dir 생성
    os.makedirs(working_dir, exist_ok=True)

    _rag_instance = LightRAG(
        working_dir=working_dir,
        llm_model_func=_anthropic_complete_no_stream,              # Anthropic Claude (비스트리밍)
        llm_model_name=config.get("llm_model", "claude-haiku-4-5-20251001"),
        llm_model_kwargs={"max_tokens": 4096},                   # Anthropic API 필수 파라미터
        embedding_func=_build_embedding_func(config),            # Ollama 로컬 embedding
        chunk_token_size=config.get("chunk_token_size", 1200),
        llm_model_max_async=config.get("max_async", 1),
    )

    return _rag_instance


async def _query_with_init(rag: LightRAG, query: str, mode: str, top_k: int) -> str:
    """스토리지를 초기화한 후 쿼리를 실행한다.

    LightRAG v1.4+에서는 스토리지를 명시적으로 초기화해야 한다.
    초기화는 최초 1회만 수행하고, finalize는 하지 않는다.
    (finalize는 event loop 리소스를 해제하므로 호출하면 안 된다.)

    Args:
        rag: LightRAG 인스턴스.
        query: 검색 쿼리 문자열.
        mode: 검색 모드.
        top_k: 반환할 최대 결과 수.

    Returns:
        검색 결과 문자열.
    """
    global _storages_initialized
    if not _storages_initialized:
        await rag.initialize_storages()
        _storages_initialized = True
    return await rag.aquery(query, param=QueryParam(mode=mode, top_k=top_k))


@observe(name="query_knowledge")
def query_knowledge(
    query: str,
    mode: str = "mix",
    top_k: int = 10,
) -> dict:
    """LightRAG Knowledge Graph를 검색한다.

    @observe 데코레이터로 Langfuse에 검색 활동을 기록한다.
    전용 event loop에서 비동기 쿼리를 실행하여 asyncio.run()의
    event loop 재생성 문제를 방지한다.

    Args:
        query: 검색 쿼리 문자열.
        mode: 검색 모드 ("local", "global", "hybrid", "mix").
        top_k: 반환할 최대 결과 수.

    Returns:
        표준 도구 반환 형식:
        {
            "source": "lightrag:{mode}",
            "summary": 검색 결과 요약 텍스트,
            "data": 원본 결과,
            "relevance": 관련성 점수 (0.0~1.0),
            "relevance_reason": 관련성 판단 근거,
        }
    """
    try:
        rag = get_rag()

        # 전용 event loop에서 쿼리 실행 (asyncio.run() 대신)
        # LightRAG 내부의 Lock/PriorityQueue가 동일 event loop에서 동작하도록 보장
        result = _run_async(_query_with_init(rag, query, mode, top_k))

        # Langfuse에 검색 메타데이터 기록
        get_client().update_current_span(
            metadata={
                "rag.query": query,
                "rag.mode": mode,
                "rag.top_k": top_k,
                "rag.result_length": len(result) if result else 0,
            }
        )

        # 결과 텍스트 길이 제한: agent_config.yaml의 token_budget.rag_max_chars 참조
        max_chars = get_token_budget()["rag_max_chars"]
        result_text = result if isinstance(result, str) else str(result)
        if len(result_text) > max_chars:
            result_text = result_text[:max_chars] + " [truncated]"

        return {
            "source": f"lightrag:{mode}",
            "summary": result_text,
            "data": result_text,
            "relevance": 0.8,  # LightRAG 결과는 기본 높은 관련성
            "relevance_reason": f"LightRAG {mode} mode 검색 결과",
        }

    except Exception as e:
        get_client().update_current_span(
            metadata={
                "rag.query": query,
                "rag.mode": mode,
                "rag.error": str(e),
            }
        )
        return {
            "source": f"lightrag:{mode}",
            "summary": f"검색 실패: {e}",
            "data": None,
            "relevance": 0.0,
            "relevance_reason": f"LightRAG 검색 오류: {e}",
        }
