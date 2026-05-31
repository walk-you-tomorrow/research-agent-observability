"""
agent/llm.py — Claude LLM 인스턴스 생성 및 재시도 호출

이 모듈은 프로젝트 전체에서 사용하는 LLM 관련 유틸리티를 제공한다:
1. create_llm(): Claude 모델 인스턴스를 생성한다
2. invoke_with_retry(): 네트워크/API 오류 시 지수 백오프로 재시도하고
   **Langfuse v3 GENERATION observation을 자동 부착**한다 (analysis/39 P0)

모든 노드는 LLM 호출 시 반드시 invoke_with_retry()를 경유해야 한다.
이를 통해:
1. 일시적 API 오류(429 Rate Limit, 500 Server Error 등) 자동 처리
2. 모든 LLM 호출이 메인 trace에 GENERATION으로 부착되어 Tab 7에서 가시화
"""
import os
import time

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langfuse import get_client


def _load_llm_config() -> dict:
    """config/agent_config.yaml에서 LLM 설정을 로드한다."""
    try:
        with open("config/agent_config.yaml") as f:
            return yaml.safe_load(f).get("llm", {})
    except FileNotFoundError:
        return {}


def create_llm(purpose: str = "agent", model_override: str | None = None) -> ChatAnthropic:
    """Claude LLM 인스턴스를 생성하여 반환한다.

    Args:
        purpose: API 키 선택 목적. 토큰 사용량을 키별로 추적할 수 있다.
            "agent" → ANTHROPIC_API_KEY_AGENT (노드 런타임)
            "evaluation" → ANTHROPIC_API_KEY_EVALUATION (4D 평가)
            그 외 → ANTHROPIC_API_KEY (기본 폴백)
        model_override: 명시적 모델명 (예: "claude-sonnet-4-6"). v3 cross-model 평가용.
            None이면 yaml config의 model 사용.

    Returns:
        ChatAnthropic 인스턴스.
    """
    cfg = _load_llm_config()
    model = model_override or cfg.get("model", "claude-haiku-4-5-20251001")

    # 목적별 API 키 분리: 각 키가 없으면 기본 ANTHROPIC_API_KEY로 폴백
    key_map = {
        "agent": "ANTHROPIC_API_KEY_AGENT",
        "evaluation": "ANTHROPIC_API_KEY_EVALUATION",
    }
    env_var = key_map.get(purpose, "ANTHROPIC_API_KEY")
    api_key = os.environ.get(env_var) or os.environ.get("ANTHROPIC_API_KEY")

    return ChatAnthropic(
        model=model,
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 4096),
        timeout=cfg.get("timeout_seconds", 120),
        api_key=api_key,
    )


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """메시지 리스트를 trace 부착용 dict 리스트로 직렬화한다."""
    out = []
    for m in messages:
        try:
            content = m.content if isinstance(m.content, str) else str(m.content)
            role = getattr(m, "type", m.__class__.__name__.lower().replace("message", ""))
            out.append({"role": role, "content": content})
        except Exception:
            out.append({"role": "unknown", "content": str(m)})
    return out


def _extract_usage(result: BaseMessage) -> dict:
    """LLM 응답에서 input/output 토큰 사용량을 추출한다."""
    usage_metadata = getattr(result, "usage_metadata", None) or {}
    if usage_metadata:
        return {
            "input": usage_metadata.get("input_tokens", 0),
            "output": usage_metadata.get("output_tokens", 0),
        }
    response_metadata = getattr(result, "response_metadata", None) or {}
    usage = response_metadata.get("usage", {})
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
    }


def invoke_with_retry(
    llm: ChatAnthropic,
    messages: list[BaseMessage],
    max_retries: int = 2,
    generation_name: str = "llm_call",
    trace_id: str | None = None,
) -> BaseMessage:
    """LLM을 호출하되, 실패 시 지수 백오프로 재시도하고 Langfuse GENERATION으로 trace한다.

    Args:
        llm: ChatAnthropic 인스턴스 (create_llm()으로 생성)
        messages: LLM에 전달할 메시지 리스트 (SystemMessage + HumanMessage 등)
        max_retries: 최대 재시도 횟수 (기본값: 2). 총 시도 = max_retries + 1.
        generation_name: Langfuse GENERATION observation 이름. 호출 노드명 + 목적
            (예: "evaluate_context_sufficiency", "verify_result_interpretation",
             "respond_to_user_compose").
        trace_id: 명시적 부착 대상 trace ID. None이면 현재 활성 컨텍스트(부모 SPAN)에
            자동 부착한다. 노드 런타임 호출은 @observe() 컨텍스트 안이므로 None으로 충분하나,
            4D Judge처럼 턴 trace 컨텍스트가 닫힌 뒤(밖에서) 호출되는 경우는
            trace_id를 명시해야 generation이 별개 root trace로 떨어지지 않고
            해당 턴 trace의 자식으로 부착된다 (Tab 7 가시화).

    Returns:
        LLM의 응답 메시지 (BaseMessage).

    Raises:
        Exception: max_retries를 모두 소진한 후에도 실패하면 마지막 예외를 그대로 raise.

    Langfuse 부착 (analysis/39 P0):
        Langfuse v3의 `start_as_current_observation(as_type="generation", ...)`로
        매 LLM 호출을 GENERATION으로 trace에 부착한다. 부모 SPAN의 자식으로 자동 등록.
        Tab 7 LLM 호출 로그에서 evaluate_context / verify_result / respond_to_user 모두
        가시화된다. input/output/model/usage가 자동 기록.
    """
    client = get_client()
    serialized_input = _serialize_messages(messages)

    # trace_id가 주어지면 해당 turn trace에 명시적으로 부착 (컨텍스트 밖 호출용)
    obs_kwargs = {
        "as_type": "generation",
        "name": generation_name,
        "input": serialized_input,
        "model": llm.model,
    }
    if trace_id:
        obs_kwargs["trace_context"] = {"trace_id": trace_id}

    with client.start_as_current_observation(**obs_kwargs) as generation:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                print(f"    🤖 LLM 호출 (model={llm.model}, attempt={attempt+1}/{max_retries+1}, gen={generation_name})")
                result = llm.invoke(messages)
                print(f"    🤖 LLM 응답 수신 ({len(result.content)}자)")
                generation.update(
                    output=result.content,
                    usage_details=_extract_usage(result),
                )
                return result
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    wait = 2**attempt
                    print(f"    ⚠ LLM 오류: {type(e).__name__}: {str(e)[:100]}  → {wait}초 후 재시도")
                    time.sleep(wait)
                else:
                    print(f"    ✘ LLM 재시도 소진: {type(e).__name__}: {str(e)[:100]}")
                    generation.update(
                        output=f"ERROR: {type(e).__name__}: {str(e)[:200]}",
                        level="ERROR",
                    )
                    raise
        # 정상 흐름에서 도달 불가 (raise 또는 return)
        raise RuntimeError("invoke_with_retry: unreachable") from last_exc
