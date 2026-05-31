"""
agent/config_loader.py — agent_config.yaml 중앙 로더

역할:
    agent_config.yaml을 한 번만 로드하고, 모듈 전역에서 재사용한다.
    도구/노드에서 설정값을 참조할 때 이 모듈을 import한다.

데이터 흐름:
    입력: config/agent_config.yaml
    출력: get_config() → dict (전체 설정), get_token_budget() → dict (토큰 제한 설정)
"""
import yaml

_config_cache: dict | None = None


def get_config() -> dict:
    """agent_config.yaml 전체를 로드하여 반환한다. 캐시된 결과를 재사용.

    Returns:
        YAML 설정 딕셔너리. 파일 없으면 빈 딕셔너리.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    try:
        with open("config/agent_config.yaml") as f:
            _config_cache = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _config_cache = {}
    return _config_cache


def get_token_budget() -> dict:
    """token_budget 섹션을 반환한다.

    Returns:
        {pandas_max_rows, api_max_rows, rag_max_chars} 딕셔너리.
        설정이 없으면 기본값 반환.
    """
    defaults = {
        "pandas_max_rows": 50,
        "api_max_rows": 100,
        "rag_max_chars": 8000,
    }
    budget = get_config().get("token_budget", {})
    return {k: budget.get(k, v) for k, v in defaults.items()}
