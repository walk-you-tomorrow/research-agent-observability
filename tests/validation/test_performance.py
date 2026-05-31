"""
tests/validation/test_performance.py — 성능 기준선 검증

에이전트와 개별 컴포넌트의 성능 특성을 검증한다.
실제 API 호출 시간은 환경에 따라 다르므로, 이 테스트는
컴포넌트의 기본 성능과 설정의 합리성을 검증한다.

테스트 대상 (G1~G4):
    G1: 단일 턴 레이턴시 구성요소 (토큰 카운팅, CSV 로딩 속도)
    G2: 7턴 세션 시간 — 설정 기반 추정
    G3: API 캐싱 효과 검증
    G4: 토큰 예산 준수 — context_window 설정 검증

실행 방법:
    python -m pytest tests/validation/test_performance.py -v
"""
import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

KB_DATA_DIR = os.path.join("knowledge_base", "data")


# ═══════════════════════════════════════
# G1: 단일 턴 레이턴시 구성요소
# ═══════════════════════════════════════
class TestG1SingleTurnLatency:
    """G1: 개별 컴포넌트의 처리 시간을 확인한다."""

    def test_token_counting_speed(self):
        """토큰 카운팅이 10ms 이내에 완료된다."""
        from agent.token_counter import count_tokens

        text = "서울 강남구 카페 상권 분석 데이터 " * 100
        start = time.monotonic()
        count_tokens(text)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100, f"Token counting took {elapsed_ms:.1f}ms (expected < 100ms)"

    def test_csv_loading_speed(self):
        """store_info.csv (182K행) 로딩이 5초 이내에 완료된다."""
        fpath = os.path.join(KB_DATA_DIR, "store_info.csv")
        if not os.path.exists(fpath):
            pytest.skip("store_info.csv not found")

        start = time.monotonic()
        df = pd.read_csv(fpath)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 5000, f"CSV loading took {elapsed_ms:.1f}ms (expected < 5000ms)"
        assert len(df) > 1000, f"store_info.csv has only {len(df)} rows"

    def test_small_csv_loading_speed(self):
        """소형 CSV(rent.csv, 22행) 로딩이 100ms 이내에 완료된다."""
        fpath = os.path.join(KB_DATA_DIR, "rent.csv")
        if not os.path.exists(fpath):
            pytest.skip("rent.csv not found")

        start = time.monotonic()
        df = pd.read_csv(fpath)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100, f"Small CSV loading took {elapsed_ms:.1f}ms"

    def test_monitoring_schema_load_speed(self):
        """monitoring_schema.yaml 로드가 100ms 이내에 완료된다."""
        import importlib

        start = time.monotonic()
        # 모듈을 다시 로드하여 YAML 파싱 시간을 측정한다
        import agent.monitoring_schema
        importlib.reload(agent.monitoring_schema)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100, f"Schema loading took {elapsed_ms:.1f}ms"


# ═══════════════════════════════════════
# G2: 7턴 세션 시간 추정
# ═══════════════════════════════════════
class TestG2SessionTimeEstimate:
    """G2: 세션 시간을 결정하는 설정이 합리적인지 검증."""

    def test_inter_turn_delay_reasonable(self):
        """턴 간 대기 시간이 합리적인 범위(0~60초)에 있다."""
        import yaml

        with open("config/agent_config.yaml") as f:
            config = yaml.safe_load(f)

        delay = config.get("retry", {}).get("inter_turn_delay_seconds", 0)
        assert 0 <= delay <= 60, f"Inter-turn delay {delay}s is unreasonable"

    def test_max_turns_reasonable(self):
        """max_turns 설정이 합리적인 범위(5~20)에 있다."""
        import yaml

        with open("config/agent_config.yaml") as f:
            config = yaml.safe_load(f)

        max_turns = config.get("retry", {}).get("max_turns", 15)
        assert 5 <= max_turns <= 20, f"max_turns {max_turns} is unreasonable"


# ═══════════════════════════════════════
# G3: API 캐싱 효과
# ═══════════════════════════════════════
class TestG3APICaching:
    """G3: API 세션 내 캐싱이 동작하는지 검증."""

    def test_cache_starts_empty(self):
        """캐시가 clear 후 비어있다."""
        from agent.tools.api_tools import _api_cache, clear_api_cache

        clear_api_cache()
        assert len(_api_cache) == 0

    def test_cache_key_generation(self):
        """동일한 (api_key, params) 조합에 대해 동일한 캐시 키를 생성한다."""
        from agent.tools.api_tools import _make_cache_key

        key1 = _make_cache_key("commercial_change", {"STDR_YYQU_CD": "20244"})
        key2 = _make_cache_key("commercial_change", {"STDR_YYQU_CD": "20244"})
        assert key1 == key2, "Same params should produce same cache key"

    def test_different_params_different_key(self):
        """다른 파라미터는 다른 캐시 키를 생성한다."""
        from agent.tools.api_tools import _make_cache_key

        key1 = _make_cache_key("commercial_change", {"STDR_YYQU_CD": "20244"})
        key2 = _make_cache_key("commercial_change", {"STDR_YYQU_CD": "20243"})
        assert key1 != key2, "Different params should produce different cache keys"

    def test_api_registry_completeness(self):
        """API 레지스트리에 5개 API가 모두 등록되어 있다."""
        from agent.tools.api_tools import API_REGISTRY

        expected_apis = [
            "estimated_sales",
            "commercial_change",
            "store_openclose",
            "crowd_facility",
            "resident_population",
        ]
        for api in expected_apis:
            assert api in API_REGISTRY, f"API '{api}' not in registry"
            assert "service" in API_REGISTRY[api], f"API '{api}' missing 'service'"
            assert "description" in API_REGISTRY[api], f"API '{api}' missing 'description'"


# ═══════════════════════════════════════
# G4: 토큰 예산 준수
# ═══════════════════════════════════════
class TestG4TokenBudget:
    """G4: 컨텍스트 윈도우 설정이 합리적이고 일관적인지 검증."""

    def test_context_window_max_tokens(self):
        """CONTEXT_WINDOW_MAX_TOKENS가 합리적인 범위(100K~200K)에 있다."""
        from agent.monitoring_schema import CONTEXT_WINDOW_MAX_TOKENS

        assert 100_000 <= CONTEXT_WINDOW_MAX_TOKENS <= 200_000, \
            f"CONTEXT_WINDOW_MAX_TOKENS = {CONTEXT_WINDOW_MAX_TOKENS} is unreasonable"

    def test_yaml_and_python_consistent(self):
        """YAML과 Python 로더의 max_tokens 값이 일치한다 (v2/v3 yaml 양립)."""
        import yaml

        from agent.monitoring_schema import CONTEXT_WINDOW_MAX_TOKENS

        with open("config/monitoring_schema.yaml", encoding="utf-8") as f:
            schema = yaml.safe_load(f)

        # v2: schema.context_window.max_tokens / v3: schema.constants.context_window.max_tokens
        cw = schema.get("context_window") or schema.get("constants", {}).get("context_window", {})
        yaml_max = cw.get("max_tokens")
        assert yaml_max == CONTEXT_WINDOW_MAX_TOKENS, \
            f"YAML ({yaml_max}) != Python ({CONTEXT_WINDOW_MAX_TOKENS})"

    def test_config_context_setting(self):
        """agent_config.yaml의 context 설정이 존재하고 합리적이다."""
        import yaml

        with open("config/agent_config.yaml") as f:
            config = yaml.safe_load(f)

        ctx = config.get("context", {})
        max_tokens = ctx.get("max_context_tokens", 0)
        assert max_tokens > 0, "max_context_tokens not configured"

        target_util = ctx.get("target_utilization", 0)
        assert 0.0 < target_util <= 1.0, \
            f"target_utilization {target_util} is unreasonable"
