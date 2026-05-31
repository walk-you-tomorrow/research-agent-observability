"""tests/conftest.py — pytest 공통 설정.

프로젝트 루트를 sys.path에 추가하고 pytest 마커를 등록한다.
"""
import os
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def pytest_configure(config):
    """pytest 마커 등록."""
    config.addinivalue_line("markers", "unit: 단위 테스트 (API 불필요)")
    config.addinivalue_line("markers", "validation: 정적 검증 (API 불필요)")
    config.addinivalue_line("markers", "scenario: 통합 시나리오 (API 필요)")
    config.addinivalue_line("markers", "slow: 실행 시간 > 30초")
