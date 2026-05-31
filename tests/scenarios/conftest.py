"""tests/scenarios/conftest.py — 시나리오 공통 설정.

시나리오 실행에 필요한 환경(프로젝트 루트, 환경변수)을 설정한다.
"""
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

load_dotenv()
