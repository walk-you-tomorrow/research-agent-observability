SHELL := /bin/bash
VENV := source .venv/bin/activate &&

# --- 테스트 리포트 ---

.PHONY: test-report test-report-only qa-report qa-report-sessions

## 전체 시나리오 실행 + 테스트 리포트 생성
test-report:
	$(VENV) python -m evaluation.run_all_scenarios

## 기존 세션 ID로 리포트만 재생성 (예: make test-report-only S="sess_abc sess_def")
test-report-only:
	$(VENV) python -m evaluation.run_all_scenarios --report-only --sessions $(S)

## 최근 로그에서 세션 자동 탐지 → QA 리포트 생성
qa-report:
	$(VENV) python -m evaluation.generate_qa_report --auto-detect

## 세션 ID 지정 QA 리포트 (예: make qa-report-sessions S="sess_abc sess_def")
qa-report-sessions:
	$(VENV) python -m evaluation.generate_qa_report --sessions $(S)

# --- 시나리오 개별 실행 ---

.PHONY: test-regather test-numeric test-all

test-regather:
	$(VENV) python tests/scenarios/pipeline/insufficient_regather.py

test-numeric:
	$(VENV) python tests/scenarios/pipeline/numeric_verify_fail.py

test-all:
	$(VENV) python -m evaluation.run_all_scenarios

# --- Unit + Validation (API 불필요) ---

.PHONY: test-unit test-validation

test-unit:
	$(VENV) pytest tests/unit/ -v

test-validation:
	$(VENV) pytest tests/validation/ -v

# --- 서비스 (고객 채팅 + 관측 대시보드, 별도 프로세스) ---

.PHONY: chat dashboard

## 고객용 채팅 앱 (기본 포트 8501)
chat:
	$(VENV) streamlit run chat_app.py --server.port 8501

## 관측 대시보드 (운영자용, 기본 포트 8502)
dashboard:
	$(VENV) streamlit run dashboard/app.py --server.port 8502

# --- 시각화 (예: make visualize S=sess_abc) ---

.PHONY: visualize

visualize:
	$(VENV) python -m evaluation.visualize_session --session-id $(S)

# --- 도움말 ---

.PHONY: help

help:
	@echo ""
	@echo "  테스트 리포트"
	@echo "    make test-report            전체 시나리오 실행 + 리포트 생성"
	@echo "    make test-report-only S=\"sess_a sess_b ...\"  기존 세션으로 리포트만"
	@echo "    make qa-report              최근 로그 자동 탐지 → QA 리포트"
	@echo "    make qa-report-sessions S=\"sess_a sess_b\""
	@echo ""
	@echo "  시나리오 (개별)"
	@echo "    make test-regather          P1: Insufficient Re-gather"
	@echo "    make test-numeric           P2: Numeric Verify Fail"
	@echo "    make test-all               전체 12개 시나리오 순차 실행"
	@echo ""
	@echo "  Unit / Validation (API 불필요)"
	@echo "    make test-unit              Unit 테스트 (B1-B9, C1-C7)"
	@echo "    make test-validation        Validation 테스트 (E,F,G)"
	@echo ""
	@echo "  서비스 (별도 프로세스)"
	@echo "    make chat                   고객용 채팅 앱 (포트 8501)"
	@echo "    make dashboard              관측 대시보드 (포트 8502)"
	@echo ""
	@echo "  시각화"
	@echo "    make visualize S=sess_xxx   세션 시각화"
	@echo ""
