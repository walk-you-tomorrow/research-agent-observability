# Tests — Observable Research Agent

## 테스트 구조 개요

| Category | Directory | Count | 실행 환경 | API 필요 |
|----------|-----------|:-----:|----------|:--------:|
| Pipeline (P1-P2) | `scenarios/pipeline/` | 2 | Full pipeline | Yes |
| Multi-Source (A1-A6) | `scenarios/multi_source/` | 6 | Full pipeline | Yes |
| Multi-Turn (D1-D4) | `scenarios/multi_turn/` | 4 | Full pipeline | Yes |
| Unit (B1-B9, C1-C7) | `unit/` | 2 files | Isolated | Partial |
| Validation (E,F,G) | `validation/` | 3 files | Static | No |

## Quick Start

```bash
cd observable-research-agent && source .venv/bin/activate

# Unit + Validation (API 불필요, ~30초)
pytest tests/unit/ tests/validation/ -v

# 단일 시나리오 실행 (API 필요)
python tests/scenarios/pipeline/insufficient_regather.py

# 전체 시나리오 + 리포트 생성
python -m evaluation.run_all_scenarios
```

## 시나리오 상세

### Pipeline (2개) — 그래프 분기 경로 검증

| ID | File | Turns | 검증 포인트 |
|----|------|:-----:|------------|
| P1 | `pipeline/insufficient_regather.py` | 1 | 분기①: 데이터 부족 → 재수집 루프, confidence_delta |
| P2 | `pipeline/numeric_verify_fail.py` | 1 | 분기②: 수치 검증 실패 → 재생성, verify_retry_count |

### Multi-Source A (6개) — 소스 선택 검증

| ID | File | Turns | 검증 포인트 |
|----|------|:-----:|------------|
| A1 | `multi_source/test_a1_csv_only.py` | 1 | CSV 전용, pandas_query만 호출 |
| A2 | `multi_source/test_a2_rag_only.py` | 1 | RAG 전용, LightRAG 시맨틱 검색 |
| A3 | `multi_source/test_a3_api_trigger.py` | 1 | API 호출, api_query 도구 |
| A4 | `multi_source/test_a4_web_trigger.py` | 1 | 웹 검색, web_search 도구 |
| A5 | `multi_source/test_a5_fusion.py` | 1 | 3개+ 소스 통합 분석 |
| A6 | `multi_source/test_a6_source_conflict.py` | 1 | 소스 간 충돌 감지 |

### Multi-Turn D (4개) — 턴 간 일관성 검증

| ID | File | Turns | 검증 포인트 |
|----|------|:-----:|------------|
| D1 | `multi_turn/test_d1_progressive.py` | 3 | 점진적 세밀화 (구→동→업종) |
| D2 | `multi_turn/test_d2_contradiction.py` | 2 | 모순 감지 + 해결 (Consistency Pattern B) |
| D3 | `multi_turn/test_d3_reference.py` | 3 | 이전 턴 참조 정확성 (lookup_previous) |
| D4 | `multi_turn/test_d4_memory.py` | 3 | turn_conclusions 누적 + messages 전달 |

### Unit Tests (B/C) — 도구 단위 테스트

| ID | File | Count | 검증 포인트 |
|----|------|:-----:|------------|
| B1-B9 | `unit/test_tools.py` | 9 groups | 9개 도구 반환값 형식, 기본 동작 |
| C1-C7 | `unit/test_error_handling.py` | 7 groups | 에러 처리, graceful degradation |

### Validation Tests (E/F/G) — 정적 검증

| ID | File | Count | 검증 포인트 |
|----|------|:-----:|------------|
| E1-E4 | `validation/test_monitoring_coverage.py` | 4 groups | 64개 ATTRS 등록, 생애주기 커버리지 |
| F1-F4 | `validation/test_data_pipeline.py` | 4 groups | CSV 인코딩, KG 무결성, PDF 추출 |
| G1-G4 | `validation/test_performance.py` | 4 groups | 레이턴시, 캐싱, 토큰 예산 |

## 리포트

테스트 리포트는 `docs/testReport/` 에 생성된다.

```bash
# 전체 시나리오 실행 + 리포트 생성
python -m evaluation.run_all_scenarios

# 기존 세션 ID로 리포트만 생성
python -m evaluation.run_all_scenarios --report-only --sessions sess_abc sess_def
```

출력: `docs/testReport/YYYYMMDD_HHMM_TEST_REPORT.md`
