# Observable Research Agent

> **AI Agent가 단계별로 컨텍스트를 어떻게 구성·변형·전달하는지를 4가지 품질(완전성·효율성·관련성·일관성)로 들여다보는 관측 프레임워크 — 서울 상권 분석 Research Agent를 도메인 #1로 검증.**

"컨텍스트 품질을 측정하면 Agent 신뢰성을 데이터 기반으로 개선할 수 있다"는 가설을 검증하기 위한 Phase 1 구현체입니다.

## 핵심 특징

- **6노드 LangGraph 파이프라인**: 질의 분석 → 데이터 수집 → 컨텍스트 평가 → 분석 생성 → 결과 검증 → 응답
- **9개 도구**: RAG 4종(시맨틱 검색, 상세, 글로벌, 비교) + CSV pandas 쿼리 + 계산 + 이전 턴 조회 + 웹 검색 + 서울시 API
- **Multi-Source 아키텍처**: 4개 이질적 소스(CSV, LightRAG/PDF, Claude 웹 검색, 서울시 API)를 통합하여 소스 선택을 관측 가능한 의사결정으로 구현
- **Context Monitoring**: AI Agent Execution Process 5단계 × 품질 4차원을 Langfuse에 자동 기록 (64개 모니터링 속성)
- **LightRAG Knowledge Graph**: 서울시 25개 구 상권분석보고서 PDF를 Knowledge Graph로 인덱싱하여 시맨틱 검색 + 엔티티 관계 탐색
- **다중 턴 지원**: 이전 턴 결론을 누적하며 교차 턴 일관성(Consistency Pattern B) 추적
- **이중 검증**: pandas 수치 대조 + LLM 해석 검증으로 분석 결과의 정확성 보장

## 아키텍처

```
START → analyze_query → gather_data → evaluate_context ─┐
               ↑                                         │
               │  (컨텍스트 부족)                          │ (충분)
               └─────────────────────────────────────────┘
                                                          ↓
       ┌── gather_data ← (수치 오류) ── verify_result ← generate_analysis
       │                                     │
       │   generate_analysis ← (해석 오류) ──┘
       │                                     │ (통과)
       └─────────────────────────────────────→ respond_to_user → END
```

**조건부 분기:**

1. `evaluate_context` 이후 — 컨텍스트 충분한가? (`confidence >= 0.7`)
2. `verify_result` 이후 — 검증 통과했는가? (수치/해석 각각 판정)

## 빠른 시작

### 1. 환경 설정

```bash
cd observable-research-agent
python3 -m venv .venv       # Windows: python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 실제 키 입력
```

필요한 키:
| 환경변수 | 용도 | 필수 |
|---------|------|------|
| `ANTHROPIC_API_KEY` | Claude LLM 호출 (폴백 키) | Yes |
| `ANTHROPIC_API_KEY_AGENT` | Agent 노드 런타임 전용 | No (폴백: ANTHROPIC_API_KEY) |
| `ANTHROPIC_API_KEY_LIGHTRAG` | LightRAG 인덱싱+쿼리 전용 | No (폴백: ANTHROPIC_API_KEY) |
| `ANTHROPIC_API_KEY_WEBSEARCH` | 웹 검색 전용 | No (폴백: ANTHROPIC_API_KEY) |
| `ANTHROPIC_API_KEY_EVALUATION` | 4D 평가 Judge 전용 | No (폴백: ANTHROPIC_API_KEY) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 트레이스 기록 | Yes |
| `LANGFUSE_SECRET_KEY` | Langfuse 인증 | Yes |
| `LANGFUSE_HOST` | Langfuse 서버 주소 | Yes |
| `OPEN_GOV_API_KEY` | 서울시 상권분석서비스 API | No (api_query 사용 시) |

**사전 요구사항:** Ollama 설치 + `nomic-embed-text` 모델 다운로드 (LightRAG 임베딩용)
```bash
brew install ollama && brew services start ollama
ollama pull nomic-embed-text
```

### 3. 실행

```bash
python3 main.py
```

기본 질문 "서울에서 카페 창업하려는데 어디가 좋을까?"로 단일 턴 세션이 실행됩니다.

## 프로젝트 구조

```
observable-research-agent/
├── main.py                          # 진입점: 세션 실행 + Langfuse 연결
├── Makefile                         # 테스트/리포트/시나리오 실행 타겟
├── config/
│   ├── agent_config.yaml            # LLM, 재시도, 컨텍스트, LightRAG, token_budget 설정
│   └── monitoring_schema.yaml       # 64개 모니터링 속성 SSOT (속성명, 타입, 프로세스 단계, 품질차원) v2.0
├── agent/
│   ├── state.py                     # AgentState 타입 정의 (공유 상태)
│   ├── graph.py                     # LangGraph 그래프 구성 + 분기 로직
│   ├── models.py                    # Pydantic 모델 (LLM 출력 파싱용)
│   ├── llm.py                       # Claude LLM 인스턴스 + 재시도 + 목적별 API 키 분리
│   ├── config_loader.py             # agent_config.yaml 중앙 로더 (get_config, get_token_budget)
│   ├── lightrag_adapter.py          # LightRAG 싱글턴 + 쿼리 인터페이스 + event loop 관리
│   ├── lightrag_indexer.py          # 마크다운 로드 + PDF 텍스트 추출 (pdfplumber)
│   ├── monitoring_schema.py         # YAML 스키마 로더 (ATTRS, THRESHOLDS, ATTR_META)
│   ├── parser.py                    # JSON 추출 + Pydantic 파싱
│   ├── token_counter.py             # tiktoken 기반 토큰 카운터
│   ├── log_writer.py                # 로그 파일 작성
│   ├── nodes/
│   │   ├── analyze_query.py         # 노드 1: 질의 분석 + 소스 선택 + 도구 계획
│   │   ├── gather_data.py           # 노드 2: 9개 도구 호출로 데이터 수집
│   │   ├── evaluate_context.py      # 노드 3: 컨텍스트 충분성 평가 (★ 핵심)
│   │   ├── generate_analysis.py     # 노드 4: 분석 생성 + 일관성 추적 + 소스 충돌 감지
│   │   ├── verify_result.py         # 노드 5: 이중 검증 (수치 + 해석)
│   │   └── respond_to_user.py       # 노드 6: 최종 응답 + 턴 결론 저장
│   └── tools/
│       ├── rag_tools.py             # RAG 도구 4종: search, deep_read, global_summary, compare
│       ├── data_tools.py            # CSV 도구: pandas_query (정확한 수치 집계)
│       ├── result_tools.py          # 결과 도구: calculate, lookup_previous
│       ├── web_tools.py             # Claude 웹 검색 (실시간 뉴스/트렌드)
│       └── api_tools.py             # 서울시 상권분석서비스 API 5종
├── scripts/
│   ├── index_knowledge_base.py      # LightRAG 인덱싱 CLI (--force, --all)
│   ├── index_remaining.py           # 증분 인덱싱 (미처리 PDF만 추가)
│   └── legacy/                      # 아카이브된 레거시 스크립트
├── evaluation/
│   ├── run_evaluation.py            # 4D 평가 실행기 (Langfuse Score 부착)
│   ├── run_all_scenarios.py         # 12개 시나리오 일괄 실행 + 리포트 생성
│   ├── generate_qa_report.py        # QA 리포트 생성기
│   ├── visualize_session.py         # 세션 시각화 (Langfuse API 연동)
│   ├── generate_charts.py           # 성과 차트 생성
│   ├── judge_completeness.py        # 완전성 평가 Judge
│   ├── judge_efficiency.py          # 효율성 평가 Judge
│   ├── judge_relevance.py           # 관련성 평가 Judge
│   └── judge_consistency.py         # 일관성 평가 Judge
├── tests/
│   ├── conftest.py                  # pytest 마커 등록
│   ├── unit/                        # 단위 테스트
│   │   ├── test_tools.py            #   B: 9개 도구 단위 테스트 (33 methods)
│   │   └── test_error_handling.py   #   C: 에러 핸들링 테스트 (16 methods)
│   ├── validation/                  # 검증 테스트
│   │   ├── test_monitoring_coverage.py  # E: 모니터링 커버리지 (17 methods)
│   │   ├── test_data_pipeline.py    #   F: 데이터 파이프라인 무결성 (35 methods)
│   │   └── test_performance.py      #   G: 성능 기준선 (13 methods)
│   └── scenarios/                   # 통합 시나리오 (12개)
│       ├── pipeline/                #   P: 파이프라인 분기 테스트
│       ├── multi_source/            #   A: 소스 선택 시나리오 (A1~A6)
│       └── multi_turn/              #   D: 다중 턴 일관성 (D1~D4)
├── knowledge_base/                  # 서울 상권 데이터
│   └── data/                        # CSV 파일들 (공공데이터 포털에서 가공)
│       ├── store_info.csv           #   상가 정보 (182,746행 — 5개 구)
│       ├── foot_traffic.csv         #   유동인구 (91행 — 행정동별 시간/연령)
│       ├── rent.csv                 #   임대료 (22행 — 구/지역별 분기 추이)
│       ├── demographics.csv         #   인구통계 (91행 — 행정동별 성별/연령)
│       ├── business_codes.csv       #   업종코드 (247행)
│       ├── dong_summary.csv         #   동별 집계 (91행 — 상가+유동인구+인구)
│       ├── commercial_district.csv  #   상권 정보
│       ├── closed_store_info.csv    #   폐업 상가 정보
│       └── store_summary_by_dong.csv #  동별 상가 요약
├── lightrag_storage/                # Knowledge Graph 저장소 (git 추적, 환경 간 재현성 보장)
├── docs/
│   ├── USER_GUIDE.md                # 사용자 가이드
│   ├── CONTEXT_MONITORING_GUIDE.md  # Context Monitoring 가이드
│   ├── analysis/                    # 분석 리포트
│   ├── workLog/                     # 세션별 작업 로그
│   └── testReport/                  # 테스트 리포트
└── requirements.txt                 # Python 의존성
```

## 데이터 소스

4개 이질적 데이터 소스를 통합한다.

### CSV 데이터 (`knowledge_base/data/`)

공공데이터 포털에서 수집하여 전처리한 것. `pandas_query` 도구가 직접 쿼리한다.

| 원본 데이터                 | 출처                 | 가공 결과                                         |
| --------------------------- | -------------------- | ------------------------------------------------- |
| 상가(상권)정보 서울         | 소상공인시장진흥공단 | `store_info.csv` (535K → 182K rows, 5개 구 필터)  |
| 길단위인구-행정동           | 서울열린데이터광장   | `foot_traffic.csv` (2025Q3, 동코드 기반 5개 구)   |
| 임대동향 지역별 임대료      | 한국부동산원         | `rent.csv` (하위지역→구 매핑)                     |
| 성별 연령별 주민등록 인구수 | 행정안전부           | `demographics.csv` (110개 연령→8개 그룹 집계)     |
| 상가정보 업종코드           | 소상공인시장진흥공단 | `business_codes.csv` (cp949→UTF-8)                |
| (위 3개 조인)               | —                    | `dong_summary.csv` (상가+유동인구+인구 동별 집계) |

대상 구: **강남구, 마포구, 서초구, 종로구, 영등포구** / 원본 인코딩: cp949 → UTF-8로 변환

### LightRAG Knowledge Graph (`lightrag_storage/`)

서울시 25개 구 상권분석보고서 PDF + 마크다운 문서를 LightRAG로 인덱싱한 Knowledge Graph. `rag_*` 도구가 시맨틱 검색한다.

| 항목 | 값 |
|------|------|
| 인덱싱 문서 | 30개 PDF (서울시 25개 구 상권분석보고서 + 소상공인 금융리포트 등) |
| 엔티티 | 11,684개 |
| 관계 | 15,891개 |
| 텍스트 청크 | 3,906개 |

### 웹 검색 (실시간)

Claude의 내장 웹 검색(`web_search_20250305`)으로 최신 뉴스/트렌드를 실시간 검색한다.

### 서울시 API (공공 데이터)

서울시 상권분석서비스 API 5종(매출, 상권변화, 개폐업, 집객시설, 상주인구)을 런타임에 조회한다.

## 기술 스택

| 분류             | 기술                         | 용도                                                    |
| ---------------- | ---------------------------- | ------------------------------------------------------- |
| LLM              | Claude (langchain-anthropic) | 질의 분석, 충분성 판단, 분석 생성, 해석 검증            |
| Agent 프레임워크 | LangGraph                    | 6노드 상태 기계 그래프                                  |
| LLM 프레임워크   | LangChain                    | LLM 호출 체인, Langfuse 콜백 연동                       |
| RAG              | LightRAG (lightrag-hku)      | Knowledge Graph + 벡터 RAG (시맨틱 검색, 엔티티 관계 탐색) |
| 임베딩           | Ollama (nomic-embed-text)    | 로컬 임베딩 서버 (768차원, 무료)                        |
| PDF 추출         | pdfplumber                   | 상권분석보고서 PDF 텍스트 추출                          |
| 관측성           | Langfuse SDK v3              | 트레이스, 스팬, 메타데이터 (`@observe`, `get_client()`) |
| 데이터 처리      | pandas                       | CSV 로드, 쿼리, 수치 검증                               |
| 토큰 계산        | tiktoken                     | 컨텍스트 윈도우 사용률 측정                             |
| 스키마 검증      | Pydantic                     | LLM JSON 출력 파싱                                      |
| 속성 관리        | PyYAML                       | 64개 모니터링 속성 SSOT (monitoring_schema.yaml v2.0)   |

## 문서

- **[사용자 가이드](docs/USER_GUIDE.md)** — 설치부터 실행, 테스트까지 단계별 안내
- **[Context Monitoring 가이드](docs/CONTEXT_MONITORING_GUIDE.md)** — 모니터링 원리와 Langfuse 대시보드 활용법

## Context Monitoring 개요

### AI Agent Execution Process (5단계)

| 프로세스 단계 | 설명                              | 담당 노드           |
| ------------- | --------------------------------- | -------------------- |
| ① Plan        | 질문 분석, 소스·도구 선택          | `analyze_query`      |
| ② Collect     | 도구 호출로 데이터/문서 수집       | `gather_data`        |
| ③ Organize    | 수집 데이터 조합, 프롬프트 구성    | `evaluate_context`   |
| ④ Generate    | LLM이 분석·답변 생성              | `generate_analysis`  |
| ⑤ Memory      | 결론 압축·저장, 다음 턴 기반 지식  | `respond_to_user`    |

> **모니터링 체크포인트:** 충분성 평가(③ 이후)와 검증(`verify_result`)은 프로세스 단계가 아닌 모니터링 활동이다.

### 4차원 품질 (4D)

| 품질 차원 | 핵심 질문                            | 대표 지표                    | 주요 관측 단계   |
| --------- | ------------------------------------ | ---------------------------- | ---------------- |
| 완전성    | 필요한 데이터가 모두 있는가?         | `context.is_sufficient`      | ② Collect, ③ Organize |
| 효율성    | 컨텍스트 윈도우를 적절히 사용하는가? | `context.window_utilization` | ③ Organize       |
| 관련성    | 노이즈 비율이 적절한가?              | `context.noise_ratio`        | ① Plan, ② Collect |
| 일관성    | 반복/턴 간 판단이 일관적인가?        | `context.confidence_delta`   | ④ Generate, ⑤ Memory |
