"""
agent/tools/api_tools.py — API 도구: api_query

서울시 상권분석서비스 REST API를 런타임에 호출하여 CSV에 없는 데이터를 조회한다.
CSV 파이프라인을 거치지 않고 gathered_data에 직접 합류한다.

반환값 구조:
    {
        "source": str,              # "api_query:{api_key}" 형태
        "summary": str,             # 조회 결과 요약
        "data": list[dict],         # API 응답 row 리스트
        "relevance": str,           # "relevant" 또는 "irrelevant"
        "relevance_reason": str,    # 관련성 판단 사유
    }

지원하는 API (5종):
    - estimated_sales: 추정매출 (OA-15572, VwsmTrdarSelngQq)
    - commercial_change: 상권변화지표 (OA-15576, VwsmTrdarIxQq)
    - store_openclose: 점포 개폐업 (OA-15577, VwsmTrdarStorQq)
    - crowd_facility: 집객시설 (OA-15580, VwsmTrdarFcltyQq)
    - resident_population: 상주인구 (OA-15584, VwsmTrdarRepopQq)
"""
import logging
import os
import time

import requests
from dotenv import load_dotenv

from agent.config_loader import get_token_budget

load_dotenv()

logger = logging.getLogger(__name__)

# --- API 인증키 ---
API_KEY = os.getenv("OPEN_GOV_API_KEY", "")

# --- API 레지스트리 ---
# 각 API의 서비스명, 선택 파라미터, 설명을 정의한다.
# URL 패턴: http://openapi.seoul.go.kr:8088/{KEY}/json/{SERVICE}/{START}/{END}/{OPTIONAL}
API_REGISTRY: dict[str, dict] = {
    "estimated_sales": {
        "service": "VwsmTrdarSelngQq",          # 추정매출-상권
        "optional_params": ["STDR_YYQU_CD"],    # 기준 년분기 코드 (예: 20241)
        "description": "상권별 추정 매출 (금액/건수, 요일별, 시간대별, 성별, 연령대별)",
    },
    "commercial_change": {
        "service": "VwsmTrdarIxQq",             # 상권변화지표-상권
        "optional_params": ["STDR_YYQU_CD"],    # 기준 년분기 코드
        "description": "상권 변화 지표 (확장/축소/정체, 개폐업 영업개월 평균)",
    },
    "store_openclose": {
        "service": "VwsmTrdarStorQq",           # 점포-상권
        "optional_params": ["STDR_YYQU_CD", "TRDAR_CD"],  # 분기 코드, 상권 코드
        "description": "상권별 점포 수, 개업률/폐업률, 프랜차이즈 수",
    },
    "crowd_facility": {
        "service": "VwsmTrdarFcltyQq",          # 집객시설-상권
        "optional_params": [],
        "description": "상권별 집객시설 (관공서, 병원, 학교, 지하철역, 버스정류장 등)",
    },
    "resident_population": {
        "service": "VwsmTrdarRepopQq",          # 상주인구-상권
        "optional_params": [],
        "description": "상권별 상주인구 (성별, 연령대별, 세대수)",
    },
}

# --- 세션 내 캐시 ---
# 동일 (api_key, params) 조합은 세션 내 1회만 호출한다.
_api_cache: dict[str, dict] = {}

# --- 요청 설정 ---
MAX_ROWS_PER_REQUEST = 1000   # 서울시 API 최대 요청 건수
REQUEST_TIMEOUT = 30          # 요청 타임아웃 (초)
MAX_RETRIES = 2               # 재시도 횟수
RETRY_DELAY = 1.0             # 재시도 대기 (초)

BASE_URL = "http://openapi.seoul.go.kr:8088"


def _build_url(service: str, start: int, end: int, optional_values: list[str] | None = None) -> str:
    """서울시 Open API URL을 구성한다.

    Args:
        service: 서비스명 (예: VwsmTrdarIxQq)
        start: 요청 시작 위치
        end: 요청 종료 위치
        optional_values: 선택 파라미터 값 리스트 (URL 경로에 순서대로 추가)

    Returns:
        완성된 API URL 문자열
    """
    url = f"{BASE_URL}/{API_KEY}/json/{service}/{start}/{end}"
    if optional_values:
        for val in optional_values:
            if val:
                url += f"/{val}"
    return url


def _call_api(service: str, start: int = 1, end: int = MAX_ROWS_PER_REQUEST,
              optional_values: list[str] | None = None) -> dict:
    """서울시 Open API를 호출하고 JSON 응답을 반환한다.

    Args:
        service: 서비스명
        start: 시작 위치 (기본 1)
        end: 종료 위치 (기본 1000)
        optional_values: 선택 파라미터 값 리스트

    Returns:
        파싱된 JSON 응답 딕셔너리

    Raises:
        requests.RequestException: HTTP 요청 실패
        ValueError: API 에러 응답 (INFO-100, ERROR-* 등)
    """
    url = _build_url(service, start, end, optional_values)

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            # 서울시 API 응답 구조: {서비스명: {list_total_count, RESULT, row}}
            service_data = data.get(service, {})
            result_info = service_data.get("RESULT", {})
            code = result_info.get("CODE", "")

            if code == "INFO-000":
                return service_data
            elif code == "INFO-200":
                # 해당하는 데이터 없음
                return {"list_total_count": 0, "row": []}
            else:
                raise ValueError(
                    f"API 에러: {code} - {result_info.get('MESSAGE', '알 수 없는 에러')}"
                )

        except (requests.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES:
                logger.warning("API 호출 재시도 %d/%d: %s", attempt + 1, MAX_RETRIES, e)
                time.sleep(RETRY_DELAY)
            else:
                raise


def _make_cache_key(api_key: str, params: dict) -> str:
    """캐시 키를 생성한다."""
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v)
    return f"{api_key}:{param_str}"


def api_query(query_analysis: dict, state: dict) -> dict:
    """서울시 상권분석서비스 API를 호출하여 데이터를 조회한다.

    query_analysis의 api_params에서 호출할 API와 파라미터를 가져온다.

    api_params 형식:
        {
            "api": "commercial_change",           # API 레지스트리 키
            "params": {                           # 선택 파라미터
                "STDR_YYQU_CD": "20244",          # 기준 분기 코드
                "TRDAR_CD": "3110017"             # 상권 코드 (일부 API만)
            }
        }

    api_params가 없으면 intent 기반으로 적절한 API를 추론한다.

    Args:
        query_analysis: 질의 분석 결과. api_params 또는 intent를 참조.
        state: 현재 AgentState (이 도구에서는 사용하지 않음).

    Returns:
        표준 도구 반환 형식. data에 API 응답 row 리스트 포함.
    """
    if not API_KEY:
        return {
            "source": "api_query",
            "summary": "API 키가 설정되지 않음 (OPEN_GOV_API_KEY)",
            "data": [],
            "relevance": "irrelevant",
            "relevance_reason": "API 키 미설정",
        }

    # --- API 파라미터 결정 ---
    api_params = query_analysis.get("api_params", {})
    api_key = api_params.get("api", "")
    params = api_params.get("params", {})

    # api_params가 없으면 intent 기반 추론
    if not api_key:
        api_key = _infer_api_from_intent(query_analysis.get("intent", ""))

    if not api_key or api_key not in API_REGISTRY:
        return {
            "source": "api_query",
            "summary": f"지원하지 않는 API: '{api_key}'",
            "data": [],
            "relevance": "irrelevant",
            "relevance_reason": f"API '{api_key}'가 레지스트리에 없음",
        }

    # --- 캐시 확인 ---
    cache_key = _make_cache_key(api_key, params)
    if cache_key in _api_cache:
        cached = _api_cache[cache_key]
        return {
            "source": f"api_query:{api_key} (cached)",
            "summary": cached["summary"],
            "data": cached["data"],
            "relevance": "relevant",
            "relevance_reason": f"캐시 히트: {api_key}",
        }

    # --- API 호출 ---
    registry = API_REGISTRY[api_key]
    service = registry["service"]

    # 선택 파라미터를 URL 경로 순서에 맞게 정렬
    optional_values = []
    for param_name in registry["optional_params"]:
        optional_values.append(params.get(param_name, ""))

    try:
        response_data = _call_api(
            service=service,
            optional_values=optional_values if any(optional_values) else None,
        )

        rows = response_data.get("row", [])
        total_count = response_data.get("list_total_count", len(rows))

        # 결과 행수 제한: 토큰 초과 방지 (agent_config.yaml의 token_budget 참조)
        max_rows = get_token_budget()["api_max_rows"]
        if len(rows) > max_rows:
            rows = rows[:max_rows]

        summary = _build_summary(api_key, rows, total_count, params)

        result = {
            "source": f"api_query:{api_key}",
            "summary": summary,
            "data": rows,
            "relevance": "relevant",
            "relevance_reason": f"API 직접 조회: {registry['description']}",
        }

        # 캐시 저장
        _api_cache[cache_key] = {"summary": summary, "data": rows}

        return result

    except Exception as e:
        logger.error("API 호출 실패 (%s): %s", api_key, e)
        return {
            "source": f"api_query:{api_key}",
            "summary": f"API 호출 실패: {str(e)[:200]}",
            "data": [],
            "relevance": "irrelevant",
            "relevance_reason": f"API 에러: {str(e)[:100]}",
        }


def _infer_api_from_intent(intent: str) -> str:
    """사용자 의도에서 적절한 API를 추론한다.

    Args:
        intent: analyze_query가 판단한 사용자 의도.

    Returns:
        API 레지스트리 키. 추론 불가 시 빈 문자열.
    """
    # 의도 키워드 → API 매핑
    intent_lower = intent.lower()
    keyword_map = {
        "commercial_change": ["변화", "성장", "쇠퇴", "추세", "트렌드", "change"],
        "estimated_sales": ["매출", "sales", "revenue"],
        "store_openclose": ["개업", "폐업", "개폐", "창업률", "폐업률", "open", "close"],
        "crowd_facility": ["시설", "학교", "병원", "지하철", "facility"],
        "resident_population": ["인구", "주민", "거주", "population", "resident"],
    }

    for api_key, keywords in keyword_map.items():
        if any(kw in intent_lower for kw in keywords):
            return api_key

    return ""


def _build_summary(api_key: str, rows: list[dict], total_count: int, params: dict) -> str:
    """API 응답을 요약 문자열로 변환한다.

    Args:
        api_key: API 레지스트리 키
        rows: API 응답 row 리스트
        total_count: 전체 데이터 건수
        params: 요청 파라미터

    Returns:
        한국어 요약 문자열
    """
    param_desc = ", ".join(f"{k}={v}" for k, v in params.items() if v)
    base = f"{API_REGISTRY[api_key]['description']}"

    if param_desc:
        base += f" (조건: {param_desc})"

    if not rows:
        return f"{base} — 조회 결과 없음"

    row_count = len(rows)
    if total_count > row_count:
        return f"{base} — {total_count}건 중 {row_count}건 조회"
    return f"{base} — {row_count}건 조회"


def clear_api_cache() -> None:
    """세션 내 API 캐시를 초기화한다. 새 세션 시작 시 호출."""
    _api_cache.clear()


def get_available_apis() -> dict[str, str]:
    """사용 가능한 API 목록과 설명을 반환한다.

    Returns:
        {api_key: description} 딕셔너리
    """
    return {k: v["description"] for k, v in API_REGISTRY.items()}
