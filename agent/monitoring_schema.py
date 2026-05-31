"""
agent/monitoring_schema.py — 모니터링 속성 스키마 로더 (v2/v3 호환)

역할:
    config/monitoring_schema.yaml을 로드하여 모니터링 속성의 단일 진실 원본을 제공한다.
    v2(평면 namespace 구조)와 v3(core/domain/partial 분할 + 8축 메타) 자동 감지.
    모듈 임포트 시 YAML을 한 번 로드하고, 이후에는 메모리에서 참조한다.

데이터 흐름:
    입력: config/monitoring_schema.yaml (v2 또는 v3)
    출력: ATTRS, ATTR_META, THRESHOLDS, CORE/DOMAIN/PARTIAL 분류 상수

제공하는 상수/함수:
    ATTRS                                — 속성 이름 → 속성 이름 (KeyError 기반 오타 방지)
    ATTR_META                            — 속성 이름 → 메타 (v2 호환 키: type, lifecycle, quality, producer, judge_input, computation)
    THRESHOLDS                           — 평가 차원 → 임계값
    CONTEXT_WINDOW_MAX_TOKENS            — 컨텍스트 윈도우 최대 토큰
    ROT_GATE_THRESHOLD                   — Rot Gate 트리거 임계값
    CUSTOMIZABLE_THRESHOLDS              — 도메인별 customization 가능 임계값 (v3 only)
    CORE_ATTRS / DOMAIN_ATTRS / PARTIAL_ATTRS  — 도메인 의존도 분류 (v3)
    CROSS_MODEL_EVALUATORS               — judge → 별도 모델 매핑 (v3, self-referential 차단)
    CONSISTENCY_PATTERN_OWNERS           — pattern A/B/C/D → attr 리스트 (v3)
    SCHEMA_VERSION                       — "v2" 또는 "v3"

함수:
    get_attr(name)                       — 알 수 없는 이름이면 KeyError
    validate_metadata(metadata)          — 스키마에 없는 키 발견 시 ValueError
    attrs_for_producer(producer)         — 특정 노드의 속성 목록
    get_judge_attributes(judge)          — judge 입력 속성 목록 (judge_input/judge_inputs 모두 인식)
    extract_judge_metadata(judge, meta)  — judge 입력만 추출
    is_rot_gate_enabled()                — 환경변수 기반 ON/OFF
    get_measurement_meta(name)           — v3 measurement 블록 (method/formula/oracle/...)
    get_tier(name)                       — Tier (1/2/3)
    get_domain_dependency(name)          — "agnostic"/"dependent"/"partial"
    get_core_attributes()                — Portable Core 속성 목록
    get_domain_attributes()              — 도메인 의존 속성 목록
    get_partial_attributes()             — 부분 의존 속성 목록
    get_consistency_pattern_owners(p)    — 일관성 패턴(A/B/C/D)별 책임 attr 목록
    get_customizable_threshold(key)      — 도메인 customizable 임계값 조회
    get_cross_model_evaluator(judge)     — judge별 cross-model 평가자 모델명 (없으면 None)
"""
import os
from typing import Optional

import yaml

# --- YAML 파일 로드 ---
# 환경변수 MONITORING_SCHEMA_PATH로 다른 yaml을 지정할 수 있다 (마이그레이션 검증용).
_SCHEMA_PATH = os.environ.get(
    "MONITORING_SCHEMA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "config", "monitoring_schema.yaml"),
)

with open(_SCHEMA_PATH, encoding="utf-8") as _f:
    _SCHEMA = yaml.safe_load(_f)


# --- 스키마 버전 감지 ---
# v3는 core_attributes / domain_attributes / partial_dependent_attributes 분할 구조
# v2는 attributes (namespace 평면 구조)
def _detect_version(schema: dict) -> str:
    """yaml 최상위 키로 v2/v3를 자동 감지한다."""
    if "core_attributes" in schema:
        return "v3"
    if "attributes" in schema:
        return "v2"
    raise ValueError("monitoring_schema.yaml: 'core_attributes'(v3) 또는 'attributes'(v2) 키가 필요합니다.")


SCHEMA_VERSION: str = _detect_version(_SCHEMA)


# --- THRESHOLDS / 컨텍스트 윈도우 (공통) ---
# v3는 constants.* 하위로 옮겨졌고, v2는 evaluation.thresholds + context_window
def _get_thresholds(schema: dict) -> dict[str, float]:
    if SCHEMA_VERSION == "v3":
        return schema.get("constants", {}).get("thresholds", {})
    return schema["evaluation"]["thresholds"]


def _get_context_window(schema: dict) -> dict:
    if SCHEMA_VERSION == "v3":
        return schema.get("constants", {}).get("context_window", {})
    return schema.get("context_window", {})


THRESHOLDS: dict[str, float] = _get_thresholds(_SCHEMA)
_CW = _get_context_window(_SCHEMA)
CONTEXT_WINDOW_MAX_TOKENS: int = _CW.get("max_tokens", 180000)
ROT_GATE_THRESHOLD: float = _CW.get("rot_gate_threshold", 0.3)

# --- 이탈 감지 임계값 (Tab 6, Tab 8 SSOT) ---
DRIFT_CONTINUITY_THRESHOLD: float = 0.5   # query.session_continuity 미만 → User Pivot
DRIFT_ALIGNMENT_THRESHOLD: float = 0.7    # analysis/response.query_alignment 미만 → Agent Drift


# --- v3 전용 상수 (v2일 때는 빈 값) ---
CUSTOMIZABLE_THRESHOLDS: dict[str, float] = (
    _SCHEMA.get("constants", {}).get("customizable_thresholds", {})
    if SCHEMA_VERSION == "v3"
    else {}
)
# customizable_thresholds 안의 메타 플래그 제거
CUSTOMIZABLE_THRESHOLDS = {
    k: v for k, v in CUSTOMIZABLE_THRESHOLDS.items() if not isinstance(v, bool)
}

# --- 대시보드 UI 임계값 (Tab 3/5 SSOT) ---
# constants.dashboard_thresholds를 노출. v2일 때는 기본값 fallback.
_DASHBOARD_DEFAULTS: dict[str, float] = {
    "previous_turns_warn": 0.25,
    "previous_turns_danger": 0.50,
    "window_optimal_min": 0.40,
    "window_optimal_max": 0.85,
    "noise_warn": 0.50,
    "redundancy_warn": 0.85,
    # Tab 4 변형 — fidelity 색상 분기 + 여정 표 new_data_ratio 색상 분기
    "fidelity_good": 0.80,
    "fidelity_warn": 0.50,
    "new_data_good": 0.30,
    "new_data_warn": 0.15,
    # Tab 5 전달 — 여정 표 색상 분기 (window는 "전달 부담" 관점, Tab 3와 의미 다름)
    "window_high_warn": 0.60,
    "window_high_danger": 0.80,
    "density_good": 0.50,
    "density_warn": 0.20,
    "noise_good": 0.30,
    "grounded_good": 0.70,
    "grounded_warn": 0.50,
}
DASHBOARD_THRESHOLDS: dict[str, float] = (
    {**_DASHBOARD_DEFAULTS,
     **_SCHEMA.get("constants", {}).get("dashboard_thresholds", {})}
    if SCHEMA_VERSION == "v3"
    else dict(_DASHBOARD_DEFAULTS)
)

_FIDELITY_WEIGHTS_DEFAULT: dict[str, float] = {
    "cond_score": 0.4, "claims_ratio": 0.3, "compression_penalty": 0.3,
}
FIDELITY_SCORE_WEIGHTS: dict[str, float] = (
    _SCHEMA.get("constants", {}).get("fidelity_score_weights", _FIDELITY_WEIGHTS_DEFAULT)
    if SCHEMA_VERSION == "v3"
    else _FIDELITY_WEIGHTS_DEFAULT
)

CROSS_MODEL_EVALUATORS: dict[str, str] = {}
if SCHEMA_VERSION == "v3":
    for _judge, _spec in _SCHEMA.get("cross_model_evaluators", {}).items():
        if isinstance(_spec, dict) and "model" in _spec:
            CROSS_MODEL_EVALUATORS[_judge] = _spec["model"]


def is_rot_gate_enabled() -> bool:
    """환경변수 ROT_GATE_ENABLED를 런타임에 확인하여 Rot Gate 활성화 여부를 반환한다."""
    return os.environ.get("ROT_GATE_ENABLED", "1").lower() not in ("0", "false", "off", "no")


# ═══════════════════════════════════════════════════════════════
# ATTRS / ATTR_META 빌드 — v2/v3 분기
# ═══════════════════════════════════════════════════════════════

ATTRS: dict[str, str] = {}
ATTR_META: dict[str, dict] = {}

# v3 전용 분류 상수
CORE_ATTRS: list[str] = []
DOMAIN_ATTRS: list[str] = []
PARTIAL_ATTRS: list[str] = []
CONSISTENCY_PATTERN_OWNERS: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}


def _normalize_v3_meta(meta: dict, domain_dep: str) -> dict:
    """v3 attribute 메타를 정규화하고 v2 호환 별칭을 추가한다.

    v3 → v2 호환 매핑:
        judge_inputs → judge_input (v2 코드 호환)
        quality_dimension → quality
        measurement.formula → computation

    Args:
        meta: v3 yaml에서 읽은 attribute 메타 dict.
        domain_dep: "agnostic"/"dependent"/"partial" — yaml 위치에서 도출.

    Returns:
        v3 키 + v2 호환 별칭이 모두 포함된 dict.
    """
    normalized = dict(meta)  # 원본 보존
    # v3 → v2 호환 별칭
    if "judge_inputs" in meta and "judge_input" not in meta:
        normalized["judge_input"] = meta["judge_inputs"]
    if "quality_dimension" in meta and "quality" not in meta:
        normalized["quality"] = meta["quality_dimension"]
    measurement = meta.get("measurement") or {}
    if "computation" not in normalized:
        normalized["computation"] = measurement.get("formula", "")
    # domain_dependency가 yaml에 명시 안 됐으면 위치에서 도출한 값 채움
    if "domain_dependency" not in normalized:
        normalized["domain_dependency"] = domain_dep
    return normalized


if SCHEMA_VERSION == "v3":
    # core / domain / partial 3개 섹션을 모두 ATTRS에 등록
    for _attr_name, _meta in (_SCHEMA.get("core_attributes") or {}).items():
        ATTRS[_attr_name] = _attr_name
        ATTR_META[_attr_name] = _normalize_v3_meta(_meta, "agnostic")
        CORE_ATTRS.append(_attr_name)
    for _attr_name, _meta in (_SCHEMA.get("domain_attributes") or {}).items():
        ATTRS[_attr_name] = _attr_name
        ATTR_META[_attr_name] = _normalize_v3_meta(_meta, "dependent")
        DOMAIN_ATTRS.append(_attr_name)
    for _attr_name, _meta in (_SCHEMA.get("partial_dependent_attributes") or {}).items():
        ATTRS[_attr_name] = _attr_name
        ATTR_META[_attr_name] = _normalize_v3_meta(_meta, "partial")
        PARTIAL_ATTRS.append(_attr_name)

    # Consistency 패턴 책임자 매핑
    # yaml 구조: consistency_patterns.{A,B,C,D}.{name, description, attributes: [...]}
    for _pattern, _spec in (_SCHEMA.get("consistency_patterns") or {}).items():
        if not isinstance(_pattern, str) or not _pattern:
            continue
        _key = _pattern[0].upper()
        if _key not in CONSISTENCY_PATTERN_OWNERS:
            continue
        if isinstance(_spec, dict):
            _attrs = _spec.get("attributes") or _spec.get("owners") or []
            if isinstance(_attrs, list):
                CONSISTENCY_PATTERN_OWNERS[_key].extend(_attrs)
        elif isinstance(_spec, list):
            CONSISTENCY_PATTERN_OWNERS[_key].extend(_spec)

else:
    # v2: 기존 평면 구조
    for _namespace, _attrs in _SCHEMA["attributes"].items():
        for _attr_name, _meta in _attrs.items():
            ATTRS[_attr_name] = _attr_name
            if "computation" not in _meta:
                _meta["computation"] = ""
            ATTR_META[_attr_name] = _meta


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def get_attr(name: str) -> str:
    """속성 이름을 반환한다. 알 수 없는 이름이면 KeyError를 발생시킨다."""
    return ATTRS[name]


def validate_metadata(metadata: dict) -> None:
    """메타데이터 딕셔너리의 키가 모두 스키마에 정의되어 있는지 검증한다.

    "tags" 키는 Langfuse 전용이므로 검증에서 제외한다.

    Raises:
        ValueError: 스키마에 정의되지 않은 키가 발견된 경우.
    """
    unknown = {k for k in metadata if k != "tags" and k not in ATTRS}
    if unknown:
        raise ValueError(f"스키마에 정의되지 않은 모니터링 속성: {sorted(unknown)}")


def get_judge_attributes(judge_name: str) -> list[str]:
    """특정 judge가 사용하는 속성 이름 리스트를 반환한다.

    v2 (judge_input) / v3 (judge_inputs) 모두 인식한다 (호환 별칭 처리됨).

    Args:
        judge_name: judge 이름 (예: "completeness", "efficiency", "relevance", "consistency").
    """
    return [
        name for name, meta in ATTR_META.items()
        if judge_name in (meta.get("judge_input") or meta.get("judge_inputs") or [])
    ]


def extract_judge_metadata(judge_name: str, metadata: dict) -> dict:
    """메타데이터에서 특정 judge에 필요한 속성만 추출한다."""
    attrs = get_judge_attributes(judge_name)
    return {attr: metadata.get(attr) for attr in attrs if attr in metadata}


def attrs_for_producer(producer: str) -> list[str]:
    """특정 생산자(노드)의 속성 목록을 반환한다."""
    return [
        name for name, meta in ATTR_META.items()
        if meta.get("producer") == producer
    ]


# --- v3 전용 API (v2일 때는 합리적 기본값 반환) ---

def get_measurement_meta(name: str) -> dict:
    """v3 measurement 블록을 반환한다 (method, formula, independence, oracle 등).

    v2일 때는 빈 dict를 반환한다.
    """
    return ATTR_META.get(name, {}).get("measurement") or {}


def get_tier(name: str) -> Optional[int]:
    """attribute의 Tier를 반환한다 (v3 1/2/3, v2일 때는 None)."""
    return ATTR_META.get(name, {}).get("tier")


def get_domain_dependency(name: str) -> str:
    """도메인 의존도를 반환한다.

    Returns:
        "agnostic" / "dependent" / "partial" (v3)
        v2일 때는 "unknown".
    """
    return ATTR_META.get(name, {}).get("domain_dependency", "unknown")


def get_core_attributes() -> list[str]:
    """Portable Core 속성 목록을 반환한다 (v3 only, v2는 빈 리스트)."""
    return list(CORE_ATTRS)


def get_domain_attributes() -> list[str]:
    """도메인 의존 속성 목록을 반환한다."""
    return list(DOMAIN_ATTRS)


def get_partial_attributes() -> list[str]:
    """부분 의존 속성 목록을 반환한다."""
    return list(PARTIAL_ATTRS)


def get_consistency_pattern_owners(pattern: str) -> list[str]:
    """일관성 4패턴(A/B/C/D)별 책임 attribute 목록을 반환한다."""
    return list(CONSISTENCY_PATTERN_OWNERS.get(pattern.upper(), []))


def get_customizable_threshold(key: str, default: Optional[float] = None) -> Optional[float]:
    """도메인별 customization 가능 임계값을 조회한다.

    예시:
        numeric_diff_threshold (verify_result 수치 비교, 기본 0.05)
        redundancy_cosine_threshold (redundancy 임베딩 cosine 컷, 기본 0.85)
        conflict_detection_threshold (source 충돌 감지 컷, 기본 0.1)
    """
    return CUSTOMIZABLE_THRESHOLDS.get(key, default)


def get_cross_model_evaluator(judge: str) -> Optional[str]:
    """judge별 cross-model 평가자 모델명을 반환한다.

    self-referential 차단을 위해 다른 모델로 평가해야 하는 경우 사용한다.
    예: verify.interpretation_score → claude-sonnet-4-6
    """
    return CROSS_MODEL_EVALUATORS.get(judge)


# ═══════════════════════════════════════════════════════════════
# Trace 호환 helper (v2/v3 양립)
# ═══════════════════════════════════════════════════════════════
# v3 통합 (2026-04-29): analysis.conflict_tracking dict가 분리 3 attribute를 흡수.
# dashboard/evaluation은 과거 trace(v2 분리 키)와 미래 trace(v3 dict) 양쪽을 graceful 처리.

def get_contradicts_from_metadata(metadata: dict) -> Optional[bool]:
    """trace metadata에서 "이전 턴과 모순 여부"를 반환한다 (v2/v3 양립)."""
    v = metadata.get("analysis.contradicts_previous")
    if v is not None:
        return v
    ct = metadata.get("analysis.conflict_tracking")
    if isinstance(ct, dict):
        return ct.get("detected")
    return None


def get_contradiction_resolved_from_metadata(metadata: dict) -> Optional[bool]:
    """trace metadata에서 "모순 해결 여부"를 반환한다 (v2/v3 양립)."""
    v = metadata.get("analysis.contradiction_resolved")
    if v is not None:
        return v
    ct = metadata.get("analysis.conflict_tracking")
    if isinstance(ct, dict):
        return (ct.get("resolution") or {}).get("has_explanation")
    return None


def get_previous_conclusion_from_metadata(metadata: dict) -> Optional[str]:
    """trace metadata에서 "모순된 이전 결론 요약"을 반환한다 (v2/v3 양립)."""
    v = metadata.get("analysis.previous_conclusion")
    if v:
        return v
    ct = metadata.get("analysis.conflict_tracking")
    if isinstance(ct, dict):
        return (ct.get("resolution") or {}).get("conflict_summary")
    return None
