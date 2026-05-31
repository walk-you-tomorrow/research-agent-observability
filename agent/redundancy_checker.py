"""
agent/redundancy_checker.py — 의미적 중복 측정 (v3 REDEFINE 2026-04-29)

★ 4D 효율성(Efficiency) — 자원 낭비 측정 ★

역할:
    gathered_data 항목 간 의미적 중복도를 측정한다.
    Ollama 가용 시 nomic-embed-text 임베딩 + cosine similarity,
    실패 시 lexical Jaccard fallback (회귀 안전망).

데이터 흐름:
    입력: list[dict] (gathered_data, data_summary 필드 필요)
    출력: float in [0.0, 1.0] — cosine ≥ threshold 페어 / 전체 페어

설계 결정:
    - cosine_threshold는 customizable_thresholds.redundancy_cosine_threshold (기본 0.85)
    - 임베딩 호출 실패 시 lexical fallback — 단위 테스트와 외부 의존 분리.
    - 페어 수가 0이면 0.0 (항목 < 2).

REDEFINE 동기 (analysis/33):
    AS-IS: Jaccard 단어 set 교집합 — "동의어/번역" 무시 (의미 중복 놓침)
    TO-BE: 임베딩 cosine — 의미적 중복 포착, 도메인 무관 (모델 교체로 다국어/도메인 customization)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def compute_redundancy_ratio(
    items: list[dict],
    cosine_threshold: float = 0.85,
) -> float:
    """gathered_data 항목 간 의미적 중복 비율을 측정한다.

    Args:
        items: gathered_data 리스트. 각 항목은 data_summary 필드를 가져야 한다.
        cosine_threshold: 중복 판정 cosine similarity 컷 (yaml customizable_thresholds 기본 0.85).

    Returns:
        cosine ≥ threshold 페어 / 전체 페어. 항목 수 < 2이면 0.0.
        임베딩 실패 시 lexical Jaccard로 fallback (값 자체는 의미가 다르지만 회귀 차단).
    """
    if not items or len(items) < 2:
        return 0.0

    texts = [item.get("data_summary", "") for item in items if item.get("data_summary")]
    if len(texts) < 2:
        return 0.0

    embeddings = _try_embed(texts)
    if embeddings is not None:
        return _pairwise_cosine_redundancy(embeddings, cosine_threshold)

    # 외부 임베딩 미가용 시 lexical fallback (회귀 안전망).
    return _lexical_redundancy(texts)


# ═══════════════════════════════════════════════════════════════
# 내부 헬퍼
# ═══════════════════════════════════════════════════════════════

def _try_embed(texts: list[str]) -> Optional[list[list[float]]]:
    """Ollama 임베딩을 시도한다. 실패 시 None.

    LightRAG 공유 event loop를 우회하여 requests로 직접 HTTP 호출한다.
    배치 실행 중 LightRAG KG 쿼리와 Ollama 경합을 방지하기 위한 독립 경로.
    Ollama 미가용, 타임아웃(15s/텍스트), 네트워크 오류 시 None 반환.
    """
    try:
        import requests
        import yaml
        from pathlib import Path

        config_path = Path(__file__).resolve().parent.parent / "config" / "agent_config.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        embed_cfg = config.get("lightrag", {}).get("embedding", {})
        model = embed_cfg.get("model", "nomic-embed-text:latest")
        host = embed_cfg.get("host", "http://localhost:11434")

        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{host}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=15,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings
    except Exception as exc:
        logger.warning(f"[redundancy_checker] 임베딩 fallback to lexical: {exc}")
        return None


def _pairwise_cosine_redundancy(
    embeddings: list[list[float]],
    threshold: float,
) -> float:
    """페어별 cosine similarity 계산 후 threshold 이상 비율을 반환한다."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    total_pairs = 0
    redundant_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(embeddings[i], embeddings[j])
            total_pairs += 1
            if sim >= threshold:
                redundant_pairs += 1
    return round(redundant_pairs / max(total_pairs, 1), 3)


def _cosine(a: list[float], b: list[float]) -> float:
    """두 벡터의 cosine similarity. 0벡터는 0 반환."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical_redundancy(texts: list[str]) -> float:
    """Jaccard 기반 lexical fallback (구공식, 회귀 안전망).

    단어 길이 ≤ 2인 stop-like 토큰은 무시한다.
    """
    keyword_sets: list[set[str]] = [
        {w for w in text.lower().split() if len(w) > 2}
        for text in texts
    ]
    keyword_sets = [k for k in keyword_sets if k]
    if len(keyword_sets) < 2:
        return 0.0
    total_overlap = 0
    total_words = 0
    for i, kw in enumerate(keyword_sets):
        others = set().union(
            *(keyword_sets[j] for j in range(len(keyword_sets)) if j != i)
        )
        total_overlap += len(kw & others)
        total_words += len(kw)
    return round(total_overlap / max(total_words, 1), 3)
