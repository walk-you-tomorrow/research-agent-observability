"""
agent/token_counter.py — tiktoken 기반 토큰 카운터

이 모듈은 텍스트의 토큰 수를 정확하게 계산하는 유틸리티를 제공한다.
Context Monitoring에서 컨텍스트 윈도우 사용률(window_utilization)과
소스별 토큰 분배를 측정하는 데 핵심적으로 사용된다.

사용하는 곳:
    - evaluate_context 노드: 시스템 프롬프트, 수집 데이터, 이전 턴 등의 토큰 수 측정
    - gather_data 노드: 수집된 각 항목의 토큰 수 기록

인코딩:
    cl100k_base 인코딩을 사용한다. 이는 GPT-4 / Claude 계열 모델의
    토크나이저와 유사한 결과를 제공한다.
    (정확한 Claude 토크나이저는 공개되지 않았으나, cl100k_base가 근사치로 충분하다)
"""
import tiktoken

# cl100k_base 인코딩을 한 번만 로드하여 모듈 레벨에서 캐시한다.
# 매번 get_encoding()을 호출하면 불필요한 오버헤드가 발생한다.
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 계산하여 반환한다.

    Args:
        text: 토큰 수를 계산할 텍스트. 빈 문자열이나 None-like 값도 안전하게 처리한다.

    Returns:
        토큰 수 (int). 빈 입력 시 0을 반환한다.

    예시:
        count_tokens("서울에서 카페 창업") → 약 8 (인코딩에 따라 다름)
        count_tokens("") → 0
    """
    if not text:
        return 0
    return len(_enc.encode(text))
