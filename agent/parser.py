"""
agent/parser.py — LLM 응답에서 JSON을 추출하고 Pydantic 모델로 파싱

이 모듈은 LLM이 반환하는 텍스트 응답에서 JSON 데이터를 추출하여
Pydantic 모델 인스턴스로 변환하는 유틸리티를 제공한다.

문제 배경:
    LLM은 JSON만 반환하도록 프롬프트해도 가끔 마크다운 코드 블록(```json ... ```)으로
    감싸거나, 앞뒤에 설명 텍스트를 추가하는 경우가 있다. 이 모듈은 이런 불완전한
    출력을 정리하여 안정적으로 파싱한다.

사용 흐름:
    LLM 응답 텍스트 → parse_llm_json(text, ModelClass) → ModelClass 인스턴스
"""
import json
from pydantic import BaseModel


def parse_llm_json(text: str, model_class: type[BaseModel]) -> BaseModel:
    """LLM 응답에서 JSON을 추출하고 Pydantic 모델로 파싱한다.

    Args:
        text: LLM의 원본 응답 텍스트. 순수 JSON이거나 ```json 코드 블록으로 감싸져 있을 수 있다.
        model_class: 파싱 대상 Pydantic 모델 클래스 (예: QueryAnalysis, ContextEvaluation).

    Returns:
        model_class의 인스턴스. JSON 필드가 모델 속성에 매핑된다.

    Raises:
        ValueError: JSON 파싱 실패 또는 Pydantic 검증 실패 시.
                    에러 메시지에 원본 텍스트 앞 500자가 포함되어 디버깅에 도움이 된다.

    처리 과정:
        1. 앞뒤 공백 제거
        2. 마크다운 코드 블록(```) 제거 (첫 줄의 ```json과 마지막 줄의 ``` 삭제)
        3. json.loads()로 파싱
        4. Pydantic 모델로 검증 및 변환
    """
    # Step 1: 앞뒤 공백 제거
    cleaned = text.strip()

    # Step 2: 마크다운 코드 블록 제거
    # LLM이 ```json\n{...}\n```\n추가설명... 형태로 반환하는 경우를 처리한다.
    # 첫 번째 ```와 그에 대응하는 닫는 ``` 사이의 내용만 추출한다.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 첫 줄(```json) 제거
        lines = lines[1:]
        # 닫는 ``` 를 찾아서 그 이전까지만 사용 (뒤에 추가 텍스트가 있을 수 있음)
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "```":
                end_idx = i
                break
        cleaned = "\n".join(lines[:end_idx])

    # Step 2b: 코드 블록 없이 JSON 앞뒤에 텍스트가 있는 경우
    # 첫 번째 { 부터 마지막 } 까지만 추출한다.
    if not cleaned.startswith(("{", "[")):
        first_brace = cleaned.find("{")
        if first_brace == -1:
            first_brace = cleaned.find("[")
        if first_brace != -1:
            # 마지막 닫는 괄호를 찾는다
            last_brace = cleaned.rfind("}" if cleaned[first_brace] == "{" else "]")
            if last_brace != -1:
                cleaned = cleaned[first_brace:last_brace + 1]

    # Step 3-4: JSON 파싱 + Pydantic 모델 변환
    try:
        data = json.loads(cleaned)
        return model_class(**data)
    except (json.JSONDecodeError, Exception) as e:
        # 파싱 실패 시 원본 텍스트의 앞 500자를 포함한 에러 메시지를 반환한다.
        # 이를 통해 LLM이 어떤 잘못된 출력을 했는지 디버깅할 수 있다.
        raise ValueError(f"JSON 파싱 실패: {e}\n원본: {text[:500]}")
