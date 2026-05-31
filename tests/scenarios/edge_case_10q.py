#!/usr/bin/env python3
"""
tests/scenarios/edge_case_10q.py — 컨텍스트 모니터링 엣지 케이스 10문항 테스트

역할:
    10가지 질문을 단일 세션에서 순서대로 실행하여
    re-gather, re-verify, 멀티소스, 교차 턴 일관성, 토큰 효율성 등을 테스트한다.

데이터 흐름:
    입력: 10개 질문 리스트
    출력: 세션 결과 (session_id, 턴별 요약)
"""
import os
import sys

# 프로젝트 루트로 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from main import run_session, load_config

QUERIES = [
    # Q1: Baseline — 정상 흐름
    "강남구 음식점 수 알려줘",
    # Q2: 멀티소스 (CSV + RAG + Web)
    "마포구 홍대 상권의 최근 트렌드와 임대료 변화 추이를 분석해줘",
    # Q3: Re-gather 유발 (불충분한 데이터)
    "강남구와 서초구의 업종별 폐업률을 비교하고, 각 구의 상위 5개 폐업 업종의 공통점을 분석해줘",
    # Q4: 존재하지 않는 데이터 (graceful degradation)
    "성북구 장위동 카페 임대료 알려줘",
    # Q5: 수치 검증 (Re-verify 가능)
    "종로구 전체 상가 수와 업종 비율을 계산해서, 소매업이 차지하는 비중을 퍼센트로 알려줘",
    # Q6: 교차 턴 모순 탐지 (Q1 도전)
    "아까 강남구 음식점 수가 틀린 것 같은데, 실제로는 훨씬 적지 않아?",
    # Q7: 거짓 모순 방지 (Q1 부분집합)
    "그러면 강남구에서 한식당만 몇 개야?",
    # Q8: 소스 간 충돌 (CSV vs RAG)
    "영등포구 여의도동 상권 현황을 CSV 데이터와 상권분석보고서 내용으로 비교해줘",
    # Q9: 토큰 효율성 압박
    "지금까지 분석한 모든 구의 결과를 종합해서, 5개 구(강남/마포/서초/종로/영등포) 전체의 업종 분포, 유동인구, 임대료를 한 번에 비교 분석해줘",
    # Q10: 모호한 질의
    "여기서 장사하면 괜찮을까?",
]


def main():
    config = load_config()
    print(f"=== 엣지 케이스 10문항 테스트 시작 ({len(QUERIES)}개 질문) ===\n")

    session_state = run_session(QUERIES, config=config)

    print("\n" + "=" * 60)
    print(f"SESSION ID: {session_state['session_id']}")
    print(f"총 턴 수: {len(session_state.get('turn_conclusions', []))}")
    print("=" * 60)

    for i, result in enumerate(session_state.get("turn_results", []), 1):
        ctx = result.get("context_evaluation") or {}
        ver = result.get("verification") or {}
        print(f"\nTurn {i}: {QUERIES[i-1][:40]}...")
        print(f"  gather_iteration: {result.get('gather_iteration', '?')}")
        print(f"  is_sufficient: {ctx.get('is_sufficient', '?')}")
        print(f"  confidence: {ctx.get('confidence_score', ctx.get('sufficiency_confidence', '?'))}")
        print(f"  verify_verdict: {ver.get('overall_verdict', '?')}")
        print(f"  contradicts_previous: {result.get('contradicts_previous', '?')}")
        resp = result.get("response", "")
        print(f"  response: {resp[:80]}..." if len(resp) > 80 else f"  response: {resp}")


if __name__ == "__main__":
    main()
