"""scripts/replay_p0_5sessions.py — P0 보강 후 5세션 × 5턴 전수 재실행.

목적: agent/llm.py P0 변경(LLM trace GENERATION hook)이 운영 데이터에서
모든 process 노드 + 4D Judge에 부착되는지 5세션 × 5턴 = 25턴으로 검증.

5개 도메인 시나리오 (서울 상권 분석):
  1. 카페 창업
  2. 도시락/혼밥
  3. 미용실
  4. 편의점
  5. 한식 음식점

각 세션은 마지막 턴이 비교/종합 질문 — 이전 턴 결론 인용 패턴 유발.
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from main import load_config, run_session


SESSIONS = [
    {
        "name": "cafe",
        "queries": [
            "강남구 카페 시장 현황은 어떤가요?",
            "마포구 카페와 비교하면 어떻게 다른가요?",
            "임대료 부담을 고려할 때 어디가 유리한가요?",
            "유동인구와 객단가 측면에서 평가해주세요.",
            "위 분석을 종합해 추천 입지 3곳을 알려주세요.",
        ],
    },
    {
        "name": "dosirak",
        "queries": [
            "종로구 도시락/혼밥 업종 분포는 어떤가요?",
            "강남구와 비교하면 어떤 차이가 있나요?",
            "오피스 상권 시간대별 유동인구 패턴을 알려주세요.",
            "임대료와 인테리어 비용을 고려한 손익 분기는?",
            "이전 분석을 종합해 창업 우선순위를 정해주세요.",
        ],
    },
    {
        "name": "beauty",
        "queries": [
            "강남구 미용실 시장 규모와 성장 추이는?",
            "영등포구 미용실 시장과 비교해주세요.",
            "마포구 트렌드와 연결지어 분석해주세요.",
            "객단가와 재방문율 측면 평가는?",
            "위 비교를 종합한 입지 추천을 알려주세요.",
        ],
    },
    {
        "name": "convenience",
        "queries": [
            "서울 5개구 편의점 밀도 비교를 알려주세요.",
            "강남구가 다른 구와 어떻게 다른가요?",
            "종로구 편의점 매출 변화 추이는?",
            "영등포구 신규 진입 가능성 평가는?",
            "위 분석 기반 우선 진입 동을 추천해주세요.",
        ],
    },
    {
        "name": "korean_food",
        "queries": [
            "마포구 한식 음식점 시장 동향은?",
            "강남구 한식과 메뉴/가격대 차이를 비교해주세요.",
            "서초구 직장인 상권 한식 수요는?",
            "종로구 관광 상권의 한식 트렌드는?",
            "위 4개구 분석을 종합한 신규 진입 전략을 알려주세요.",
        ],
    },
]


def main() -> None:
    config = load_config()
    print("=" * 60)
    print(f"P0 5-session replay — {len(SESSIONS)} 세션 × 5턴 = {len(SESSIONS)*5} 턴")
    print("=" * 60)
    completed = []
    for i, sess in enumerate(SESSIONS, 1):
        print(f"\n[{i}/{len(SESSIONS)}] 세션: {sess['name']}")
        print("-" * 60)
        t0 = time.time()
        result = run_session(sess["queries"], config=config)
        sid = result.get("session_id")
        completed.append((sess["name"], sid))
        elapsed = time.time() - t0
        print(f"\n✅ {sess['name']} 완료: {sid} ({elapsed/60:.1f}분)")

        # 세션 간 휴식 (rate limit 회피)
        if i < len(SESSIONS):
            print("   다음 세션 전 30초 대기...")
            time.sleep(30)

    print()
    print("=" * 60)
    print("전체 완료 — 새 세션 ID:")
    for name, sid in completed:
        print(f"  {name}: {sid}")
    print("=" * 60)


if __name__ == "__main__":
    main()
