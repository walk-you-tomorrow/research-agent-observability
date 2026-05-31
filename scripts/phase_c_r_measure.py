"""
scripts/phase_c_r_measure.py — Phase C 옵션 A 2단계: r 측정용 5턴 세션 실행

도메인별 5턴 시나리오 1개를 인자로 받아 실행. 4개 백그라운드 병렬 실행으로
sess_f55d9481(이미 완료)와 합쳐 총 n=25 (5세션 × 5턴) 데이터 확보.

사용:
    python scripts/phase_c_r_measure.py --domain mapo
    python scripts/phase_c_r_measure.py --domain jongno
    python scripts/phase_c_r_measure.py --domain seocho
    python scripts/phase_c_r_measure.py --domain ydp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_session, load_config

SCENARIOS = {
    "mapo": [
        "마포구 합정동에서 음식점 창업하기 좋은 위치는 어디인가?",
        "한식과 양식 중 어떤 업종이 유리한가?",
        "유동인구와 매출 추세는 어떤가?",
        "임대료 부담은 다른 동과 비교해 어떤가?",
        "지금까지 분석을 종합하면 추천 업종과 위치는?",
    ],
    "jongno": [
        "종로구 도시락/혼밥 수요가 높은 지역은 어디인가?",
        "오피스 밀집 지역의 점심 매출 패턴은?",
        "유사 업종의 폐업률은 어떻게 되나?",
        "임대료와 인테리어 비용을 고려한 손익 분기는?",
        "종합 추천 입지와 주의사항은?",
    ],
    "seocho": [
        "서초구에서 교육 관련 창업이 유리한 동은?",
        "학원 대비 스터디카페의 수요는 어떤가?",
        "주말과 평일 유동인구 차이는?",
        "임대료와 학부모 동선을 고려한 입지 조건은?",
        "종합적으로 어디를 추천하는가?",
    ],
    "ydp": [
        "영등포구에서 헬스장/필라테스 창업 적합 지역은?",
        "오피스 직장인 vs 주거 지역 수요 차이는?",
        "유사 업종 밀집도와 경쟁 강도는?",
        "임대료 대비 회원 수 손익 분기는?",
        "종합 분석으로 추천 입지는?",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(SCENARIOS.keys()))
    args = parser.parse_args()

    queries = SCENARIOS[args.domain]
    print(f"[{args.domain}] 5턴 시나리오 시작")
    config = load_config()
    final_state = run_session(queries, config=config)
    print(f"[{args.domain}] 완료: turn={final_state.get('current_turn')}")


if __name__ == "__main__":
    main()
