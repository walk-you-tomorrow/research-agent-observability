"""scripts/collect_phase38_data.py — Phase 3.8 데이터 수집 배치 러너

이탈 감지 3개 속성(query.session_continuity, analysis.query_alignment,
response.query_alignment)의 임계값 보정을 위해 다양한 패턴의 세션을 실행한다.

목표: ≥30세션. 3가지 패턴 균등 분포:
    - HIGH continuity: 주제 일관 유지
    - PIVOT: 세션 중간 주제 전환
    - DRIFT: 점진적 주제 이동

실행:
    source venv/bin/activate
    python scripts/collect_phase38_data.py [--sessions N] [--dry-run]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import truststore
truststore.inject_into_ssl()

# --- 쿼리셋 정의 ---

# 패턴 1: HIGH continuity — 동일 주제 지속 (session_continuity 0.7+ 기대)
HIGH_CONTINUITY_SESSIONS = [
    {
        "label": "강남_카페_연속",
        "queries": [
            "강남구 카페 시장 현황과 최근 트렌드를 알려주세요.",
            "강남구 카페 창업 시 예상 임대료와 보증금은 얼마인가요?",
            "강남구 카페 경쟁 강도와 폐업률 데이터를 알려주세요.",
            "강남구에서 카페 창업이 유리한 세부 상권은 어디인가요?",
        ],
    },
    {
        "label": "홍대_음식점_연속",
        "queries": [
            "홍대 인근 음식점 시장 특성을 분석해주세요.",
            "홍대 음식점의 평균 매출과 수익성은 어떻게 되나요?",
            "홍대 음식점 창업에 필요한 초기 투자금은 얼마인가요?",
        ],
    },
    {
        "label": "마포_뷰티_연속",
        "queries": [
            "마포구 뷰티 업종(미용실, 네일샵 등) 현황을 알려주세요.",
            "마포구 뷰티 업종의 임대료 수준은 어떻게 되나요?",
            "마포구에서 미용실 창업 시 성공 가능성은 어떻게 볼 수 있나요?",
            "마포구 뷰티 업종 중 수익성이 높은 세부 업종은 무엇인가요?",
        ],
    },
    {
        "label": "이태원_의류_연속",
        "queries": [
            "이태원 의류 상권의 특성과 타겟 고객층을 알려주세요.",
            "이태원 의류 매장의 평균 매출 규모는 어떻게 되나요?",
            "이태원 의류 창업 비용과 임대료 현황은 어떤가요?",
        ],
    },
    {
        "label": "종로_카페_연속",
        "queries": [
            "종로구 카페 상권의 특성은 무엇인가요?",
            "종로구 카페 창업 시 유동인구와 매출 예상치는 어떻게 되나요?",
            "종로구와 강남구 카페 시장을 비교 분석해주세요.",
            "종로구 카페 창업에 유리한 입지 조건은 무엇인가요?",
        ],
    },
    {
        "label": "강남_음식점_연속",
        "queries": [
            "강남구 음식점 업종별 현황을 알려주세요.",
            "강남구 음식점 평균 임대료와 권리금은 얼마인가요?",
            "강남구 음식점 폐업률과 생존율 데이터를 알려주세요.",
        ],
    },
    {
        "label": "서초_뷰티_연속",
        "queries": [
            "서초구 뷰티 상권 현황과 특성을 알려주세요.",
            "서초구 미용실 창업 비용은 어떻게 되나요?",
            "서초구 뷰티 업종의 경쟁 강도는 어느 정도인가요?",
            "서초구와 강남구 뷰티 상권 비교를 해주세요.",
        ],
    },
    {
        "label": "송파_카페_연속",
        "queries": [
            "송파구 카페 상권 현황을 분석해주세요.",
            "송파구 카페 창업에 필요한 초기 투자 비용은 얼마인가요?",
            "송파구 카페 유동인구와 매출 데이터를 알려주세요.",
        ],
    },
    {
        "label": "영등포_음식점_연속",
        "queries": [
            "영등포구 음식점 상권의 주요 특성을 알려주세요.",
            "영등포 타임스퀘어 인근 음식점 매출 현황은 어떤가요?",
            "영등포구 음식점 창업 시 유망 업종은 무엇인가요?",
            "영등포구 음식점 임대료 추이를 알려주세요.",
        ],
    },
    {
        "label": "신촌_카페_연속",
        "queries": [
            "신촌 카페 상권의 특성과 주요 타겟층은 누구인가요?",
            "신촌 카페 평균 매출과 임대료 수준은 어떻게 되나요?",
            "신촌 카페 창업 시 유의해야 할 사항은 무엇인가요?",
        ],
    },
]

# 패턴 2: PIVOT — 세션 중반 주제 전환 (session_continuity 급락 기대)
PIVOT_SESSIONS = [
    {
        "label": "카페→미용실_피벗",
        "queries": [
            "강남구 카페 시장 현황은 어떤가요?",
            "강남구 카페의 임대료 수준은 어떻게 되나요?",
            "강남구 미용실 창업 비용은 얼마나 드나요?",    # PIVOT
            "마포구 미용실과 비교해주세요.",
        ],
    },
    {
        "label": "음식점→의류_피벗",
        "queries": [
            "홍대 음식점 창업 유망 업종은 무엇인가요?",
            "홍대 음식점 평균 매출은 어떻게 되나요?",
            "이태원 의류 상권으로 창업하면 어떨까요?",     # PIVOT
            "이태원 의류 매장 초기 투자 비용은 얼마인가요?",
        ],
    },
    {
        "label": "뷰티→카페_피벗",
        "queries": [
            "마포구 미용실 창업 전망은 어떤가요?",
            "마포구 뷰티 업종 경쟁 현황을 알려주세요.",
            "종로구 카페 창업은 어떻게 생각하시나요?",     # PIVOT
            "종로구 카페 유동인구 데이터를 알려주세요.",
        ],
    },
    {
        "label": "강남→홍대_지역피벗",
        "queries": [
            "강남구 창업 유망 상권은 어디인가요?",
            "강남구 음식점 임대료 현황은 어떤가요?",
            "홍대 지역 창업은 어떤가요?",                 # PIVOT (지역 전환)
            "홍대 카페와 음식점 중 어떤 업종이 유리한가요?",
        ],
    },
    {
        "label": "카페→헬스케어_피벗",
        "queries": [
            "강남 카페 창업 전망을 알려주세요.",
            "강남 카페 평균 매출은 얼마인가요?",
            "강남 헬스케어(의원, 약국 등) 업종은 어떤가요?",  # PIVOT
            "강남 헬스케어 업종 창업 비용은 얼마인가요?",
        ],
    },
    {
        "label": "음식점→뷰티_피벗",
        "queries": [
            "서초구 음식점 상권 현황을 알려주세요.",
            "서초구 음식점 매출 데이터가 있나요?",
            "서초구 뷰티숍 창업은 어떨까요?",              # PIVOT
            "서초구 뷰티 업종 임대료는 얼마인가요?",
        ],
    },
    {
        "label": "의류→카페_피벗",
        "queries": [
            "이태원 의류 상권 최근 트렌드는 어떻게 되나요?",
            "이태원 의류 매장 매출 현황을 알려주세요.",
            "강남 카페 창업이 더 좋을까요?",               # PIVOT
            "강남 카페 성공 사례와 실패 요인은 무엇인가요?",
        ],
    },
    {
        "label": "강남→마포_지역피벗",
        "queries": [
            "강남 상권에서 창업하면 어떤 업종이 유리한가요?",
            "강남 임대료와 권리금 현황을 알려주세요.",
            "마포구로 방향을 바꾸면 어떨까요?",            # PIVOT
            "마포구 창업 유망 업종과 임대료를 알려주세요.",
        ],
    },
    {
        "label": "카페→학원_피벗",
        "queries": [
            "노원구 카페 창업 전망을 알려주세요.",
            "노원구 카페 임대료 수준은 어떻게 되나요?",
            "노원구 학원가 창업은 어떤가요?",              # PIVOT
            "노원구 학원 업종 수익성은 어떻게 되나요?",
        ],
    },
    {
        "label": "음식점→카페_피벗",
        "queries": [
            "용산구 음식점 상권 현황을 알려주세요.",
            "용산구 음식점 창업 비용은 얼마인가요?",
            "용산구 카페 창업이 더 나을까요?",             # PIVOT
            "용산구 카페 임대료와 매출 현황을 알려주세요.",
        ],
    },
]

# 패턴 3: GRADUAL DRIFT — 점진적 주제 이동 (중간 정도 session_continuity 기대)
DRIFT_SESSIONS = [
    {
        "label": "강남→서울전체_드리프트",
        "queries": [
            "강남구 카페 시장 현황은 어떤가요?",
            "강남구와 서초구 카페를 비교해주세요.",
            "서울 전체 카페 시장 트렌드는 어떤가요?",
            "서울에서 카페 창업이 가장 유망한 지역은 어디인가요?",
        ],
    },
    {
        "label": "카페→F&B전체_드리프트",
        "queries": [
            "강남 카페 창업 현황을 알려주세요.",
            "강남 음식점과 카페 중 어느 쪽이 더 유망한가요?",
            "강남 F&B 업종 전반의 트렌드는 어떤가요?",
            "강남 F&B 창업 유망 업종 순위를 알려주세요.",
        ],
    },
    {
        "label": "강남→강남권_드리프트",
        "queries": [
            "강남구 상권 현황은 어떤가요?",
            "서초구 상권은 강남구와 어떻게 다른가요?",
            "송파구를 포함한 강남3구 상권 비교를 해주세요.",
        ],
    },
    {
        "label": "카페→창업전반_드리프트",
        "queries": [
            "강남 카페 창업 비용은 얼마인가요?",
            "강남 카페 외에 초기 투자비가 적은 업종은 무엇인가요?",
            "서울에서 소자본 창업이 가능한 업종과 지역을 알려주세요.",
            "소자본 창업 성공률을 높이는 방법은 무엇인가요?",
        ],
    },
    {
        "label": "뷰티→웰니스_드리프트",
        "queries": [
            "강남 미용실 창업 전망을 알려주세요.",
            "강남 헬스장/피트니스센터 시장은 어떤가요?",
            "강남 웰니스 관련 업종 전반 트렌드를 알려주세요.",
        ],
    },
    {
        "label": "음식점→외식업전반_드리프트",
        "queries": [
            "마포구 음식점 창업이 유망한가요?",
            "마포구 음식점 중 배달 전문점 현황은 어떤가요?",
            "서울 배달 음식점 시장 트렌드는 어떻게 되나요?",
            "배달 음식점과 홀 음식점의 수익성을 비교해주세요.",
        ],
    },
    {
        "label": "강남→서울→전국_드리프트",
        "queries": [
            "강남구 상권 특성은 무엇인가요?",
            "서울 주요 상권의 특성을 비교해주세요.",
            "전국적으로 유망한 창업 상권은 어디인가요?",
        ],
    },
    {
        "label": "카페→리테일_드리프트",
        "queries": [
            "홍대 카페 창업 현황을 알려주세요.",
            "홍대 편의점이나 소매점 창업은 어떤가요?",
            "홍대 리테일 상권 전반 트렌드를 알려주세요.",
            "홍대에서 리테일과 F&B 중 어느 쪽이 더 유망한가요?",
        ],
    },
    {
        "label": "임대료→수익성_드리프트",
        "queries": [
            "강남 카페 임대료 현황을 알려주세요.",
            "강남 카페 매출 대비 임대료 비율은 어떻게 되나요?",
            "강남 카페의 실제 수익률은 얼마나 되나요?",
            "강남 카페 투자 대비 회수 기간(ROI)은 얼마나 걸리나요?",
        ],
    },
    {
        "label": "지역비교_드리프트",
        "queries": [
            "강남구 창업 환경은 어떤가요?",
            "강남구와 마포구의 창업 환경을 비교해주세요.",
            "마포구, 강남구, 종로구 중 창업하기 가장 좋은 지역은 어디인가요?",
        ],
    },
]

ALL_SESSIONS = (
    [(s, "HIGH") for s in HIGH_CONTINUITY_SESSIONS]
    + [(s, "PIVOT") for s in PIVOT_SESSIONS]
    + [(s, "DRIFT") for s in DRIFT_SESSIONS]
)


def run_batch(target_sessions: int = 27, dry_run: bool = False) -> None:
    """배치 세션 실행.

    Args:
        target_sessions: 실행할 세션 수 (이미 수집된 것을 제외한 추가 목표).
        dry_run: True면 쿼리셋만 출력하고 실제 실행은 하지 않는다.
    """
    from main import load_config, run_session

    config = load_config()
    log_path = Path("logs/phase38_collection.jsonl")
    log_path.parent.mkdir(exist_ok=True)

    # 기존 수집 로그 로드
    collected: list[dict] = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        collected.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    already_done = len(collected)
    print(f"\n{'='*60}")
    print(f"Phase 3.8 데이터 수집 배치 러너")
    print(f"{'='*60}")
    print(f"기존 수집 세션: {already_done}개")
    print(f"목표 추가 세션: {target_sessions}개")
    print(f"총 목표:        {already_done + target_sessions}개")
    print(f"전체 쿼리셋:   {len(ALL_SESSIONS)}개")
    if dry_run:
        print("\n[DRY RUN 모드 — 실제 실행 없음]")
        for i, (sess, pattern) in enumerate(ALL_SESSIONS):
            print(f"\n  [{i+1:02d}] {pattern} — {sess['label']}")
            for j, q in enumerate(sess["queries"], 1):
                print(f"       T{j}: {q}")
        return

    print(f"\n{'─'*60}")

    session_pool = list(ALL_SESSIONS)
    # 이미 완료된 레이블은 건너뜀
    done_labels = {c["label"] for c in collected}
    remaining = [(s, p) for s, p in session_pool if s["label"] not in done_labels]

    success_count = 0
    fail_count = 0
    start_total = time.time()

    for idx, (sess_def, pattern) in enumerate(remaining):
        if success_count >= target_sessions:
            break

        label = sess_def["label"]
        queries = sess_def["queries"]
        n = success_count + 1

        print(f"\n[{n:02d}/{target_sessions}] {pattern} — {label} ({len(queries)}턴)")
        print(f"  쿼리: {queries[0][:50]}...")

        sess_start = time.time()
        try:
            session = run_session(queries, config)
            elapsed = time.time() - sess_start

            session_id = session.get("session_id", "unknown")
            intent_history = session.get("session_intent_history", [])
            continuities = [
                t.get("session_continuity")
                for t in session.get("turn_results", [])
            ]

            record = {
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "pattern": pattern,
                "session_id": session_id,
                "turns": len(queries),
                "elapsed_seconds": round(elapsed),
                "continuities": continuities,
                "intent_history_count": len(intent_history),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            success_count += 1
            continuity_str = ", ".join(
                f"{c:.3f}" if c is not None else "None"
                for c in continuities
            )
            print(f"  ✅ 완료 ({elapsed:.0f}초) | session_id={session_id}")
            print(f"     continuity: [{continuity_str}]")

        except KeyboardInterrupt:
            print("\n\n⚠ 사용자 중단. 지금까지 수집된 세션을 보존합니다.")
            break
        except Exception as e:
            elapsed = time.time() - sess_start
            print(f"  ❌ 실패 ({elapsed:.0f}초): {e}")
            fail_count += 1
            # 실패한 세션도 기록 (재시도 방지)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "label": label,
                    "pattern": pattern,
                    "session_id": None,
                    "error": str(e),
                    "status": "failed",
                }) + "\n")

        # 세션 간 대기 (Rate Limit 여유)
        if success_count < target_sessions and idx < len(remaining) - 1:
            wait = 10
            print(f"  ⏳ 다음 세션까지 {wait}초 대기...")
            time.sleep(wait)

    total_elapsed = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"수집 완료: 성공 {success_count}개, 실패 {fail_count}개")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print(f"수집 로그: {log_path.absolute()}")

    total_collected = already_done + success_count
    remaining_needed = max(0, 30 - total_collected)
    if remaining_needed > 0:
        print(f"\n⚠ 목표(30세션)까지 {remaining_needed}개 더 필요합니다.")
    else:
        print(f"\n✅ Phase 3.8 데이터 수집 목표 달성 ({total_collected}세션)")
        print("   다음 단계: threshold_status uncalibrated → calibrated 전환")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.8 데이터 수집 배치 러너")
    parser.add_argument(
        "--sessions", type=int, default=27,
        help="추가 수집할 세션 수 (기본값: 27, 기존 3 + 27 = 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="쿼리셋만 출력하고 실제 실행은 하지 않음",
    )
    args = parser.parse_args()
    run_batch(target_sessions=args.sessions, dry_run=args.dry_run)
