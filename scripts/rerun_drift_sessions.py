"""scripts/rerun_drift_sessions.py — DRIFT 세션 재실행

collect_phase38_data.py에서 Ollama 타임아웃으로 continuity가 모두 null이었던
DRIFT 세션 7개를 재실행한다.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import truststore
truststore.inject_into_ssl()

DRIFT_SESSIONS = [
    {
        "label": "강남→서울전체_드리프트_retry",
        "queries": [
            "강남구 카페 시장 현황은 어떤가요?",
            "강남구와 서초구 카페를 비교해주세요.",
            "서울 전체 카페 시장 트렌드는 어떤가요?",
            "서울에서 카페 창업이 가장 유망한 지역은 어디인가요?",
        ],
    },
    {
        "label": "카페→F&B전체_드리프트_retry",
        "queries": [
            "강남 카페 창업 현황을 알려주세요.",
            "강남 음식점과 카페 중 어느 쪽이 더 유망한가요?",
            "강남 F&B 업종 전반의 트렌드는 어떤가요?",
            "강남 F&B 창업 유망 업종 순위를 알려주세요.",
        ],
    },
    {
        "label": "강남→강남권_드리프트_retry",
        "queries": [
            "강남구 상권 현황은 어떤가요?",
            "서초구 상권은 강남구와 어떻게 다른가요?",
            "송파구를 포함한 강남3구 상권 비교를 해주세요.",
        ],
    },
    {
        "label": "카페→창업전반_드리프트_retry",
        "queries": [
            "강남 카페 창업 비용은 얼마인가요?",
            "강남 카페 외에 초기 투자비가 적은 업종은 무엇인가요?",
            "서울에서 소자본 창업이 가능한 업종과 지역을 알려주세요.",
            "소자본 창업 성공률을 높이는 방법은 무엇인가요?",
        ],
    },
    {
        "label": "뷰티→웰니스_드리프트_retry",
        "queries": [
            "강남 미용실 창업 전망을 알려주세요.",
            "강남 헬스장/피트니스센터 시장은 어떤가요?",
            "강남 웰니스 관련 업종 전반 트렌드를 알려주세요.",
        ],
    },
    {
        "label": "음식점→외식업전반_드리프트_retry",
        "queries": [
            "마포구 음식점 창업이 유망한가요?",
            "마포구 음식점 중 배달 전문점 현황은 어떤가요?",
            "서울 배달 음식점 시장 트렌드는 어떻게 되나요?",
            "배달 음식점과 홀 음식점의 수익성을 비교해주세요.",
        ],
    },
    {
        "label": "강남→서울→전국_드리프트_retry",
        "queries": [
            "강남구 상권 특성은 무엇인가요?",
            "서울 주요 상권의 특성을 비교해주세요.",
            "전국적으로 유망한 창업 상권은 어디인가요?",
        ],
    },
]

def main():
    from main import load_config, run_session

    config = load_config()
    log_path = Path("logs/phase38_collection.jsonl")

    # 유효한 continuity 값(non-null)이 하나라도 있는 세션만 완료로 간주.
    # 전부 null인 세션은 재실행 대상 (크레딧 소진 등 외부 오류로 실패한 케이스).
    done_labels: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    continuities = rec.get("continuities", [])
                    has_valid = any(c is not None for c in continuities)
                    if has_valid:
                        done_labels.add(rec.get("label", ""))
                except json.JSONDecodeError:
                    pass

    pending = [s for s in DRIFT_SESSIONS if s["label"] not in done_labels]

    print(f"\n{'='*60}")
    print(f"DRIFT 세션 재실행")
    print(f"  전체: {len(DRIFT_SESSIONS)}개 / 이미 완료: {len(done_labels) & set(s['label'] for s in DRIFT_SESSIONS).__len__()} → 대기: {len(pending)}개")
    print(f"{'='*60}")

    if not pending:
        print("실행할 세션이 없습니다.")
        return

    for idx, sess_def in enumerate(pending):
        label = sess_def["label"]
        queries = sess_def["queries"]
        print(f"\n[{idx+1:02d}/{len(pending)}] DRIFT — {label} ({len(queries)}턴)")
        print(f"  T1: {queries[0][:50]}...")

        sess_start = time.time()
        try:
            session = run_session(queries, config)
            elapsed = time.time() - sess_start
            session_id = session.get("session_id", "unknown")
            turn_results = session.get("turn_results", [])
            continuities = [t.get("session_continuity") for t in turn_results]
            intent_history = session.get("session_intent_history", [])

            record = {
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "pattern": "DRIFT",
                "session_id": session_id,
                "turns": len(queries),
                "elapsed_seconds": round(elapsed),
                "continuities": continuities,
                "intent_history_count": len(intent_history),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            c_str = ", ".join(f"{c:.3f}" if c is not None else "None" for c in continuities)
            print(f"  ✅ {session_id} ({elapsed:.0f}초) | continuity: [{c_str}]")

        except Exception as e:
            print(f"  ❌ 실패: {e}")

        if idx < len(pending) - 1:
            print("  ⏳ 15초 대기...")
            time.sleep(15)

    print(f"\n{'='*60}")
    print("DRIFT 재실행 완료")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
