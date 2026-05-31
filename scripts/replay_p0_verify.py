"""scripts/replay_p0_verify.py — P0 (LLM trace GENERATION hook) 부착 검증용 짧은 세션 실행.

목적: agent/llm.py의 generation_name 패턴이 5개 process 노드 + 4D Judge 모두에
GENERATION을 trace에 부착시키는지 1세션 2턴으로 빠르게 검증.

검증 후 본격 5세션 재실행은 replay_p0_5sessions.py 참조.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from main import load_config, run_session

VERIFY_QUERIES = [
    "강남구 카페 시장은 어떤가요?",
    "마포구와 비교하면 어떨까?",
]


def main() -> None:
    config = load_config()
    print("=" * 60)
    print("P0 verify run — 1 session × 2 turns")
    print("=" * 60)
    result = run_session(VERIFY_QUERIES, config=config)
    sid = result.get("session_id")
    print()
    print(f"✅ 세션 완료: {sid}")
    print(f"   완료 턴: {result.get('current_turn')}")
    print(f"   다음 단계: dashboard에서 세션 {sid} 선택 → Tab 7에서")
    print(f"     - 6개 process 노드 모두 GENERATION 부착 확인")
    print(f"     - 4개 4D Judge LLM 부착 확인")


if __name__ == "__main__":
    main()
