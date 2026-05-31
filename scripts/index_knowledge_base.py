"""
scripts/index_knowledge_base.py — LightRAG Knowledge Base 인덱싱 CLI

역할:
    knowledge_base/docs/ 마크다운 문서와 documents/data/상권분석보고서/ PDF를
    LightRAG Knowledge Graph에 삽입한다.

데이터 흐름:
    입력: knowledge_base/docs/*.md + documents/data/상권분석보고서/*.pdf
    처리: lightrag_indexer.py (PDF→텍스트, 마크다운 로드) + lightrag_adapter.py (LightRAG 인스턴스)
    출력: lightrag_storage/ (Knowledge Graph)

사용법:
    python scripts/index_knowledge_base.py           # 인덱싱 (이미 있으면 건너뜀)
    python scripts/index_knowledge_base.py --force   # 기존 KG 삭제 후 재인덱싱
    python scripts/index_knowledge_base.py --all     # 전체 PDF 포함 (우선순위 외)
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

# --- 회사 네트워크 SSL 인증서 호환 (macOS Keychain 신뢰) ---
import truststore
truststore.inject_into_ssl()

# --- 프로젝트 루트를 sys.path에 추가 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.lightrag_adapter import get_rag
from agent.lightrag_indexer import generate_all_documents


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    Returns:
        파싱된 인자 네임스페이스.
    """
    parser = argparse.ArgumentParser(
        description="LightRAG Knowledge Base 인덱싱",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 lightrag_storage/ 삭제 후 재인덱싱",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="전체 PDF 포함 (우선순위 외 PDF도 인덱싱)",
    )
    return parser.parse_args()


def main() -> None:
    """메인 실행 함수."""
    args = parse_args()

    storage_dir = PROJECT_ROOT / "lightrag_storage"

    # --- Force 모드: 기존 스토리지 삭제 ---
    if args.force and storage_dir.exists():
        print(f"[Force] 기존 스토리지 삭제: {storage_dir}")
        shutil.rmtree(storage_dir)

    # --- 이미 인덱싱된 경우 확인 ---
    if not args.force and storage_dir.exists():
        # graph_chunk_entity_relation.graphml 파일이 있으면 이미 인덱싱됨
        graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
        if graph_file.exists():
            print(f"이미 인덱싱된 KG가 존재합니다: {storage_dir}")
            print("재인덱싱하려면 --force 옵션을 사용하세요.")
            return

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: 마크다운 + PDF → 자연어 문서 변환
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1: 문서 변환")
    print("=" * 60)

    priority_only = not args.all
    documents = generate_all_documents(priority_only=priority_only)

    if not documents:
        print("변환된 문서가 없습니다. knowledge_base/docs/ 또는 PDF 디렉토리를 확인하세요.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: LightRAG 인스턴스 생성
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2: LightRAG 인스턴스 생성")
    print("=" * 60)

    rag = get_rag()
    print(f"  working_dir: {rag.working_dir}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Knowledge Graph에 문서 삽입
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"STEP 3: {len(documents)}개 문서를 Knowledge Graph에 삽입")
    print("=" * 60)

    count = asyncio.run(_index_all(rag, documents))

    # ═══════════════════════════════════════════════════════════════
    # 결과 요약
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("인덱싱 완료")
    print("=" * 60)
    print(f"  삽입 성공: {count}/{len(documents)}개")
    print(f"  스토리지: {storage_dir}")

    # 스토리지 파일 목록
    if storage_dir.exists():
        files = list(storage_dir.iterdir())
        print(f"  생성된 파일: {len(files)}개")
        for f in sorted(files):
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"    {f.name}: {size_str}")


async def _index_all(rag, documents: list[str]) -> int:
    """모든 문서를 LightRAG에 삽입한다.

    스토리지 초기화 → 문서 삽입 → 스토리지 종료의 전체 생애주기를 관리한다.

    Args:
        rag: LightRAG 인스턴스.
        documents: 삽입할 텍스트 문서 리스트.

    Returns:
        삽입된 문서 수.
    """
    from agent.lightrag_indexer import index_documents

    # LightRAG v1.4+ 에서는 스토리지를 명시적으로 초기화해야 함
    await rag.initialize_storages()
    try:
        count = await index_documents(rag, documents)
    finally:
        await rag.finalize_storages()
    return count


if __name__ == "__main__":
    main()
