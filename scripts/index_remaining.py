"""
scripts/index_remaining.py — 미인덱싱 PDF 증분 인덱싱

역할:
    이미 인덱싱된 PDF를 건너뛰고 나머지만 추가 인덱싱한다.
    기존 Knowledge Graph를 유지하면서 새 문서를 추가한다.

사용법:
    python scripts/index_remaining.py
"""

import asyncio
import json
import sys
from pathlib import Path

# --- 회사 네트워크 SSL 인증서 호환 ---
import truststore
truststore.inject_into_ssl()

# --- 프로젝트 루트를 sys.path에 추가 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pdfplumber

from agent.lightrag_adapter import get_rag

PDF_DIR = PROJECT_ROOT.parent / "documents" / "data" / "상권분석보고서"
STORAGE_DIR = PROJECT_ROOT / "lightrag_storage"


def get_indexed_filenames() -> set[str]:
    """doc_status에서 processed 상태인 PDF 파일명을 반환한다."""
    doc_status_path = STORAGE_DIR / "kv_store_doc_status.json"
    if not doc_status_path.exists():
        return set()

    with open(doc_status_path, encoding="utf-8") as f:
        data = json.load(f)

    indexed = set()
    for val in data.values():
        if not isinstance(val, dict):
            continue
        if val.get("status") != "processed":
            continue
        summary = val.get("content_summary", "")
        if "[출처:" in summary and ".pdf]" in summary:
            start = summary.find("[출처:") + 5
            end = summary.find("]", start)
            if end > start:
                indexed.add(summary[start:end].strip())
    return indexed


def extract_pdf_text(pdf_path: Path) -> str | None:
    """PDF에서 텍스트를 추출하여 단일 문서 문자열로 반환한다."""
    try:
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text.strip())

        if pages_text:
            full_text = f"[출처: {pdf_path.name}]\n\n" + "\n\n---\n\n".join(pages_text)
            print(f"  PDF: {pdf_path.name} ({len(pages_text)}페이지, {len(full_text):,}자)")
            return full_text
        else:
            print(f"  PDF: {pdf_path.name} — 텍스트 추출 실패")
            return None
    except Exception as e:
        print(f"  PDF: {pdf_path.name} — 에러: {e}")
        return None


async def index_remaining() -> int:
    """미인덱싱 PDF를 추가 인덱싱한다."""
    # 이미 인덱싱된 파일 확인
    indexed = get_indexed_filenames()
    print(f"이미 인덱싱된 PDF: {len(indexed)}개")
    for name in sorted(indexed):
        print(f"  OK {name}")

    # 전체 PDF 목록
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"\n전체 PDF: {len(all_pdfs)}개")

    # 미인덱싱 PDF 필터링
    remaining = [p for p in all_pdfs if p.name not in indexed]
    print(f"추가 인덱싱 대상: {len(remaining)}개")
    for p in remaining:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  -> {p.name} ({size_mb:.1f}MB)")

    if not remaining:
        print("\n모든 PDF가 이미 인덱싱되었습니다.")
        return 0

    # PDF 텍스트 추출 (대상 파일만)
    print("\n" + "=" * 60)
    print("STEP 1: PDF 텍스트 추출")
    print("=" * 60)

    new_docs = []
    for pdf_path in remaining:
        text = extract_pdf_text(pdf_path)
        if text:
            new_docs.append(text)

    print(f"\n삽입 대상: {len(new_docs)}개")

    if not new_docs:
        print("추출된 문서가 없습니다.")
        return 0

    # LightRAG에 삽입
    print("\n" + "=" * 60)
    print(f"STEP 2: LightRAG에 {len(new_docs)}개 문서 삽입")
    print("=" * 60)

    rag = get_rag()
    await rag.initialize_storages()

    count = 0
    for i, doc in enumerate(new_docs):
        first_line = doc.split("\n")[0]
        try:
            await rag.ainsert(doc)
            count += 1
            print(f"  [{i+1}/{len(new_docs)}] 성공: {first_line[:60]}")
        except Exception as e:
            print(f"  [{i+1}/{len(new_docs)}] 실패: {first_line[:60]} — {e}")

    await rag.finalize_storages()

    print("\n" + "=" * 60)
    print(f"완료: {count}/{len(new_docs)}개 추가 인덱싱")
    print("=" * 60)
    return count


if __name__ == "__main__":
    asyncio.run(index_remaining())
