"""
agent/lightrag_indexer.py — LightRAG Knowledge Base 인덱서

역할:
    knowledge_base/docs/ 마크다운 문서와 documents/data/상권분석보고서/ PDF를
    LightRAG에 삽입한다. PDF는 pdfplumber로 텍스트를 추출한다.

데이터 흐름:
    입력: knowledge_base/docs/*.md + documents/data/상권분석보고서/*.pdf
    출력: lightrag_storage/ (Knowledge Graph)

인덱싱 전략:
    - 마크다운 → 그대로 삽입 (이미 자연어)
    - PDF → pdfplumber로 텍스트 추출 후 페이지 단위로 삽입
    - CSV는 LightRAG에 인덱싱하지 않음 (pandas_query가 직접 쿼리)
"""

from pathlib import Path

import pdfplumber


# --- 프로젝트 경로 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DOCS = PROJECT_ROOT / "knowledge_base" / "docs"

# 상권분석보고서 PDF 디렉토리 (프로젝트 외부 documents/ 공유 디렉토리)
PDF_DIR = PROJECT_ROOT.parent / "documents" / "data" / "상권분석보고서"

# --- 우선순위 PDF 목록 ---
# 5개 포커스 구(강남, 마포, 서초, 종로, 영등포) + 핵심 보고서를 우선 인덱싱한다.
# 전체 31개 PDF 중 ~10개를 우선 처리하여 인덱싱 시간과 토큰 비용을 절약한다.
PRIORITY_PDFS = [
    "2025 강남구 상권분석 보고서.pdf",
    "2025 마포구 상권분석 보고서.pdf",
    "2025 서초구 상권분석 보고서.pdf",
    "2025 종로구 상권분석 보고서.pdf",
    "2025 영등포구 상권분석 보고서.pdf",
    "2024 소상공인 금융리포트.pdf",
]


def load_markdown_docs(docs_dir: Path) -> list[str]:
    """마크다운 문서를 그대로 로드한다.

    Args:
        docs_dir: knowledge_base/docs/ 경로.

    Returns:
        마크다운 문서 내용 리스트.
    """
    docs = []
    if not docs_dir.exists():
        return docs

    for md_file in sorted(docs_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if content.strip():
            docs.append(content)
            print(f"  마크다운: {md_file.name} ({len(content):,}자)")

    return docs


def load_pdf_documents(
    pdf_dir: Path | None = None,
    priority_only: bool = True,
) -> list[str]:
    """상권분석보고서 PDF를 텍스트로 변환한다.

    pdfplumber를 사용하여 PDF에서 텍스트를 추출한다.
    각 PDF는 하나의 문서로 합쳐진다 (페이지 구분 유지).

    Args:
        pdf_dir: PDF 디렉토리 경로. None이면 기본 경로 사용.
        priority_only: True이면 PRIORITY_PDFS만 처리. False이면 전체 PDF 처리.

    Returns:
        PDF에서 추출한 텍스트 문서 리스트.
    """
    pdf_dir = pdf_dir or PDF_DIR
    docs = []

    if not pdf_dir.exists():
        print(f"  PDF 디렉토리 없음: {pdf_dir}")
        return docs

    # 대상 PDF 결정
    if priority_only:
        pdf_files = [pdf_dir / name for name in PRIORITY_PDFS if (pdf_dir / name).exists()]
    else:
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            pages_text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(text.strip())

            if pages_text:
                # PDF 전체를 하나의 문서로 합침 (페이지 구분 포함)
                full_text = f"[출처: {pdf_path.name}]\n\n" + "\n\n---\n\n".join(pages_text)
                docs.append(full_text)
                print(f"  PDF: {pdf_path.name} ({len(pages_text)}페이지, {len(full_text):,}자)")
            else:
                print(f"  PDF: {pdf_path.name} — 텍스트 추출 실패 (스캔 이미지?)")

        except Exception as e:
            print(f"  PDF: {pdf_path.name} — 에러: {e}")

    return docs


# ═══════════════════════════════════════════════════════════════
# 메인 인덱싱 함수
# ═══════════════════════════════════════════════════════════════

def generate_all_documents(
    docs_dir: Path | None = None,
    pdf_dir: Path | None = None,
    priority_only: bool = True,
) -> list[str]:
    """마크다운과 PDF를 자연어 문서로 변환한다.

    Args:
        docs_dir: 마크다운 디렉토리. None이면 기본 경로 사용.
        pdf_dir: PDF 디렉토리. None이면 기본 경로 사용.
        priority_only: True이면 우선순위 PDF만 처리.

    Returns:
        LightRAG에 삽입할 텍스트 문서 리스트.
    """
    docs_dir = docs_dir or KB_DOCS
    all_docs = []

    print("마크다운 문서 로드:")
    md_docs = load_markdown_docs(docs_dir)
    all_docs.extend(md_docs)

    print("\nPDF 문서 로드:")
    pdf_docs = load_pdf_documents(pdf_dir, priority_only=priority_only)
    all_docs.extend(pdf_docs)

    print(f"\n총 {len(all_docs)}개 문서 준비 완료")
    return all_docs


async def index_documents(rag, documents: list[str]) -> int:
    """문서 리스트를 LightRAG에 삽입한다.

    Args:
        rag: LightRAG 인스턴스.
        documents: 삽입할 텍스트 문서 리스트.

    Returns:
        삽입된 문서 수.
    """
    count = 0
    total = len(documents)

    for i, doc in enumerate(documents):
        try:
            await rag.ainsert(doc)
            count += 1
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  진행: {i+1}/{total} ({count}개 성공)")
        except Exception as e:
            print(f"  ⚠ 문서 {i+1} 삽입 실패: {e}")

    return count
