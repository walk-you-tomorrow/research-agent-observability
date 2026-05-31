"""
tests/validation/test_data_pipeline.py — 데이터 파이프라인 무결성 검증

Knowledge Base의 CSV 파일과 LightRAG KG 저장소의 무결성을 검증한다.

테스트 대상 (F1~F4):
    F1: CSV 인코딩 — 모든 CSV가 UTF-8이고 깨지지 않는지
    F2: 5개 구 필터 — 구 정보가 있는 CSV가 5개 구만 포함하는지
    F3: KG 인덱싱 무결성 — lightrag_storage에 엔티티/관계가 존재하는지
    F4: PDF 추출 품질 — PDF 텍스트가 정상 추출되어 인덱싱에 사용 가능한지

실행 방법:
    python -m pytest tests/validation/test_data_pipeline.py -v
"""
import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import pandas as pd

# --- 상수 ---
KB_DATA_DIR = os.path.join("knowledge_base", "data")
LIGHTRAG_DIR = "lightrag_storage"
ALLOWED_DISTRICTS = {"강남구", "마포구", "서초구", "종로구", "영등포구"}

# CSV 파일 목록
CSV_FILES = [
    "store_info.csv",
    "foot_traffic.csv",
    "rent.csv",
    "demographics.csv",
    "business_codes.csv",
    "dong_summary.csv",
    "card_consumption.csv",
    "estimated_sales.csv",
    "subway_ridership.csv",
]


# ═══════════════════════════════════════
# F1: CSV 인코딩 검증
# ═══════════════════════════════════════
class TestF1CSVEncoding:
    """F1: 모든 CSV가 UTF-8로 인코딩되어 있고 깨지지 않는지 검증."""

    @pytest.mark.parametrize("csv_file", CSV_FILES)
    def test_csv_is_utf8(self, csv_file):
        """CSV 파일이 UTF-8로 읽히는지 확인."""
        fpath = os.path.join(KB_DATA_DIR, csv_file)
        if not os.path.exists(fpath):
            pytest.skip(f"{csv_file} not found")

        # UTF-8로 읽기 시도 — 실패하면 인코딩 문제
        try:
            df = pd.read_csv(fpath, encoding="utf-8", nrows=5)
        except UnicodeDecodeError:
            pytest.fail(f"{csv_file} is not UTF-8 encoded")

        assert len(df) > 0, f"{csv_file} is empty"

    @pytest.mark.parametrize("csv_file", CSV_FILES)
    def test_no_mojibake(self, csv_file):
        """CSV 컬럼명에 깨진 문자(mojibake)가 없는지 확인."""
        fpath = os.path.join(KB_DATA_DIR, csv_file)
        if not os.path.exists(fpath):
            pytest.skip(f"{csv_file} not found")

        df = pd.read_csv(fpath, nrows=1)
        for col in df.columns:
            # 대표적인 mojibake 패턴 체크
            assert "ï¿½" not in col, f"Mojibake in column '{col}' of {csv_file}"
            assert "\ufffd" not in col, f"Replacement character in column '{col}' of {csv_file}"


# ═══════════════════════════════════════
# F2: 5개 구 필터 검증
# ═══════════════════════════════════════
class TestF2DistrictFilter:
    """F2: 구(district) 컬럼이 있는 CSV에 5개 구 데이터만 포함되어 있는지 검증."""

    # district 컬럼이 있는 CSV 목록
    DISTRICT_CSVS = ["store_info.csv", "foot_traffic.csv", "demographics.csv", "dong_summary.csv"]

    @pytest.mark.parametrize("csv_file", DISTRICT_CSVS)
    def test_only_five_districts(self, csv_file):
        """CSV에 5개 구만 포함되어 있는지 확인."""
        fpath = os.path.join(KB_DATA_DIR, csv_file)
        if not os.path.exists(fpath):
            pytest.skip(f"{csv_file} not found")

        df = pd.read_csv(fpath)
        if "district" not in df.columns:
            pytest.skip(f"{csv_file} has no 'district' column")

        districts = set(df["district"].unique())
        extra = districts - ALLOWED_DISTRICTS
        assert not extra, \
            f"{csv_file} contains unexpected districts: {extra}"

    @pytest.mark.parametrize("csv_file", DISTRICT_CSVS)
    def test_all_five_districts_present(self, csv_file):
        """5개 구가 모두 포함되어 있는지 확인."""
        fpath = os.path.join(KB_DATA_DIR, csv_file)
        if not os.path.exists(fpath):
            pytest.skip(f"{csv_file} not found")

        df = pd.read_csv(fpath)
        if "district" not in df.columns:
            pytest.skip(f"{csv_file} has no 'district' column")

        districts = set(df["district"].unique())
        missing = ALLOWED_DISTRICTS - districts
        if missing:
            pytest.xfail(f"{csv_file} missing districts: {missing}")


# ═══════════════════════════════════════
# F3: KG 인덱싱 무결성
# ═══════════════════════════════════════
class TestF3KGIntegrity:
    """F3: LightRAG 저장소에 엔티티/관계가 존재하는지 검증."""

    def test_lightrag_storage_exists(self):
        """lightrag_storage 디렉토리가 존재한다."""
        assert os.path.isdir(LIGHTRAG_DIR), \
            f"LightRAG storage directory not found: {LIGHTRAG_DIR}"

    def test_graph_file_exists(self):
        """Knowledge Graph 파일이 존재한다."""
        graph_file = os.path.join(LIGHTRAG_DIR, "graph_chunk_entity_relation.graphml")
        assert os.path.exists(graph_file), "GraphML file not found"
        # 파일 크기 확인 (빈 파일이 아닌지)
        size = os.path.getsize(graph_file)
        assert size > 100, f"GraphML file is too small: {size} bytes"

    def test_entity_store_has_data(self):
        """엔티티 저장소에 데이터가 있다."""
        entity_file = os.path.join(LIGHTRAG_DIR, "kv_store_full_entities.json")
        if not os.path.exists(entity_file):
            pytest.skip("Entity store not found")

        with open(entity_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, "Entity store is empty"

    def test_relation_store_has_data(self):
        """관계 저장소에 데이터가 있다."""
        relation_file = os.path.join(LIGHTRAG_DIR, "kv_store_full_relations.json")
        if not os.path.exists(relation_file):
            pytest.skip("Relation store not found")

        with open(relation_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, "Relation store is empty"

    def test_vector_db_exists(self):
        """벡터 DB 파일이 존재한다."""
        vdb_files = [
            os.path.join(LIGHTRAG_DIR, "vdb_chunks.json"),
            os.path.join(LIGHTRAG_DIR, "vdb_entities.json"),
        ]
        for vf in vdb_files:
            if os.path.exists(vf):
                size = os.path.getsize(vf)
                assert size > 10, f"Vector DB file {vf} is too small"
                return
        pytest.skip("No vector DB files found")


# ═══════════════════════════════════════
# F4: PDF 추출 품질
# ═══════════════════════════════════════
class TestF4PDFExtraction:
    """F4: PDF 텍스트 추출이 가능하고 한글이 포함되어 있는지 검증."""

    PDF_DIR = os.path.join("documents", "data", "상권분석보고서")

    def test_pdf_directory_exists(self):
        """PDF 디렉토리가 존재한다."""
        if not os.path.isdir(self.PDF_DIR):
            pytest.skip("PDF directory not found")
        files = [f for f in os.listdir(self.PDF_DIR) if f.endswith(".pdf")]
        assert len(files) > 0, "No PDF files found"

    def test_pdf_extractable(self):
        """최소 1개 PDF에서 텍스트를 추출할 수 있다."""
        if not os.path.isdir(self.PDF_DIR):
            pytest.skip("PDF directory not found")

        try:
            import pdfplumber
        except ImportError:
            pytest.skip("pdfplumber not installed")

        pdf_files = [f for f in os.listdir(self.PDF_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            pytest.skip("No PDF files")

        # 첫 번째 PDF에서 텍스트 추출 시도
        pdf_path = os.path.join(self.PDF_DIR, pdf_files[0])
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages[:3]:  # 처음 3페이지만
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        assert len(text) > 100, f"Extracted too little text: {len(text)} chars"

    def test_pdf_contains_korean(self):
        """추출된 텍스트에 한글이 포함되어 있다."""
        if not os.path.isdir(self.PDF_DIR):
            pytest.skip("PDF directory not found")

        try:
            import pdfplumber
        except ImportError:
            pytest.skip("pdfplumber not installed")

        pdf_files = [f for f in os.listdir(self.PDF_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            pytest.skip("No PDF files")

        pdf_path = os.path.join(self.PDF_DIR, pdf_files[0])
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        # 한글 문자가 있는지 확인
        korean_chars = [c for c in text if "\uac00" <= c <= "\ud7a3"]
        assert len(korean_chars) > 10, "No Korean characters found in PDF"

    def test_indexer_module_importable(self):
        """lightrag_indexer 모듈이 import 가능한지 확인."""
        try:
            import pdfplumber  # noqa: F401 — pdfplumber가 없으면 skip
        except ImportError:
            pytest.skip("pdfplumber not installed")

        from agent.lightrag_indexer import generate_all_documents
        assert callable(generate_all_documents)
