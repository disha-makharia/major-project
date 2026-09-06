import pytest

from backend.rag.document_loader import DocumentLoadError, detect_doc_type, extract_text


def test_detect_doc_type():
    assert detect_doc_type("report.pdf") == "pdf"
    assert detect_doc_type("notes.txt") == "txt"
    assert detect_doc_type("memo.docx") == "docx"


def test_detect_doc_type_rejects_unsupported():
    with pytest.raises(DocumentLoadError):
        detect_doc_type("archive.zip")


def test_extract_text_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Q3 revenue declined due to a delayed marketing campaign.")
    text = extract_text(path, "txt")
    assert "Q3 revenue declined" in text


def test_extract_text_pdf(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 6, "Sales grew twelve percent in the West region during Q4.")
    path = tmp_path / "report.pdf"
    pdf.output(str(path))

    text = extract_text(path, "pdf")
    assert "West region" in text


def test_extract_text_docx(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("The East region was the most resilient during the Q3 slowdown.")
    path = tmp_path / "memo.docx"
    document.save(str(path))

    text = extract_text(path, "docx")
    assert "East region" in text


def test_extract_text_empty_pdf_raises(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    path = tmp_path / "blank.pdf"
    pdf.output(str(path))

    with pytest.raises(DocumentLoadError):
        extract_text(path, "pdf")
