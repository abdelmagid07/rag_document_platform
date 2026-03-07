import fitz  # PyMuPDF


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def load_text(file_path: str) -> str:
    """Load plain text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file_path: str) -> str:
    """Load document text based on file extension."""
    ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
    if ext == "pdf":
        return load_pdf(file_path)
    elif ext == "txt":
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")