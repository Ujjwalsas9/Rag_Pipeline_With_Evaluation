from langchain.document_loaders import PyMuPDFLoader

def load_pdfs(pdf_paths):
    """Load content from multiple PDF files."""
    all_documents = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        print(f"Loaded {len(documents)} chunks from {path}")
        all_documents.extend(documents)
    print(f"Total loaded documents from all PDFs: {len(all_documents)}")
    return all_documents