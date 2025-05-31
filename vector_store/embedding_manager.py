import os
from langchain.vectorstores import Chroma

def store_embeddings(persist_directory, embeddings, docs=None):
    """Create or use existing vector store for embeddings."""
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        print(f"Creating new vector store in {persist_directory}")
        vector_store = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
    return vector_store

def get_processed_document_names(persist_directory, embeddings):
    """Retrieve names of processed documents from vector store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    all_metadatas = vectorstore.get()["metadatas"]
    processed_sources = {metadata["source"] for metadata in all_metadatas if metadata and "source" in metadata}
    return processed_sources

def filter_new_pdfs(pdf_paths, persist_directory, embeddings):
    """Filter out PDFs that have already been processed."""
    processed_sources = get_processed_document_names(persist_directory, embeddings)
    new_pdfs = [path for path in pdf_paths if path not in processed_sources]
    print(f"Found {len(new_pdfs)} new PDFs to process: {new_pdfs}" if new_pdfs else "No new PDFs to process.")
    return new_pdfs