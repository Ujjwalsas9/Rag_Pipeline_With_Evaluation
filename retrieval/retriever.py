def retrieve_chunks(query, vectorstore, top_k=5):
    """Retrieve top-k semantically relevant documents from ChromaDB."""
    results = vectorstore.similarity_search(query, k=top_k*2)
    unique_results = []
    seen_contents = set()

    for doc in results:
        if doc.page_content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(doc.page_content)
        if len(unique_results) >= top_k:
            break

    return unique_results