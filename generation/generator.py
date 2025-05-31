def generate_answer(query, top_chunks, chat_client, model_name):
    """Generate an answer based on retrieved chunks."""
    context = "\n\n".join([doc.page_content for doc in top_chunks])
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer (generate an answer strictly based on the above context; do not use your own knowledge. "
        f"If the query is not covered in the context, respond with: 'This query is not as per the PDF.'):"
    )
    
    response = chat_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        model=model_name
    )
    
    return response.choices[0].message.content