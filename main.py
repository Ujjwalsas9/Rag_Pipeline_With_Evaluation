import os
import warnings
from config.settings import load_config
from models.azure_model import initialize_models
from document_processing.loader import load_pdfs
from document_processing.chunker import chunk_documents
from vector_store.embedding_manager import store_embeddings, filter_new_pdfs
from retrieval.retriever import retrieve_chunks
from generation.generator import generate_answer
from evaluation.evaluator import extract_rag_metadata, evaluate_response

warnings.filterwarnings("ignore")
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

def pdf_chatbot_pipeline(pdf_paths, user_query, config, embeddings, chat_client, chat_model, wrapped_model):
    """Full RAG pipeline: Load → Chunk → Embed → Retrieve → Generate."""
    persist_directory = config['persist_directory']
    
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(f"Using existing embeddings from {persist_directory}")
        new_pdfs = filter_new_pdfs(pdf_paths, persist_directory, embeddings)
        
        if new_pdfs:
            print(f"Processing {len(new_pdfs)} new PDFs...")
            raw_docs = load_pdfs(new_pdfs)
            chunks = chunk_documents(raw_docs)
            vectorstore = store_embeddings(persist_directory, embeddings)
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            print(f"Added {len(chunks)} new chunks to existing vector store")
        else:
            vectorstore = store_embeddings(persist_directory, embeddings)
    else:
        print(f"No existing embeddings found. Processing PDFs...")
        raw_docs = load_pdfs(pdf_paths)
        chunks = chunk_documents(raw_docs)
        vectorstore = store_embeddings(persist_directory, embeddings, chunks)

    retrieved = retrieve_chunks(user_query, vectorstore)
    answer = generate_answer(user_query, retrieved, chat_client, config['chat_deployment_name'])

    return {
        'context': retrieved,
        'question': user_query,
        'AI_generated_response': answer
    }

def main():
    config = load_config()
    embeddings, chat_client, chat_model, wrapped_model = initialize_models(config)
    
    pdf_paths = ['digital_transformation.pdf', 'HealthCareSectorinindia-AnOverview.pdf']
    user_query = "What are the main challenges facing the healthcare sector in India today?"
    
    response = pdf_chatbot_pipeline(pdf_paths, user_query, config, embeddings, chat_client, chat_model, wrapped_model)
    
    print("\nAI-Generated Response:\n")
    print("-" * 80)
    print(response["AI_generated_response"])
    
    extract_rag_metadata(response['context'])
    
    human_answer = """MIoT (Medical Internet of Things) improves hospital safety by creating a connected environment where medical devices and systems can communicate seamlessly. This connectivity allows real-time monitoring of patients through biometric sensors and smart devices, which helps detect critical changes in a patient’s condition more quickly. As a result, healthcare providers can respond faster and more accurately. Additionally, MIoT reduces human errors by automating data collection and ensuring that medical information is accurate and readily available across different care settings—from hospital wards to home care. Overall, this leads to better coordination, quicker interventions, and enhanced patient safety."""
    
    evaluate_response(response, human_answer, wrapped_model)

if __name__ == "__main__":
    main()