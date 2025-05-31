import os
import warnings
import logging
import streamlit as st
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pdf_chatbot_pipeline(pdf_paths, user_query, config, embeddings, chat_client, chat_model, wrapped_model):
    """Full RAG pipeline: Load → Chunk → Embed → Retrieve → Generate."""
    try:
        persist_directory = config['persist_directory']
        
        # Verify PDF paths
        for pdf in pdf_paths:
            if not os.path.exists(pdf):
                raise FileNotFoundError(f"PDF not found: {pdf}")
        
        if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
            st.write(f"Using existing embeddings from {persist_directory}")
            logger.info(f"Using existing embeddings from {persist_directory}")
            new_pdfs = filter_new_pdfs(pdf_paths, persist_directory, embeddings)
            
            if new_pdfs:
                st.write(f"Processing {len(new_pdfs)} new PDFs...")
                logger.info(f"Processing {len(new_pdfs)} new PDFs: {new_pdfs}")
                raw_docs = load_pdfs(new_pdfs)
                chunks = chunk_documents(raw_docs)
                vectorstore = store_embeddings(persist_directory, embeddings)
                vectorstore.add_documents(chunks)
                vectorstore.persist()
                st.write(f"Added {len(chunks)} new chunks to existing vector store")
                logger.info(f"Added {len(chunks)} new chunks to existing vector store")
            else:
                vectorstore = store_embeddings(persist_directory, embeddings)
        else:
            st.write(f"No existing embeddings found. Processing PDFs...")
            logger.info(f"No existing embeddings found. Processing PDFs...")
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
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        return None

def main():
    st.title("Healthcare RAG System")
    st.markdown("Ask questions about healthcare based on PDFs in the `documents/` folder.")

    try:
        # Load configuration
        config = load_config()
        
        # Initialize models
        logger.info("Initializing models...")
        embeddings, chat_client, chat_model, wrapped_model = initialize_models(config)
        
        # Define PDF paths
        pdf_paths = [
            'documents/digital_transformation.pdf',
            'documents/HealthCareSectorinindia-AnOverview.pdf'
        ]
        
        # Human-written reference answer aligned with the query
        human_answer = """The main challenges facing the healthcare sector in India today include a shortage of trained medical personnel, such as doctors, nurses, and paramedics, particularly in rural areas. There is also inadequate infrastructure, with limited access to advanced medical facilities and equipment in many regions. The high cost of private healthcare services makes them unaffordable for a large portion of the population, while public healthcare systems are often overburdened. Additionally, the rise in non-communicable diseases, such as diabetes and heart disease, alongside existing communicable diseases, poses a dual burden. Digital transformation efforts face hurdles due to low digital literacy, data privacy concerns, and insufficient investment in healthcare technology."""
        
        # User input
        user_query = st.text_input("Enter your question:", value="What are the main challenges facing the healthcare sector in India today?")
        
        if st.button("Get Answer"):
            if not user_query:
                st.warning("Please enter a question.")
                return
            
            with st.spinner("Processing..."):
                # Run RAG pipeline
                logger.info(f"Processing query: {user_query}")
                response = pdf_chatbot_pipeline(pdf_paths, user_query, config, embeddings, chat_client, chat_model, wrapped_model)
                
                if response and response.get('context') and response.get('AI_generated_response'):
                    # Display AI-generated response
                    st.subheader("AI-Generated Response")
                    st.write(response["AI_generated_response"])
                    
                    # Display retrieved context
                    st.subheader("Retrieved Context")
                    extracted_metadata = extract_rag_metadata(response['context'])
                    for i, entry in enumerate(extracted_metadata, 1):
                        st.write(f"**Chunk {i}**")
                        st.write(f"- File Path: {entry['file_path']}")
                        st.write(f"- Source: {entry['source']}")
                        st.write(f"- Page: {entry['page']}")
                        st.write(f"- Content: {entry['chunk'][:200]}...")
                    
                    # Evaluate response
                    st.subheader("Evaluation Metrics")
                    try:
                        logger.info("Running evaluation...")
                        evaluation_results = evaluate_response(response, human_answer, wrapped_model)
                        for result in evaluation_results:
                            st.write(f"**{result['Metric']}**")
                            st.write(f"- Success: {result['Success']}")
                            st.write(f"- Score: {result['Score']:.2f}")
                            st.write(f"- Reason: {result['Reason']}")
                    except Exception as e:
                        st.error(f"Evaluation error: {str(e)}")
                        logger.error(f"Evaluation error: {str(e)}", exc_info=True)
                else:
                    st.error("Failed to generate response. Check console for details.")
                    logger.error("Response object is invalid or missing context/AI_generated_response")

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        logger.error(f"Initialization error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()