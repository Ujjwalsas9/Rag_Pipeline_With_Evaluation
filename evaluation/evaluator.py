from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, GEval
from deepeval.test_case import LLMTestCaseParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_rag_metadata(docs):
    """Extract metadata and content from retrieved documents."""
    try:
        extracted = []
        for doc in docs:
            meta = doc.metadata
            chunk_data = {
                "file_path": meta.get("file_path", "N/A"),
                "source": meta.get("source", "N/A"),
                "page": meta.get("page", "N/A"),
                "chunk": doc.page_content.strip()
            }
            extracted.append(chunk_data)
        return extracted
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}", exc_info=True)
        return []

def evaluate_response(response, human_answer, wrapped_model):
    """Evaluate RAG pipeline response using DeepEval metrics."""
    try:
        retrieved_context = [doc.page_content for doc in response['context']] if response.get('context') else []
        
        # Test case for most metrics
        test_case = LLMTestCase(
            input=response['question'],
            actual_output=response['AI_generated_response'],
            expected_output=human_answer,
            retrieval_context=retrieved_context
        )
        
        # Test case for HallucinationMetric (uses human_answer as context)
        hallucination_test_case = LLMTestCase(
            input=response['question'],
            actual_output=response['AI_generated_response'],
            context=[human_answer]
        )
        
        metrics = [
            ContextualPrecisionMetric(threshold=0.6, model=wrapped_model, include_reason=True, verbose_mode=True),
            ContextualRecallMetric(threshold=0.6, model=wrapped_model, include_reason=True, verbose_mode=True),
            ContextualRelevancyMetric(threshold=0.6, model=wrapped_model, include_reason=True, verbose_mode=True),
            AnswerRelevancyMetric(threshold=0.6, model=wrapped_model, include_reason=True, verbose_mode=True),
            FaithfulnessMetric(threshold=0.6, model=wrapped_model, include_reason=True, verbose_mode=True),
            HallucinationMetric(threshold=0.6, model=wrapped_model, include_reason=True, verbose_mode=True),
            GEval(
                threshold=0.6,
                model=wrapped_model,
                name="RAG Fact Checker",
                evaluation_steps=[
                    "Create a list of statements from 'actual output'",
                    "Validate if they are relevant and answers the given question in 'input', penalize if any statements are irrelevant",
                    "Also Validate if they exist in 'expected output', penalize if any statements are missing or factually wrong",
                    "Also validate if these statements are grounded in the 'retrieval context' and penalize if they are missing or factually wrong",
                    "Finally also penalize if any statements seem to be invented or made up and do not make sense factually given the 'input' and 'retrieval context'"
                ],
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
                verbose_mode=True
            )
        ]
        
        evaluation_results = []
        
        # Evaluate metrics
        for metric in metrics:
            try:
                test_case_to_use = hallucination_test_case if isinstance(metric, HallucinationMetric) else test_case
                results = evaluate([test_case_to_use], [metric])
                metric_data = results.test_results[0].metrics_data[0]
                evaluation_results.append({
                    "Metric": metric_data.name,
                    "Success": metric_data.success,
                    "Score": metric_data.score,
                    "Reason": metric_data.reason
                })
                logger.info(f"Evaluated {metric_data.name}: Score={metric_data.score}, Success={metric_data.success}")
            except Exception as e:
                logger.error(f"Error evaluating {metric.__class__.__name__}: {str(e)}", exc_info=True)
                evaluation_results.append({
                    "Metric": metric.__class__.__name__,
                    "Success": False,
                    "Score": 0.0,
                    "Reason": f"Evaluation failed: {str(e)}"
                })
        
        return evaluation_results
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}", exc_info=True)
        return [{"Metric": "Evaluation", "Success": False, "Score": 0.0, "Reason": f"Evaluation failed: {str(e)}"}]