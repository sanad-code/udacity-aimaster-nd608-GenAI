from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os
import json
import logging
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops for Streamlit
nest_asyncio.apply()

# Configure logging
logger = logging.getLogger(__name__)

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _load_ground_truth(question: str) -> Dict:
    """Look up ground truth reference answer and contexts from test_questions.json"""
    try:
        with open("test_questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        question_lower = question.strip().lower().rstrip("?")
        for tq in data.get("test_questions", []):
            if tq.get("question", "").strip().lower().rstrip("?") == question_lower:
                return {
                    "reference_answer": tq.get("reference_answer", ""),
                    "reference_contexts": tq.get("reference_contexts", [])
                }
    except Exception as e:
        logger.warning(f"Could not load ground truth: {e}")
    return {}


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # Get API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"error": "OPENAI_API_KEY not set in environment"}
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key))
    
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    )
    
    # Define an instance for each metric to evaluate, using the metrics found in imports of rags.metrics
    bleu_score = BleuScore()
    context_precision = NonLLMContextPrecisionWithReference()
    response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    rouge_score = RougeScore()
    
    
    metrics = [bleu_score, context_precision, response_relevancy, faithfulness_metric, rouge_score]
    
    # Look up ground truth from test_questions.json for meaningful evaluation
    ground_truth = _load_ground_truth(question)
    reference = ground_truth.get("reference_answer", answer)
    reference_contexts = ground_truth.get("reference_contexts", contexts)
    
    # Evaluate the response using the metrics
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
        reference_contexts=reference_contexts
    )
    
    results = {}
    for metric in metrics:
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            score = loop.run_until_complete(metric.single_turn_ascore(sample))
            results[metric.name] = float(score)
            logger.info(f"Metric {metric.name}: {score}")
        except Exception as e:
            error_msg = f"Error evaluating {metric.name}: {str(e)}"
            logger.error(error_msg)
            print(f"DEBUG: {error_msg}")
            results[metric.name] = 0.0
    
    # Return the evaluation results
    return results
