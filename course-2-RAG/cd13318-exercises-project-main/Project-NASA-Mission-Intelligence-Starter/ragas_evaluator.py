from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
    # Define an instance for each metric to evaluate, using the metrics found in imports of rags.metrics
    bleu_score = BleuScore()
    context_precision = NonLLMContextPrecisionWithReference()
    response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    rouge_score = RougeScore()
    
    
    metrics = [bleu_score, context_precision, response_relevancy, faithfulness_metric, rouge_score]
    
    # Evaluate the response using the metrics
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer
    )
    
    results = {}
    for metric in metrics:
        try:
            import asyncio
            score = asyncio.get_event_loop().run_until_complete(metric.single_turn_ascore(sample))
            results[metric.name] = float(score)
        except Exception as e:
            results[metric.name] = 0.0
    
    # Return the evaluation results
    return results
