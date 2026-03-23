from typing import Dict, List, Optional
import os
import json
import logging
import asyncio

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logger = logging.getLogger(__name__)

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics.collections import BleuScore, NonLLMStringSimilarity, AnswerRelevancy, Faithfulness, RougeScore
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings
    from openai import AsyncOpenAI
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

    # Create evaluator LLM and embeddings using the RAGAS 0.4.x API
    async_openai_client = AsyncOpenAI(api_key=openai_api_key)
    evaluator_llm = llm_factory("gpt-3.5-turbo", client=async_openai_client)
    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", client=async_openai_client
    )

    # Define an instance for each metric to evaluate
    bleu_score = BleuScore()
    string_similarity = NonLLMStringSimilarity()
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    rouge_score = RougeScore()

    # Look up ground truth from test_questions.json for meaningful evaluation
    ground_truth = _load_ground_truth(question)
    reference = ground_truth.get("reference_answer", answer)

    # Each metric in RAGAS 0.4.x uses ascore() with its own specific kwargs:
    #   BleuScore / RougeScore / NonLLMStringSimilarity: (reference, response)
    #   AnswerRelevancy:                                 (user_input, response)
    #   Faithfulness:                                    (user_input, response, retrieved_contexts)
    metric_calls = [
        (bleu_score,         {"reference": reference, "response": answer}),
        (string_similarity,  {"reference": reference, "response": answer}),
        (answer_relevancy,   {"user_input": question,  "response": answer}),
        (faithfulness_metric,{"user_input": question,  "response": answer, "retrieved_contexts": contexts}),
        (rouge_score,        {"reference": reference, "response": answer}),
    ]

    async def _run_metrics_async():
        """Run all metrics inside a proper async Task so asyncio.timeout works correctly."""
        results = {}
        for metric, kwargs in metric_calls:
            try:
                result = await metric.ascore(**kwargs)
                score = float(result.score) if hasattr(result, "score") else float(result)
                results[metric.name] = score
                logger.info(f"Metric {metric.name}: {score}")
            except Exception as e:
                error_msg = f"Error evaluating {metric.name}: {str(e)}"
                logger.error(error_msg)
                results[metric.name] = 0.0
        # Explicitly close the async client before the event loop shuts down
        await async_openai_client.close()
        return results

    try:
        results = asyncio.run(_run_metrics_async())
    except RuntimeError:
        # Fallback for environments with an already-running event loop (e.g. Streamlit)
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(_run_metrics_async())
        results = loop.run_until_complete(future)

    # Return the evaluation results
    return results


def evaluate_all_questions(collection=None, model: str = "gpt-3.5-turbo") -> Dict:
    """Run batch evaluation over all test questions and return per-question + aggregate results.

    If *collection* is provided, the full RAG pipeline (retrieve → format → generate)
    is executed for each question.  Otherwise, reference answers from the test file are
    used directly so that metrics can still be computed without a live database.

    Returns a dict with:
        - "per_question": list of per-question result dicts
        - "aggregates": dict mapping metric name → mean score
    """
    import rag_client
    import llm_client

    # Load test questions
    try:
        with open("test_questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "test_questions.json not found"}

    test_questions = data.get("test_questions", [])
    if not test_questions:
        return {"error": "No test questions found in file"}

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        return {"error": "OPENAI_API_KEY not set in environment"}

    per_question_results: List[Dict] = []
    all_metric_scores: Dict[str, List[float]] = {}

    for tq in test_questions:
        question = tq["question"]
        category = tq.get("category", "unknown")
        qid = tq.get("id", "?")

        # Retrieve context and generate a response via the live RAG pipeline
        # when a collection is available; fall back to reference data otherwise.
        if collection is not None:
            docs_result = rag_client.retrieve_documents(collection, question, n_results=3)
            if docs_result and docs_result.get("documents"):
                documents = docs_result["documents"][0]
                metadatas = docs_result["metadatas"][0]
                distances = docs_result.get("distances", [None])[0]
                context_str = rag_client.format_context(documents, metadatas, distances)
                contexts_list = documents
            else:
                context_str = ""
                contexts_list = []

            answer = llm_client.generate_response(
                openai_key, question, context_str, [], model
            )
        else:
            # Use reference data when no collection is provided
            answer = tq.get("reference_answer", "")
            contexts_list = tq.get("reference_contexts", [])

        # Evaluate
        scores = evaluate_response_quality(question, answer, contexts_list)

        result_entry = {
            "id": qid,
            "category": category,
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "scores": scores,
        }
        per_question_results.append(result_entry)

        # Accumulate scores for aggregation
        for metric_name, score in scores.items():
            if metric_name == "error":
                continue
            if isinstance(score, (int, float)):
                all_metric_scores.setdefault(metric_name, []).append(score)

        logger.info(f"[{qid}] {category}: {scores}")

    # Compute aggregates (mean for each metric)
    aggregates: Dict[str, float] = {}
    for metric_name, score_list in all_metric_scores.items():
        if score_list:
            aggregates[metric_name] = sum(score_list) / len(score_list)

    return {
        "per_question": per_question_results,
        "aggregates": aggregates,
        "total_questions": len(test_questions),
    }


if __name__ == "__main__":
    """CLI entry point: run batch evaluation and print results."""
    import sys

    print("=" * 60)
    print("NASA Mission Intelligence – Batch Evaluation")
    print("=" * 60)

    # Optionally initialise a live RAG collection
    collection = None
    try:
        import rag_client
        backends = rag_client.discover_chroma_backends()
        if backends:
            key = next(iter(backends))
            backend = backends[key]
            collection, success, err = rag_client.initialize_rag_system(
                backend["directory"], backend["collection_name"]
            )
            if not success:
                print(f"Could not initialise RAG backend: {err}")
                collection = None
            else:
                print(f"Using backend: {backend['display_name']}")
    except Exception as e:
        print(f"Backend discovery failed ({e}), using reference answers only.")

    results = evaluate_all_questions(collection=collection)

    if "error" in results:
        print(f"\n Error: {results['error']}")
        sys.exit(1)

    # Per-question summary
    print(f"\n Per-Question Results ({results['total_questions']} questions)")
    print("-" * 60)
    for entry in results["per_question"]:
        print(f"\n[Q{entry['id']}] ({entry['category']}) {entry['question']}")
        for metric, score in entry["scores"].items():
            print(f"    {metric:40s} {score:.4f}" if isinstance(score, float) else f"    {metric}: {score}")

    # Aggregate summary
    print("\n" + "=" * 60)
    print(" Aggregate Metrics (Mean)")
    print("-" * 60)
    for metric, mean_score in results["aggregates"].items():
        print(f"    {metric:40s} {mean_score:.4f}")
    print("=" * 60)
