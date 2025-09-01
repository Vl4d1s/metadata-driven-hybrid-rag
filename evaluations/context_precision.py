from ragas.evaluation import evaluate
from ragas.metrics import context_precision
from datasets import Dataset

def evaluate_context_precision(question: str, answer: str, ground_truth: str, contexts: list):
    """
    Evaluate the context precision using Ragas.
    
    Context Precision measures how relevant and focused the retrieved context is 
    to the given question. It evaluates whether the retrieved contexts are 
    useful for answering the question, with higher precision indicating that 
    more of the retrieved contexts are relevant.
    
    Args:
        question (str): The input question
        answer (str): The generated answer from the RAG system
        ground_truth (str): The expected/correct answer
        contexts (list): List of retrieved context strings
        
    Returns:
        dict: Evaluation results containing context_precision score
    """
    # Prepare data in the format expected by Ragas
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer], 
        "ground_truth": [ground_truth],
        "contexts": [contexts]  # List of context strings
    })
    
    # Evaluate using context_precision metric
    results = evaluate(data, metrics=[context_precision])
    
    # Extract and print the context precision score
    precision_score = results['context_precision']
    
    # print("=" * 60)
    # print("🎯 CONTEXT PRECISION EVALUATION RESULTS")
    # print("=" * 60)
    # print(f"Question: {question}")
    # print(f"Generated Answer: {answer}")
    # print(f"Ground Truth: {ground_truth}")
    # print(f"Number of Context Items: {len(contexts)}")
    print("-" * 60)
    print(f"🎯 Context Precision Score: {precision_score}")    
    print("-" * 60)
    
    # Print context details with relevance assessment
    # print("📝 Retrieved Context Analysis:")
    # for i, context in enumerate(contexts, 1):
    #     print(f"  Context {i}: {context[:200]}{'...' if len(context) > 200 else ''}")

    
    return results


