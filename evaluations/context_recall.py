from ragas.evaluation import evaluate
from ragas.metrics import context_recall
from datasets import Dataset

def evaluate_context_recall(question: str, answer: str, ground_truth: str, contexts: list):
    """
    Evaluate the context recall using Ragas.
    
    Context Recall measures how well the retrieved context supports the ground truth answer.
    It calculates the proportion of statements in the ground truth that can be attributed 
    to the retrieved context.
    
    Args:
        question (str): The input question
        answer (str): The generated answer from the RAG system
        ground_truth (str): The expected/correct answer
        contexts (list): List of retrieved context strings
        
    Returns:
        dict: Evaluation results containing context_recall score
    """
    # Prepare data in the format expected by Ragas
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer], 
        "ground_truth": [ground_truth],
        "contexts": [contexts]  # List of context strings
    })
    
    # Evaluate using context_recall metric
    results = evaluate(data, metrics=[context_recall])
    
    # Extract and print the context recall score
    recall_score = results['context_recall']
    
    # print("=" * 60)
    # print("📊 CONTEXT RECALL EVALUATION RESULTS")
    # print("=" * 60)
    # print(f"Question: {question}")
    # print(f"Generated Answer: {answer}")
    # print(f"Ground Truth: {ground_truth}")
    # print(f"Number of Context Items: {len(contexts)}")
    print("-" * 60)
    print(f"🎯 Context Recall Score: {recall_score}")
    print("-" * 60)
    
    # Print context details
    # print("📝 Retrieved Context:")
    # for i, context in enumerate(contexts, 1):
    #     print(f"  Context {i}: {context[:200]}{'...' if len(context) > 200 else ''}")
    
    # print("=" * 60)
    
    return results





