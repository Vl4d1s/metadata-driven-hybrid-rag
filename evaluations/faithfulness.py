from ragas.evaluation import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset

def evaluate_faithfulness(question: str, answer: str, contexts: list):
    """
    Evaluate the faithfulness using Ragas.
    
    Faithfulness measures how well the generated answer is grounded in the 
    provided context. It calculates the proportion of claims in the answer 
    that can be inferred from the given context. Higher faithfulness scores 
    indicate that the answer is more factually consistent with the context.
    
    Args:
        question (str): The input question
        answer (str): The generated answer from the RAG system
        contexts (list): List of retrieved context strings
        
    Returns:
        dict: Evaluation results containing faithfulness score
    """
    # Prepare data in the format expected by Ragas
    # Note: faithfulness doesn't require ground_truth, only question, answer, and contexts
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer], 
        "contexts": [contexts]  # List of context strings
    })
    
    # Evaluate using faithfulness metric
    results = evaluate(data, metrics=[faithfulness])
    
    # Extract and print the faithfulness score
    faithfulness_score = results['faithfulness']
    
    # print("=" * 60)
    # print("🔍 FAITHFULNESS EVALUATION RESULTS")
    # print("=" * 60)
    # print(f"Question: {question}")
    # print(f"Generated Answer: {answer}")
    # print(f"Number of Context Items: {len(contexts)}")
    print("-" * 60)
    print(f"🎯 Faithfulness Score: {faithfulness_score}")
    print("-" * 60)
    
    # Print context details
    # print("📝 Source Context:")
    # for i, context in enumerate(contexts, 1):
    #     print(f"  Context {i}: {context[:200]}{'...' if len(context) > 200 else ''}")
        
    return results
