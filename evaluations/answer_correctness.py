from ragas.evaluation import evaluate
from ragas.metrics import answer_correctness
from datasets import Dataset

def evaluate_answer_correctness(agent_query: str, agent_summary: str, ground_truth_summary: str):
    """
    Evaluate the answer correctness using Ragas.
    """
    data = Dataset.from_dict({
        "question": [agent_query],
        "answer": [agent_summary],
        "ground_truth": [ground_truth_summary],
    })
    results = evaluate(data, metrics=[answer_correctness])
    return results
