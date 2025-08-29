import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')

from agents.router.agent import get_router_agent
from agents.summary.agent import get_summary_agent
router_examples = """
Question: "Explain the additional benefits offered under comprehensive cover"
Classification: summary
Reasoning: Requests an overview of multiple benefits in the policy.

Question: "Summarize the accident of the driver 445-78-9012"
Classification: summary
Reasoning: Requests a broad narrative/overview of an accident report.

Question: "Does this insurance include roadside assistance?"
Classification: qna
Reasoning: Asks for a specific yes/no fact from the policy.

Question: "What is the car of the driver 445-78-9012?"
Classification: qna
Reasoning: Asks for a specific piece of information about the driver’s vehicle.

Question: "What is the maximum amount the insurer will pay for windscreen replacement?"
Classification: needle
Reasoning: Looking for a very specific limit (number) buried in the policy text.

Question: "What injuries were reported in the accident involving driver 445-78-9012?"
Classification: needle
Reasoning: Looking for a specific detail inside an accident report.
"""


def start_chat():
    print("Type 'quit' to exit")
    router_agent = get_router_agent(options=["summary", "qna" , "needle"] ,examples=router_examples)
    summary_agent = get_summary_agent(data_path="C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag\\flows\\insurance\\data\\policy\\policy.pdf")
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_question:            
            print("Routing the question...")
            result = router_agent.invoke({"question": user_question})
            answer = result["output"]
            print(f"\n💡 Answer: {answer}")
            if answer.startswith("summary"):
                print("Routing to Summary Agent...")
                result = summary_agent.invoke({"input": user_question})
                answer = result["output"]
                print(f"\n💡 Answer: {answer}")
            # Classify the question
            # classification = classify_for_agents(user_question, ["summery", "qna"], "qna")
            
            # Route to appropriate agent
            # if classification == "summery":
            #     print("Routing to Timeline Agent...")
            #     timeline_agent = get_timeline_agent(llm=llm)
            #     result = timeline_agent.invoke({"input": user_question})
            # elif classification == "qna":
            #     print("Routing to QnA Agent...")
            #     qna_agent = get_qna_agent(llm=llm)
            #     result = qna_agent.invoke({"input": user_question})
            # else:
            #     print(f"\u274C Unknown classification: {classification}")
            #     continue
                
            # answer = result["output"]
            # print(f"\n💡 Answer: {answer}")
            # print("-" * 50)

start_chat()