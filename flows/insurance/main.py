import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')

from agents.router.agent import get_router_agent
from agents.summary.agent import get_summary_agent
from agents.qna.agent import get_qna_agent
from agents.needle.agent import get_needle_agent
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

Question: "Find the section about collision coverage"
Classification: needle
Reasoning: Looking for a specific section location in the policy.

Question: "What injuries were reported in the accident involving driver 445-78-9012?"
Classification: needle
Reasoning: Looking for a specific detail about a specific driver ID in an accident report.

Question: "Locate information about driver John Smith"
Classification: needle
Reasoning: Looking for specific entity location using driver name identifier.
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
            elif answer.startswith("qna"):
                print("Routing to QnA Agent...")
                qna_agent = get_qna_agent()
                result = qna_agent.invoke({"input": user_question})
                answer = result["output"]
                print(f"\n💡 Answer: {answer}")
            elif answer.startswith("needle"):
                print("Routing to Needle Agent...")
                needle_agent = get_needle_agent()
                result = needle_agent.invoke({"input": user_question})
                needle_result = result["output"]
                
                # Try to parse and display the needle results nicely
                try:
                    import json
                    needle_data = json.loads(needle_result)
                    
                    print(f"\n🔍 Needle Search Results:")
                    print("=" * 60)
                    print(f"Query: {needle_data.get('query', 'N/A')}")
                    print(f"Classification: {needle_data.get('classification', 'N/A')}")
                    print(f"Total Locations Found: {needle_data.get('total_found', 0)}")
                    
                    # Display generated answer if available
                    if needle_data.get('generated_answer'):
                        print(f"\n💡 Generated Answer:")
                        print(f"   {needle_data.get('generated_answer')}")
                    
                    print("-" * 60)
                    
                    for i, location in enumerate(needle_data.get('locations', []), 1):
                        print(f"\nLocation {i}:")
                        print(f"  📍 Anchor: {location.get('anchor', 'N/A')}")
                        
                        if 'document_id' in location:
                            # Policy document location
                            print(f"  📄 Document: {location.get('document_id', 'N/A').split('\\')[-1]}")
                            print(f"  📖 Page: {location.get('page_number', 'N/A')}")
                            print(f"  📑 Section: {location.get('section_name', 'N/A')}")
                            print(f"  🔢 Chunk: {location.get('chunk_index', 'N/A')}/{location.get('total_chunks_in_section', 'N/A')}")
                        elif 'entity_id' in location:
                            # Entity location
                            print(f"  🆔 Entity ID: {location.get('entity_id', 'N/A')}")
                            print(f"  📍 Location: {location.get('location', 'N/A')}")
                            print(f"  🏙️ City: {location.get('city', 'N/A')}")
                            print(f"  📅 Date: {location.get('date', 'N/A')}")
                        
                        print(f"  ⭐ Score: {location.get('similarity_score', 'N/A')}")
                        print(f"  📝 Content:")
                        full_content = location.get('full_content', location.get('description', 'N/A'))
                        print(f"     \"{full_content[:300]}{'...' if len(full_content) > 300 else ''}\"")
                        print("-" * 40)
                    
                    print("=" * 60)
                    
                except (json.JSONDecodeError, Exception) as e:
                    print(f"\n🔍 Raw Needle Result:")
                    print("=" * 50)
                    print(needle_result)
                    print("=" * 50)

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