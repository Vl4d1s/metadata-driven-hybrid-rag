import os
from typing import List
import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')
from langchain.tools import Tool

from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from .utils import get_hybrid_retriever_with_indexes, get_hybrid_cypher_retriever_with_indexes
from .prompts import router_examples , retrieval_query_entity
# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD") 
    )
)



# # Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


def graph_rag_tool(question: str,return_context: bool = False):
    from agents.router.agent import get_router_agent  
    # Run the router agent with the question
    options = ["policy", "entity", "both"]
    examples = router_examples
    router_agent = get_router_agent(options, examples)
    result = router_agent.invoke({"question": question})
    print(f"Router Result: {result}")
    classification = result.get('output', 'unknown')

    if classification == "policy":
        # hybrid retriever for policy
        retriever = get_hybrid_retriever_with_indexes(
            "PolicyVectorIndex","PolicyFullTextIndex","Chunk","embedding",1536,"cosine",["Chunk_Content"]
        )
        llm = OpenAILLM(model_name="gpt-4o-mini",model_params={"temperature":0.2})
        rag = GraphRAG(retriever=retriever, llm=llm)
        response = rag.search(
            query_text=question,
            retriever_config={"top_k": 3 , "effective_search_ratio": 2, "ranker": "LINEAR", "alpha": 0.7},
            return_context=True
        )

        if not return_context:
            print("\n--- LLM Generated Answer ---")
        # print(response)
                        
        if response.retriever_result:
            if not return_context:
                print("\n--- Retrieved Context Details ---")
                print(f"Number of retrieved items: {len(response.retriever_result.items)}")
            
            for i, item in enumerate(response.retriever_result.items):
                if not return_context:
                    print(f"\nItem {i+1}:")
                    print(f"  Score: {item.metadata.get('score', 'N/A')}")  # Score is in metadata
                
                # Parse the content string to extract relevant information
                try:
                    import json
                    content_data = json.loads(item.content)
                except (json.JSONDecodeError, ValueError):
                    content_data = eval(item.content)
                
                if not return_context:
                    print(f"  Document ID: {content_data.get('DocumentId', 'N/A')}")
                    print(f"  Page Number: {content_data.get('PageNumber', 'N/A')}")
                    print(f"  Section: {content_data.get('SectionName', 'N/A')}")
                    print(f"  Chunk Index: {content_data.get('Chunk_Index', 'N/A')}")
                    print(f"  Total Chunks in Section: {content_data.get('TotalChunksInSection', 'N/A')}")
                    print(f"  Content (first 200 chars): {content_data.get('Chunk_Content', '')}")
        else:
            print("\nNo context was retrieved for the query.")
        print("ANSWER:", response.answer)
        if return_context:
            return response.answer, response.retriever_result.items
        else:
            return response.answer
        
    elif classification == "entity":
        retrieval_query = retrieval_query_entity
        retriever = get_hybrid_cypher_retriever_with_indexes(
            "AccidentVectorIndex","AccidentFullTextIndex","Accident","embedding",1536,"cosine",["Description"],retrieval_query
        )
        llm = OpenAILLM(model_name="gpt-4o-mini",model_params={"temperature":0.2})
        rag = GraphRAG(retriever=retriever, llm=llm)
        response = rag.search(
            query_text=question,
            retriever_config={"top_k": 3 , "ranker": "LINEAR", "alpha": 0.7},
            return_context=True
        )
        if not return_context:
            print("ANSWER:", response.answer)
            print("\nCONTEXT:")
        for item in response.retriever_result.items:
            print(item)
        if return_context:
            return response.answer, response.retriever_result.items
        else:
            return response.answer
    
    elif classification == "both":
        policy_retriever = get_hybrid_retriever_with_indexes(
            "PolicyVectorIndex","PolicyFullTextIndex","Chunk","embedding",1536,"cosine",["Chunk_Content"]
        )
        policy_response = policy_retriever.search(
            query_text=question,
            top_k=2,
            effective_search_ratio=2,
            ranker="LINEAR",
            alpha=0.7,
        )
        
        # 2. Get entity retriever and search
        entity_retrieval = retrieval_query_entity
        entity_retriever = get_hybrid_cypher_retriever_with_indexes(
            "AccidentVectorIndex","AccidentFullTextIndex","Accident","embedding",1536,"cosine",["Description"],entity_retrieval
        )
        entity_response = entity_retriever.search(
            query_text=question,
            top_k=2,
            effective_search_ratio=2,
            ranker="LINEAR",
            alpha=0.7,
        )
        
        # 3. Combine contexts from both retrievers
        combined_context = []
        
        # Add policy context
        if policy_response and policy_response.items:
            for item in policy_response.items:
                combined_context.append({
                    "source": "policy",
                    "content": item.content,
                    "metadata": item.metadata
                })
        
        # Add entity context
        if entity_response and entity_response.items:
            for item in entity_response.items:
                combined_context.append({
                    "source": "entity", 
                    "content": item.content,
                    "metadata": item.metadata
                })
        
        # 4. Create combined context string for the LLM
        context_text = ""
        for i, ctx in enumerate(combined_context):
            context_text += f"\n--- Context {i+1} (from {ctx['source']}) ---\n"
            context_text += str(ctx['content'])
            context_text += "\n"
        
        # 5. Use LLM to answer based on combined context
        llm = OpenAILLM(model_name="gpt-4o-mini",model_params={"temperature":0.2})
        
        # Create a prompt that includes both contexts
        prompt = f"""
You are an expert insurance claims analyst and policy specialist with years of experience in handling complex insurance cases. You have access to comprehensive policy documentation and detailed accident/entity records.

Your task is to provide a thorough, professional analysis by examining both policy terms and specific case details. You should approach this as a seasoned professional who can seamlessly connect policy provisions with real-world scenarios.

QUESTION TO ANALYZE: {question}

AVAILABLE INFORMATION:
{context_text}

INSTRUCTIONS FOR YOUR ANALYSIS:
1. As an insurance expert, first identify the key policy provisions that apply to this situation
2. Then examine the specific entity details (accidents, drivers, vehicles) that are relevant
3. Provide a comprehensive professional assessment that connects the policy terms to the specific case details
4. If there are any coverage determinations to be made, explain your reasoning clearly
5. Maintain a professional, authoritative tone befitting an experienced insurance analyst

Please provide your expert analysis and recommendations based on the available information.
"""
        
        # Get LLM response
        llm_response = llm.invoke(prompt)
        
        if not return_context:
            print("ANSWER:", llm_response.content)
            print(f"\nCONTEXT (Policy: {len([c for c in combined_context if c['source'] == 'policy'])}, Entity: {len([c for c in combined_context if c['source'] == 'entity'])}):")
            for i, ctx in enumerate(combined_context):
                print(f"Context {i+1} ({ctx['source']}): {str(ctx['content'])[:200]}...")
        
        if return_context:
            return llm_response.content, combined_context
        else:
            return llm_response.content
    
    else:
        return "unknown"
    

def get_graph_rag_tool() -> Tool:

    tool_name = "graph_rag_tool"
    tool_description = "Use this tool to answer QnA questions about insurance policies or entities (drivers, cars, accidents) and return answers."
    
    graph_rag_tool_instance = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda question: graph_rag_tool(question),
        args_schema=None,
    )
    return graph_rag_tool_instance


