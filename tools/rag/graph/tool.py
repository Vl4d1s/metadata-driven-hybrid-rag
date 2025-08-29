import os

# testing 
import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')
from flows.insurance.data.models.accident_case import AccidentCase

# the tool will get files path and schema
from pydantic import BaseModel
from tools._functions.llama_extractor import extract_file
from langchain.tools import Tool

from dotenv import load_dotenv
load_dotenv()




from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever , VectorCypherRetriever , Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)





# # Create embedder
# embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


# def vector_retriever():
#     # return the vector retriver
#     # retrive the chunk by vector similarity, with the properties selected
#     retriever = VectorRetriever(
#         driver,
#         index_name="moviePlots",
#         embedder=embedder,
#         return_properties=["title", "plot"],
#     )
#     return retriever

# def graph_enhanced_vector_retriever():
#     # Define retrieval query
#     #The query traverses the graph to find related nodes for genres and actors, as well as sorting the results by the user rating.
#     retrieval_query = """
#     MATCH (node)<-[r:RATED]-()
#     RETURN 
#     node.title AS title, node.plot AS plot, score AS similarityScore, 
#     collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, 
#     collect { MATCH (node)<-[:ACTED_IN]->(a) RETURN a.name } as actors, 
#     avg(r.rating) as userRating
#     ORDER BY userRating DESC
#     """

#     # Create retriever
#     retriever = VectorCypherRetriever(
#         driver,
#         index_name="moviePlots",
#         embedder=embedder,
#         retrieval_query=retrieval_query,
#     )
#     return retriever

# def text2cypher_retriver():
#     # Cypher examples as input/query pairs
#     examples = [
#         "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
#     ]
#     t2c_llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature":0})
#     retriever = Text2CypherRetriever(
#         driver=driver,
#         llm=t2c_llm,
#         examples=examples,
#     )
#     return retriever


# retriever = graph_enhanced_vector_retriever()

# # Search for similar items
# result = retriever.search(query_text="Toys coming alive", top_k=5)

# # Parse results
# for item in result.items:
#     print(item.content, item.metadata["score"])

# # Create the LLM
# llm = OpenAILLM(model_name="gpt-4o-mini",model_params={"temperature":0.2})

# # Create GraphRAG pipeline
# rag = GraphRAG(retriever=retriever, llm=llm)

# # Search
# query_text = "Find me movies about toys coming alive"
 
# response = rag.search(
#     query_text=query_text, 
#     retriever_config={"top_k": 5},
#     return_context=True
# )

# print(response.answer)
# print("CONTEXT:", response.retriever_result.items)
# # CLose the database connection
# driver.close()


def graph_rag_tool(question: str) -> str:
    """
    Graph RAG tool that uses a router agent to classify questions.
    
    Args:
        question: The question to classify
        
    Returns:
        Empty string (for now)
    """
    from agents.router.agent import get_router_agent
    
    # Define options for the router
    options = ["policy", "entity"]
    
    # Define examples to help the router classify questions
    examples = """
Question: "What is the coverage amount for collision damage?"
Classification: policy
Reasoning: Asks about policy terms and coverage details

Question: "What are the deductibles mentioned in the policy?"
Classification: policy
Reasoning: Seeks information about policy terms and conditions

Question: "Tell me about the driver John Smith"
Classification: entity
Reasoning: Asks about a specific person/driver entity

Question: "What cars are involved in the accident?"
Classification: entity
Reasoning: Asks about specific vehicles/car entities

Question: "Show me details about accident case ID 12345"
Classification: entity
Reasoning: Asks about a specific accident entity

Question: "What happened in the accident on Main Street?"
Classification: entity
Reasoning: Asks about a specific accident event/entity
"""
    
    # Get the router agent
    router_agent = get_router_agent(options, examples)
    
    # Run the router agent with the question
    result = router_agent.invoke({"question": question})
    
    # Print the router result
    print(f"Router Result: {result}")
    
    # Return empty string as requested
    return ""


def get_graph_rag_tool() -> Tool:
    """
    Creates a LangChain Tool that uses a router agent to classify questions about insurance policies or entities.
    
    Returns:
        Tool: LangChain Tool instance for graph RAG functionality
    """
    tool_name = "graph_rag_tool"
    tool_description = "Use this tool to answer QnA questions about insurance policies or entities (drivers, cars, accidents) and return answers."
    
    graph_rag_tool_instance = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda question: graph_rag_tool(question),
        args_schema=None,
    )
    return graph_rag_tool_instance


