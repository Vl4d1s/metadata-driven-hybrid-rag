import os
from typing import List

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
from neo4j_graphrag.retrievers import HybridCypherRetriever,VectorRetriever , VectorCypherRetriever , Text2CypherRetriever  , HybridRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index , retrieve_vector_index_info , create_fulltext_index , retrieve_fulltext_index_info
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

def get_hybrid_retriever_with_indexes(
    vector_index_name: str,
    fulltext_index_name: str,
    node_label: str,
    embedding_property: str,
    vector_dimensions: int,
    vector_similarity_fn: str,
    fulltext_node_properties: List[str],
):

    # 1. Check and Create Vector Index
    print(f"Checking for vector index '{vector_index_name}'...")
    vector_index_info = retrieve_vector_index_info(
        driver,
        index_name=vector_index_name,
        label_or_type=node_label,
        embedding_property=embedding_property,
    ) # [9, 10]

    if vector_index_info is None:
        print(f"Vector index '{vector_index_name}' does not exist. Creating it now...")
        create_vector_index(
            driver,
            name=vector_index_name,
            label=node_label,
            embedding_property=embedding_property,
            dimensions=vector_dimensions,
            similarity_fn=vector_similarity_fn,
            fail_if_exists=False,  # Prevent error if index already exists [1]
        ) # [1, 7, 11, 12]
        print(f"Vector index '{vector_index_name}' created successfully.")
    else:
        print(f"Vector index '{vector_index_name}' already exists.")

    # 2. Check and Create Fulltext Index
    print(f"Checking for fulltext index '{fulltext_index_name}'...")
    fulltext_index_info = retrieve_fulltext_index_info(
        driver,
        index_name=fulltext_index_name,
        label_or_type=node_label,
        text_properties=fulltext_node_properties,
    ) # [10, 13, 14]

    if fulltext_index_info is None:
        print(f"Fulltext index '{fulltext_index_name}' does not exist. Creating it now...")
        create_fulltext_index(
            driver,
            name=fulltext_index_name,
            label=node_label,
            node_properties=fulltext_node_properties,
            fail_if_exists=False,  # Prevent error if index already exists [2]
        ) # [2, 7, 15, 16]
        print(f"Fulltext index '{fulltext_index_name}' created successfully.")
    else:
        print(f"Fulltext index '{fulltext_index_name}' already exists.")

    # 3. Initialize and Return HybridRetriever
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(
        driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        embedder=embedder, # An embedder is required to query by text [5, 17]
    ) # [3, 5, 18]
    print("HybridRetriever initialized.")
    return retriever

def get_hybrid_cypher_retriever_with_indexes(
    vector_index_name: str,
    fulltext_index_name: str,
    node_label: str,
    embedding_property: str,
    vector_dimensions: int,
    vector_similarity_fn: str,
    fulltext_node_properties: List[str],
    retrieval_query: str,
) -> HybridCypherRetriever:

    # 1. Check and Create Vector Index
    print(f"Checking for vector index '{vector_index_name}'...")
    vector_index_info = retrieve_vector_index_info(
        driver,
        index_name=vector_index_name,
        label_or_type=node_label,
        embedding_property=embedding_property,
    ) # [9, 10]

    if vector_index_info is None:
        print(f"Vector index '{vector_index_name}' does not exist. Creating it now...")
        create_vector_index(
            driver,
            name=vector_index_name,
            label=node_label,
            embedding_property=embedding_property,
            dimensions=vector_dimensions,
            similarity_fn=vector_similarity_fn,
            fail_if_exists=False,  # Prevent error if index already exists [1]
        ) # [1, 7, 11, 12]
        print(f"Vector index '{vector_index_name}' created successfully.")
    else:
        print(f"Vector index '{vector_index_name}' already exists.")

    # 2. Check and Create Fulltext Index
    print(f"Checking for fulltext index '{fulltext_index_name}'...")
    fulltext_index_info = retrieve_fulltext_index_info(
        driver,
        index_name=fulltext_index_name,
        label_or_type=node_label,
        text_properties=fulltext_node_properties,
    ) # [10, 13, 14]

    if fulltext_index_info is None:
        print(f"Fulltext index '{fulltext_index_name}' does not exist. Creating it now...")
        create_fulltext_index(
            driver,
            name=fulltext_index_name,
            label=node_label,
            node_properties=fulltext_node_properties,
            fail_if_exists=False,  # Prevent error if index already exists [2]
        ) # [2, 7, 15, 16]
        print(f"Fulltext index '{fulltext_index_name}' created successfully.")
    else:
        print(f"Fulltext index '{fulltext_index_name}' already exists.")

    # 3. Initialize and Return HybridRetriever
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(
        driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        embedder=embedder, # An embedder is required to query by text [5, 17]
        retrieval_query=retrieval_query,
    ) # [3, 5, 18]
    print("HybridRetriever initialized.")
    return retriever

def vector_retriever(index_name: str,label: str,embedding_property: str,return_properties: list):
    # return the vector retriver
    # retrive the chunk by vector similarity, with the properties selected
    index_info = retrieve_vector_index_info(
        driver,
        index_name,
        label_or_type=label,
        embedding_property=embedding_property
    )
    print("index_info",index_info)
    if index_info is None:
        print(f"Vector index '{index_name}' does not exist. Creating it now...")
        create_vector_index(
            driver,
            index_name,
            label=label,
            embedding_property=embedding_property,
            dimensions=1536,
            similarity_fn="cosine",
            fail_if_exists=False, # Set to False to prevent error if it somehow exists
        )
        print(f"Vector index '{index_name}' created.")
    else:
        print(f"Vector index '{index_name}' already exists: {index_info}")
    retriever = VectorRetriever(
        driver,
        index_name=index_name,
        embedder=embedder,
        # return_properties=return_properties,
    )
    return retriever

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
    
    # Extract the classification from the result
    classification = result.get('output', 'unknown')

    if classification == "policy":
        # vector retriever for policy
        # retriever = vector_retriever("PolicyVectorIndex","Chunk","embedding",["Chunk_Content"])
        # llm = OpenAILLM(model_name="gpt-4o-mini",model_params={"temperature":0.2})
        # rag = GraphRAG(retriever=retriever, llm=llm)
        # response = rag.search(
        #     query_text=question, 
        #     retriever_config={"top_k": 3},
        #     return_context=True
        #     )


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

        print("\n--- LLM Generated Answer ---")
        # print(response)
                        
        if response.retriever_result:
            print("\n--- Retrieved Context Details ---")
            print(f"Number of retrieved items: {len(response.retriever_result.items)}")
            
            # Print retriever metadata if available
            # if response.retriever_result.metadata:
            #     print(f"Retriever metadata: {response.retriever_result.metadata}")
            
            for i, item in enumerate(response.retriever_result.items):
                print(f"\nItem {i+1}:")
                print(f"  Score: {item.metadata.get('score', 'N/A')}")  # Score is in metadata
                
                # Parse the content string to extract relevant information
                try:
                    import json
                    content_data = json.loads(item.content)
                except (json.JSONDecodeError, ValueError):
                    # Fallback to eval if not valid JSON (use with caution)
                    content_data = eval(item.content)
                
                print(f"  Document ID: {content_data.get('DocumentId', 'N/A')}")
                print(f"  Page Number: {content_data.get('PageNumber', 'N/A')}")
                print(f"  Section: {content_data.get('SectionName', 'N/A')}")
                print(f"  Chunk Index: {content_data.get('Chunk_Index', 'N/A')}")
                print(f"  Total Chunks in Section: {content_data.get('TotalChunksInSection', 'N/A')}")
                print(f"  Content (first 200 chars): {content_data.get('Chunk_Content', '')}")
        else:
            print("\nNo context was retrieved for the query.")
        # context = []
        # for item in response.retriever_result.items:
        #     context.append(item.content)
        # print("CONTEXT:", context)
        print("ANSWER:", response.answer)

        return response.answer
        
    elif classification == "entity":
         print("ANSWER:", response.answer)
         print("\nCONTEXT:")
         for item in response.retriever_result.items:
             print(f"Accident ID: {item['accidentId']}")
             print(f"Description: {item['accidentDescription']}")
             print(f"Location: {item['location']}")
             print(f"City: {item['city']}")
             print(f"Date: {item['date']} at {item['time']}")
             print(f"Vehicles Involved: {item['numberOfVehicles']}")
             print(f"Police Agency: {item['policeAgency']}")
             print(f"Police Report Made: {item['policeReportMade']}")
             print(f"Similarity Score: {item['similarityScore']}")
             
             print("\nInvolved Cars:")
             for car in item['involvedCars']:
                 print(f"  - {car['makeAndModel']} ({car['year']}) - License: {car['licensePlate']}")
                 print(f"    Total Accidents: {car['totalAccidents']}")
                 if car['otherAccidents']:
                     print("    Other Accidents:")
                     for acc in car['otherAccidents']:
                         print(f"      * {acc['accidentId']} - {acc['date']} {acc['time']} at {acc['location']}, {acc['city']}")
                 else:
                     print("    No other accidents found")
             
             print("\nInvolved Drivers:")
             for driver in item['involvedDrivers']:
                 print(f"  - {driver['firstName']} {driver['lastName']} (DOB: {driver['dateOfBirth']})")
                 print(f"    License: {driver['licenseNumber']}")
                 print(f"    Address: {driver['houseNumber']} {driver['street']}, {driver['city']} {driver['zipCode']}")
                 print(f"    Phone: {driver['phone']}")
                 print(f"    Total Accidents: {driver['totalAccidents']}")
                 if driver['otherAccidents']:
                     print("    Other Accidents:")
                     for acc in driver['otherAccidents']:
                         print(f"      * {acc['accidentId']} - {acc['date']} {acc['time']} at {acc['location']}, {acc['city']}")
                 else:
                     print("    No other accidents found")
             
             print("-" * 50)
         
         return "no answer found for this question"
    else:
        return "unknown"
    
    # Return a meaningful response based on classification
    return f"Question classified as: {classification}. This question is about {classification} and should be routed to the appropriate handler."


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


