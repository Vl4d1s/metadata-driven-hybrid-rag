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
    # print(f"Checking for vector index '{vector_index_name}'...")
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
    # else:
        # print(f"Vector index '{vector_index_name}' already exists.")

    # 2. Check and Create Fulltext Index
    # print(f"Checking for fulltext index '{fulltext_index_name}'...")
    fulltext_index_info = retrieve_fulltext_index_info(
        driver,
        index_name=fulltext_index_name,
        label_or_type=node_label,
        text_properties=fulltext_node_properties,
    ) # [10, 13, 14]

    if fulltext_index_info is None:
        # print(f"Fulltext index '{fulltext_index_name}' does not exist. Creating it now...")
        create_fulltext_index(
            driver,
            name=fulltext_index_name,
            label=node_label,
            node_properties=fulltext_node_properties,
            fail_if_exists=False,  # Prevent error if index already exists [2]
        ) # [2, 7, 15, 16]
        print(f"Fulltext index '{fulltext_index_name}' created successfully.")
    # else:
    #     print(f"Fulltext index '{fulltext_index_name}' already exists.")

    # 3. Initialize and Return HybridRetriever
    # print("Initializing HybridRetriever...")
    retriever = HybridRetriever(
        driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        embedder=embedder, # An embedder is required to query by text [5, 17]
    ) # [3, 5, 18]
    # print("HybridRetriever initialized.")
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
    retriever = HybridCypherRetriever(
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



def graph_rag_tool(question: str,return_context: bool = False):
    """
    Graph RAG tool that uses a router agent to classify questions.
    
    Args:
        question: The question to classify
        
    Returns:
        Empty string (for now)
    """
    from agents.router.agent import get_router_agent
    
    # Define options for the router
    options = ["policy", "entity", "both"]
    
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

Question: "What is the coverage for the accident that happened on Main Street?"
Classification: both
Reasoning: Asks about policy coverage AND specific accident entity

Question: "Are the vehicles involved in accident 12345 covered under the policy?"
Classification: both
Reasoning: Asks about policy coverage AND specific accident/vehicle entities
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
            
            # Print retriever metadata if available
            # if response.retriever_result.metadata:
            #     print(f"Retriever metadata: {response.retriever_result.metadata}")
            
            for i, item in enumerate(response.retriever_result.items):
                if not return_context:
                    print(f"\nItem {i+1}:")
                    print(f"  Score: {item.metadata.get('score', 'N/A')}")  # Score is in metadata
                
                # Parse the content string to extract relevant information
                try:
                    import json
                    content_data = json.loads(item.content)
                except (json.JSONDecodeError, ValueError):
                    # Fallback to eval if not valid JSON (use with caution)
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
        # context = []
        # for item in response.retriever_result.items:
        #     context.append(item.content)
        # print("CONTEXT:", context)
        print("ANSWER:", response.answer)
        if return_context:
            return response.answer, response.retriever_result.items
        else:
            return response.answer
        return response.answer
        
    elif classification == "entity":
        retrieval_query = """
RETURN 
  node.Id AS accidentId,
  node.Description AS accidentDescription,
  node.Location AS location,
  node.Date AS date,
  node.Time AS time,
  node.City AS city,
  node.NumberOfVehiclesInvolved AS numberOfVehicles,
  node.PoliceAgency AS policeAgency,
  node.PoliceReportMade AS policeReportMade,
  score AS similarityScore,
  collect { 
    MATCH (node)<-[:INVOLVED_AT]-(c:Car) 
    OPTIONAL MATCH (c)-[:INVOLVED_AT]->(otherAcc:Accident)
    WHERE otherAcc <> node
    WITH c, collect(DISTINCT otherAcc) AS otherAccidents
    RETURN {
      nodeType: 'Car',
      makeAndModel: c.MakeAndModel,
      year: c.Year,
      licensePlate: c.LicensePlate,
      otherAccidents: [acc IN otherAccidents | {
        accidentId: acc.Id,
        date: acc.Date,
        time: acc.Time,
        location: acc.Location,
        city: acc.City
      }],
      totalAccidents: size(otherAccidents) + 1
    }
  } as involvedCars,
  collect { 
    MATCH (node)<-[:INVOLVED_AT]-(d:Driver) 
    OPTIONAL MATCH (d)-[:INVOLVED_AT]->(otherAcc:Accident)
    WHERE otherAcc <> node
    WITH d, collect(DISTINCT otherAcc) AS otherAccidents
    RETURN {
      nodeType: 'Driver',
      firstName: d.FirstName,
      lastName: d.LastName,
      dateOfBirth: d.DateOfBirth,
      licenseNumber: d.LicenseNumber,
      phone: d.Phone,
      city: d.City,
      street: d.Street,
      houseNumber: d.HouseNumber,
      zipCode: d.ZipCode,
      idNumber: d.IdNumber,
      otherAccidents: [acc IN otherAccidents | {
        accidentId: acc.Id,
        date: acc.Date,
        time: acc.Time,
        location: acc.Location,
        city: acc.City
      }],
      totalAccidents: size(otherAccidents) + 1
    }
  } as involvedDrivers
ORDER BY similarityScore DESC
"""
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
        return "no answer found for this question"
    
    elif classification == "both":
        # Retrieve from both policy and entity indexes with 2 chunks each
        
        # 1. Get policy retriever and search
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
        entity_retrieval_query = """
RETURN 
  node.Id AS accidentId,
  node.Description AS accidentDescription,
  node.Location AS location,
  node.Date AS date,
  node.Time AS time,
  node.City AS city,
  node.NumberOfVehiclesInvolved AS numberOfVehicles,
  node.PoliceAgency AS policeAgency,
  node.PoliceReportMade AS policeReportMade,
  score AS similarityScore,
  collect { 
    MATCH (node)<-[:INVOLVED_AT]-(c:Car) 
    OPTIONAL MATCH (c)-[:INVOLVED_AT]->(otherAcc:Accident)
    WHERE otherAcc <> node
    WITH c, collect(DISTINCT otherAcc) AS otherAccidents
    RETURN {
      nodeType: 'Car',
      makeAndModel: c.MakeAndModel,
      year: c.Year,
      licensePlate: c.LicensePlate,
      otherAccidents: [acc IN otherAccidents | {
        accidentId: acc.Id,
        date: acc.Date,
        time: acc.Time,
        location: acc.Location,
        city: acc.City
      }],
      totalAccidents: size(otherAccidents) + 1
    }
  } as involvedCars,
  collect { 
    MATCH (node)<-[:INVOLVED_AT]-(d:Driver) 
    OPTIONAL MATCH (d)-[:INVOLVED_AT]->(otherAcc:Accident)
    WHERE otherAcc <> node
    WITH d, collect(DISTINCT otherAcc) AS otherAccidents
    RETURN {
      nodeType: 'Driver',
      firstName: d.FirstName,
      lastName: d.LastName,
      dateOfBirth: d.DateOfBirth,
      licenseNumber: d.LicenseNumber,
      phone: d.Phone,
      city: d.City,
      street: d.Street,
      houseNumber: d.HouseNumber,
      zipCode: d.ZipCode,
      idNumber: d.IdNumber,
      otherAccidents: [acc IN otherAccidents | {
        accidentId: acc.Id,
        date: acc.Date,
        time: acc.Time,
        location: acc.Location,
        city: acc.City
      }],
      totalAccidents: size(otherAccidents) + 1
    }
  } as involvedDrivers
ORDER BY similarityScore DESC
"""
        entity_retriever = get_hybrid_cypher_retriever_with_indexes(
            "AccidentVectorIndex","AccidentFullTextIndex","Accident","embedding",1536,"cosine",["Description"],entity_retrieval_query
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


