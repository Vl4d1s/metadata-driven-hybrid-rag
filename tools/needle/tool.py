import os
from typing import List, Dict, Any
import json

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
from neo4j_graphrag.retrievers import HybridCypherRetriever, VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, HybridRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index, retrieve_vector_index_info, create_fulltext_index, retrieve_fulltext_index_info

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD") 
    )
)

# Create embedder
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
    """
    Creates a hybrid retriever with vector and fulltext indexes for needle functionality.
    """
    # 1. Check and Create Vector Index
    print(f"Checking for vector index '{vector_index_name}'...")
    vector_index_info = retrieve_vector_index_info(
        driver,
        index_name=vector_index_name,
        label_or_type=node_label,
        embedding_property=embedding_property,
    )

    if vector_index_info is None:
        print(f"Vector index '{vector_index_name}' does not exist. Creating it now...")
        create_vector_index(
            driver,
            name=vector_index_name,
            label=node_label,
            embedding_property=embedding_property,
            dimensions=vector_dimensions,
            similarity_fn=vector_similarity_fn,
            fail_if_exists=False,
        )
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
    )

    if fulltext_index_info is None:
        print(f"Fulltext index '{fulltext_index_name}' does not exist. Creating it now...")
        create_fulltext_index(
            driver,
            name=fulltext_index_name,
            label=node_label,
            node_properties=fulltext_node_properties,
            fail_if_exists=False,
        )
        print(f"Fulltext index '{fulltext_index_name}' created successfully.")
    else:
        print(f"Fulltext index '{fulltext_index_name}' already exists.")

    # 3. Initialize and Return HybridRetriever
    print("Initializing HybridRetriever for needle...")
    retriever = HybridRetriever(
        driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        embedder=embedder,
    )
    print("HybridRetriever initialized for needle.")
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
    """
    Creates a hybrid cypher retriever with custom query for needle functionality.
    """
    # 1. Check and Create Vector Index
    print(f"Checking for vector index '{vector_index_name}'...")
    vector_index_info = retrieve_vector_index_info(
        driver,
        index_name=vector_index_name,
        label_or_type=node_label,
        embedding_property=embedding_property,
    )

    if vector_index_info is None:
        print(f"Vector index '{vector_index_name}' does not exist. Creating it now...")
        create_vector_index(
            driver,
            name=vector_index_name,
            label=node_label,
            embedding_property=embedding_property,
            dimensions=vector_dimensions,
            similarity_fn=vector_similarity_fn,
            fail_if_exists=False,
        )
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
    )

    if fulltext_index_info is None:
        print(f"Fulltext index '{fulltext_index_name}' does not exist. Creating it now...")
        create_fulltext_index(
            driver,
            name=fulltext_index_name,
            label=node_label,
            node_properties=fulltext_node_properties,
            fail_if_exists=False,
        )
        print(f"Fulltext index '{fulltext_index_name}' created successfully.")
    else:
        print(f"Fulltext index '{fulltext_index_name}' already exists.")

    # 3. Initialize and Return HybridCypherRetriever
    print("Initializing HybridCypherRetriever for needle...")
    retriever = HybridCypherRetriever(
        driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        embedder=embedder,
        retrieval_query=retrieval_query,
    )
    print("HybridCypherRetriever initialized for needle.")
    return retriever

def generate_answer_from_locations(query: str, locations: List[Dict[str, Any]]) -> str:
    """
    Generate an answer based on the found locations using LLM.
    
    Args:
        query: The original query
        locations: List of location dictionaries with content
        
    Returns:
        Generated answer based on the locations
    """
    if not locations:
        return "No relevant locations found to answer the query."
    
    # Prepare context from all locations with structured information
    context_parts = []
    context_metadata = []
    
    for i, location in enumerate(locations, 1):
        if 'full_content' in location:
            # Policy document location
            page = location.get('page_number', 'N/A')
            chunk = location.get('chunk_index', 'N/A')
            score = location.get('similarity_score', 'N/A')
            section = location.get('section_name', 'N/A')
            content = location['full_content']
            doc_id = location.get('document_id', 'N/A')
            
            context_parts.append(f"Location {i} (Page {page}, Section: {section}):\n{content}")
            context_metadata.append({
                'page': page,
                'chunk': chunk,
                'score': score,
                'section': section,
                'content': content,
                'document': doc_id.split('\\')[-1] if '\\' in str(doc_id) else doc_id
            })
        elif 'description' in location:
            # Entity location
            entity_id = location.get('entity_id', 'N/A')
            entity_type = location.get('entity_type', 'N/A')
            description = location.get('description', 'N/A')
            score = location.get('similarity_score', 'N/A')
            
            context_parts.append(f"Entity {i} (ID: {entity_id}, Type: {entity_type}):\n{description}")
            context_metadata.append({
                'entity_id': entity_id,
                'entity_type': entity_type,
                'score': score,
                'description': description
            })
    
    context = "\n\n".join(context_parts)
    
    # Prepare metadata for the prompt
    metadata_info = []
    for meta in context_metadata:
        if 'page' in meta:
            metadata_info.append(f"(page {meta['page']}, chunk {meta['chunk']}, score {meta['score']}, section \"{meta['section']}\", document \"{meta['document']}\")")
        else:
            metadata_info.append(f"(entity {meta['entity_id']}, type {meta['entity_type']}, score {meta['score']})")
    
    # Create prompt for answer generation
    prompt = f"""Based on the following context from insurance documents, answer the question: "{query}"

Context:
{context}

Available Context Metadata:
{chr(10).join(metadata_info)}

Instructions:
1. Start your answer by stating which document the information was found in
2. Provide a direct, specific answer to the question
3. Quote relevant text from the context in your answer
4. If specific amounts, dates, or details are mentioned, include them
5. Be concise but complete
6. After your answer, add a section titled "Context used to answer:" and list ONLY the most relevant context that was actually used to answer the question in this exact format:
   (page X, chunk Y, score Z, section "Section Name", content "most relevant text excerpt used for the answer")

Answer:"""
    
    try:
        # Initialize LLM for answer generation
        llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.1})
        response = llm.invoke(prompt)
        # Extract the actual text content from the LLM response
        if hasattr(response, 'content'):
            answer = response.content
        elif hasattr(response, 'text'):
            answer = response.text
        else:
            answer = str(response)
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

def extract_location_info(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts location/anchor information from content data.
    
    Args:
        content_data: Dictionary containing chunk/content information
        
    Returns:
        Dictionary with location anchor information
    """
    location_info = {
        "document_id": content_data.get('DocumentId', 'N/A'),
        "page_number": content_data.get('PageNumber', 'N/A'),
        "section_name": content_data.get('SectionName', 'N/A'),
        "chunk_index": content_data.get('Chunk_Index', 'N/A'),
        "total_chunks_in_section": content_data.get('TotalChunksInSection', 'N/A'),
        "anchor": f"doc_{content_data.get('DocumentId', 'unknown')}_page_{content_data.get('PageNumber', 'unknown')}_section_{content_data.get('SectionName', 'unknown').replace(' ', '_')}_chunk_{content_data.get('Chunk_Index', 'unknown')}",
        "content_preview": content_data.get('Chunk_Content', '')[:100] + "..." if len(content_data.get('Chunk_Content', '')) > 100 else content_data.get('Chunk_Content', ''),
        "full_content": content_data.get('Chunk_Content', '')
    }
    return location_info

def needle_tool(query: str) -> str:
    """
    Needle tool that finds and returns specific anchor/paragraph/section locators.
    
    Args:
        query: The search query to find specific content locations
        
    Returns:
        JSON string containing location information and anchors
    """
    from agents.router.agent import get_router_agent
    
    # Define options for the router
    options = ["policy", "entity"]
    
    # Define examples to help the router classify questions for needle search
    examples = """
Question: "Find the section about collision coverage"
Classification: policy
Reasoning: Looking for policy section, no specific entity mentioned

Question: "Locate deductible information in the policy"
Classification: policy
Reasoning: Seeking policy terms, no specific entity identifier

Question: "Find the paragraph about vehicle damage"
Classification: policy
Reasoning: General topic search in policy, no specific entity name/ID mentioned

Question: "Find information about driver John Smith"
Classification: entity
Reasoning: Specific driver name mentioned - clear entity identifier

Question: "Locate accident details for accident ID ACC-001"
Classification: entity
Reasoning: Specific accident ID mentioned - clear entity identifier

Question: "Find driver with license number 445-78-9012"
Classification: entity
Reasoning: Specific driver license number mentioned - clear entity identifier

Question: "Show me details about car with plate number ABC123"
Classification: entity
Reasoning: Specific license plate mentioned - clear entity identifier

Question: "Find accident on Main Street on 2023-01-15"
Classification: entity
Reasoning: Specific location and date mentioned - clear entity identifiers

Question: "Locate driver named Sarah Johnson"
Classification: entity
Reasoning: Specific driver name mentioned - clear entity identifier

Question: "Find accident involving BMW X5"
Classification: entity
Reasoning: Specific car model mentioned - entity identifier

IMPORTANT RULES:
- Only classify as 'entity' if the question contains specific identifiers like:
  * Driver names 
  * Driver IDs or license numbers 
  * Car details (license plates, specific models)
  * Accident IDs (ACC-001, ACC-002 , etc.)
  * Specific locations with dates of accidents or cities of drivers
  * Any other unique entity identifiers

- Classify as 'policy' for general topics or concepts without specific entity identifiers:
  * General coverage topics (vehicle damage, collision, theft)
  * Policy terms and conditions
  * Benefits and limits
  * General procedures
"""
    
    # Get the router agent
    router_agent = get_router_agent(options, examples)
    
    # Run the router agent with the query
    result = router_agent.invoke({"question": query})
    
    # Print the router result
    print(f"Needle Router Result: {result}")
    
    # Extract the classification from the result
    classification = result.get('output', 'unknown')
    
    needle_results = {
        "query": query,
        "classification": classification,
        "locations": [],
        "anchors": [],
        "total_found": 0
    }

    if classification == "policy":
        # Use hybrid retriever for policy content location finding
        retriever = get_hybrid_retriever_with_indexes(
            "PolicyVectorIndex", "PolicyFullTextIndex", "Chunk", "embedding", 1536, "cosine", ["Chunk_Content"]
        )
        
        # Search for relevant chunks without generating answers
        search_results = retriever.search(
            query_text=query,
            top_k=3  # Get top 3 most relevant results for needle functionality
        )
        
        print(f"\n--- Needle Search Results for Policy ---")
        # print(f"Found {len(search_results.items)} relevant locations")
        
        for i, item in enumerate(search_results.items):
            try:
                # Parse the content string to extract relevant information
                try:
                    content_data = json.loads(item.content)
                except (json.JSONDecodeError, ValueError):
                    # Fallback to eval if not valid JSON (use with caution)
                    content_data = eval(item.content)
                
                # Extract location information
                location_info = extract_location_info(content_data)
                location_info["similarity_score"] = item.metadata.get('score', 'N/A')
                location_info["rank"] = i + 1
                
                needle_results["locations"].append(location_info)
                needle_results["anchors"].append(location_info["anchor"])
                
                # print(f"\nLocation {i+1}:")
                # print(f"  Anchor: {location_info['anchor']}")
                # print(f"  Document: {location_info['document_id']}")
                # print(f"  Page: {location_info['page_number']}")
                # print(f"  Section: {location_info['section_name']}")
                # print(f"  Chunk: {location_info['chunk_index']}/{location_info['total_chunks_in_section']}")
                # print(f"  Score: {location_info['similarity_score']}")
                # print(f"  Preview: {location_info['content_preview']}")
                
            except Exception as e:
                print(f"Error processing item {i+1}: {e}")
                
        needle_results["total_found"] = len(needle_results["locations"])
        
        # Generate answer based on found locations
        if needle_results["locations"]:
            needle_results["generated_answer"] = generate_answer_from_locations(query, needle_results["locations"])
        
    elif classification == "entity":
        # Custom retrieval query for entity location finding
        retrieval_query = """
        RETURN 
          node.Id AS entityId,
          node.Description AS description,
          node.Location AS location,
          node.Date AS date,
          node.Time AS time,
          node.City AS city,
          score AS similarityScore,
          'Accident' AS entityType,
          'accident_' + node.Id AS anchor
        ORDER BY similarityScore DESC
        """
        
        retriever = get_hybrid_cypher_retriever_with_indexes(
            "AccidentVectorIndex", "AccidentFullTextIndex", "Accident", "embedding", 1536, "cosine", ["Description"], retrieval_query
        )
        
        # Search for relevant entities
        search_results = retriever.search(
            query_text=query,
            top_k=3
        )
        
        print(f"\n--- Needle Search Results for Entities ---")
        print(f"Found {len(search_results.items)} relevant entity locations")
        
        for i, item in enumerate(search_results.items):
            try:
                # Parse entity information
                if isinstance(item.content, str):
                    try:
                        entity_data = json.loads(item.content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as plain text
                        entity_data = {
                            'entityId': f'entity_{i+1}',
                            'entityType': 'Accident',
                            'description': item.content,
                            'anchor': f'entity_accident_{i+1}'
                        }
                else:
                    entity_data = item.content
                
                location_info = {
                    "entity_id": entity_data.get('entityId', 'N/A'),
                    "entity_type": entity_data.get('entityType', 'N/A'),
                    "description": entity_data.get('description', 'N/A'),
                    "location": entity_data.get('location', 'N/A'),
                    "date": entity_data.get('date', 'N/A'),
                    "time": entity_data.get('time', 'N/A'),
                    "city": entity_data.get('city', 'N/A'),
                    "anchor": entity_data.get('anchor', f"entity_{entity_data.get('entityId', 'unknown')}"),
                    "similarity_score": entity_data.get('similarityScore', item.metadata.get('score', 'N/A')),
                    "rank": i + 1
                }
                
                needle_results["locations"].append(location_info)
                needle_results["anchors"].append(location_info["anchor"])
                
                print(f"\nEntity Location {i+1}:")
                print(f"  Anchor: {location_info['anchor']}")
                print(f"  Entity ID: {location_info['entity_id']}")
                print(f"  Type: {location_info['entity_type']}")
                print(f"  Location: {location_info['location']}")
                print(f"  Date/Time: {location_info['date']} {location_info['time']}")
                print(f"  City: {location_info['city']}")
                print(f"  Score: {location_info['similarity_score']}")
                print(f"  Description: {location_info['description'][:100]}...")
                
            except Exception as e:
                print(f"Error processing entity {i+1}: {e}")
                
        needle_results["total_found"] = len(needle_results["locations"])
        
        # Generate answer based on found entity locations
        if needle_results["locations"]:
            needle_results["generated_answer"] = generate_answer_from_locations(query, needle_results["locations"])
        
    else:
        needle_results["error"] = f"Unknown classification: {classification}"
        
    # Return results as JSON string
    return json.dumps(needle_results, indent=2)

def get_needle_tool() -> Tool:
    """
    Creates a LangChain Tool for needle functionality that finds and returns specific anchor/paragraph/section locators.
    
    Returns:
        Tool: LangChain Tool instance for needle functionality
    """
    tool_name = "needle_tool"
    tool_description = "Use this tool to find and return specific anchor/paragraph/section locators for content in insurance policies or entity data. Returns detailed location information including anchors, page numbers, sections, and content previews."
    
    needle_tool_instance = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda query: needle_tool(query),
        args_schema=None,
    )
    return needle_tool_instance
