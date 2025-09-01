import os
from typing import List
import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')

from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever,VectorRetriever , VectorCypherRetriever , Text2CypherRetriever  , HybridRetriever
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
