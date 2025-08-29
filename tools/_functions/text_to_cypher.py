"""
Text to Cypher Retrieval Function

This module provides a reusable function for converting natural language queries
to Cypher queries and retrieving data from Neo4j databases using the neo4j_graphrag
Text2CypherRetriever.
"""

import os
from typing import Optional, List, Dict, Any, Callable
from neo4j import Driver, Record, GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import LLMInterface, OpenAILLM
from neo4j_graphrag.retrievers.base import RetrieverResultItem, RawSearchResult

# Neo4j driver configuration
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# OpenAI LLM configuration
llm = OpenAILLM(
    model_name="gpt-4",
    model_params={"temperature": 0}
)


def create_text_to_cypher_retriever(
    neo4j_schema: Optional[str] = None,
    examples: Optional[List[str]] = None,
    result_formatter: Optional[Callable[[Record], RetrieverResultItem]] = None,
    custom_prompt: Optional[str] = None,
    neo4j_database: Optional[str] = None
) -> Text2CypherRetriever:
    """
    Create a configured Text2CypherRetriever instance using the global driver and LLM.
    
    Args:
        neo4j_schema: Neo4j schema used to generate the Cypher query
        examples: Optional user input/query pairs for the LLM to use as examples
        result_formatter: Optional function to format the results
        custom_prompt: Optional custom prompt to use instead of auto generated prompt
        neo4j_database: Optional Neo4j database name
        
    Returns:
        Configured Text2CypherRetriever instance
        
    Raises:
        RetrieverInitializationError: If validation of the input arguments fail
    """
    return Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
        result_formatter=result_formatter,
        custom_prompt=custom_prompt,
        neo4j_database=neo4j_database
    )


def search_with_text_to_cypher(
    retriever: Text2CypherRetriever,
    query_text: str,
    prompt_params: Optional[Dict[str, Any]] = None
) -> RawSearchResult:
    """
    Search the Neo4j database using natural language query.
    
    Converts the natural language query to a Cypher query using an LLM,
    then retrieves records from the Neo4j database.
    
    Args:
        retriever: Configured Text2CypherRetriever instance
        query_text: The natural language query used to search the Neo4j database
        prompt_params: Additional values to inject into the custom prompt.
                      If the schema or examples parameter is specified, it will 
                      overwrite the corresponding value passed during initialization.
                      Example: {'schema': 'this is the graph schema'}
                      
    Returns:
        The results of the search query as a list of neo4j.Record and optional metadata
        
    Raises:
        SearchValidationError: If validation of the input arguments fail
        Text2CypherRetrievalError: If the LLM fails to generate a correct Cypher query
    """
    return retriever.search(query_text=query_text, prompt_params=prompt_params)


def text_to_cypher_pipeline(
    query_text: str,
    neo4j_schema: Optional[str] = None,
    examples: Optional[List[str]] = None,
    result_formatter: Optional[Callable[[Record], RetrieverResultItem]] = None,
    custom_prompt: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    prompt_params: Optional[Dict[str, Any]] = None
) -> RawSearchResult:
    """
    Complete pipeline for text-to-cypher retrieval in a single function call.
    
    This is a convenience function that creates a retriever and performs the search
    in one step using the global driver and LLM.
    
    Args:
        query_text: The natural language query used to search the Neo4j database
        neo4j_schema: Neo4j schema used to generate the Cypher query
        examples: Optional user input/query pairs for the LLM to use as examples
        result_formatter: Optional function to format the results
        custom_prompt: Optional custom prompt to use instead of auto generated prompt
        neo4j_database: Optional Neo4j database name
        prompt_params: Additional values to inject into the custom prompt
        
    Returns:
        The results of the search query as a list of neo4j.Record and optional metadata
        
    Raises:
        RetrieverInitializationError: If validation of the input arguments fail
        SearchValidationError: If validation of the input arguments fail
        Text2CypherRetrievalError: If the LLM fails to generate a correct Cypher query
    """
    # Create the retriever
    retriever = create_text_to_cypher_retriever(
        neo4j_schema=neo4j_schema,
        examples=examples,
        result_formatter=result_formatter,
        custom_prompt=custom_prompt,
        neo4j_database=neo4j_database
    )
    
    # Perform the search
    return search_with_text_to_cypher(
        retriever=retriever,
        query_text=query_text,
        prompt_params=prompt_params
    )


# Example usage functions for common scenarios

def create_schema_aware_retriever(
    schema: str,
    database: Optional[str] = None
) -> Text2CypherRetriever:
    """
    Create a retriever with a predefined schema for better query generation.
    
    Args:
        schema: The Neo4j schema as a string
        database: Optional Neo4j database name
        
    Returns:
        Configured Text2CypherRetriever instance with schema
    """
    return create_text_to_cypher_retriever(
        neo4j_schema=schema,
        neo4j_database=database
    )


def create_example_guided_retriever(
    examples: List[str],
    schema: Optional[str] = None,
    database: Optional[str] = None
) -> Text2CypherRetriever:
    """
    Create a retriever with predefined examples for few-shot learning.
    
    Args:
        examples: List of user input/query pairs for the LLM to use as examples
        schema: Optional Neo4j schema
        database: Optional Neo4j database name
        
    Returns:
        Configured Text2CypherRetriever instance with examples
    """
    return create_text_to_cypher_retriever(
        neo4j_schema=schema,
        examples=examples,
        neo4j_database=database
    )


def quick_text_to_cypher_search(
    query: str,
    schema: Optional[str] = None
) -> RawSearchResult:
    """
    Quick utility function for simple text-to-cypher searches.
    
    Args:
        query: The natural language query
        schema: Optional Neo4j schema
        
    Returns:
        Search results
    """
    return text_to_cypher_pipeline(
        query_text=query,
        neo4j_schema=schema
    )
