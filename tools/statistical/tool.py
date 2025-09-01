from typing import List, Dict, Any, Optional
from langchain.tools import Tool
from .._functions.text_to_cypher import text_to_cypher_pipeline


def statistical_tool(question: str) -> str:
    """
    Statistical tool that converts natural language questions to Cypher queries 
    and returns statistical information from a Neo4j database.
    
    Args:
        question: Natural language question about statistical data
        
    Returns:
        String answer to the question
    """
    try:
        # Use the existing text_to_cypher pipeline with statistical examples
        search_result = text_to_cypher_pipeline(
            query_text=question,
            # examples=examples,
        )
        
        # Debug: Print the type and attributes of search_result
        return search_result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing statistical query: {str(e)}"


def get_statistical_tool() -> Tool:
    tool_name = "statistical_tool"
    tool_description = """Use this tool to answer statistical questions about data stored in a Neo4j database.
The tool converts natural language questions to Cypher queries and returns human-readable answers.

Examples of questions you can ask:
- "How many accidents occurred in 2024?"
- "Analyze the drivers involved in accidents, list them in descending order of the number of accidents"
- "What car models were involved in the most accidents?"
- "How many drivers were involved in more than one accident?"
- "How many sections are in the policy document?"

The tool automatically handles query generation, execution, and result formatting."""
    
    statistical_tool_instance = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda question: statistical_tool(question),
        args_schema=None,
    )
    return statistical_tool_instance
