from typing import List, Dict, Any, Optional
from langchain.tools import Tool
from .prompts import examples
from .._functions.text_to_cypher import text_to_cypher_pipeline


def format_results(records: List, question: str) -> str:
    """
    Format the Cypher query results into a human-readable answer.
    
    Args:
        records: List of Neo4j records from the query
        question: Original question for context
        
    Returns:
        Formatted string answer
    """
    if not records:
        return "No results found for your question."
    
    # Handle different types of results
    if len(records) == 1 and len(records[0].values()) == 1:
        # Single value result (like COUNT queries)
        value = records[0].values()[0]
        if isinstance(value, (int, float)):
            return f"The answer is: {value}"
        else:
            return str(value)
    
    elif len(records) == 1 and len(records[0].values()) > 1:
        # Single record with multiple fields
        result_dict = records[0].data()
        formatted_pairs = [f"{key}: {value}" for key, value in result_dict.items()]
        return ", ".join(formatted_pairs)
    
    else:
        # Multiple records - create a formatted list
        results = []
        for i, record in enumerate(records[:10]):  # Limit to first 10 results
            if len(record.values()) == 1:
                results.append(str(record.values()[0]))
            else:
                result_dict = record.data()
                formatted_pairs = [f"{key}: {value}" for key, value in result_dict.items()]
                results.append("(" + ", ".join(formatted_pairs) + ")")
        
        answer = "Results:\n" + "\n".join(f"{i+1}. {result}" for i, result in enumerate(results))
        
        if len(records) > 10:
            answer += f"\n... and {len(records) - 10} more results"
        
        return answer


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
            examples=examples,
        )
        
        # Debug: Print the type and attributes of search_result
        return search_result
        print(f"DEBUG: search_result type: {type(search_result)}")
        print(f"DEBUG: search_result attributes: {search_result}")
        
        # RawSearchResult should have 'records' attribute
        if hasattr(search_result, 'records'):
            records = search_result.records
            print(f"DEBUG: Found {len(records)} records")
        else:
            print("DEBUG: No 'records' attribute found")
            records = []
        
        # Format the results into a readable answer
        formatted_answer = format_results(records, question)
        
        return formatted_answer
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing statistical query: {str(e)}"


def get_statistical_tool() -> Tool:
    """
    Creates a LangChain Tool for answering statistical questions using text-to-cypher conversion.
    
    Returns:
        Tool: LangChain Tool instance for statistical functionality
    """
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
