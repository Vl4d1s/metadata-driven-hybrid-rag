"""
Unified Summary Tool that uses Router Agent for intelligent routing
"""

from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from agents.router.agent import get_router_agent
from .prompts import create_router_examples
import re
from tools.summary.map_reduce.tool import create_mapreduce_chain
from tools._functions.text_to_cypher import quick_text_to_cypher_search

class SummaryToolInput(BaseModel):
    """Input schema for the summary tool"""
    question: str = Field(description="The question from the user")


def process_summary_question(question: str, data_path: str) -> str:
    """
    Process the summary question using router agent to determine if it's:
    1. Document summary (policy documents)
    2. Entity-specific summary (driver, accident, car with IDs)
    """
    
    # Define routing options
    routing_options = ["document_summary", "entity_summary"]
    
    # Define examples for router agent
    router_examples = create_router_examples()
    
    # Create router agent with examples
    router_agent = get_router_agent(options=routing_options, examples=router_examples)
    
    # Use router to classify the question
    router_result = router_agent.invoke({"question": question})
    print(f"Router Result: {router_result}")
    classification = router_result.get("output", "").strip().lower()
    print(f"Router Result: {classification}")
    summary = ""
    if classification == "document_summary":
        summary = create_mapreduce_chain(data_path)
        print("Document Summary Request - Processing policy documents")
    elif classification == "entity_summary":
        print("Entity Summary Request - Processing entity summaries")
        
        # Use text_to_cypher to find the entity and its relationships
        try:
            # Define the custom Cypher query template and instructions
            cypher_instructions = """
            Use this Cypher query pattern to find the node without embedding:
            
            MATCH (n:YourLabel {yourProperty: 'someValue'})
            RETURN apoc.map.removeKey(properties(n), 'embedding') AS filteredNode
            
            Label and property mappings:
            - Driver label uses property: idNumber
            - Accident label uses property: Id  
            - Car label uses property: LicensePlate
            
            Extract the entity type and value from the user question and replace YourLabel and yourProperty accordingly.
            """
            
            # Create enhanced query with instructions
            enhanced_query = f"""
            {cypher_instructions}
            
            User question: {question}
            
            Generate the appropriate Cypher query to find the entity mentioned in the question and all its relationships.
            """
            
            # Use text_to_cypher with the enhanced instructions
            entity_results = quick_text_to_cypher_search(enhanced_query)
            print(f"Entity Search Results: {entity_results}")
            
            # Print first and last 100 characters of entity_results
            entity_str = str(entity_results)
            print(f"\nFirst 100 chars: {entity_str[:100]}")
            print(f"Last 100 chars: {entity_str[-100:]}")
            
            summary = f"Entity search completed for: {question}"
                
        except Exception as e:
            print(f"Error in entity summary processing: {e}")
            summary = f"Error processing entity summary: {str(e)}"
    else:
        print(f"Unknown classification: {classification}")
    
    
    # For now, just return empty string as requested
    return summary


def get_unified_summary_tool(data_path: str) -> Tool:
    """Create and return the unified summary tool that uses router agent"""
    tool_name = "unified_summary"
    tool_description = """Use this tool to get a complete summary for any user question.     
    Simply send the user's question and receive the complete summary response."""
    
    def process_with_data_path(question: str) -> str:
        """Wrapper function that passes data_path to the processing function"""
        return process_summary_question(question, data_path)
    
    unified_summary_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=process_with_data_path,
        args_schema=SummaryToolInput,
    )
    return unified_summary_tool
