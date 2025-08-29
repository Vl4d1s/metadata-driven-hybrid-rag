"""
Unified Summary Tool that uses Router Agent for intelligent routing
"""

from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from agents.router.agent import get_router_agent
from .prompts import create_router_examples
import re


class SummaryToolInput(BaseModel):
    """Input schema for the summary tool"""
    question: str = Field(description="The question from the user")


def process_summary_question(question: str) -> str:
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
    classification = router_result.get("output", "").strip().lower()
    print(f"Router Result: {router_result}")
    
    
    
    # For now, just return empty string as requested
    return ""


def get_unified_summary_tool(        llm=None, 
        data_path=str,
        timeline_examples = None,
        timeline_rules = None,
        summary_examples = None,
        summary_rules = None) -> Tool:
    """Create and return the unified summary tool that uses router agent"""
    tool_name = "unified_summary"
    tool_description = """Use this tool to get a complete summary for any user question.     
    Simply send the user's question and receive the complete summary response."""
    
    unified_summary_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=process_summary_question,
        args_schema=SummaryToolInput,
    )
    return unified_summary_tool
