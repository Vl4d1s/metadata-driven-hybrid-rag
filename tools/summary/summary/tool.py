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
from tools.summary.refine.tool import create_refine_chain
from tools._functions.get_document_sections import get_document_sections
from tools.summary.refine.tool import create_refine_chain_from_strings

class SummaryToolInput(BaseModel):
    """Input schema for the summary tool"""
    question: str = Field(description="The question from the user")

# Concise Section-Based Summarization

SECTION_SUMMARY_EXAMPLES = """
Example 1:
Input sections:

Page: 1-3, Subject: Coverage, Content: Vehicle damage, theft, liability coverage. Collision $50k, theft/vandalism $30k, liability $1M.
Page: 4-5, Subject: Premiums, Content: Based on age, vehicle, history, location. Base $800 annually.
Page: 6, Subject: Claims, Content: Report within 24hrs. Need police report, photos, witnesses.

Output:
Page: 1-3, Coverage
Comprehensive vehicle protection including collision damage up to $50,000, theft/vandalism coverage up to $30,000, and third-party liability coverage up to $1,000,000.
Page: 4-5, Premiums
Annual premium calculation starting at $800 base rate, adjusted based on driver age, vehicle type, driving history, and geographic location.
Page: 6, Claims
Claims reporting process requiring notification within 24 hours of incident with mandatory documentation including police report, photos, and witness statements.
Example 2:
Input sections:

Page: 1, Subject: Incident, Content: March 15 collision at Main/Oak, 3:30 PM. Honda Civic vs Ford F-150.
Page: 2-3, Subject: Drivers, Content: Smith (35, clean record) and Johnson (28, one ticket).
Page: 4, Subject: Damages, Content: Civic $8,500 front damage, F-150 $2,300 rear damage.

Output:
Page: 1, Incident
March 15, 3:30 PM collision at Main/Oak intersection between Honda Civic and Ford F-150.
Page: 2-3, Drivers
John Smith (35, clean record) and Sarah Johnson (28, one 2022 speeding violation).
Page: 4, Damages
Total damages $10,800: Honda Civic front-end $8,500, Ford F-150 rear bumper $2,300.
"""

SECTION_SUMMARY_RULES = """
CONCISE SECTION RULES:

1. FORMAT: "Page: X-Y, Section Name \n section summary description"
2. PAGE REFS: Use "Page: X-Y" for ranges, "Page: X" for single pages
3. SECTION NAMES: Clear, descriptive titles without ALL CAPS
4. SUMMARIES: One detailed sentence per section with key facts and figures
5. OUTPUT: List each section separately with page reference and summary only
"""


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
        print("Document Summary Request - Processing policy documents")
        sections = get_document_sections(data_path)
        if not sections or len(sections) == 0 :
            summary = create_mapreduce_chain(data_path)
        else: 
            sections_summary = []
            for section in sections:
                print(f"Section: {section['sectionName']}")
                # print(f"Section Content: {section['sectionContent'][:100]}")
                print(f"Section Pages: {section['sectionPages']}")
                section_summary = create_mapreduce_chain(section['sectionContent'])
                # print(f"Section Summary: {section_summary[:100]}")
                print("-" * 100)
                sections_summary.append(f"Page: {section['sectionPages']}Subject: {section['sectionName']} Content: {section_summary}")
            summary = create_refine_chain_from_strings(sections_summary,SECTION_SUMMARY_EXAMPLES,SECTION_SUMMARY_RULES)
             
        # print(f"Document Summary: {summary}")
    elif classification == "entity_summary":
        print("Entity Summary Request - Processing entity summaries")
        
        # Use text_to_cypher to find the entity and its relationships
        try:
            # Define the custom Cypher query template and instructions
            cypher_instructions = """
            Use this Cypher query pattern to find the node and its relationships without embeddings:
            
            MATCH (n:YourLabel {yourProperty: 'someValue'})
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN 
              apoc.map.removeKey(properties(n), 'embedding') AS filteredNode,
              r,
              apoc.map.removeKey(properties(m), 'embedding') AS filteredConnectedNode
            
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
            # print(f"Entity Search Results: {entity_results}")
            
            # Print first and last 100 characters of entity_results
            entity_str = str(entity_results)
            print(f"\nFirst 100 chars: {entity_str[:100]}")
            print(f"Last 100 chars: {entity_str[-100:]}")
            
            # Extract content from entity_results and send to refine chain
            if entity_results and hasattr(entity_results, 'items') and entity_results.items:
                # Combine all content from the results
                combined_content = ""
                for item in entity_results.items:
                    if hasattr(item, 'content'):
                        combined_content += f"{item.content}\n\n"
                
                print(f"\nCombined content length: {len(combined_content)}")
                print(f"Combined content preview: {combined_content[:200]}...")
                
                # Send to refine chain for summarization
                if combined_content.strip():
                    summary = create_refine_chain(exist_data=combined_content)
                    print(f"\nRefine chain summary: {summary}")
                else:
                    summary = f"No content found for entity query: {question}"
            else:
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
