"""
Prompts for the unified summary tool
"""
from langchain_core.prompts import PromptTemplate


def create_router_examples() -> str:
    """Create examples for the router agent to understand summary vs entity classification"""
    return """
Question: "Summarize the insurance policy document"
Classification: document_summary
Reasoning: Requests summary of policy document

Question: "Give me an overview of the accident reports"
Classification: document_summary  
Reasoning: Requests summary of report documents

Question: "Summarize the accident 12345"
Classification: entity_summary
Reasoning: Requests summary of specific accident with ID

Question: "Tell me about driver D001"
Classification: entity_summary
Reasoning: Requests information about specific driver with ID

Question: "Summarize the info of car ABC123"
Classification: entity_summary
Reasoning: Requests summary of specific car with license plate

Question: "What happened in the accident with ID ACC-2023-001?"
Classification: entity_summary
Reasoning: Requests information about specific accident with ID

Question: "Give me a summary of all the policy terms"
Classification: document_summary
Reasoning: Requests summary of policy document content
"""



def create_entity_extraction_prompt() -> PromptTemplate:
    """Create prompt for extracting entity IDs from questions"""
    return PromptTemplate.from_template("""
Extract the entity ID or identifier from the following question:

Question: {question}

Look for:
- Driver IDs (driver_id, driver ID, driver: XXX)
- Accident IDs (accident_id, accident ID, accident: XXX)  
- Car license plates or IDs (car: XXX, license: XXX, plate: XXX)

Return the extracted ID and entity type, or "NONE" if no specific ID is found.

Format: ENTITY_TYPE:ID or NONE

Entity ID:
""")
