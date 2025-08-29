from langchain_core.prompts import ChatPromptTemplate


DEFAULT_EXAMPLES = """
Example 1:
Input: "Project started Jan 1, 2023. Milestone reached March 5, 2023. Project completed March 20, 2023."
Output:
2023-01-01 00:00:00 - PROJECT_START - Project initiation
2023-03-05 00:00:00 - MILESTONE_REACHED - Key milestone achieved
2023-03-20 00:00:00 - PROJECT_COMPLETED - Project finalized

Example 2:
Input: "Meeting scheduled 2023-01-15. Presentation delivered Feb 2023. Follow-up meeting March 2023."
Output:
2023-01-15 00:00:00 - MEETING_SCHEDULED - Meeting arranged
2023-02-01 00:00:00 - PRESENTATION_DELIVERED - Presentation completed
2023-03-01 00:00:00 - FOLLOWUP_MEETING - Follow-up discussion held
"""

DEFAULT_RULES = """
- Format: YYYY-MM-DD - EVENT_TYPE - Description
- Use descriptive event types that match the content domain
- Estimate dates if not exact (use first day of month/year when approximate)
- Integrate new events chronologically
- Remove duplicates and keep the most detailed version
- Output ONLY the timeline events, no explanations
"""


def create_initial_refine_prompt(user_examples=None, user_rules=None) -> ChatPromptTemplate:
    """Create the initial prompt for refine pattern"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert analyst. Create a timeline from this text.
Examples:
{user_examples if user_examples else DEFAULT_EXAMPLES}

Rules:
{user_rules if user_rules else DEFAULT_RULES}

Text: {{text}}

Timeline events:"""
    )


def create_refine_prompt(user_examples=None, user_rules=None) -> ChatPromptTemplate:
    """Create the refine prompt template for iteratively updating timeline"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert analyst. Refine the existing timeline with new information.
Examples:
{user_examples if user_examples else DEFAULT_EXAMPLES}

Rules:
{user_rules if user_rules else DEFAULT_RULES}


Existing timeline:
{{existing_timeline}}

New text to integrate:
{{new_text}}

Refined timeline:"""
    )


