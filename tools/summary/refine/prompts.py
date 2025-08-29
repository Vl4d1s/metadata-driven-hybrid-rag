from langchain_core.prompts import ChatPromptTemplate


DEFAULT_EXAMPLES = """
Example 1:
Input: "Project kickoff meeting at 9:00 AM on Jan 1, 2023. First milestone demo at 2:30 PM March 5, 2023. Project completed late evening March 20, 2023."
Output:
2023-01-01 09:00:00 - PROJECT_START - Project kickoff meeting
2023-03-05 14:30:00 - MILESTONE_REACHED - First milestone demo
2023-03-20 22:00:00 - PROJECT_COMPLETED - Project finalized

Example 2:
Input: "Conference call scheduled for 10:15 AM on 2023-01-15. Presentation delivered around noon in Feb 2023. Follow-up meeting in the morning of March 15, 2023."
Output:
2023-01-15 10:15:00 - MEETING_SCHEDULED - Conference call arranged
2023-02-01 12:00:00 - PRESENTATION_DELIVERED - Presentation completed
2023-03-15 09:00:00 - FOLLOWUP_MEETING - Follow-up discussion held

Example 3:
Input: "System went down at 11:47 PM yesterday. Emergency patch deployed at 3:22 AM today. Full service restored this afternoon at 4:15 PM."
Output:
2023-12-14 23:47:00 - SYSTEM_DOWN - System outage occurred
2023-12-15 03:22:00 - PATCH_DEPLOYED - Emergency fix implemented
2023-12-15 16:15:00 - SERVICE_RESTORED - Full functionality recovered
"""

DEFAULT_RULES = """
- Format: YYYY-MM-DD HH:MM:SS - EVENT_TYPE - Description
- Always include specific times when available; estimate reasonable times if not provided
- Use 24-hour format (HH:MM:SS) for all timestamps
- Use descriptive event types that match the content domain
- Estimate dates and times if not exact (use first day of month/year and 00:00:00 when approximate)
- For vague times like "morning" use 09:00:00, "noon" use 12:00:00, "afternoon" use 15:00:00, "evening" use 18:00:00, "night" use 21:00:00
- Integrate new events chronologically by both date and time
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


