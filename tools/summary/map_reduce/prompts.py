from langchain.prompts import ChatPromptTemplate

DEFAULT_EXAMPLES = """
Example 1:
Input: "The quarterly report shows revenue increased by 15%. Customer satisfaction improved to 92%. New product launch was successful with 10,000 units sold."
Output:
The quarterly performance demonstrates strong growth with revenue up 15% and high customer satisfaction at 92%. The new product launch exceeded expectations with 10,000 units sold, indicating positive market reception.

Example 2:
Input: "Team completed three major milestones this month. Budget utilization at 85%. Risk assessment identified two potential issues requiring attention."
Output:
Monthly progress includes three completed milestones with efficient budget utilization at 85%. Risk management has identified two areas needing attention for proactive resolution.
"""

DEFAULT_RULES = """
- Create concise, coherent summaries that capture key points
- Focus on the most important information and insights
- Use professional, clear language
- Maintain factual accuracy from the source text
- Output ONLY the summary, no explanations or meta-commentary
"""

DEFAULT_FINAL_RULES = """
- Integrate all key points from the fragments
- Create a cohesive, well-structured summary
- Avoid repetition and redundancy
- Maintain professional tone and clarity
- Output ONLY the final summary, no explanations
"""


def create_map_prompt(user_example = None,user_rules = None) -> ChatPromptTemplate:
    """Create the map prompt template for extracting key information"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert analyst. Extract and summarize the key information from this text.

Examples:
{user_example if user_example else DEFAULT_EXAMPLES}

Rules:
{user_rules if user_rules else DEFAULT_RULES}

Text: {{text}}

Summary:"""
    )

def create_reduce_prompt(user_example = None,user_rules = None) -> ChatPromptTemplate:
    """Create the reduce prompt template for combining summaries"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert analyst. Combine these summary fragments into a comprehensive final summary.

Examples:
{user_example if user_example else DEFAULT_EXAMPLES}

Rules:
{user_rules if user_rules else DEFAULT_FINAL_RULES}

Summary fragments:
{{text}}

Final comprehensive summary:"""
    )