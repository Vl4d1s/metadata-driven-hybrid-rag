from langchain_core.prompts import PromptTemplate


def router_agent_prompt(options: list[str], examples: str) -> PromptTemplate:
    """Create the classifier prompt template with few-shot examples and role-playing"""
    
    # Format options as a numbered list
    formatted_options = ", ".join(options)
    
    # Create base examples
    base_examples = """
Question: "What happened during the American Civil War?"
Classification: qna
Reasoning: Seeks specific events or facts, not an explicit summary or overview

Question: "When did the American Civil War start?"
Classification: qna
Reasoning: Seeks a specific date/fact

Question: "Can you give me a timeline of the company's growth?"
Classification: summary
Reasoning: Requests chronological overview

Question: "Who is the CEO of the company?"
Classification: qna
Reasoning: Asks for specific person/information

Question: "Tell me about the history of artificial intelligence"
Classification: summary
Reasoning: Asks for broad overview/narrative"""
    
    all_examples = f"{base_examples}\n\n{examples}"
    
    template = """You are an expert question classifier working for a knowledge management system. 
Your role is to accurately categorize incoming questions to route them to the appropriate response handlers.
You have years of experience in natural language processing and question analysis.

Available Options:
{formatted_options}

TOOLS YOU CAN USE (may be empty):
{{tools}}

TOOL NAMES:
{{tool_names}}

EXAMPLES:

{all_examples}

NOW CLASSIFY THIS QUESTION:
Question: {{question}}

Reasoning and previous actions:
{{agent_scratchpad}}

Use the following format:
    Thought: think step-by-step about which option fits best
    Action: the action to take, should be one of {{tool_names}} if tools are available, otherwise skip to Final Answer
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat as needed)
    Final Answer: [Return ONLY the classification option, nothing else. Must be one of: {formatted_options}]

CRITICAL: Your Final Answer must contain ONLY the option name, no additional text, explanation, or formatting."""

    return PromptTemplate.from_template(template.format(
        formatted_options=formatted_options,
        all_examples=all_examples
    ))

