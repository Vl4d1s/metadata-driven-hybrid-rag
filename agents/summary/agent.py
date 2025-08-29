from langchain.agents import create_react_agent, AgentExecutor
from llms.base_llm import get_llm
from tools.summary.summary.tool import get_unified_summary_tool
from .prompts import summary_agent_prompt

def get_summary_agent(
        llm=None, 
        data_path: str = None,
        timeline_examples = None,
        timeline_rules = None,
        summary_examples = None,
        summary_rules = None) -> AgentExecutor:
    """Create and return the summary AgentExecutor. Optionally accept an LLM instance."""
    if not data_path:
        raise ValueError("data_path is required for summary agent")
    llm = llm if llm else get_llm()
    unified_tool = get_unified_summary_tool(data_path)
    tools = [unified_tool]
    agent = create_react_agent(llm=llm, tools=tools, prompt=summary_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    return agent_executor
