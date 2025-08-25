from langchain.agents import create_react_agent, AgentExecutor
from llms.base_llm import get_llm
from tools.summary.map_reduce.tool import get_map_reduce_summary_tool
from tools.summary.refine.tool import get_refine_summary_tool
from .prompts import summary_agent_prompt

def get_summary_agent(
        llm=None, 
        data_path=str,
        timeline_examples = None,
        timeline_rules = None,
        summary_examples = None,
        summary_rules = None) -> AgentExecutor:
    """Create and return the summary AgentExecutor. Optionally accept an LLM instance."""
    llm = llm if llm else get_llm()
    timeline_tool = get_refine_summary_tool(data_path,timeline_examples,timeline_rules)
    summary_tool = get_map_reduce_summary_tool(data_path,summary_examples,summary_rules)
    tools = [timeline_tool,summary_tool]
    agent = create_react_agent(llm=llm, tools=tools, prompt=summary_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    return agent_executor
