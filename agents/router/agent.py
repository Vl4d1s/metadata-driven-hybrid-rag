from llms.base_llm import get_llm
from langchain.agents import create_react_agent, AgentExecutor
from .prompts import router_agent_prompt

def get_router_agent(options: list[str], examples: str = "",rules: str = ""):
    """Create and return the Router AgentExecutor"""
    llm = get_llm()
    tools = []
    prompt = router_agent_prompt(options, examples,rules)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor