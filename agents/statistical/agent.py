from langchain.agents import create_react_agent, AgentExecutor
from tools.statistical.tool import get_statistical_tool
from llms.base_llm import get_llm
from agents.statistical.prompts import statistical_agent_prompt

def get_statistical_agent(LLM=None, data_path=str):
    """Create and return the Statistical AgentExecutor. Optionally accept an LLM instance."""
    llm = LLM if LLM else get_llm()
    tools = [get_statistical_tool()]
    prompt = statistical_agent_prompt()
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor
