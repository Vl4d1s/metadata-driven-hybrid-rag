from langchain.agents import create_react_agent, AgentExecutor
# from tools.rag.basic.qna_rag_basic import qna_rag_basic_tool
from tools.rag.graph.tool import get_graph_rag_tool
from llms.base_llm import get_llm
from agents.qna.prompts import qna_agent_prompt

def get_qna_agent(LLM=None,data_path=str):
    """Create and return the QnA AgentExecutor. Optionally accept an LLM instance."""
    llm = LLM if LLM else get_llm()
    tools = [get_graph_rag_tool()]
    prompt= qna_agent_prompt()
    agent = create_react_agent(llm , tools, prompt )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor