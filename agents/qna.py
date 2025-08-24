from langchain.agents import create_react_agent, AgentExecutor
from tools.rag.qna_rag_basic import qna_rag_basic_tool
from langchain_core.prompts import PromptTemplate
from llms.base_llm import get_llm

def qna_agent_prompt() -> PromptTemplate:
    """Create the agent prompt template for QnA agent"""
    return PromptTemplate.from_template("""
You are an insurance QnA agent. 
Your job is simple: always use the RAG tool to answer any question and return the tool's result directly to the user.

You have access to the following tool:
{tools}

IMPORTANT: You must ALWAYS use the tool for every question. Do not try to answer questions yourself. 
Simply:
    1. Call the tool with the user's question
    2. Return the tool's result as your final answer

Use the following format:
    Question: the input question you must answer
    Thought: I need to use the RAG tool to answer this question
    Action: the action to take, should be one of {tool_names}
    Action Input: the input to the action
    Observation: the result of the action
    Thought: I now have the result from the tool
    Final Answer: the result from the tool
Question: {input}
{agent_scratchpad}
""")

def get_qna_agent(LLM=None):
    """Create and return the QnA AgentExecutor. Optionally accept an LLM instance."""
    llm = LLM if LLM else get_llm()
    tools = [qna_rag_basic_tool()]
    prompt= qna_agent_prompt()
    agent = create_react_agent(llm , tools, prompt )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor