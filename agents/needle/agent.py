from langchain.agents import create_react_agent, AgentExecutor
from tools.needle.tool import get_needle_tool, needle_tool
from llms.base_llm import get_llm
from agents.needle.prompts import needle_agent_prompt

class DirectNeedleAgent:
    """
    Direct needle agent that bypasses LLM interpretation and returns raw needle tool results.
    """
    def __init__(self):
        self.needle_tool = get_needle_tool()
    
    def invoke(self, inputs):
        """
        Directly invoke the needle tool and return raw results without LLM interpretation.
        """
        question = inputs.get("input", "")
        print(f"🔍 Direct Needle Search for: {question}")
        
        # Call needle tool directly
        raw_result = needle_tool(question)
        
        # Return in the expected format
        return {
            "output": raw_result,
            "intermediate_steps": []
        }

def get_needle_agent(LLM=None, data_path=str, use_direct=True):
    """
    Create and return the Needle Agent. 
    
    Args:
        LLM: Optional LLM instance
        data_path: Data path (unused for needle)
        use_direct: If True, use direct needle agent that bypasses LLM interpretation
    """
    if use_direct:
        return DirectNeedleAgent()
    else:
        # Original LangChain agent (may interpret results)
        llm = LLM if LLM else get_llm()
        tools = [get_needle_tool()]
        prompt = needle_agent_prompt()
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        return agent_executor
