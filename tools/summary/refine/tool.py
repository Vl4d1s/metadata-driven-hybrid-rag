from llms.base_llm import get_llm
from langchain_core.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from tools.summary.refine.prompts import (
    create_initial_refine_prompt,
    create_refine_prompt,
)

def create_refine_chain(data_path = None, user_examples=None, user_rules=None,exist_data=None) -> str:
    """Create timeline using refine pattern, reading events.txt directly."""
    if exist_data:
        text = exist_data
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    llm = get_llm()
    output_parser = StrOutputParser()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    initial_chain = create_initial_refine_prompt(user_examples, user_rules) | llm | output_parser
    refine_chain = create_refine_prompt(user_examples, user_rules) | llm | output_parser
    current_result = initial_chain.invoke({"text": docs[0].page_content})
    for doc in docs[1:]:
        current_result = refine_chain.invoke({
            "existing_timeline": current_result,
            "new_text": doc.page_content
        })
    return current_result.strip()

def create_refine_chain_from_strings(string_list, user_examples=None, user_rules=None) -> str:
    """Create summary using refine pattern from a list of strings."""
    if not string_list:
        return ""
    
    llm = get_llm()
    output_parser = StrOutputParser()
    
    # Create chains
    initial_chain = create_initial_refine_prompt(user_examples, user_rules) | llm | output_parser
    refine_chain = create_refine_prompt(user_examples, user_rules) | llm | output_parser
    
    # Start with the first string
    current_result = initial_chain.invoke({"text": string_list[0]})
    
    # Refine with each subsequent string
    for text in string_list[1:]:
        if text.strip():  # Only process non-empty strings
            current_result = refine_chain.invoke({
                "existing_timeline": current_result,
                "new_text": text
            })
    
    return current_result.strip()

def get_refine_summary_tool(data_path = str, examples = None, rules = None) -> Tool:
    tool_name = "refine_timeline"
    tool_description = "Use this tool to generate a detailed timeline of insurance events from the provided text data."
    refine_summary_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda _: create_refine_chain(data_path, examples, rules),
        args_schema=None,
    )
    return refine_summary_tool

def get_refine_strings_tool(string_list, examples = None, rules = None) -> Tool:
    """Create a tool that summarizes a list of strings using refine pattern."""
    tool_name = "refine_strings_summary"
    tool_description = "Use this tool to generate a refined summary from a list of strings."
    refine_strings_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda _: create_refine_chain_from_strings(string_list, examples, rules),
        args_schema=None,
    )
    return refine_strings_tool 