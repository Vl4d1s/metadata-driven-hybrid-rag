"""
Timeline Tools using Map-Reduce and Refine Chains
"""
from langchain_core.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from llms.base_llm import get_llm
from .prompts import create_map_prompt, create_reduce_prompt
import os

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        PyPDF2 = pypdf
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False


def get_file_content(file_path: str) -> str:
    """Read file content, handling both text and PDF files"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return ""
    
    # Check if it's a PDF file
    if file_path.lower().endswith('.pdf'):
        if not PDF_AVAILABLE:
            print("PDF reading library not available. Install PyPDF2 or pypdf.")
            return ""
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""
    
    # Try to read as text file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # If UTF-8 fails, try other encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read().strip()
            except:
                continue
        print(f"Could not decode file: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def create_mapreduce_chain(data_path, user_examples=None, user_rules=None) -> str:
    """Create regular summary using map-reduce pattern from provided data path."""
    print(f"Creating map-reduce chain for data path: {data_path}")
    
    # Use the new file content reader that handles PDF and text files
    text = get_file_content(data_path)
    if not text:
        print(f"Could not read content from: {data_path}")
        return ""
    llm = get_llm()
    output_parser = StrOutputParser()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    map_chain = create_map_prompt(user_examples, user_rules) | llm | output_parser
    reduce_chain = create_reduce_prompt(user_examples, user_rules) | llm | output_parser
    map_results = []
    for doc in docs:
        result = map_chain.invoke({"text": doc.page_content})
        map_results.append(result)
    combined_text = "\n".join(map_results)
    final_result = reduce_chain.invoke({"text": combined_text})
    return final_result.strip()


def get_map_reduce_summary_tool(data_path = str, examples = None, rules = None) -> Tool:
    tool_name = "mapreduce_summary"
    tool_description = "Use this tool to generate a regular, coherent summary using a map-reduce approach."
    map_reduce_summary_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=lambda _: create_mapreduce_chain(data_path, examples, rules),
        args_schema=None,
    )
    return map_reduce_summary_tool 