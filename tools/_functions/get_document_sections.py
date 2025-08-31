import PyPDF2
import json
import re
import os
from typing import List, Dict, Any, Optional 
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI


def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" Error: OPENAI_API_KEY environment variable is not set!")
        return False
    return True


def initialize_llm():
    """Initialize the LangChain OpenAI client."""
    if not check_api_key():
        return None
    
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        return llm
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {str(e)}")
        return None


def extract_first_pages(file_path: str, num_pages: int = 3) -> str:
    """Extract text from the first N pages of a PDF."""
    text = ""
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            pages_to_read = min(num_pages, len(pdf_reader.pages))
            
            for page_num in range(pages_to_read):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
                
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return ""
    
    return text


def get_toc_from_llm(text: str) -> Optional[List[Dict[str, Any]]]:
    """Send text to OpenAI and request Table of Contents in JSON format."""
    
    # Get the LLM instance
    llm = initialize_llm()
    if not llm:
        return None
    
    prompt = f"""
You are an expert document analyst with years of experience in reading and interpreting PDF structures. 
Your role is to carefully analyze the following text (from the first 3 pages of a PDF document) and determine 
whether it contains a Table of Contents.

Return the Table of Contents in this exact JSON format:
[
    {{"subject": "Chapter/Section Title", "start_page": "page_number", "end_page": "page_number"}},
    {{"subject": "Last Chapter/Section", "start_page": "page_number", "end_page": "-1"}}
]

Important notes:
- The last entry should have "end_page": "-1"
- Use actual page numbers from the document
- If no Table of Contents is found, return an empty array: []
- Return ONLY the JSON array, no additional text

Text to analyze:
{text}
"""
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        toc_json = json.loads(response_text)
        
        if isinstance(toc_json, list) and all(
            isinstance(item, dict) and 
            "subject" in item and 
            "start_page" in item and 
            "end_page" in item 
            for item in toc_json
        ):
            return toc_json
        else:
            print("Invalid JSON structure received from LLM.")
            return None
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM response: {str(e)}")
        return None
    except Exception as e:
        print(f"Error communicating with LLM: {str(e)}")
        return None


def extract_section_content(file_path: str, start_page: int, end_page: int) -> str:
    """Extract content from a specific page range in a PDF."""
    content = ""
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Convert to 0-based indexing
            start_idx = max(0, start_page - 1)
            if end_page == -1:
                end_idx = total_pages - 1
            else:
                end_idx = min(total_pages - 1, end_page - 1)
            
            for page_num in range(start_idx, end_idx + 1):
                if page_num < total_pages:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    content += page_text + "\n"
                    
    except Exception as e:
        print(f"Error extracting section content: {str(e)}")
        return ""
    
    return content.strip()


def get_document_sections(file_path: str) -> List[Dict[str, str]]:
    """
    Extract document sections and return them as an array of dictionaries.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        List[Dict[str, str]]: Array of sections in format [{"sectionName": "", "sectionContent": ""}]
                             Returns empty array if no sections found or on error
    """
    
    # Validate file path
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return []
    
    try:
        # Step 1: Extract first 3 pages to find Table of Contents
        first_three_pages_text = extract_first_pages(file_path, num_pages=3)
        
        if not first_three_pages_text:
            print("Error: Could not extract text from PDF.")
            return []
        
        # Step 2: Get Table of Contents from LLM
        toc_json = get_toc_from_llm(first_three_pages_text)
        
        if not toc_json or len(toc_json) == 0:
            print("No Table of Contents found in document.")
            return []
        
        # Step 3: Extract content for each section
        sections = []
        
        for section in toc_json:
            section_name = section.get("subject", "Unknown")
            start_page = section.get("start_page", "1")
            end_page = section.get("end_page", "-1")
            
            try:
                start_page_num = int(start_page)
                end_page_num = int(end_page) if end_page != "-1" else -1
                
                # Extract section content
                section_content = extract_section_content(file_path, start_page_num, end_page_num)
                
                if section_content:
                    sections.append({
                        "sectionName": section_name,
                        "sectionContent": section_content,
                        "sectionPages": f"{start_page_num}-{end_page_num if end_page_num != -1 else 'end'}"
                    })
                    
            except ValueError:
                print(f"Invalid page numbers for section '{section_name}'. Skipping.")
                continue
        
        print(f"Successfully extracted {len(sections)} sections from document.")
        return sections
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return []


# Example usage
if __name__ == "__main__":
    print("Document Sections Extractor")
    print("=" * 40)
    
    # Check API key before proceeding
    if not check_api_key():
        exit(1)
    
    print("OpenAI API key found!")
    
    # Example usage with the policy PDF
    file_path = "flows/insurance/data/policy/policy.pdf"
    sections = get_document_sections(file_path)
    
    if sections:
        print(f"\nFound {len(sections)} sections:")
        for i, section in enumerate(sections, 1):
            print(f"\n{i}. Section: {section['sectionName']}")
            print(f"   Content preview: {section['sectionContent'][:200]}...")
    else:
        print("\nNo sections found or error occurred.")
