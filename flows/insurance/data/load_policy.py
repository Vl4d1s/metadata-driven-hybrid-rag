import PyPDF2
import json
import re
import os
from typing import List, Dict, Any, Optional 
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# You'll need to install these packages:
# pip install PyPDF2 langchain-openai

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)


# Check if OpenAI API key is set
def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" Error: OPENAI_API_KEY environment variable is not set!")
        print("Please set your OpenAI API key using one of these methods:")
        print("1. Set environment variable: export OPENAI_API_KEY='your-api-key'")
        print("2. In PowerShell: $env:OPENAI_API_KEY='your-api-key'")
        print("3. Create a .env file with: OPENAI_API_KEY=your-api-key")
        print("\nYou can get your API key from: https://platform.openai.com/api-keys")
        return False
    return True

# Initialize the LangChain OpenAI client
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


def process_pdf_with_toc(file_path: str) -> None:
    """
    Process a PDF file to extract Table of Contents and print content summaries.
    
    Args:
        file_path (str): Path to the PDF file
    """
    
    # Step 1: Get the file path and validate
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return
    
    
    try:
        # Step 2 & 3: Pull PDF file and extract first 3 pages
        first_three_pages_text = extract_first_pages(file_path, num_pages=3)
        
        if not first_three_pages_text:
            print("Error: Could not extract text.")
            return
        
        print("Extracted first 3 pages successfully.")
        
        # Step 4: Send to LLM and ask for Table of Contents
        toc_json = get_toc_from_llm(first_three_pages_text)
        print("toc_json:", toc_json)
        if not toc_json:
            print("No Table of Contents found or could not parse LLM response.")
            return
        
        print("Table of Contents extracted successfully.")
        print(f"Found {len(toc_json)} sections.")
        
        process_sections_by_toc_with_chunks(file_path, toc_json)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")


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
        print(f"LLM Response: {response_text}")
        return None
    except Exception as e:
        print(f"Error communicating with LLM: {str(e)}")
        return None


def process_sections_by_toc_with_chunks(
    file_path: str, 
    toc_json: List[Dict[str, Any]], 
    chunk_size: int = 1500, 
    chunk_overlap: int = 300
) -> List[Dict[str, Any]]:
    """
    Process PDF sections based on Table of Contents and split into chunks.
    
    Args:
        file_path: Path to the PDF file
        toc_json: Table of contents with section information
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of dictionaries with chunk information in the format:
        {
            "DocumentId": str,
            "PageNumber": int,
            "SectionName": str,
            "Chunk_Index": int,
            "Chunk_Content": str,
            "TotalChunksInSection": int
        }
    """
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            print("\n" + "="*50)
            print("PROCESSING SECTIONS BY TABLE OF CONTENTS")
            print("="*50)
            
            for section in toc_json:
                subject = section.get("subject", "Unknown")
                start_page = section.get("start_page", "1")
                end_page = section.get("end_page", "-1")
                
                print(f"\n📖 SECTION: {subject}")
                print("-" * 40)
                
                try:
                    start_page_num = int(start_page) - 1
                    if end_page == "-1":
                        end_page_num = total_pages - 1
                    else:
                        end_page_num = int(end_page) - 1
                    
                    start_page_num = max(0, start_page_num)
                    end_page_num = min(total_pages - 1, end_page_num)
                    
                except ValueError:
                    print(f"❌ Invalid page numbers for section '{subject}'. Skipping.")
                    continue
                
                # Extract all text from section pages
                section_text = ""
                section_pages = []
                
                for page_num in range(start_page_num, end_page_num + 1):
                    if page_num < total_pages:
                        actual_page_num = page_num + 1  # Convert to 1-based
                        section_pages.append(actual_page_num)
                        
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        # Add page marker to help track pages in chunks
                        section_text += f"\n[PAGE_{actual_page_num}]\n{page_text}\n"
                
                print(f"📄 Pages: {start_page} - {end_page if end_page != '-1' else total_pages}")
                print(f"🔍 Processing {len(section_pages)} page(s)")
                
                # Create a document for the entire section
                section_doc = Document(
                    page_content=section_text.strip(),
                    metadata={
                        "documentId": file_path,
                        "sectionName": subject,
                        "pages": section_pages,
                        "section_start_page": section_pages[0] if section_pages else 1,
                        "section_end_page": section_pages[-1] if section_pages else 1
                    }
                )
                
                # Split section into chunks
                section_chunks = text_splitter.split_documents([section_doc])
                embedder = OpenAIEmbeddings(model="text-embedding-ada-002")  # Move outside loop

                with driver.session() as session:
                    for i, chunk in enumerate(section_chunks):
                        chunk_page_num = extract_page_from_chunk(chunk.page_content, section_pages)
                        chunk_content = clean_chunk_content(chunk.page_content)
                        chunk_embedding = embedder.embed_query(chunk_content)  # Fixed method
        
                         # Use parameters for all dynamic content
                        chunk_query = '''
                        MERGE (d:Document {Id: $document_id})
                        SET d.Title = "Policy"
                        MERGE (s:Section {name: $section_name})
                        SET s.start_page = $start_page
                        SET s.end_page = $end_page
                        MERGE (d)-[:HAS_SECTION]->(s)
                        MERGE (s)-[:SOURCE]->(d)
                        MERGE (c:Chunk {DocumentId: $document_id, SectionName: $section_name, Chunk_Index: $chunk_index})
                        SET c.DocumentId = $document_id
                        SET c.PageNumber = $page_number
                        SET c.SectionName = $section_name
                        SET c.Chunk_Index = $chunk_index
                        SET c.Chunk_Content = $chunk_content
                        SET c.TotalChunksInSection = $total_chunks
                        SET c.embedding = $chunk_embedding
                        MERGE (s)-[:HAS_CHUNK]->(c)
                        MERGE (c)-[:SOURCE]->(s)
                        '''
        
                        session.run(chunk_query, parameters={
                            "document_id": file_path,
                            "section_name": subject,
                            "start_page": start_page_num,
                            "end_page": end_page_num,
                            "page_number": chunk_page_num,
                            "chunk_index": i,
                            "chunk_content": chunk_content,
                            "total_chunks": len(section_chunks),
                            "chunk_embedding": chunk_embedding
                        })
        
                        chunk_dict = {
                            "DocumentId": file_path,
                            "PageNumber": chunk_page_num,
                            "SectionName": subject,
                            "Chunk_Index": i,
                            "Chunk_Content": chunk_content,
                            "TotalChunksInSection": len(section_chunks)
                            }
                        all_chunks.append(chunk_dict)
                
                    print(f"✅ Created {len(section_chunks)} chunk(s) for section '{subject}'")
                
                    
    except Exception as e:
        print(f"❌ Error processing sections: {str(e)}")
        return []
    
    print(f"\n🎉 Total chunks created: {len(all_chunks)}")
    # print("="*50 + "\n")
    # print(all_chunks) 
        
    return all_chunks


def extract_page_from_chunk(chunk_content: str, section_pages: List[int]) -> int:
    """
    Extract the most likely page number for a chunk based on page markers.
    
    Args:
        chunk_content: The content of the chunk
        section_pages: List of page numbers in the section
    
    Returns:
        The most likely page number for this chunk
    """
    import re
    
    # Find all page markers in the chunk
    page_markers = re.findall(r'\[PAGE_(\d+)\]', chunk_content)
    
    if page_markers:
        # Convert to integers and filter to only pages in this section
        found_pages = [int(p) for p in page_markers if int(p) in section_pages]
        
        if found_pages:
            # Return the first page found (chunks usually start at a page boundary)
            return found_pages[0]
    
    # Fallback: return the first page of the section
    return section_pages[0] if section_pages else 1


def clean_chunk_content(content: str) -> str:
    """
    Remove page markers from chunk content.
    
    Args:
        content: Raw chunk content with page markers
    
    Returns:
        Cleaned content without page markers
    """
    import re
    
    # Remove page markers
    cleaned = re.sub(r'\n?\[PAGE_\d+\]\n?', '\n', content)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def get_first_n_words(text: str, n: int) -> str:
    """Helper function to get first n words from text."""
    words = text.split()
    return ' '.join(words[:n]) if len(words) >= n else ' '.join(words)


# Example usage function
def demonstrate_chunking(file_path: str, toc_json: List[Dict[str, Any]]):
    """Demonstrate the chunking functionality."""
    
    # Process sections and get chunks
    chunks = process_sections_by_toc_with_chunks(
        file_path=file_path,
        toc_json=toc_json,
        chunk_size=500,  # Smaller chunks for demo
        chunk_overlap=100
    )
    
    # Display results
    print("\n" + "="*60)
    print("CHUNK ANALYSIS")
    print("="*60)
    
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\n🔸 Chunk {i+1}:")
        print(f"   📄 Page: {chunk['PageNumber']}")
        print(f"   📖 Section: {chunk['SectionName']}")
        print(f"   📁 Document: {chunk['DocumentId']}")
        print(f"   🔢 Chunk Index: {chunk['Chunk_Index']}/{chunk['TotalChunksInSection']-1}")
        print(f"   📝 Content preview: {chunk['Chunk_Content'][:150]}...")
        print("-" * 40)
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    
    # Group chunks by section
    section_stats = {}
    for chunk in chunks:
        section = chunk['SectionName']
        if section not in section_stats:
            section_stats[section] = {'count': 0, 'pages': set()}
        section_stats[section]['count'] += 1
        section_stats[section]['pages'].add(chunk['PageNumber'])
    
    print("\n📊 SECTION STATISTICS:")
    for section, stats in section_stats.items():
        pages_list = sorted(list(stats['pages']))
        print(f"   📖 {section}: {stats['count']} chunks across pages {pages_list}")
    
    return chunks





def get_first_n_words(text: str, n: int = 10) -> str:
    """Extract the first N words from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    first_words = words[:n]
    
    if len(first_words) < n:
        return " ".join(first_words) + " [End of page content]"
    else:
        return " ".join(first_words) + "..."


# Example usage
if __name__ == "__main__":
    print(" PDF Table of Contents Processor")
    print("=" * 40)
    
    # Check API key before proceeding
    if not check_api_key():
        exit(1)
    
    print(" OpenAI API key found!")
    print(" Starting PDF processing...\n")
    
    # process_pdf_with_toc('C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag\\flows\\insurance\\data\\policy\\policy.pdf')
    
    print("\n✨ PDF processing completed!")
