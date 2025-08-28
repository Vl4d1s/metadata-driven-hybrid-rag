from llama_cloud_services import LlamaExtract
from pydantic import BaseModel, Field
# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()


# # Initialize client
# extractor = LlamaExtract()


# # Define schema using Pydantic
# class Resume(BaseModel):
#     name: str = Field(description="Full name of candidate")
#     email: str = Field(description="Email address")
#     skills: list[str] = Field(description="Technical skills and technologies")


# # Create extraction agent
# agent = extractor.create_agent(name="resume-parser", data_schema=Resume)

# # Extract data from document
# result = agent.extract("resume.pdf")
# print(result.data)


def extract_file(file_path: str = "C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag\\flows\\insurance\\data\\reports\\001.pdf" , schema: BaseModel = BaseModel , extract_name: str="file-parser"):
    """Extract resume data from a PDF file and return as a Resume object."""
    print(f"Extracting data from {file_path} using schema {schema.__name__} with agent name '{extract_name}'")
    try:
        extractor = LlamaExtract()
        agent = extractor.get_agent(name=extract_name)
        if not agent:
            agent = extractor.create_agent(name="file-parser", data_schema=schema)
        result = agent.extract(file_path)
        print(f"Extracted data from {file_path}: {result.data}")
        return result.data
    except Exception as e:
        print(f"Error extracting data from {file_path}: {e}")