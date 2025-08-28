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


def extract_file(file_path: str , schema: BaseModel):
    """Extract resume data from a PDF file and return as a Resume object."""
    try:
        extractor = LlamaExtract()
        agent = extractor.create_agent(name="file-parser", data_schema=schema)
        result = agent.extract(file_path)
        return result.data
    except Exception as e:
        print(f"Error extracting data from {file_path}: {e}")