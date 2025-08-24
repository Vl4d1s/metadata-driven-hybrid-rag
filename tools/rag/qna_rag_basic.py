from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from tools._functions.answer_with_rag_basic import answer_with_rag_basic

class QnAInput(BaseModel):
    """Input schema for QnA tool"""
    question: str = Field(description="The question to answer")

def qna_rag_basic_tool():
    """Get QnA tool that accepts a question and returns an answer using RAG"""
    qna_tool = Tool(
        name="qna_rag_basic",
        description="Answer questions using RAG pipeline. Input should be a question string.",
        func=answer_with_rag_basic,
        args_schema= QnAInput,
    )
    return qna_rag_basic_tool