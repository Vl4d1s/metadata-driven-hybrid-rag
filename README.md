# Metadata-Driven Hybrid RAG

A sophisticated hybrid Retrieval-Augmented Generation (RAG) system that combines multiple AI agents for document analysis, question answering, and statistical insights.

## Features

- **Router Agent**: Intelligently routes queries to the appropriate specialized agent
- **Summary Agent**: Provides comprehensive summaries of documents
- **QnA Agent**: Answers specific questions using retrieved context
- **Needle Agent**: Locates specific information within documents
- **Statistical Agent**: Performs statistical analysis and pattern recognition

## Prerequisites

Before running the application, ensure you have the following:

1. Python 3.8 or higher
2. Required API keys and database credentials (see Environment Variables section)

## Environment Variables

Create a `.env` file in the project root directory with the following variables:

```env
# OpenAI API Key for language model access
OPENAI_API_KEY=your_openai_api_key_here

# LlamaCloud API Key for document processing
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Neo4j Database Configuration
NEO4J_URI=your_neo4j_uri_here
NEO4J_USERNAME=your_neo4j_username_here
NEO4J_PASSWORD=your_neo4j_password_here
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd metadata-driven-hybrid-rag
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables as described above.

## Usage

### Running the Insurance Flow

To start the interactive chat interface for the insurance flow:

```bash
cd flows/insurance
python main.py
```

This will launch an interactive session where you can:

- Ask questions about insurance policies
- Request summaries of documents
- Search for specific information (needle search)
- Perform statistical analysis on accident data

### Example Queries

The system can handle various types of queries:

**Summary Questions:**

- "Summarize the additional benefits offered under comprehensive cover"
- "Summarize the accident of the driver 445-78-9012"

**QnA Questions:**

- "Does this insurance include roadside assistance?"
- "What is the car of the driver 445-78-9012?"

**Needle Search Questions:**

- "Find the maximum amount the insurer will pay for windscreen replacement"
- "Locate information about driver John Smith"

**Statistical Questions:**

- "Analyze the accident patterns in 2024"
- "Give me the statistics of how many drivers were involved in more than one accident"

## Project Structure

```
metadata-driven-hybrid-rag/
├── agents/          # Specialized AI agents
├── evaluations/     # Evaluation metrics
├── flows/           # Application flows (insurance, etc.)
├── llms/            # Language model configurations
├── tools/           # RAG tools and utilities
└── requirements.txt # Python dependencies
```

## Exiting the Application

Type `quit`, `exit`, or `q` to stop the interactive session.
