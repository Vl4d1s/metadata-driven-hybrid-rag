import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever , VectorCypherRetriever , Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


def vector_retriever():
    # return the vector retriver
    # retrive the chunk by vector similarity, with the properties selected
    retriever = VectorRetriever(
        driver,
        index_name="moviePlots",
        embedder=embedder,
        return_properties=["title", "plot"],
    )
    return retriever

def graph_enhanced_vector_retriever():
    # Define retrieval query
    #The query traverses the graph to find related nodes for genres and actors, as well as sorting the results by the user rating.
    retrieval_query = """
    MATCH (node)<-[r:RATED]-()
    RETURN 
    node.title AS title, node.plot AS plot, score AS similarityScore, 
    collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, 
    collect { MATCH (node)<-[:ACTED_IN]->(a) RETURN a.name } as actors, 
    avg(r.rating) as userRating
    ORDER BY userRating DESC
    """

    # Create retriever
    retriever = VectorCypherRetriever(
        driver,
        index_name="moviePlots",
        embedder=embedder,
        retrieval_query=retrieval_query,
    )
    return retriever

def text2cypher_retriver():
    # Cypher examples as input/query pairs
    examples = [
        "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
    ]
    t2c_llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature":0})
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=t2c_llm,
        examples=examples,
    )
    return retriever


retriever = graph_enhanced_vector_retriever()

# Search for similar items
result = retriever.search(query_text="Toys coming alive", top_k=5)

# Parse results
for item in result.items:
    print(item.content, item.metadata["score"])

# Create the LLM
llm = OpenAILLM(model_name="gpt-4o-mini",model_params={"temperature":0.2})

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
query_text = "Find me movies about toys coming alive"
 
response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)
# CLose the database connection
driver.close()