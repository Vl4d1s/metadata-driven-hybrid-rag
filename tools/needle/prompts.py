"""
Prompts for the needle tool functionality.
The needle tool is designed to find and return specific anchor/paragraph/section locators
rather than generating answers like the RAG tool.
"""

NEEDLE_SYSTEM_PROMPT = """
You are a needle tool that specializes in finding and returning specific location information 
for content within documents. Your role is to:

1. Locate specific paragraphs, sections, or anchors in documents
2. Return precise location information including:
   - Document ID and page numbers
   - Section names and chunk indices
   - Anchor identifiers for direct referencing
   - Content previews for verification
3. Provide structured location data that can be used for direct navigation

You do NOT generate answers or summaries - you only return location information.
"""

NEEDLE_CLASSIFICATION_PROMPT = """
Classify the following query for needle search functionality:

Query: {question}

Classification options:
- policy: For finding locations in insurance policy documents
- entity: For finding locations of specific entities (drivers, cars, accidents)

Examples:
{examples}

Provide only the classification (policy/entity) and brief reasoning.
"""

NEEDLE_POLICY_SEARCH_PROMPT = """
Find the specific locations in policy documents that contain information relevant to: {query}

Return location information including:
- Document anchors
- Page and section references  
- Chunk positions
- Content previews

Focus on precise location identification, not content interpretation.
"""

NEEDLE_ENTITY_SEARCH_PROMPT = """
Find the specific locations of entities that match: {query}

Return entity location information including:
- Entity identifiers and anchors
- Associated document references
- Relationship connections
- Location metadata

Focus on entity identification and location mapping, not detailed analysis.
"""

NEEDLE_RESULT_FORMAT_PROMPT = """
Format needle search results as JSON with the following structure:

{
  "query": "original search query",
  "classification": "policy|entity", 
  "locations": [
    {
      "anchor": "unique_anchor_identifier",
      "document_id": "document_identifier",
      "page_number": "page_reference",
      "section_name": "section_identifier", 
      "chunk_index": "position_in_section",
      "content_preview": "first_100_chars...",
      "similarity_score": "relevance_score",
      "rank": "result_ranking"
    }
  ],
  "anchors": ["list_of_anchor_ids"],
  "total_found": "number_of_results"
}

Ensure all location identifiers are suitable for direct navigation and referencing.
"""
