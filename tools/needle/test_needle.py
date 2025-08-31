"""
Test script for the needle tool functionality.
This demonstrates how to use the needle tool to find specific locations.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.needle.tool import get_needle_tool, needle_tool
import json

def test_needle_tool():
    """
    Test the needle tool with sample queries.
    """
    print("=" * 60)
    print("NEEDLE TOOL TEST")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "Find the section about collision coverage",
        "Locate deductible information in the policy", 
        "Find information about driver John Smith",
        "Locate accident details on Main Street",
        "Find the paragraph about vehicle damage"
    ]
    
    # Get the needle tool
    needle_tool_instance = get_needle_tool()
    
    print(f"Tool Name: {needle_tool_instance.name}")
    print(f"Tool Description: {needle_tool_instance.description}")
    print("\n" + "=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTEST {i}: {query}")
        print("-" * 40)
        
        try:
            # Test the needle tool function directly
            result = needle_tool(query)
            
            # Parse and display results
            result_data = json.loads(result)
            
            print(f"Classification: {result_data.get('classification', 'N/A')}")
            print(f"Total Found: {result_data.get('total_found', 0)}")
            print(f"Anchors: {result_data.get('anchors', [])}")
            
            if result_data.get('locations'):
                print("\nLocation Details:")
                for loc in result_data['locations'][:2]:  # Show first 2 results
                    print(f"  - Anchor: {loc.get('anchor', 'N/A')}")
                    if 'document_id' in loc:
                        print(f"    Document: {loc['document_id']}, Page: {loc['page_number']}")
                        print(f"    Section: {loc['section_name']}, Chunk: {loc['chunk_index']}")
                    elif 'entity_id' in loc:
                        print(f"    Entity: {loc['entity_id']}, Type: {loc['entity_type']}")
                        print(f"    Location: {loc['location']}, City: {loc['city']}")
                    print(f"    Score: {loc.get('similarity_score', 'N/A')}")
            
            if result_data.get('error'):
                print(f"Error: {result_data['error']}")
                
        except Exception as e:
            print(f"Error testing query: {e}")
        
        print("-" * 40)

def test_tool_integration():
    """
    Test the LangChain Tool integration.
    """
    print("\n" + "=" * 60)
    print("LANGCHAIN TOOL INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Get the tool instance
        needle_tool_instance = get_needle_tool()
        
        # Test using the tool's func method
        test_query = "Find collision coverage information"
        print(f"\nTesting with query: {test_query}")
        
        result = needle_tool_instance.func(test_query)
        result_data = json.loads(result)
        
        print(f"Result type: {type(result)}")
        print(f"Classification: {result_data.get('classification')}")
        print(f"Total locations found: {result_data.get('total_found', 0)}")
        print(f"Number of anchors: {len(result_data.get('anchors', []))}")
        
        print("\nTool integration successful!")
        
    except Exception as e:
        print(f"Tool integration error: {e}")

if __name__ == "__main__":
    print("Note: This test requires a properly configured Neo4j database")
    print("and the insurance data to be loaded for full functionality.\n")
    
    # For demonstration purposes, we'll show the tool structure
    print("NEEDLE TOOL STRUCTURE DEMONSTRATION")
    print("=" * 60)
    
    try:
        needle_tool_instance = get_needle_tool()
        print(f"✓ Tool created successfully")
        print(f"  Name: {needle_tool_instance.name}")
        print(f"  Description: {needle_tool_instance.description}")
        print(f"  Function: {needle_tool_instance.func}")
        
        print(f"\n✓ Needle tool is ready for use!")
        print(f"  Use get_needle_tool() to get the LangChain Tool instance")
        print(f"  Use needle_tool(query) to search for specific locations")
        
    except Exception as e:
        print(f"✗ Error creating needle tool: {e}")
    
    # Uncomment the following lines to run actual tests with database
    # test_needle_tool()
    # test_tool_integration()
