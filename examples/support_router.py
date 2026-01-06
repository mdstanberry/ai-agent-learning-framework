"""
Support Router Example

This example demonstrates the Routing pattern - classifying incoming queries
and routing them to specialized handlers based on the query type.

The router classifies queries into categories:
- Tech Support: Technical questions and troubleshooting
- Sales: Product inquiries and pricing questions
- Billing: Payment and account questions
- General: Everything else

Run this example with:
    python examples/support_router.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from patterns.routing import QueryRouter
from utils.agent_logging import logger


def main():
    """
    Main function demonstrating the Routing pattern.
    
    This workflow:
    1. Takes a user query as input
    2. Classifies the query into a category
    3. Routes to the appropriate specialized handler
    4. Returns a response from the specialized handler
    """
    print("=" * 70)
    print("Support Router - Routing Pattern Demo")
    print("=" * 70)
    print("\nThis example demonstrates query classification and routing:")
    print("  - Classifies queries into: Tech Support, Sales, Billing, General")
    print("  - Routes to specialized handlers for each category")
    print("\n" + "-" * 70)
    
    # Initialize the router
    router = QueryRouter()
    
    # Example queries for different categories
    example_queries = [
        {
            "query": "How do I reset my password?",
            "expected_category": "Tech Support",
            "description": "Technical support question"
        },
        {
            "query": "What are your pricing plans?",
            "expected_category": "Sales",
            "description": "Sales inquiry"
        },
        {
            "query": "I need to update my payment method",
            "expected_category": "Billing",
            "description": "Billing question"
        },
        {
            "query": "What is your company's mission?",
            "expected_category": "General",
            "description": "General inquiry"
        },
        {
            "query": "My API is returning 500 errors",
            "expected_category": "Tech Support",
            "description": "Technical troubleshooting"
        },
        {
            "query": "Do you offer enterprise discounts?",
            "expected_category": "Sales",
            "description": "Sales inquiry"
        }
    ]
    
    print("\n[Example Queries]")
    print("Testing router with various query types:")
    print("-" * 70)
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n[Example {i}] {example['description']}")
        print(f"Query: \"{example['query']}\"")
        print(f"Expected Category: {example['expected_category']}")
        print("-" * 70)
        
        try:
            # Route the query
            result = router.route_query(example['query'])
            
            if result.is_success:
                classification = result.result
                print(f"\n[Classification Result]")
                print(f"  Category: {classification.category}")
                print(f"  Confidence: {classification.confidence:.2%}")
                print(f"  Reasoning: {classification.reasoning}")
                
                # Get response from specialized handler
                print(f"\n[Handler Response]")
                handler_response = router.get_handler_response(classification)
                
                if handler_response.is_success:
                    response = handler_response.result
                    print(f"  Handler Type: {response.handler_type}")
                    print(f"  Response:")
                    # Print first 200 characters of response
                    response_text = response.response if hasattr(response, 'response') else str(response)
                    print(f"    {response_text[:200]}...")
                else:
                    print(f"  [FAIL] Handler response failed: {handler_response.error}")
            else:
                print(f"\n[FAIL] Routing failed: {result.error}")
        
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
    
    print("\n" + "=" * 70)
    print("SUCCESS: Support router demonstrated Routing pattern!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  - Query classification using LLM")
    print("  - Confidence scoring for classifications")
    print("  - Specialized handlers for each category")
    print("  - Fallback to general handler when needed")


if __name__ == "__main__":
    main()


