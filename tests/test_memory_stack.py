"""
Comprehensive Tests for Memory Stack

This module tests all four memory types with sample data to verify
they work correctly and demonstrate their usage.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable logging to avoid Unicode encoding issues in tests
os.environ["LOG_LEVEL"] = "ERROR"

from memory.short_term import ShortTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory


def test_short_term_memory():
    """Test Short-Term Memory with conversation history."""
    print("\n" + "=" * 60)
    print("Testing Short-Term Memory")
    print("=" * 60)
    
    memory = ShortTermMemory(max_messages=10, summarize_threshold=8)
    
    # Add conversation messages
    print("\n[Test 1] Adding Conversation Messages")
    memory.add_message("user", "Hello, I need help with Python.")
    memory.add_message("assistant", "I'd be happy to help with Python!")
    memory.add_message("user", "How do I read a file?")
    memory.add_message("assistant", "You can use the open() function...")
    memory.add_message("user", "What about writing to a file?")
    memory.add_message("assistant", "You can use open() with 'w' mode...")
    
    assert len(memory.conversation.messages) == 6, "Should have 6 messages"
    print(f"[OK] Added {len(memory.conversation.messages)} messages")
    
    # Test context retrieval
    print("\n[Test 2] Getting Context for LLM")
    system_prompt = "You are a helpful Python assistant."
    retrieved_context = "Python file operations documentation..."
    
    context = memory.get_context_for_llm(system_prompt, retrieved_context)
    assert len(context) > 0, "Context should not be empty"
    assert context[0]["role"] == "system", "First message should be system prompt"
    print(f"[OK] Retrieved {len(context)} context messages")
    
    # Test token usage tracking
    print("\n[Test 3] Token Usage Tracking")
    usage = memory.get_token_usage()
    assert "total_context" in usage, "Should track total context tokens"
    assert "context_budget" in usage, "Should track context budget"
    print(f"[OK] Token usage tracked: {usage['total_context']}/{usage['context_budget']}")
    
    # Test clearing
    print("\n[Test 4] Clearing Memory")
    memory.clear()
    assert len(memory.conversation.messages) == 0, "Memory should be cleared"
    print("[OK] Memory cleared successfully")
    
    print("\n[OK] Short-Term Memory tests passed!")


def test_episodic_memory():
    """Test Episodic Memory with event storage and search."""
    print("\n" + "=" * 60)
    print("Testing Episodic Memory")
    print("=" * 60)
    
    memory = EpisodicMemory()
    
    # Store events
    print("\n[Test 1] Storing Events")
    event1 = memory.store_event(
        content="User asked about Python file operations",
        event_type="conversation",
        metadata={"topic": "python", "user_id": "user123"}
    )
    event2 = memory.store_event(
        content="User requested help with API integration",
        event_type="conversation",
        metadata={"topic": "api", "user_id": "user123"}
    )
    event3 = memory.store_event(
        content="User completed a task successfully",
        event_type="interaction",
        metadata={"action": "task_completion", "user_id": "user123"}
    )
    
    assert event1 != "", "Should return event ID"
    print(f"[OK] Stored 3 events")
    
    # Test semantic search
    print("\n[Test 2] Semantic Search")
    results = memory.search("Python programming help", limit=3)
    assert len(results) > 0, "Should find matching events"
    print(f"[OK] Found {len(results)} matching events")
    
    # Test recent events
    print("\n[Test 3] Getting Recent Events")
    recent = memory.get_recent_events(limit=5)
    assert len(recent) > 0, "Should retrieve recent events"
    print(f"[OK] Retrieved {len(recent)} recent events")
    
    # Test stats
    print("\n[Test 4] Memory Statistics")
    stats = memory.get_stats()
    assert stats["enabled"] == True, "Memory should be enabled"
    assert stats["total_events"] >= 3, "Should have at least 3 events"
    print(f"[OK] Stats: {stats['total_events']} events stored")
    
    # Test deletion
    print("\n[Test 5] Deleting Event")
    deleted = memory.delete_event(event1)
    assert deleted == True, "Should delete event successfully"
    print("[OK] Event deleted successfully")
    
    print("\n[OK] Episodic Memory tests passed!")


def test_semantic_memory():
    """Test Semantic Memory with facts and relationships."""
    print("\n" + "=" * 60)
    print("Testing Semantic Memory")
    print("=" * 60)
    
    memory = SemanticMemory()
    
    # Store facts
    print("\n[Test 1] Storing Facts")
    memory.add_fact("user123", "name", "Alice", entity_type="user")
    memory.add_fact("user123", "age", 30, entity_type="user")
    memory.add_fact("user123", "location", "San Francisco", entity_type="user")
    memory.add_fact("product456", "name", "Widget Pro", entity_type="product")
    memory.add_fact("product456", "price", 99.99, entity_type="product")
    
    name = memory.get_fact("user123", "name")
    assert name == "Alice", "Should retrieve stored fact"
    print(f"[OK] Stored facts about user123 and product456")
    
    # Store relationships
    print("\n[Test 2] Storing Relationships")
    memory.add_relationship("user123", "product456", "purchased")
    memory.add_relationship("user123", "user789", "knows")
    memory.add_relationship("user789", "product456", "recommended")
    
    relationships = memory.get_relationships("user123")
    assert len(relationships) > 0, "Should retrieve relationships"
    print(f"[OK] Stored {len(relationships)} relationships")
    
    # Test relationship retrieval
    print("\n[Test 3] Getting Relationships")
    user_rels = memory.get_relationships("user123")
    assert len(user_rels) >= 2, "Should have at least 2 relationships"
    print(f"[OK] Retrieved {len(user_rels)} relationships for user123")
    
    # Test graph traversal
    print("\n[Test 4] Finding Related Entities")
    related = memory.find_related_entities("user123", max_depth=2)
    assert len(related) > 0, "Should find related entities"
    print(f"[OK] Found {len(related)} related entities")
    
    # Test querying
    print("\n[Test 5] Querying Entities")
    users = memory.query(entity_type="user")
    assert len(users) > 0, "Should find users"
    print(f"[OK] Found {len(users)} users")
    
    # Test stats
    print("\n[Test 6] Memory Statistics")
    stats = memory.get_stats()
    assert stats["enabled"] == True, "Memory should be enabled"
    assert stats["total_entities"] >= 3, "Should have at least 3 entities"
    print(f"[OK] Stats: {stats['total_entities']} entities, {stats['total_relationships']} relationships")
    
    print("\n[OK] Semantic Memory tests passed!")


def test_procedural_memory():
    """Test Procedural Memory with templates, snippets, and procedures."""
    print("\n" + "=" * 60)
    print("Testing Procedural Memory")
    print("=" * 60)
    
    memory = ProceduralMemory()
    
    # Store templates
    print("\n[Test 1] Storing Templates")
    template_id = memory.store_template(
        name="email_greeting",
        content="Hello {{name}},\n\nThank you for your interest in {{product}}.",
        category="email",
        tags=["greeting", "customer-service"],
        description="Standard email greeting template"
    )
    assert template_id == "email_greeting", "Should return template name"
    print("[OK] Stored email_greeting template")
    
    # Store snippets
    print("\n[Test 2] Storing Snippets")
    snippet1_id = memory.store_snippet(
        name="python_hello",
        content="print('Hello, World!')",
        language="python",
        category="examples",
        tags=["hello-world", "beginner"]
    )
    snippet2_id = memory.store_snippet(
        name="json_parser",
        content="import json\ndata = json.loads(json_string)",
        language="python",
        category="utilities",
        tags=["json", "parsing"]
    )
    assert snippet1_id == "python_hello", "Should return snippet name"
    print(f"[OK] Stored 2 snippets")
    
    # Store procedures
    print("\n[Test 3] Storing Procedures")
    procedure_id = memory.store_procedure(
        name="user_onboarding",
        steps=[
            "1. Create user account",
            "2. Send welcome email",
            "3. Set up initial preferences",
            "4. Log first activity"
        ],
        category="workflow",
        tags=["onboarding", "user-management"]
    )
    assert procedure_id == "user_onboarding", "Should return procedure name"
    print("[OK] Stored user_onboarding procedure")
    
    # Test retrieval
    print("\n[Test 4] Retrieving Items")
    template = memory.get_template("email_greeting")
    assert template is not None, "Should retrieve template"
    assert template["name"] == "email_greeting", "Should have correct name"
    
    snippet = memory.get_snippet("python_hello")
    assert snippet is not None, "Should retrieve snippet"
    assert snippet["language"] == "python", "Should have correct language"
    
    procedure = memory.get_procedure("user_onboarding")
    assert procedure is not None, "Should retrieve procedure"
    assert len(procedure["steps"]) == 4, "Should have 4 steps"
    print("[OK] Retrieved all item types successfully")
    
    # Test search
    print("\n[Test 5] Searching Items")
    python_results = memory.search("python", item_type="snippet")
    assert len(python_results) > 0, "Should find Python snippets"
    print(f"[OK] Found {len(python_results)} Python snippets")
    
    # Test listing
    print("\n[Test 6] Listing All Items")
    all_items = memory.list_all()
    assert len(all_items) >= 4, "Should list all items"
    print(f"[OK] Listed {len(all_items)} items")
    
    # Test stats
    print("\n[Test 7] Memory Statistics")
    stats = memory.get_stats()
    assert stats["enabled"] == True, "Memory should be enabled"
    assert stats["total_templates"] >= 1, "Should have at least 1 template"
    assert stats["total_snippets"] >= 2, "Should have at least 2 snippets"
    assert stats["total_procedures"] >= 1, "Should have at least 1 procedure"
    print(f"[OK] Stats: {stats['total_items']} total items")
    
    print("\n[OK] Procedural Memory tests passed!")


def run_all_tests():
    """Run all memory tests."""
    print("=" * 60)
    print("Memory Stack Comprehensive Tests")
    print("=" * 60)
    
    try:
        test_short_term_memory()
        test_episodic_memory()
        test_semantic_memory()
        test_procedural_memory()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("[OK] Short-Term Memory: Conversation context management")
        print("[OK] Episodic Memory: Event storage and semantic search")
        print("[OK] Semantic Memory: Facts and relationship storage")
        print("[OK] Procedural Memory: Templates, snippets, and procedures")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

