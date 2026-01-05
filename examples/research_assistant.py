"""
Research Assistant Example

This example demonstrates the ReAct agent loop with memory integration.
The agent can:
- Think about what to do next
- Act by calling tools (search, calculator, file operations)
- Observe the results and update its understanding
- Use memory to remember past interactions and facts

The agent integrates:
- Short-term memory: Current conversation context
- Episodic memory: Past research sessions
- Semantic memory: Facts learned about topics
- Procedural memory: Research templates and procedures

Run this example with:
    python examples/research_assistant.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.react_agent import ReActAgent
from memory.short_term import ShortTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory
from utils.agent_logging import logger


def main():
    """
    Main function demonstrating the ReAct agent with memory integration.
    
    This workflow:
    1. Initializes the agent with all memory types
    2. Sets a research goal
    3. Agent thinks, acts, and observes in a loop
    4. Uses tools to gather information
    5. Stores findings in memory
    6. Returns final answer
    """
    print("=" * 70)
    print("Research Assistant - ReAct Loop with Memory Demo")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  - ReAct loop: Think → Act → Observe")
    print("  - Tool usage: Search, calculator, file operations")
    print("  - Memory integration: All 4 memory types")
    print("\n" + "-" * 70)
    
    # Initialize memory systems
    print("\n[Initialization] Setting up memory systems...")
    short_term = ShortTermMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    procedural = ProceduralMemory()
    print("[OK] All memory systems initialized")
    
    # Store some initial facts in semantic memory
    print("\n[Semantic Memory] Storing initial knowledge...")
    semantic.add_fact("python", "type", "programming_language", entity_type="concept")
    semantic.add_fact("python", "popularity", "very_high", entity_type="concept")
    semantic.add_fact("ai_agents", "type", "technology_concept", entity_type="concept")
    print("[OK] Initial facts stored")
    
    # Store a research template in procedural memory
    print("\n[Procedural Memory] Storing research template...")
    procedural.store_procedure(
        name="research_workflow",
        steps=[
            "1. Define research question",
            "2. Search for relevant information",
            "3. Analyze findings",
            "4. Synthesize answer"
        ],
        category="research",
        tags=["workflow", "research"]
    )
    print("[OK] Research template stored")
    
    # Initialize the ReAct agent
    print("\n[Agent] Initializing ReAct agent...")
    agent = ReActAgent(
        system_prompt="You are a helpful research assistant. Use tools to gather information and provide accurate answers.",
        max_iterations=5
    )
    print("[OK] Agent initialized")
    
    # Example research goals
    research_goals = [
        "What is Python and why is it popular?",
        "Calculate: What is 15 * 23 + 42?",
        "Search for information about AI agents"
    ]
    
    print("\n" + "=" * 70)
    print("[Research Session]")
    print("=" * 70)
    
    for i, goal in enumerate(research_goals, 1):
        print(f"\n[Research Goal {i}] {goal}")
        print("-" * 70)
        
        try:
            # Add goal to short-term memory
            short_term.add_message("user", goal)
            
            # Run the agent
            print("\n[Agent] Starting ReAct loop...")
            result = agent.run(goal)
            
            if result.is_success:
                print(f"\n[OK] Agent completed successfully!")
                print(f"Final Answer: {result.result}")
                print(f"Iterations: {result.metadata.get('iterations', 0)}")
                print(f"Tools Used: {result.metadata.get('tools_used', [])}")
                
                # Store the interaction in episodic memory
                episodic.store_event(
                    content=f"Research session: {goal}",
                    event_type="research",
                    metadata={
                        "goal": goal,
                        "iterations": result.metadata.get('iterations', 0),
                        "tools_used": result.metadata.get('tools_used', [])
                    }
                )
                
                # Add agent response to short-term memory
                short_term.add_message("assistant", result.result)
                
            else:
                print(f"\n[FAIL] Agent execution failed: {result.error}")
        
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
    
    # Show memory statistics
    print("\n[Memory Statistics]")
    print("-" * 70)
    
    print("\nShort-Term Memory:")
    usage = short_term.get_token_usage()
    print(f"  Messages: {len(short_term.conversation.messages)}")
    print(f"  Token usage: {usage['total_context']}/{usage['context_budget']}")
    
    print("\nEpisodic Memory:")
    episodic_stats = episodic.get_stats()
    print(f"  Total events: {episodic_stats.get('total_events', 0)}")
    
    print("\nSemantic Memory:")
    semantic_stats = semantic.get_stats()
    print(f"  Total entities: {semantic_stats.get('total_entities', 0)}")
    print(f"  Total relationships: {semantic_stats.get('total_relationships', 0)}")
    
    print("\nProcedural Memory:")
    procedural_stats = procedural.get_stats()
    print(f"  Total items: {procedural_stats.get('total_items', 0)}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Research assistant demonstrated ReAct loop with memory!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  - ReAct loop: Think → Act → Observe cycle")
    print("  - Tool integration: Search, calculator, file operations")
    print("  - Short-term memory: Conversation context")
    print("  - Episodic memory: Research session history")
    print("  - Semantic memory: Facts and knowledge")
    print("  - Procedural memory: Research templates")


if __name__ == "__main__":
    main()

