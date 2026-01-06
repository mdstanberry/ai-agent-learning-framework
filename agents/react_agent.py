"""
ReAct Agent Implementation

This module implements the ReAct (Reasoning + Acting) agent pattern.
The ReAct pattern enables agents to think, act, and observe in a loop
until they achieve their goal or reach a stopping condition.

Pattern Overview:
1. Think: Agent reasons about what to do next
2. Act: Agent executes an action (calls a tool or responds directly)
3. Observe: Agent processes the result and updates its understanding
4. Loop: Repeat until goal achieved or max iterations reached

When to Use:
- When the agent needs to dynamically decide what actions to take
- When the solution path is unpredictable
- When the agent needs to use tools to gather information
- When you need an agent that can reason about its actions

When NOT to Use:
- When the steps are predictable (use Chaining pattern instead)
- When you can classify and route (use Routing pattern instead)
- When tasks are independent (use Parallelization pattern instead)
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from utils.llm import LLMClient
from utils.agent_logging import logger
from utils.config import config
from utils.schemas import (
    Message,
    ConversationHistory,
    ToolCall,
    ToolResult,
    AgentThought,
    AgentAction,
    AgentResponse
)
from tools.registry import tool_registry


# =============================================================================
# Pydantic Models for ReAct Agent
# =============================================================================

class AgentState(BaseModel):
    """
    Represents the current state of the ReAct agent.
    """
    goal: str = Field(description="The agent's current goal")
    iteration: int = Field(default=0, description="Current iteration number")
    thoughts: List[AgentThought] = Field(
        default_factory=list,
        description="History of agent thoughts"
    )
    actions: List[AgentAction] = Field(
        default_factory=list,
        description="History of agent actions"
    )
    observations: List[str] = Field(
        default_factory=list,
        description="History of observations"
    )
    conversation: ConversationHistory = Field(
        default_factory=ConversationHistory,
        description="Conversation history"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Final answer when goal is achieved"
    )


class ThoughtAction(BaseModel):
    """
    Represents a thought-action pair in the ReAct loop.
    """
    thought: str = Field(description="Agent's reasoning")
    action_type: Literal["tool_call", "final_answer"] = Field(
        description="Type of action to take"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool to call (if action_type is tool_call)"
    )
    tool_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool arguments (if action_type is tool_call)"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Final answer (if action_type is final_answer)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Why this action was chosen"
    )


# =============================================================================
# ReAct Agent Implementation
# =============================================================================

class ReActAgent:
    """
    Implements the ReAct (Reasoning + Acting) agent pattern.
    
    This agent:
    1. Thinks about what to do next
    2. Acts by calling tools or providing an answer
    3. Observes the results
    4. Repeats until goal is achieved
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            provider: LLM provider to use (defaults to config)
            system_prompt: Custom system prompt (defaults to standard ReAct prompt)
        """
        self.client = LLMClient(provider=provider)
        self.max_iterations = config.get_nested("agent.max_iterations", 10)
        self.timeout = config.get_nested("agent.timeout", 300)
        self.enable_tools = config.get_nested("agent.enable_tools", True)
        
        # Default system prompt for ReAct agent
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Get available tools
        self.available_tools = tool_registry.get_tool_definitions(format="openai") if self.enable_tools else []
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for ReAct agent."""
        return """You are a helpful AI agent that uses the ReAct (Reasoning + Acting) pattern.

Your process:
1. THINK: Reason about what you need to do to achieve the goal
2. ACT: Either call a tool to gather information OR provide a final answer
3. OBSERVE: Process the results and update your understanding

Available tools:
{available_tools}

Guidelines:
- Think step-by-step about what information you need
- Use tools when you need to gather information or perform actions
- Provide a final answer when you have enough information
- Be clear and concise in your reasoning
- If you're stuck, explain what you tried and ask for clarification"""
    
    def run(self, goal: str) -> AgentState:
        """
        Run the ReAct agent to achieve a goal.
        
        This is the main entry point that orchestrates the Think-Act-Observe loop.
        
        Args:
            goal: The goal the agent should achieve
            
        Returns:
            Final agent state with the result
        """
        logger.section("ReAct Agent", f"Goal: {goal}")
        
        # Initialize agent state
        state = AgentState(goal=goal)
        
        # Format system prompt with available tools
        formatted_prompt = self.system_prompt.replace(
            "{available_tools}",
            self._format_tools_for_prompt()
        )
        
        state.conversation.add_message("system", formatted_prompt)
        state.conversation.add_message("user", goal)
        
        # ReAct loop
        while state.iteration < self.max_iterations:
            state.iteration += 1
            
            logger.info(f"Iteration {state.iteration}/{self.max_iterations}")
            
            # Step 1: Think
            logger.thought(f"Thinking about how to achieve: {goal}")
            thought_action = self._think(state)
            
            state.thoughts.append(AgentThought(content=thought_action.thought))
            logger.observation(f"Thought: {thought_action.thought}")
            
            # Step 2: Act
            if thought_action.action_type == "final_answer":
                logger.action("Providing final answer")
                state.final_answer = thought_action.final_answer
                state.conversation.add_message("assistant", thought_action.final_answer)
                logger.final_answer(thought_action.final_answer)
                break
            
            elif thought_action.action_type == "tool_call" and self.enable_tools:
                logger.action(
                    f"Calling tool: {thought_action.tool_name}",
                    tool_name=thought_action.tool_name,
                    arguments=thought_action.tool_arguments
                )
                
                # Execute tool
                tool_call = ToolCall(
                    id=f"call_{state.iteration}",
                    name=thought_action.tool_name,
                    arguments=thought_action.tool_arguments or {}
                )
                
                tool_result = tool_registry.execute_tool(tool_call)
                state.actions.append(AgentAction(
                    action_type="tool_call",
                    tool_call=tool_call,
                    reasoning=thought_action.reasoning
                ))
                
                # Step 3: Observe
                observation = self._observe(tool_result)
                state.observations.append(observation)
                
                logger.observation(f"Tool result: {observation[:200]}...")
                
                # Add tool result to conversation
                state.conversation.add_message(
                    "tool",
                    f"Tool {tool_call.name} returned: {tool_result.result}",
                    name=tool_call.name
                )
                
                # Add observation to conversation for next iteration
                state.conversation.add_message(
                    "assistant",
                    f"I observed: {observation}"
                )
            
            else:
                logger.warning("Invalid action type or tools disabled")
                break
        
        if state.iteration >= self.max_iterations:
            logger.warning(f"Reached max iterations ({self.max_iterations})")
            if not state.final_answer:
                state.final_answer = "I reached the maximum number of iterations without finding a complete answer."
        
        logger.success(f"Agent completed in {state.iteration} iterations")
        return state
    
    def _think(self, state: AgentState) -> ThoughtAction:
        """
        Think step: Agent reasons about what to do next.
        
        Args:
            state: Current agent state
            
        Returns:
            Thought-action pair
        """
        # Build context from conversation history
        messages = state.conversation.get_messages_for_llm()
        
        # Add instruction for ReAct pattern
        think_prompt = """Based on the conversation so far, decide what to do next.

Think step-by-step:
1. What information do I need to answer the question?
2. Do I have enough information, or do I need to use a tool?
3. If I need a tool, which one and with what arguments?
4. If I have enough information, what is my final answer?

Be clear in your reasoning."""
        
        messages.append({"role": "user", "content": think_prompt})
        
        try:
            # Call LLM with tool support
            response = self.client.call(
                messages=messages,
                tools=self.available_tools if self.enable_tools else None
            )
            
            # Check if response contains tool calls
            if isinstance(response, dict) and "tool_calls" in response:
                # LLM wants to call a tool
                tool_call_data = response["tool_calls"][0]
                return ThoughtAction(
                    thought=response.get("content", "I need to use a tool to gather information."),
                    action_type="tool_call",
                    tool_name=tool_call_data["name"],
                    tool_arguments=tool_call_data["arguments"],
                    reasoning="Need to gather information using a tool"
                )
            else:
                # LLM provided a final answer
                answer_text = response if isinstance(response, str) else response.get("content", str(response))
                return ThoughtAction(
                    thought=f"I have enough information to provide an answer.",
                    action_type="final_answer",
                    final_answer=answer_text,
                    reasoning="Have sufficient information to answer"
                )
            
        except Exception as e:
            logger.error(f"Think step failed: {e}", exception=e)
            # Fallback: try to get a text response without tools
            try:
                response = self.client.call(messages=messages)
                answer_text = response if isinstance(response, str) else str(response)
                
                # Simple parsing
                if any(phrase in answer_text.lower() for phrase in ["final answer", "answer:", "conclusion"]):
                    return ThoughtAction(
                        thought=answer_text,
                        action_type="final_answer",
                        final_answer=answer_text
                    )
                else:
                    # Default to needing more information
                    return ThoughtAction(
                        thought=answer_text,
                        action_type="tool_call",
                        tool_name="search_web",
                        tool_arguments={"query": state.goal},
                        reasoning="Need to search for information"
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback think step also failed: {fallback_error}", exception=fallback_error)
                # Last resort: return a basic action
                return ThoughtAction(
                    thought="Unable to determine next action",
                    action_type="final_answer",
                    final_answer="I encountered an error while processing your request."
                )
    
    def _observe(self, tool_result: ToolResult) -> str:
        """
        Observe step: Process tool result and extract useful information.
        
        Args:
            tool_result: Result from tool execution
            
        Returns:
            Observation string summarizing what was learned
        """
        if not tool_result.is_success:
            return f"Tool execution failed: {tool_result.error}"
        
        # Format the observation
        if isinstance(tool_result.result, dict):
            # Try to extract key information
            if "content" in tool_result.result:
                return str(tool_result.result["content"])
            elif "results" in tool_result.result:
                return f"Found {len(tool_result.result['results'])} results"
            else:
                return str(tool_result.result)
        elif isinstance(tool_result.result, list):
            return f"Received {len(tool_result.result)} items"
        else:
            return str(tool_result.result)
    
    def _format_tools_for_prompt(self) -> str:
        """
        Format available tools for inclusion in system prompt.
        
        Returns:
            Formatted string describing available tools
        """
        if not self.available_tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.available_tools:
            if "function" in tool:
                func = tool["function"]
                tool_descriptions.append(
                    f"- {func['name']}: {func['description']}"
                )
        
        return "\n".join(tool_descriptions)


if __name__ == "__main__":
    # Import tools to ensure they're registered
    import tools.calculator  # noqa: F401
    import tools.search  # noqa: F401
    import tools.file_ops  # noqa: F401
    
    # Example usage
    print("=" * 60)
    print("ReAct Agent Demo")
    print("=" * 60)
    
    agent = ReActAgent()
    
    # Test 1: Simple goal with one tool call
    print("\n" + "=" * 60)
    print("Test 1: Simple Search Goal")
    print("=" * 60)
    goal1 = "What is the weather like today? Use the search tool to find current weather information."
    
    try:
        result1 = agent.run(goal1)
        
        print("\n" + "=" * 60)
        print("Agent Execution Summary")
        print("=" * 60)
        print(f"\nGoal: {result1.goal}")
        print(f"Iterations: {result1.iteration}")
        print(f"\nThoughts: {len(result1.thoughts)}")
        for i, thought in enumerate(result1.thoughts, 1):
            print(f"  {i}. {thought.content[:100]}...")
        
        print(f"\nActions: {len(result1.actions)}")
        for i, action in enumerate(result1.actions, 1):
            if action.tool_call:
                print(f"  {i}. Called tool: {action.tool_call.name}")
        
        print(f"\nObservations: {len(result1.observations)}")
        for i, obs in enumerate(result1.observations, 1):
            print(f"  {i}. {obs[:100]}...")
        
        if result1.final_answer:
            print(f"\nFinal Answer:\n{result1.final_answer}")
        else:
            print("\nNo final answer provided.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Goal requiring multiple tool calls
    print("\n\n" + "=" * 60)
    print("Test 2: Multiple Tool Calls (Task 9.10)")
    print("=" * 60)
    print("Goal: Calculate 2 to the power of 10, then search for information about that number")
    
    goal2 = "Calculate 2 to the power of 10, then search the web for information about that number."
    
    try:
        result2 = agent.run(goal2)
        
        print("\n" + "=" * 60)
        print("Agent Execution Summary")
        print("=" * 60)
        print(f"\nGoal: {result2.goal}")
        print(f"Iterations: {result2.iteration}")
        print(f"\nThoughts: {len(result2.thoughts)}")
        for i, thought in enumerate(result2.thoughts, 1):
            print(f"  {i}. {thought.content[:150]}...")
        
        print(f"\nActions: {len(result2.actions)}")
        tool_calls_made = []
        for i, action in enumerate(result2.actions, 1):
            if action.tool_call:
                tool_name = action.tool_call.name
                tool_calls_made.append(tool_name)
                print(f"  {i}. Called tool: {tool_name}")
                if action.tool_call.arguments:
                    print(f"      Arguments: {action.tool_call.arguments}")
        
        print(f"\nObservations: {len(result2.observations)}")
        for i, obs in enumerate(result2.observations, 1):
            print(f"  {i}. {obs[:150]}...")
        
        if result2.final_answer:
            print(f"\nFinal Answer:\n{result2.final_answer}")
        else:
            print("\nNo final answer provided.")
        
        # Verify multiple tool calls were made
        print("\n" + "=" * 60)
        print("Test Verification")
        print("=" * 60)
        unique_tools = set(tool_calls_made)
        print(f"Total tool calls: {len(tool_calls_made)}")
        print(f"Unique tools used: {len(unique_tools)}")
        print(f"Tools called: {', '.join(tool_calls_made)}")
        
        if len(tool_calls_made) >= 2:
            print("\n[SUCCESS] Agent made multiple tool calls as required!")
        else:
            print(f"\n[WARNING] Agent should have made multiple tool calls but only made {len(tool_calls_made)}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your API key to .env")
        print("3. Installed all dependencies: pip install -r requirements.txt")

