"""
Common Pydantic Schemas Module

This module defines Pydantic models used throughout the AI agent framework.
These models ensure type safety and structured data validation.

Why Pydantic?
- Automatic validation of data types
- Clear error messages when data is invalid
- Easy conversion to/from JSON
- IDE autocomplete support
"""

from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Message and Conversation Schemas
# =============================================================================

class Message(BaseModel):
    """
    Represents a single message in a conversation.
    
    Used in conversation history and agent interactions.
    """
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(
        description="The message content"
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional name of the sender (useful for tool results)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "What is the weather today?",
                "timestamp": "2024-01-04T10:30:00"
            }
        }
    )


class ConversationHistory(BaseModel):
    """
    Represents a full conversation history.
    
    Manages the list of messages and provides utility methods.
    """
    messages: List[Message] = Field(
        default_factory=list,
        description="List of messages in chronological order"
    )
    max_messages: int = Field(
        default=50,
        description="Maximum number of messages to retain"
    )
    
    def add_message(self, role: str, content: str, name: Optional[str] = None) -> None:
        """Add a new message to the conversation."""
        self.messages.append(Message(role=role, content=content, name=name))
        
        # Trim if exceeds max (keep system messages)
        if len(self.messages) > self.max_messages:
            system_messages = [m for m in self.messages if m.role == "system"]
            other_messages = [m for m in self.messages if m.role != "system"]
            self.messages = system_messages + other_messages[-(self.max_messages - len(system_messages)):]
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Convert messages to format expected by LLM APIs."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]


# =============================================================================
# Tool Schemas
# =============================================================================

class ToolParameter(BaseModel):
    """
    Defines a single parameter for a tool.
    """
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, number, boolean, etc.)")
    description: str = Field(description="What this parameter does")
    required: bool = Field(default=True, description="Whether this parameter is required")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values (if applicable)")


class ToolDefinition(BaseModel):
    """
    Defines a tool that an agent can use.
    
    This schema is used to register tools and generate JSON schemas for LLMs.
    """
    name: str = Field(description="Tool name (should be snake_case)")
    description: str = Field(description="What the tool does")
    parameters: List[ToolParameter] = Field(
        default_factory=list,
        description="List of parameters the tool accepts"
    )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class ToolCall(BaseModel):
    """
    Represents a request to call a tool.
    """
    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "call_123",
                "name": "search_web",
                "arguments": {"query": "weather in San Francisco"}
            }
        }
    )


class ToolResult(BaseModel):
    """
    Represents the result of a tool execution.
    """
    tool_call_id: str = Field(description="ID of the tool call this result corresponds to")
    tool_name: str = Field(description="Name of the tool that was executed")
    result: Any = Field(description="The result returned by the tool")
    error: Optional[str] = Field(default=None, description="Error message if tool execution failed")
    execution_time: float = Field(default=0.0, description="Time taken to execute (seconds)")
    
    @property
    def is_success(self) -> bool:
        """Check if tool execution was successful."""
        return self.error is None


# =============================================================================
# Agent Response Schemas
# =============================================================================

class AgentThought(BaseModel):
    """
    Represents an agent's reasoning or thought process.
    
    Used in ReAct pattern to log what the agent is thinking.
    """
    content: str = Field(description="The agent's thought or reasoning")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "I need to search for the weather first before I can answer",
                "timestamp": "2024-01-04T10:30:00"
            }
        }
    )


class AgentAction(BaseModel):
    """
    Represents an action the agent wants to take.
    
    This could be calling a tool or providing a final answer.
    """
    action_type: Literal["tool_call", "final_answer"] = Field(
        description="Type of action to take"
    )
    tool_call: Optional[ToolCall] = Field(
        default=None,
        description="Tool to call (if action_type is tool_call)"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Final answer to return (if action_type is final_answer)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Why this action was chosen"
    )


class AgentResponse(BaseModel):
    """
    Complete response from an agent.
    
    Includes the response content, any tool calls, and metadata.
    """
    content: str = Field(description="The agent's response content")
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tools the agent wants to call"
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Why the agent stopped (e.g., 'stop', 'tool_calls', 'length')"
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Number of tokens used in this response"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model that generated this response"
    )


# =============================================================================
# Pattern-Specific Schemas
# =============================================================================

class ChainStep(BaseModel):
    """
    Represents a single step in a prompt chain.
    """
    step_number: int = Field(description="Step number in the chain")
    step_name: str = Field(description="Name of this step")
    input: str = Field(description="Input to this step")
    output: str = Field(description="Output from this step")
    model_used: Optional[str] = Field(default=None)
    tokens_used: Optional[int] = Field(default=None)
    execution_time: float = Field(default=0.0)


class RouteClassification(BaseModel):
    """
    Result of routing classification.
    """
    route: str = Field(description="The selected route/category")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Why this route was chosen"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class EvaluationScore(BaseModel):
    """
    Quality evaluation score for iterative improvement patterns.
    """
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1)"
    )
    criteria_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores for individual criteria"
    )
    feedback: str = Field(description="Detailed feedback on how to improve")
    meets_threshold: bool = Field(description="Whether the output meets quality threshold")


# =============================================================================
# Cost Tracking Schemas
# =============================================================================

class TokenUsage(BaseModel):
    """
    Tracks token usage and costs.
    """
    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")
    model: Optional[str] = Field(default=None, description="Model used")
    
    def calculate_cost(self, model: str, pricing: Dict[str, Dict[str, float]]) -> None:
        """
        Calculate estimated cost based on token usage and pricing.
        
        Args:
            model: Model name
            pricing: Pricing info (cost per 1M tokens)
        """
        self.model = model
        
        # Extract provider from model name
        provider = "openai" if "gpt" in model.lower() else "anthropic"
        
        if provider in pricing and model in pricing[provider]:
            model_pricing = pricing[provider][model]
            input_cost = (self.prompt_tokens / 1_000_000) * model_pricing["input"]
            output_cost = (self.completion_tokens / 1_000_000) * model_pricing["output"]
            self.estimated_cost = input_cost + output_cost


class SessionMetrics(BaseModel):
    """
    Tracks metrics for an entire agent session.
    """
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    num_llm_calls: int = Field(default=0)
    num_tool_calls: int = Field(default=0)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None)
    
    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


if __name__ == "__main__":
    # Test the schemas
    print("Testing Pydantic schemas...")
    
    # Test Message
    msg = Message(role="user", content="Hello!")
    print(f"[OK] Message: {msg.role} - {msg.content}")
    
    # Test ToolDefinition
    tool = ToolDefinition(
        name="search_web",
        description="Search the web for information",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query", required=True)
        ]
    )
    print(f"[OK] ToolDefinition: {tool.name}")
    print(f"     OpenAI format: {tool.to_openai_format()['function']['name']}")
    
    # Test TokenUsage
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    print(f"[OK] TokenUsage: {usage.total_tokens} tokens")
    
    print("\nAll schemas validated successfully!")

