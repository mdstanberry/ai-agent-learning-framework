"""
LLM Interface Module

This module provides a unified interface for calling different LLM providers
(OpenAI and Anthropic) with consistent error handling, retry logic, and
structured output support using Pydantic models.

Why this abstraction?
- Switch between providers with a config change
- Consistent error handling across providers
- Automatic retries with exponential backoff
- Type-safe responses with Pydantic validation
- Token tracking for cost management
"""

import time
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

# Import provider SDKs
try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Run: pip install openai")

try:
    from anthropic import Anthropic, APIError as AnthropicError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic package not installed. Run: pip install anthropic")

from utils.config import config
from utils.agent_logging import logger
from utils.schemas import TokenUsage, Message

# Type variable for generic Pydantic models
T = TypeVar('T', bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMClient:
    """
    Unified LLM client that works with multiple providers.
    
    This class handles:
    - Provider selection based on config
    - API authentication
    - Request formatting for each provider
    - Error handling and retries
    - Token usage tracking
    - Structured output parsing
    
    Usage:
        client = LLMClient()
        response = client.call(
            messages=[{"role": "user", "content": "Hello!"}],
            response_model=MyPydanticModel
        )
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider ("openai" or "anthropic"). 
                     If None, uses default from config.
        """
        self.provider = provider or config.get("llm.provider", "openai")
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Initialize the appropriate client
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise LLMError("OpenAI package not installed")
            api_key = config.get_api_key("openai")
            self.client = OpenAI(api_key=api_key)
            self.model_config = config.get_model_config("openai")
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise LLMError("Anthropic package not installed")
            api_key = config.get_api_key("anthropic")
            self.client = Anthropic(api_key=api_key)
            self.model_config = config.get_model_config("anthropic")
        else:
            raise LLMError(f"Unsupported provider: {self.provider}")
        
        logger.info(f"LLM Client initialized with provider: {self.provider}")
    
    def call(
        self,
        messages: Union[List[Dict[str, str]], List[Message]],
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Union[str, T, Dict]:
        """
        Call the LLM with retry logic and structured output support.
        
        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model for structured output
            system_prompt: Optional system prompt (will be prepended)
            temperature: Temperature setting (overrides config)
            max_tokens: Max tokens (overrides config)
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific parameters
            
        Returns:
            - String response if no response_model
            - Parsed Pydantic model if response_model provided
            - Dict with tool_calls if tools were used
            
        Raises:
            LLMError: If the call fails after all retries
        """
        # Format messages
        formatted_messages = self._format_messages(messages, system_prompt)
        
        # Get retry config
        retry_config = config.get_nested("llm.retry", {})
        max_retries = retry_config.get("max_retries", 3)
        initial_delay = retry_config.get("initial_delay", 1.0)
        max_delay = retry_config.get("max_delay", 60.0)
        exponential_base = retry_config.get("exponential_base", 2)
        
        # Attempt call with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                return self._call_provider(
                    formatted_messages,
                    response_model,
                    temperature,
                    max_tokens,
                    tools,
                    **kwargs
                )
            except (OpenAIError if OPENAI_AVAILABLE else Exception,
                    AnthropicError if ANTHROPIC_AVAILABLE else Exception) as e:
                last_error = e
                
                # Check if we should retry
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                    logger.error(f"Error details: {str(e)}", exception=e)
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts", exception=e)
        
        raise LLMError(f"Failed after {max_retries} attempts: {str(last_error)}")
    
    def _format_messages(
        self,
        messages: Union[List[Dict[str, str]], List[Message]],
        system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """
        Format messages into provider-expected format.
        
        Args:
            messages: Input messages
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Formatted messages list
        """
        # Convert Message objects to dicts
        if messages and isinstance(messages[0], Message):
            formatted = [{"role": msg.role, "content": msg.content} for msg in messages]
        else:
            formatted = list(messages)
        
        # Add system prompt if provided and not already present
        if system_prompt:
            if not formatted or formatted[0]["role"] != "system":
                formatted.insert(0, {"role": "system", "content": system_prompt})
        
        return formatted
    
    def _call_provider(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[T]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Dict]],
        **kwargs
    ) -> Union[str, T, Dict]:
        """
        Call the specific provider's API.
        
        Args:
            messages: Formatted messages
            response_model: Optional Pydantic model
            temperature: Temperature setting
            max_tokens: Max tokens
            tools: Tool definitions
            **kwargs: Additional parameters
            
        Returns:
            Response from provider
        """
        if self.provider == "openai":
            return self._call_openai(messages, response_model, temperature, max_tokens, tools, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, response_model, temperature, max_tokens, tools, **kwargs)
        else:
            raise LLMError(f"Unsupported provider: {self.provider}")
    
    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[T]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Dict]],
        **kwargs
    ) -> Union[str, T, Dict]:
        """Call OpenAI API."""
        # Prepare parameters
        params = {
            "model": self.model_config.get("default_model", "gpt-4-turbo"),
            "messages": messages,
            "temperature": temperature or self.model_config.get("temperature", 0.7),
            "max_tokens": max_tokens or self.model_config.get("max_tokens", 2000),
        }
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        # If structured output requested, use response format
        if response_model:
            # For structured outputs, we'll use JSON mode and parse manually
            params["response_format"] = {"type": "json_object"}
            # Add instruction to system message
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\nRespond with valid JSON matching this schema: {response_model.model_json_schema()}"
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": f"Respond with valid JSON matching this schema: {response_model.model_json_schema()}"
                })
        
        # Update with any additional kwargs
        params.update(kwargs)
        
        # Make the call
        response = self.client.chat.completions.create(**params)
        
        # Track token usage
        usage = response.usage
        self._track_tokens(usage.prompt_tokens, usage.completion_tokens, params["model"])
        
        # Extract content
        message = response.choices[0].message
        
        # Handle tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return {
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in message.tool_calls
                ],
                "finish_reason": response.choices[0].finish_reason
            }
        
        content = message.content
        
        # Parse structured output if requested
        if response_model:
            return self._parse_structured_output(content, response_model)
        
        return content
    
    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[T]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Dict]],
        **kwargs
    ) -> Union[str, T, Dict]:
        """Call Anthropic API."""
        # Extract system message (Anthropic handles it separately)
        system_message = None
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        
        # Prepare parameters
        params = {
            "model": self.model_config.get("default_model", "claude-3-5-sonnet-20241022"),
            "messages": messages,
            "temperature": temperature or self.model_config.get("temperature", 0.7),
            "max_tokens": max_tokens or self.model_config.get("max_tokens", 2000),
        }
        
        if system_message:
            params["system"] = system_message
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
        
        # If structured output requested, add to system prompt
        if response_model:
            schema_instruction = f"\n\nRespond with valid JSON matching this schema: {response_model.model_json_schema()}"
            if system_message:
                params["system"] = system_message + schema_instruction
            else:
                params["system"] = schema_instruction
        
        # Update with any additional kwargs
        params.update(kwargs)
        
        # Make the call
        response = self.client.messages.create(**params)
        
        # Track token usage
        self._track_tokens(response.usage.input_tokens, response.usage.output_tokens, params["model"])
        
        # Extract content
        content_block = response.content[0]
        
        # Handle tool use
        if content_block.type == "tool_use":
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": content_block.id,
                        "name": content_block.name,
                        "arguments": content_block.input
                    }
                ],
                "finish_reason": response.stop_reason
            }
        
        content = content_block.text
        
        # Parse structured output if requested
        if response_model:
            return self._parse_structured_output(content, response_model)
        
        return content
    
    def _parse_structured_output(self, content: str, response_model: Type[T]) -> T:
        """
        Parse LLM output into a Pydantic model.
        
        Args:
            content: Raw LLM output
            response_model: Pydantic model class
            
        Returns:
            Parsed and validated model instance
            
        Raises:
            LLMError: If parsing or validation fails
        """
        try:
            # Try to parse as JSON
            data = json.loads(content)
            # Validate with Pydantic
            return response_model(**data)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse JSON response: {e}\nContent: {content}")
        except ValidationError as e:
            raise LLMError(f"Response validation failed: {e}\nData: {data}")
    
    def _track_tokens(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """
        Track token usage and estimated cost.
        
        Args:
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            model: Model name
        """
        total = prompt_tokens + completion_tokens
        self.total_tokens_used += total
        
        # Calculate cost if tracking enabled
        if config.get_nested("cost_tracking.enabled", True):
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total
            )
            
            pricing = config.get_nested("cost_tracking.pricing", {})
            usage.calculate_cost(model, pricing)
            
            self.total_cost += usage.estimated_cost
            
            if config.get_nested("cost_tracking.log_token_usage", True):
                logger.info(
                    f"Token usage: {prompt_tokens} + {completion_tokens} = {total} "
                    f"(~${usage.estimated_cost:.4f})"
                )
    
    def get_total_cost(self) -> float:
        """Get total estimated cost for this session."""
        return self.total_cost
    
    def get_total_tokens(self) -> int:
        """Get total tokens used in this session."""
        return self.total_tokens_used


# Convenience function for simple calls
def call_llm(
    prompt: str,
    response_model: Optional[Type[T]] = None,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> Union[str, T]:
    """
    Convenience function for simple LLM calls.
    
    Args:
        prompt: User prompt
        response_model: Optional Pydantic model for structured output
        system_prompt: Optional system prompt
        provider: LLM provider (defaults to config)
        **kwargs: Additional parameters
        
    Returns:
        String response or parsed Pydantic model
        
    Example:
        # Simple call
        response = call_llm("What is 2+2?")
        
        # Structured output
        class Answer(BaseModel):
            result: int
        
        answer = call_llm("What is 2+2?", response_model=Answer)
        print(answer.result)  # 4
    """
    client = LLMClient(provider=provider)
    messages = [{"role": "user", "content": prompt}]
    return client.call(messages, response_model, system_prompt, **kwargs)


if __name__ == "__main__":
    # Test the LLM interface
    print("Testing LLM Interface...")
    print(f"Provider: {config.get('llm.provider')}")
    
    try:
        # Simple test call
        response = call_llm("Say 'Hello, world!' and nothing else.")
        print(f"\nSimple call result: {response}")
        
        print("\nLLM interface test completed!")
        
    except Exception as e:
        print(f"\nTest failed (this is expected if API keys are not set): {e}")
        print("\nTo use the LLM interface:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to .env")
        print("3. Run this test again")

