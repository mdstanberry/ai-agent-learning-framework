"""
Short-Term Memory Module

This module implements short-term/working memory for AI agents - managing
the immediate context window and conversation history.

## When to Use Short-Term Memory

Short-term memory is used for:
- **Active conversation context**: The current conversation thread you're having
- **Immediate context window**: What the LLM needs to see right now
- **Session state**: Temporary information that only matters for this interaction
- **Working memory**: Information being actively processed in the current turn

**Use short-term memory when:**
- You need to maintain conversation continuity within a single session
- You want to track what was said in the current conversation
- You need to manage token budgets and context window limits
- You're processing information that's only relevant for the current task

**Do NOT use short-term memory for:**
- Long-term facts about users (use Semantic Memory)
- Past events or conversations (use Episodic Memory)
- Reusable templates or procedures (use Procedural Memory)
- Information that needs to persist across sessions

## How It Works

According to .cursorrules:
- Context window management: prioritize system prompt (fixed), then retrieved
  context (dynamic), then conversation history (truncate oldest first).
  Reserve 20% for output. [file:1][file:4][file:5]

The module automatically:
- Tracks conversation history with message roles (user, assistant, system, tool)
- Manages token budgets (80% for context, 20% for output)
- Summarizes old messages when context gets too long
- Prioritizes messages: system prompt > retrieved context > conversation history

## Example Usage

```python
from memory.short_term import ShortTermMemory

memory = ShortTermMemory(max_messages=50)

# Add messages to conversation
memory.add_message("user", "What is Python?")
memory.add_message("assistant", "Python is a programming language...")

# Get formatted context for LLM
context = memory.get_context_for_llm(
    system_prompt="You are a helpful assistant.",
    retrieved_context="Python documentation..."
)
```

## Comparison with Other Memory Types

- **vs. Episodic Memory**: Short-term is for current session; Episodic stores past events
- **vs. Semantic Memory**: Short-term is temporary context; Semantic stores permanent facts
- **vs. Procedural Memory**: Short-term is conversation state; Procedural stores reusable patterns
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from utils.schemas import Message, ConversationHistory
from utils.agent_logging import logger
from utils.config import config
from utils.llm import LLMClient


def safe_log(level: str, message: str, **kwargs) -> None:
    """
    Safely log a message, handling Unicode encoding errors.
    
    Args:
        level: Log level (info, error, warning, etc.)
        message: Message to log
        **kwargs: Additional arguments for logger methods
    """
    try:
        log_method = getattr(logger, level, logger.info)
        log_method(message, **kwargs)
    except (UnicodeEncodeError, Exception):
        # Fallback to simple print for Windows console issues
        print(f"[{level.upper()}] {message}")


class ShortTermMemory:
    """
    Manages short-term/working memory for an AI agent.
    
    This handles:
    - Conversation history
    - Context window management
    - Automatic summarization when context gets too long
    - Token budget management
    """
    
    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        summarize_threshold: Optional[int] = None,
        provider: Optional[str] = None
    ):
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum number of messages to keep (defaults to config)
            max_tokens: Maximum tokens for context window (defaults to config)
            summarize_threshold: When to trigger summarization (defaults to config)
            provider: LLM provider for summarization (defaults to config)
        """
        self.conversation = ConversationHistory()
        self.max_messages = max_messages or config.get_nested(
            "memory.short_term.max_messages",
            50
        )
        self.summarize_threshold = summarize_threshold or config.get_nested(
            "memory.short_term.summarize_threshold",
            40
        )
        
        # Token budget management (following .cursorrules: reserve 20% for output)
        # If max_tokens not specified, estimate based on model
        if max_tokens:
            self.max_tokens = max_tokens
        else:
            # Default to common model context sizes
            model_config = config.get_model_config()
            default_max = model_config.get("max_tokens", 2000) * 5  # Rough estimate
            self.max_tokens = default_max
        
        self.context_token_budget = int(self.max_tokens * 0.8)  # 80% for context
        self.output_token_budget = int(self.max_tokens * 0.2)  # 20% for output
        
        self.client = LLMClient(provider=provider) if provider else None
        
        # Track context components
        self.system_prompt_tokens: int = 0
        self.retrieved_context_tokens: int = 0
        self.conversation_tokens: int = 0
    
    def add_message(
        self,
        role: str,
        content: str,
        name: Optional[str] = None
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            name: Optional name (for tool messages)
        """
        self.conversation.add_message(role, content, name)
        
        # Check if we need to summarize
        if len(self.conversation.messages) > self.summarize_threshold:
            self._check_and_summarize()
    
    def get_context_for_llm(
        self,
        system_prompt: str,
        retrieved_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get formatted context for LLM following .cursorrules priority:
        1. System prompt (fixed)
        2. Retrieved context (dynamic)
        3. Conversation history (truncate oldest first)
        
        Args:
            system_prompt: System prompt (always included)
            retrieved_context: Optional retrieved context from RAG
            
        Returns:
            List of messages formatted for LLM
        """
        messages = []
        
        # Priority 1: System prompt (fixed, always included)
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        self.system_prompt_tokens = self._estimate_tokens(system_prompt)
        
        # Priority 2: Retrieved context (dynamic, if provided)
        if retrieved_context:
            messages.append({
                "role": "user",
                "content": f"[Retrieved Context]\n{retrieved_context}"
            })
            self.retrieved_context_tokens = self._estimate_tokens(retrieved_context)
        
        # Priority 3: Conversation history (truncate oldest first if needed)
        conversation_messages = self._get_conversation_within_budget()
        messages.extend(conversation_messages)
        
        self.conversation_tokens = sum(
            self._estimate_tokens(msg.get("content", ""))
            for msg in conversation_messages
        )
        
        total_tokens = (
            self.system_prompt_tokens +
            self.retrieved_context_tokens +
            self.conversation_tokens
        )
        
        safe_log(
            "info",
            f"Context tokens: system={self.system_prompt_tokens}, "
            f"retrieved={self.retrieved_context_tokens}, "
            f"conversation={self.conversation_tokens}, "
            f"total={total_tokens}/{self.context_token_budget}"
        )
        
        return messages
    
    def _get_conversation_within_budget(self) -> List[Dict[str, str]]:
        """
        Get conversation messages that fit within token budget.
        Truncates oldest messages first (excluding system messages).
        
        Returns:
            List of conversation messages within budget
        """
        # Get all non-system messages
        non_system_messages = [
            msg for msg in self.conversation.messages
            if msg.role != "system"
        ]
        
        # Start from most recent and work backwards
        selected_messages = []
        current_tokens = 0
        
        for msg in reversed(non_system_messages):
            msg_tokens = self._estimate_tokens(msg.content)
            
            if current_tokens + msg_tokens <= self.context_token_budget:
                selected_messages.insert(0, {
                    "role": msg.role,
                    "content": msg.content
                })
                current_tokens += msg_tokens
            else:
                # Can't fit this message, stop
                break
        
        return selected_messages
    
    def _check_and_summarize(self) -> None:
        """
        Check if conversation needs summarization and perform it if needed.
        
        Summarizes old messages to reduce context size while preserving
        important information.
        """
        if len(self.conversation.messages) <= self.summarize_threshold:
            return
        
        # Count non-system messages
        non_system_messages = [
            msg for msg in self.conversation.messages
            if msg.role != "system"
        ]
        
        if len(non_system_messages) <= self.summarize_threshold:
            return
        
        safe_log("info", "Conversation history is getting long, summarizing...")
        
        # Keep recent messages, summarize older ones
        keep_count = self.summarize_threshold // 2
        recent_messages = non_system_messages[-keep_count:]
        old_messages = non_system_messages[:-keep_count]
        
        # Summarize old messages
        summary = self._summarize_messages(old_messages)
        
        # Replace old messages with summary
        system_messages = [
            msg for msg in self.conversation.messages
            if msg.role == "system"
        ]
        
        # Create summary message
        summary_message = Message(
            role="system",
            content=f"[Conversation Summary]\n{summary}",
            timestamp=datetime.now()
        )
        
        # Rebuild conversation: system messages + summary + recent messages
        self.conversation.messages = (
            system_messages +
            [summary_message] +
            recent_messages
        )
        
        safe_log("success", f"Summarized {len(old_messages)} messages into summary")
    
    def _summarize_messages(self, messages: List[Message]) -> str:
        """
        Summarize a list of messages using LLM.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary string
        """
        if not self.client:
            # Fallback: simple text summary
            return f"Previous conversation with {len(messages)} messages"
        
        # Format messages for summarization
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])
        
        system_prompt = """You are a conversation summarizer. Create a concise
summary of the conversation that preserves key information, decisions, and context."""
        
        user_prompt = f"""Summarize this conversation:

{conversation_text}

Provide a concise summary that preserves:
- Key topics discussed
- Important decisions made
- Relevant context for future interactions"""
        
        try:
            summary = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )
            return summary if isinstance(summary, str) else str(summary)
        except Exception as e:
            safe_log("error", f"Summarization failed: {e}", exception=e)
            return f"Previous conversation with {len(messages)} messages (summarization unavailable)"
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple estimation: ~4 characters per token for English text.
        More accurate methods would use actual tokenizers.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 4 characters per token
        return len(text) // 4
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get current token usage breakdown.
        
        Returns:
            Dictionary with token usage by component
        """
        return {
            "system_prompt": self.system_prompt_tokens,
            "retrieved_context": self.retrieved_context_tokens,
            "conversation": self.conversation_tokens,
            "total_context": (
                self.system_prompt_tokens +
                self.retrieved_context_tokens +
                self.conversation_tokens
            ),
            "context_budget": self.context_token_budget,
            "output_budget": self.output_token_budget,
            "total_budget": self.max_tokens
        }
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.conversation = ConversationHistory()
        self.system_prompt_tokens = 0
        self.retrieved_context_tokens = 0
        self.conversation_tokens = 0
        safe_log("info", "Short-term memory cleared")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Short-Term Memory Demo")
    print("=" * 60)
    
    memory = ShortTermMemory(max_messages=10, summarize_threshold=8)
    
    # Add some messages
    print("\n[Test 1] Adding Messages")
    memory.add_message("user", "Hello, I need help with Python.")
    memory.add_message("assistant", "I'd be happy to help with Python!")
    memory.add_message("user", "How do I read a file?")
    memory.add_message("assistant", "You can use open() function...")
    
    print(f"Messages in memory: {len(memory.conversation.messages)}")
    
    # Get context for LLM
    print("\n[Test 2] Getting Context for LLM")
    system_prompt = "You are a helpful Python assistant."
    retrieved_context = "Python file operations documentation..."
    
    context = memory.get_context_for_llm(system_prompt, retrieved_context)
    print(f"Context messages: {len(context)}")
    print(f"Message roles: {[msg['role'] for msg in context]}")
    
    # Check token usage
    print("\n[Test 3] Token Usage")
    usage = memory.get_token_usage()
    for key, value in usage.items():
        print(f"  {key}: {value}")
    
    print("\nShort-term memory demo complete!")

