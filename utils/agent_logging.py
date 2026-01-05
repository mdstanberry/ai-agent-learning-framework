"""
Rich-Formatted Logging Module

This module provides color-coded logging for AI agents using the Rich library.
The goal is to make the agent's "thought process" visible and easy to follow.

Color Scheme:
- THOUGHT (Blue): What the agent is thinking/reasoning
- ACTION (Yellow): What action the agent is taking (e.g., calling a tool)
- OBSERVATION (Green): Results/observations from actions
- ERROR (Red): Errors and warnings
- INFO (White): General information
"""

from typing import Any, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
import json


class AgentLogger:
    """
    Rich-formatted logger for AI agents.
    
    Provides methods to log different types of agent activities with
    color coding for easy visual tracking.
    
    Usage:
        logger = AgentLogger()
        logger.thought("I need to search for information")
        logger.action("Calling search_web tool")
        logger.observation("Found 5 results")
    """
    
    def __init__(self, enable_timestamps: bool = True, enable_colors: bool = True):
        """
        Initialize the agent logger.
        
        Args:
            enable_timestamps: Whether to show timestamps in logs
            enable_colors: Whether to use colored output
        """
        self.console = Console(force_terminal=enable_colors)
        self.enable_timestamps = enable_timestamps
        self.enable_colors = enable_colors
        self.step_counter = 0
    
    def _format_timestamp(self) -> str:
        """Get formatted timestamp string."""
        if not self.enable_timestamps:
            return ""
        return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
    
    def thought(self, message: str, **kwargs) -> None:
        """
        Log an agent's thought or reasoning.
        
        Args:
            message: The thought content
            **kwargs: Additional metadata to display
        """
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append("💭 THOUGHT: ", style="bold blue")
            text.append(message, style="blue")
            self.console.print(timestamp, text)
        else:
            self.console.print(f"{timestamp}THOUGHT: {message}")
        
        # Print any additional metadata
        if kwargs:
            self._print_metadata(kwargs, style="dim blue")
    
    def action(self, message: str, tool_name: Optional[str] = None, 
               arguments: Optional[dict] = None) -> None:
        """
        Log an agent's action (e.g., calling a tool).
        
        Args:
            message: Description of the action
            tool_name: Name of the tool being called (optional)
            arguments: Tool arguments (optional)
        """
        self.step_counter += 1
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append(f"⚡ ACTION #{self.step_counter}: ", style="bold yellow")
            text.append(message, style="yellow")
            self.console.print(timestamp, text)
            
            # Show tool details if provided
            if tool_name:
                self.console.print(f"   Tool: {tool_name}", style="yellow")
            if arguments:
                self.console.print(f"   Args: {json.dumps(arguments, indent=2)}", style="dim yellow")
        else:
            self.console.print(f"{timestamp}ACTION #{self.step_counter}: {message}")
            if tool_name:
                self.console.print(f"   Tool: {tool_name}")
            if arguments:
                self.console.print(f"   Args: {json.dumps(arguments, indent=2)}")
    
    def observation(self, message: str, result: Optional[Any] = None) -> None:
        """
        Log an observation or result from an action.
        
        Args:
            message: Description of the observation
            result: The actual result data (optional)
        """
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append("👁️  OBSERVATION: ", style="bold green")
            text.append(message, style="green")
            self.console.print(timestamp, text)
            
            # Show result if provided
            if result is not None:
                self._print_result(result, style="dim green")
        else:
            self.console.print(f"{timestamp}OBSERVATION: {message}")
            if result is not None:
                self.console.print(f"   Result: {result}")
    
    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error.
        
        Args:
            message: Error message
            exception: The exception object (optional)
        """
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append("❌ ERROR: ", style="bold red")
            text.append(message, style="red")
            self.console.print(timestamp, text)
            
            if exception:
                self.console.print(f"   {type(exception).__name__}: {str(exception)}", 
                                 style="dim red")
        else:
            self.console.print(f"{timestamp}ERROR: {message}")
            if exception:
                self.console.print(f"   {type(exception).__name__}: {str(exception)}")
    
    def info(self, message: str) -> None:
        """
        Log general information.
        
        Args:
            message: Information message
        """
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append("ℹ️  INFO: ", style="bold white")
            text.append(message, style="white")
            self.console.print(timestamp, text)
        else:
            self.console.print(f"{timestamp}INFO: {message}")
    
    def success(self, message: str) -> None:
        """
        Log a success message.
        
        Args:
            message: Success message
        """
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append("✅ SUCCESS: ", style="bold green")
            text.append(message, style="green")
            self.console.print(timestamp, text)
        else:
            self.console.print(f"{timestamp}SUCCESS: {message}")
    
    def warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Warning message
        """
        timestamp = self._format_timestamp()
        
        if self.enable_colors:
            text = Text()
            text.append("⚠️  WARNING: ", style="bold yellow")
            text.append(message, style="yellow")
            self.console.print(timestamp, text)
        else:
            self.console.print(f"{timestamp}WARNING: {message}")
    
    def section(self, title: str, subtitle: Optional[str] = None) -> None:
        """
        Log a section header (useful for separating different phases).
        
        Args:
            title: Section title
            subtitle: Optional subtitle
        """
        if self.enable_colors:
            panel_title = Text(title, style="bold cyan")
            if subtitle:
                panel = Panel(
                    subtitle,
                    title=panel_title,
                    border_style="cyan",
                    box=box.ROUNDED
                )
            else:
                panel = Panel(
                    "",
                    title=panel_title,
                    border_style="cyan",
                    box=box.ROUNDED
                )
            self.console.print(panel)
        else:
            self.console.print(f"\n{'='*60}")
            self.console.print(f"{title}")
            if subtitle:
                self.console.print(f"{subtitle}")
            self.console.print(f"{'='*60}\n")
    
    def final_answer(self, answer: str) -> None:
        """
        Log the agent's final answer in a highlighted panel.
        
        Args:
            answer: The final answer
        """
        if self.enable_colors:
            panel = Panel(
                answer,
                title="[bold magenta]🎯 Final Answer[/bold magenta]",
                border_style="magenta",
                box=box.DOUBLE
            )
            self.console.print(panel)
        else:
            self.console.print(f"\n{'='*60}")
            self.console.print(f"FINAL ANSWER:")
            self.console.print(answer)
            self.console.print(f"{'='*60}\n")
    
    def token_usage(self, prompt_tokens: int, completion_tokens: int, 
                    total_tokens: int, estimated_cost: float) -> None:
        """
        Log token usage and cost information.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            estimated_cost: Estimated cost in USD
        """
        if self.enable_colors:
            table = Table(title="Token Usage", box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            
            table.add_row("Prompt Tokens", str(prompt_tokens))
            table.add_row("Completion Tokens", str(completion_tokens))
            table.add_row("Total Tokens", str(total_tokens))
            table.add_row("Estimated Cost", f"${estimated_cost:.4f}")
            
            self.console.print(table)
        else:
            self.console.print(f"\nToken Usage:")
            self.console.print(f"  Prompt Tokens: {prompt_tokens}")
            self.console.print(f"  Completion Tokens: {completion_tokens}")
            self.console.print(f"  Total Tokens: {total_tokens}")
            self.console.print(f"  Estimated Cost: ${estimated_cost:.4f}\n")
    
    def _print_metadata(self, metadata: dict, style: str = "dim") -> None:
        """Print metadata in a readable format."""
        for key, value in metadata.items():
            self.console.print(f"   {key}: {value}", style=style)
    
    def _print_result(self, result: Any, style: str = "dim") -> None:
        """Print result data in a readable format."""
        if isinstance(result, dict):
            self.console.print(f"   Result: {json.dumps(result, indent=2)}", style=style)
        elif isinstance(result, str) and len(result) > 100:
            # Truncate long strings
            self.console.print(f"   Result: {result[:100]}...", style=style)
        else:
            self.console.print(f"   Result: {result}", style=style)
    
    def reset_counter(self) -> None:
        """Reset the action step counter."""
        self.step_counter = 0
    
    def print(self, *args, **kwargs) -> None:
        """Direct access to the underlying console print method."""
        self.console.print(*args, **kwargs)


# Global logger instance
# Import this in other modules: from utils.logging import logger
logger = AgentLogger()


if __name__ == "__main__":
    # Demo the logger
    logger.section("Agent Logger Demo", "Showing all available log types")
    
    logger.info("Starting agent execution...")
    
    logger.thought("I need to find the weather in San Francisco")
    
    logger.action(
        "Calling search tool",
        tool_name="search_web",
        arguments={"query": "weather San Francisco"}
    )
    
    logger.observation(
        "Search completed successfully",
        result={"temperature": "68°F", "conditions": "Sunny"}
    )
    
    logger.thought("I now have the information needed to answer the question")
    
    logger.final_answer("The weather in San Francisco is 68°F and sunny.")
    
    logger.success("Agent execution completed successfully!")
    
    logger.token_usage(
        prompt_tokens=150,
        completion_tokens=75,
        total_tokens=225,
        estimated_cost=0.0045
    )
    
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\nLogger demo complete!")

