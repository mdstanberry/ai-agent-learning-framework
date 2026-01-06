"""
Tool Registry Module

This module provides a centralized registry for managing tools that AI agents
can use. Tools are functions that agents can call to interact with the world
(e.g., search the web, read files, perform calculations).

Key Features:
- Tool registration and management
- Conversion to LLM tool formats (OpenAI, Anthropic)
- Tool execution with error handling
- Safety wrappers for destructive operations
- Tool discovery and listing

How to Add Custom Tools:
-------------------------

1. Define your tool function:
   
   def my_custom_tool(param1: str, param2: int) -> dict:
       \"\"\"Description of what your tool does.\"\"\"
       # Your tool logic here
       return {"result": "success"}

2. Register the tool:
   
   from tools.registry import tool_registry, ToolSafetyLevel, ToolParameter
   
   tool_registry.register(
       name="my_custom_tool",
       description="What your tool does (used by LLM to decide when to call it)",
       function=my_custom_tool,
       parameters=[
           ToolParameter(
               name="param1",
               type="string",
               description="What param1 does",
               required=True
           ),
           ToolParameter(
               name="param2",
               type="number",
               description="What param2 does",
               required=True
           )
       ],
       safety_level=ToolSafetyLevel.SAFE,  # or MODERATE or DESTRUCTIVE
       requires_confirmation=False,  # Set True for destructive operations
       category="my_category"  # Optional category for grouping
   )

3. For destructive operations, use the safety wrapper:
   
   from tools.safety import safety_wrapper, ToolSafetyLevel
   
   @safety_wrapper.wrap_destructive(
       safety_level=ToolSafetyLevel.DESTRUCTIVE,
       description="What this operation does"
   )
   def my_destructive_tool():
       # Your destructive operation
       pass

4. The tool will automatically be available to agents via the registry.
   Agents can discover and call your tool using its name.

Example:
--------
See the __main__ block at the bottom of this file for a complete example.
"""

from typing import Dict, List, Callable, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum
import json
import inspect
from utils.schemas import ToolDefinition, ToolParameter, ToolCall, ToolResult
from utils.agent_logging import logger


class ToolSafetyLevel(Enum):
    """Safety levels for tools."""
    SAFE = "safe"  # Read-only, no side effects
    MODERATE = "moderate"  # Some side effects, reversible
    DESTRUCTIVE = "destructive"  # Permanent changes, requires confirmation


@dataclass
class RegisteredTool:
    """
    Represents a registered tool with its metadata.
    """
    name: str
    description: str
    function: Callable
    parameters: List[ToolParameter]
    safety_level: ToolSafetyLevel
    requires_confirmation: bool = False
    category: Optional[str] = None


class ToolRegistry:
    """
    Central registry for managing tools that agents can use.
    
    This class handles:
    - Registering tools with their schemas
    - Converting tools to LLM formats
    - Executing tools safely
    - Managing tool metadata
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, RegisteredTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[List[ToolParameter]] = None,
        safety_level: ToolSafetyLevel = ToolSafetyLevel.SAFE,
        requires_confirmation: bool = False,
        category: Optional[str] = None
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Tool name (should be snake_case)
            description: What the tool does
            function: The function to call
            parameters: List of tool parameters (auto-detected if None)
            safety_level: Safety level of the tool
            requires_confirmation: Whether tool requires human confirmation
            category: Optional category for grouping tools
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered. Overwriting.")
        
        # Auto-detect parameters if not provided
        if parameters is None:
            parameters = self._extract_parameters(function)
        
        tool = RegisteredTool(
            name=name,
            description=description,
            function=function,
            parameters=parameters,
            safety_level=safety_level,
            requires_confirmation=requires_confirmation or safety_level == ToolSafetyLevel.DESTRUCTIVE,
            category=category
        )
        
        self._tools[name] = tool
        
        # Add to category
        if category:
            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)
        
        logger.info(f"Tool '{name}' registered successfully")
    
    def _extract_parameters(self, function: Callable) -> List[ToolParameter]:
        """
        Extract parameters from a function signature.
        
        Args:
            function: Function to analyze
            
        Returns:
            List of tool parameters
        """
        sig = inspect.signature(function)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' and other special parameters
            if param_name == 'self':
                continue
            
            # Determine parameter type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_str = str(param.annotation)
                if "int" in type_str or "float" in type_str:
                    param_type = "number"
                elif "bool" in type_str:
                    param_type = "boolean"
                elif "list" in type_str.lower() or "List" in type_str:
                    param_type = "array"
            
            # Check if required
            required = param.default == inspect.Parameter.empty
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter {param_name}",
                required=required
            ))
        
        return parameters
    
    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """
        Get a registered tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Registered tool or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered tool names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def get_tool_definitions(
        self,
        format: str = "openai"
    ) -> List[Dict[str, Any]]:
        """
        Get tool definitions in LLM format.
        
        Args:
            format: Format to use ("openai" or "anthropic")
            
        Returns:
            List of tool definitions in the requested format
        """
        definitions = []
        
        for tool in self._tools.values():
            tool_def = ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters
            )
            
            if format == "openai":
                definitions.append(tool_def.to_openai_format())
            elif format == "anthropic":
                definitions.append(tool_def.to_anthropic_format())
            else:
                raise ValueError(f"Unknown format: {format}")
        
        return definitions
    
    def execute_tool(
        self,
        tool_call: ToolCall,
        require_confirmation: Optional[Callable] = None
    ) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_call: The tool call to execute
            require_confirmation: Optional function to request confirmation
            
        Returns:
            Tool result with output or error
        """
        import time
        start_time = time.time()
        
        tool = self._tools.get(tool_call.name)
        
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=f"Tool '{tool_call.name}' not found",
                execution_time=time.time() - start_time
            )
        
        logger.action(
            f"Executing tool: {tool_call.name}",
            tool_name=tool_call.name,
            arguments=tool_call.arguments
        )
        
        # Check if confirmation is required
        if tool.requires_confirmation:
            logger.warning(f"Tool '{tool_call.name}' requires confirmation")
            
            # Use provided confirmation handler or default safety wrapper
            if require_confirmation:
                confirmed = require_confirmation(tool_call)
            else:
                # Use safety wrapper for confirmation
                from tools.safety import require_confirmation_for_tool_call
                confirmed = require_confirmation_for_tool_call(tool_call)
            
            if not confirmed:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=None,
                    error="Tool execution cancelled - confirmation required but not provided",
                    execution_time=time.time() - start_time
                )
        
        try:
            # Execute the tool function
            result = tool.function(**tool_call.arguments)
            
            execution_time = time.time() - start_time
            
            logger.observation(
                f"Tool '{tool_call.name}' executed successfully",
                result=result
            )
            
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                error=None,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution failed: {str(e)}"
            
            logger.error(f"Tool '{tool_call.name}' failed: {error_msg}", exception=e)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Dictionary with tool information
        """
        tool = self._tools.get(name)
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "safety_level": tool.safety_level.value,
            "requires_confirmation": tool.requires_confirmation,
            "category": tool.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required
                }
                for p in tool.parameters
            ]
        }
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            tool = self._tools.pop(name)
            
            # Remove from category
            if tool.category and tool.category in self._categories:
                if name in self._categories[tool.category]:
                    self._categories[tool.category].remove(name)
            
            logger.info(f"Tool '{name}' unregistered")
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()
        logger.info("Tool registry cleared")


# Global tool registry instance
# Import this in other modules: from tools.registry import tool_registry
tool_registry = ToolRegistry()


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Tool Registry Demo")
    print("=" * 60)
    
    # Example tool function
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    # Register a tool
    tool_registry.register(
        name="add_numbers",
        description="Add two numbers together",
        function=add_numbers,
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    print(f"\nRegistered tools: {tool_registry.list_tools()}")
    
    # Get tool definitions
    openai_tools = tool_registry.get_tool_definitions(format="openai")
    print(f"\nOpenAI format tools: {len(openai_tools)}")
    
    # Get tool info
    info = tool_registry.get_tool_info("add_numbers")
    print(f"\nTool info: {info}")
    
    # Execute a tool
    tool_call = ToolCall(
        id="call_1",
        name="add_numbers",
        arguments={"a": 5, "b": 3}
    )
    
    result = tool_registry.execute_tool(tool_call)
    print(f"\nTool execution result: {result.result}")
    print(f"Execution time: {result.execution_time:.3f}s")
    
    print("\nTool registry demo complete!")


