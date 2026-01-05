"""
Safety Wrapper Module

This module provides safety wrappers for destructive or high-risk tool operations.
It implements human-in-the-loop confirmation for operations that could cause
permanent changes or data loss.

Features:
- Confirmation prompts for destructive operations
- Safety level validation
- Operation logging and audit trail
- Configurable confirmation requirements
"""

from typing import Callable, Any, Optional, Dict
from functools import wraps
from enum import Enum
from utils.agent_logging import logger
from tools.registry import ToolSafetyLevel


class ConfirmationResult(Enum):
    """Result of a confirmation request."""
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class SafetyWrapper:
    """
    Safety wrapper for tool operations that require confirmation.
    
    This class provides a decorator pattern for wrapping functions
    that perform destructive or high-risk operations.
    """
    
    def __init__(self, confirmation_handler: Optional[Callable] = None):
        """
        Initialize the safety wrapper.
        
        Args:
            confirmation_handler: Optional function to handle confirmations.
                                 If None, uses default prompt-based handler.
        """
        self.confirmation_handler = confirmation_handler or self._default_confirmation_handler
        self.audit_log: list[Dict[str, Any]] = []
    
    def require_confirmation(
        self,
        operation_name: str,
        description: str,
        safety_level: ToolSafetyLevel = ToolSafetyLevel.DESTRUCTIVE
    ) -> bool:
        """
        Request confirmation for a potentially destructive operation.
        
        Args:
            operation_name: Name of the operation
            description: Description of what will happen
            safety_level: Safety level of the operation
            
        Returns:
            True if confirmed, False if rejected
        """
        logger.warning(
            f"CONFIRMATION REQUIRED for {operation_name}: {description}"
        )
        
        result = self.confirmation_handler(operation_name, description, safety_level)
        
        # Log the confirmation attempt
        self.audit_log.append({
            "operation": operation_name,
            "description": description,
            "safety_level": safety_level.value,
            "result": result.value if isinstance(result, ConfirmationResult) else ("approved" if result else "rejected"),
            "timestamp": logger.__class__.__module__  # Placeholder - would use actual timestamp
        })
        
        if result == ConfirmationResult.APPROVED or (isinstance(result, bool) and result):
            logger.success(f"Operation '{operation_name}' confirmed")
            return True
        else:
            logger.warning(f"Operation '{operation_name}' rejected or cancelled")
            return False
    
    def _default_confirmation_handler(
        self,
        operation_name: str,
        description: str,
        safety_level: ToolSafetyLevel
    ) -> ConfirmationResult:
        """
        Default confirmation handler that prompts for user input.
        
        In a real system, this would integrate with the UI or API
        to get actual user confirmation. For now, it's a mock.
        
        Args:
            operation_name: Name of the operation
            description: Description of what will happen
            safety_level: Safety level
            
        Returns:
            ConfirmationResult
        """
        # In a real implementation, this would:
        # 1. Display a confirmation prompt to the user
        # 2. Wait for user response
        # 3. Return APPROVED or REJECTED
        
        # For now, we'll log and return REJECTED by default for safety
        logger.warning(
            f"Default confirmation handler: Operation '{operation_name}' "
            f"requires manual confirmation. Auto-rejecting for safety."
        )
        logger.info(
            f"To approve: Call confirm_operation('{operation_name}') "
            f"or implement custom confirmation handler"
        )
        
        return ConfirmationResult.REJECTED
    
    def wrap_destructive(
        self,
        safety_level: ToolSafetyLevel = ToolSafetyLevel.DESTRUCTIVE,
        description: Optional[str] = None
    ):
        """
        Decorator to wrap a function with safety checks.
        
        Args:
            safety_level: Safety level of the operation
            description: Optional description of what the operation does
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate operation description
                op_description = description or f"{func.__name__}({', '.join(map(str, args))})"
                
                # Check if confirmation is required
                if safety_level == ToolSafetyLevel.DESTRUCTIVE:
                    if not self.require_confirmation(
                        func.__name__,
                        op_description,
                        safety_level
                    ):
                        raise PermissionError(
                            f"Operation '{func.__name__}' requires confirmation and was rejected"
                        )
                
                # Execute the function
                try:
                    logger.action(f"Executing {safety_level.value} operation: {func.__name__}")
                    result = func(*args, **kwargs)
                    logger.success(f"Operation '{func.__name__}' completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Operation '{func.__name__}' failed: {e}", exception=e)
                    raise
            
            return wrapper
        return decorator
    
    def get_audit_log(self) -> list[Dict[str, Any]]:
        """
        Get the audit log of all confirmation attempts.
        
        Returns:
            List of audit log entries
        """
        return self.audit_log.copy()


# Global safety wrapper instance
# Import this in other modules: from tools.safety import safety_wrapper
safety_wrapper = SafetyWrapper()


def require_confirmation_for_tool_call(
    tool_call,
    confirmation_handler: Optional[Callable] = None
) -> bool:
    """
    Helper function to check if a tool call requires confirmation.
    
    This is used by the tool registry when executing tools.
    
    Args:
        tool_call: ToolCall object
        confirmation_handler: Optional custom confirmation handler
        
    Returns:
        True if confirmed, False if rejected
    """
    from tools.registry import tool_registry
    
    tool = tool_registry.get_tool(tool_call.name)
    
    if not tool:
        return False
    
    if tool.requires_confirmation:
        wrapper = SafetyWrapper(confirmation_handler)
        return wrapper.require_confirmation(
            tool_call.name,
            f"Tool call: {tool_call.name} with arguments: {tool_call.arguments}",
            tool.safety_level
        )
    
    return True


# Example: Destructive tool that uses the safety wrapper
def delete_file_example(file_path: str) -> Dict[str, Any]:
    """
    Example of a destructive operation that would require confirmation.
    
    This is just an example - actual file deletion should use proper
    safety wrappers and confirmation.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        Result dictionary
    """
    # In a real implementation, this would actually delete the file
    # For now, it's just a demonstration
    logger.warning(f"Would delete file: {file_path} (example only)")
    return {
        "status": "would_delete",
        "file_path": file_path,
        "message": "This is an example - file not actually deleted"
    }


# Wrap the example with safety decorator
@safety_wrapper.wrap_destructive(
    safety_level=ToolSafetyLevel.DESTRUCTIVE,
    description="Delete a file permanently"
)
def safe_delete_file(file_path: str) -> Dict[str, Any]:
    """
    Safe version of delete_file that requires confirmation.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        Result dictionary
    """
    return delete_file_example(file_path)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Safety Wrapper Demo")
    print("=" * 60)
    
    # Test confirmation requirement
    print("\n[Test 1] Confirmation Requirement")
    result = safety_wrapper.require_confirmation(
        "delete_file",
        "Delete file: important_data.txt",
        ToolSafetyLevel.DESTRUCTIVE
    )
    print(f"Confirmation result: {result}")
    
    # Test wrapped function
    print("\n[Test 2] Wrapped Destructive Function")
    try:
        # This will require confirmation and be rejected by default
        safe_delete_file("test_file.txt")
    except PermissionError as e:
        print(f"[OK] Operation correctly rejected: {e}")
    
    # Test audit log
    print("\n[Test 3] Audit Log")
    audit_log = safety_wrapper.get_audit_log()
    print(f"Audit log entries: {len(audit_log)}")
    for entry in audit_log:
        print(f"  - {entry['operation']}: {entry['result']}")
    
    print("\nSafety wrapper demo complete!")
    print("\nNote: In a real system, implement a custom confirmation_handler")
    print("that integrates with your UI or API to get actual user confirmation.")


