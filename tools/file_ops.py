"""
File Operations Tool

This module provides file reading capabilities for AI agents with safety checks.
In a production system, you'd want additional security measures.

Safety Features:
- File size limits
- Allowed file extensions
- Path validation (prevent directory traversal)
- Read-only operations (no write/delete)
"""

from typing import Optional, Dict, Any
from pathlib import Path
import os
from utils.agent_logging import logger
from utils.config import config
from tools.registry import tool_registry, ToolSafetyLevel, ToolParameter


# Configuration from config.yaml
ALLOWED_EXTENSIONS = config.get_nested(
    "tools.file_ops.allowed_extensions",
    [".txt", ".md", ".json", ".py"]
)
MAX_FILE_SIZE_MB = config.get_nested(
    "tools.file_ops.max_file_size_mb",
    10
)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def read_file(file_path: str, max_lines: Optional[int] = None) -> Dict[str, Any]:
    """
    Read a file safely with various safety checks.
    
    Safety checks:
    - Validates file extension is allowed
    - Checks file size doesn't exceed limit
    - Validates path (prevents directory traversal)
    - Only reads files (no write/delete operations)
    
    Args:
        file_path: Path to the file to read (relative to project root)
        max_lines: Optional maximum number of lines to read (for large files)
        
    Returns:
        Dictionary containing:
        - content: File content (or first max_lines if specified)
        - file_path: Path that was read
        - size_bytes: File size in bytes
        - total_lines: Total lines in file
        - lines_read: Number of lines actually read
        - encoding: File encoding detected
        
    Raises:
        ValueError: If file doesn't meet safety requirements
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
    """
    logger.info(f"Reading file: {file_path}")
    
    # Safety check 1: Validate path (prevent directory traversal)
    normalized_path = _normalize_path(file_path)
    if normalized_path is None:
        raise ValueError(f"Invalid file path: {file_path}")
    
    file_path_obj = Path(normalized_path)
    
    # Safety check 2: Check file extension
    if not _is_allowed_extension(file_path_obj):
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise ValueError(
            f"File extension not allowed. Allowed extensions: {allowed}"
        )
    
    # Safety check 3: Check if file exists
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Safety check 4: Check if it's a file (not a directory)
    if not file_path_obj.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    # Safety check 5: Check file size
    file_size = file_path_obj.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.2f} MB "
            f"(max: {MAX_FILE_SIZE_MB} MB)"
        )
    
    logger.observation(f"File size: {file_size / 1024:.2f} KB")
    
    # Read file content
    try:
        # Try UTF-8 first
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                    total_lines = sum(1 for _ in open(file_path_obj, 'r', encoding='utf-8'))
                    lines_read = len(lines)
                else:
                    content = f.read()
                    lines_read = total_lines = len(content.splitlines())
                encoding = 'utf-8'
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            logger.warning("UTF-8 decoding failed, trying latin-1")
            with open(file_path_obj, 'r', encoding='latin-1') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                    total_lines = sum(1 for _ in open(file_path_obj, 'r', encoding='latin-1'))
                    lines_read = len(lines)
                else:
                    content = f.read()
                    lines_read = total_lines = len(content.splitlines())
                encoding = 'latin-1'
        
        logger.success(f"File read successfully: {lines_read} lines")
        
        return {
            "content": content,
            "file_path": str(file_path_obj),
            "size_bytes": file_size,
            "total_lines": total_lines,
            "lines_read": lines_read,
            "encoding": encoding,
            "truncated": max_lines is not None and lines_read < total_lines
        }
        
    except PermissionError as e:
        logger.error(f"Permission denied reading file: {file_path}", exception=e)
        raise PermissionError(f"Cannot read file (permission denied): {file_path}")
    except Exception as e:
        logger.error(f"Error reading file: {file_path}", exception=e)
        raise


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file without reading its contents.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file metadata:
        - exists: Whether file exists
        - size_bytes: File size
        - extension: File extension
        - is_allowed: Whether extension is allowed
        - is_readable: Whether file can be read
    """
    logger.info(f"Getting file info: {file_path}")
    
    normalized_path = _normalize_path(file_path)
    if normalized_path is None:
        return {
            "exists": False,
            "error": "Invalid file path"
        }
    
    file_path_obj = Path(normalized_path)
    
    info = {
        "exists": file_path_obj.exists(),
        "extension": file_path_obj.suffix,
        "is_allowed": _is_allowed_extension(file_path_obj),
        "is_readable": False
    }
    
    if file_path_obj.exists() and file_path_obj.is_file():
        try:
            info["size_bytes"] = file_path_obj.stat().st_size
            info["size_mb"] = info["size_bytes"] / 1024 / 1024
            info["is_readable"] = os.access(file_path_obj, os.R_OK)
        except Exception as e:
            info["error"] = str(e)
    
    return info


def _normalize_path(file_path: str) -> Optional[str]:
    """
    Normalize and validate file path to prevent directory traversal attacks.
    
    Args:
        file_path: Input file path
        
    Returns:
        Normalized path or None if invalid
    """
    try:
        # Convert to Path object
        path = Path(file_path)
        
        # Resolve to absolute path
        if not path.is_absolute():
            # If relative, resolve from current working directory
            path = Path.cwd() / path
        
        # Normalize the path (resolve .. and .)
        normalized = path.resolve()
        
        # Ensure it's within the project directory (basic check)
        # In production, you'd want stricter path validation
        cwd = Path.cwd()
        
        # Check if normalized path is within current directory
        try:
            normalized.relative_to(cwd)
            return str(normalized)
        except ValueError:
            # Path is outside current directory - reject for safety
            logger.warning(f"Path outside project directory rejected: {file_path}")
            return None
            
    except Exception as e:
        logger.warning(f"Path normalization failed: {e}")
        return None


def _is_allowed_extension(file_path: Path) -> bool:
    """
    Check if file extension is in the allowed list.
    
    Args:
        file_path: Path object
        
    Returns:
        True if extension is allowed
    """
    extension = file_path.suffix.lower()
    return extension in [ext.lower() for ext in ALLOWED_EXTENSIONS]


# Register tools with the registry
def register_file_tools():
    """Register file operation tools with the tool registry."""
    
    tool_registry.register(
        name="read_file",
        description="Read a file safely. Supports text files (.txt, .md, .json, .py). Has size limits and safety checks.",
        function=read_file,
        parameters=[
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to read (relative to project root)",
                required=True
            ),
            ToolParameter(
                name="max_lines",
                type="number",
                description="Optional maximum number of lines to read (for large files)",
                required=False
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="file_operations"
    )
    
    tool_registry.register(
        name="get_file_info",
        description="Get information about a file without reading its contents (size, extension, etc.).",
        function=get_file_info,
        parameters=[
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="file_operations"
    )
    
    logger.info("File operation tools registered")


# Auto-register tools when module is imported
register_file_tools()


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("File Operations Tool Demo")
    print("=" * 60)
    
    # Test file info
    print("\n[Test 1] Get File Info")
    info = get_file_info("requirements.txt")
    print(f"File exists: {info.get('exists')}")
    if info.get('exists'):
        print(f"Size: {info.get('size_bytes')} bytes")
        print(f"Extension: {info.get('extension')}")
        print(f"Allowed: {info.get('is_allowed')}")
    
    # Test reading a file
    print("\n[Test 2] Read File")
    try:
        result = read_file("requirements.txt", max_lines=5)
        print(f"Read {result['lines_read']} lines")
        print(f"Content preview:")
        print(result['content'][:200] + "...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test safety checks
    print("\n[Test 3] Safety Checks")
    
    # Test invalid extension
    try:
        read_file("test.exe")
    except ValueError as e:
        print(f"[OK] Invalid extension rejected: {e}")
    
    # Test non-existent file
    try:
        read_file("nonexistent_file.txt")
    except FileNotFoundError as e:
        print(f"[OK] Non-existent file rejected: {e}")
    
    # Test tool registry integration
    print("\n[Test 4] Tool Registry Integration")
    from tools.registry import tool_registry
    from utils.schemas import ToolCall
    
    tool_call = ToolCall(
        id="test_1",
        name="read_file",
        arguments={"file_path": "requirements.txt", "max_lines": 3}
    )
    
    result = tool_registry.execute_tool(tool_call)
    if result.is_success:
        print(f"Tool executed successfully!")
        print(f"Lines read: {result.result.get('lines_read')}")
    else:
        print(f"Tool execution failed: {result.error}")
    
    print("\nFile operations tool demo complete!")



