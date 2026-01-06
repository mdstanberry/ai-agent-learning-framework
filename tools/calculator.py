"""
Calculator Tool

This module provides a calculator tool that agents can use to perform
mathematical operations safely.

Features:
- Basic arithmetic operations
- Mathematical functions (sqrt, pow, sin, cos, etc.)
- Safe evaluation of mathematical expressions
- Error handling for invalid expressions
"""

from typing import Union, Dict, Any, Optional
import math
import re
from utils.agent_logging import logger
from utils.config import config
from tools.registry import tool_registry, ToolSafetyLevel, ToolParameter


# Configuration
PRECISION = config.get_nested("tools.calculator.precision", 10)


def calculate(expression: str) -> Dict[str, Any]:
    """
    Calculate the result of a mathematical expression.
    
    Supports:
    - Basic arithmetic: +, -, *, /, %
    - Exponentiation: ** or pow()
    - Functions: sqrt, sin, cos, tan, log, exp, abs, round, etc.
    - Constants: pi, e
    - Parentheses for grouping
    
    Safety: Only allows mathematical operations, no code execution.
    
    Args:
        expression: Mathematical expression as a string
        
    Returns:
        Dictionary containing:
        - result: The calculated result
        - expression: The original expression
        - formatted_result: Result formatted to specified precision
        
    Raises:
        ValueError: If expression is invalid or unsafe
        
    Example:
        >>> calculate("2 + 2")
        {'result': 4.0, 'expression': '2 + 2', 'formatted_result': '4.0'}
        
        >>> calculate("sqrt(16)")
        {'result': 4.0, 'expression': 'sqrt(16)', 'formatted_result': '4.0'}
    """
    logger.info(f"Calculating: {expression}")
    
    # Safety check: Validate expression contains only safe characters
    if not _is_safe_expression(expression):
        raise ValueError(
            "Expression contains unsafe characters. "
            "Only mathematical operations are allowed."
        )
    
    try:
        # Replace math constants
        expression_normalized = _normalize_expression(expression)
        
        # Evaluate safely
        result = _safe_eval(expression_normalized)
        
        # Format result
        if isinstance(result, float):
            formatted_result = f"{result:.{PRECISION}f}".rstrip('0').rstrip('.')
        else:
            formatted_result = str(result)
        
        logger.observation(f"Calculation result: {formatted_result}")
        
        return {
            "result": float(result) if isinstance(result, (int, float)) else result,
            "expression": expression,
            "formatted_result": formatted_result
        }
        
    except Exception as e:
        error_msg = f"Calculation failed: {str(e)}"
        logger.error(error_msg, exception=e)
        raise ValueError(error_msg)


def add(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    result = float(a) + float(b)
    logger.info(f"Addition: {a} + {b} = {result}")
    return result


def subtract(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Subtract b from a.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Difference of a and b
    """
    result = float(a) - float(b)
    logger.info(f"Subtraction: {a} - {b} = {result}")
    return result


def multiply(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of a and b
    """
    result = float(a) * float(b)
    logger.info(f"Multiplication: {a} * {b} = {result}")
    return result


def divide(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Divide a by b.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Quotient of a and b
        
    Raises:
        ValueError: If b is zero
    """
    if float(b) == 0:
        raise ValueError("Division by zero is not allowed")
    
    result = float(a) / float(b)
    logger.info(f"Division: {a} / {b} = {result}")
    return result


def power(base: Union[int, float], exponent: Union[int, float]) -> float:
    """
    Raise base to the power of exponent.
    
    Args:
        base: Base number
        exponent: Exponent
        
    Returns:
        base raised to the power of exponent
    """
    result = math.pow(float(base), float(exponent))
    logger.info(f"Power: {base} ** {exponent} = {result}")
    return result


def sqrt(number: Union[int, float]) -> float:
    """
    Calculate square root.
    
    Args:
        number: Number to take square root of
        
    Returns:
        Square root of number
        
    Raises:
        ValueError: If number is negative
    """
    if float(number) < 0:
        raise ValueError("Cannot calculate square root of negative number")
    
    result = math.sqrt(float(number))
    logger.info(f"Square root: sqrt({number}) = {result}")
    return result


def _is_safe_expression(expression: str) -> bool:
    """
    Check if expression contains only safe mathematical characters.
    
    Args:
        expression: Expression to check
        
    Returns:
        True if expression is safe
    """
    # Allow: numbers, operators, parentheses, math functions, spaces, decimal points
    safe_pattern = re.compile(r'^[0-9+\-*/().\s,abcdefghijklmnopqrstuvwxyz_]+$', re.IGNORECASE)
    
    if not safe_pattern.match(expression):
        return False
    
    # Additional check: no double underscores (prevents __builtins__ access)
    if '__' in expression:
        return False
    
    # Check for dangerous function names
    dangerous = ['exec', 'eval', 'compile', 'open', 'file', 'input', 'raw_input']
    expression_lower = expression.lower()
    for danger in dangerous:
        if danger in expression_lower:
            return False
    
    return True


def _normalize_expression(expression: str) -> str:
    """
    Normalize expression by replacing constants and function names.
    
    Args:
        expression: Original expression
        
    Returns:
        Normalized expression
    """
    # Replace constants
    expression = expression.replace('pi', str(math.pi))
    expression = expression.replace('e', str(math.e))
    
    # Replace ** with pow for consistency
    expression = expression.replace('**', '^')
    
    return expression


def _safe_eval(expression: str) -> Union[int, float]:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression
        
    Returns:
        Result of evaluation
        
    Raises:
        ValueError: If expression cannot be evaluated
    """
    # Create a safe namespace with only math functions
    safe_dict = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "fabs": math.fabs,
        "factorial": math.factorial,
        "pi": math.pi,
        "e": math.e,
    }
    
    # Replace ^ with ** for Python
    expression = expression.replace('^', '**')
    
    try:
        result = eval(expression, safe_dict)
        
        # Validate result is a number
        if not isinstance(result, (int, float, complex)):
            raise ValueError("Expression did not evaluate to a number")
        
        # Convert complex to float (take real part)
        if isinstance(result, complex):
            result = result.real
        
        return float(result)
        
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except NameError as e:
        raise ValueError(f"Unknown function or variable: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Evaluation error: {str(e)}")


# Register tools with the registry
def register_calculator_tools():
    """Register calculator tools with the tool registry."""
    
    # Register main calculate function
    tool_registry.register(
        name="calculate",
        description="Calculate the result of a mathematical expression. Supports basic arithmetic, functions (sqrt, sin, cos, etc.), and constants (pi, e).",
        function=calculate,
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    # Register individual operations
    tool_registry.register(
        name="add",
        description="Add two numbers together",
        function=add,
        parameters=[
            ToolParameter(
                name="a",
                type="number",
                description="First number",
                required=True
            ),
            ToolParameter(
                name="b",
                type="number",
                description="Second number",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    tool_registry.register(
        name="subtract",
        description="Subtract second number from first number",
        function=subtract,
        parameters=[
            ToolParameter(
                name="a",
                type="number",
                description="First number",
                required=True
            ),
            ToolParameter(
                name="b",
                type="number",
                description="Second number",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    tool_registry.register(
        name="multiply",
        description="Multiply two numbers",
        function=multiply,
        parameters=[
            ToolParameter(
                name="a",
                type="number",
                description="First number",
                required=True
            ),
            ToolParameter(
                name="b",
                type="number",
                description="Second number",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    tool_registry.register(
        name="divide",
        description="Divide first number by second number",
        function=divide,
        parameters=[
            ToolParameter(
                name="a",
                type="number",
                description="Dividend",
                required=True
            ),
            ToolParameter(
                name="b",
                type="number",
                description="Divisor",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    tool_registry.register(
        name="power",
        description="Raise base to the power of exponent",
        function=power,
        parameters=[
            ToolParameter(
                name="base",
                type="number",
                description="Base number",
                required=True
            ),
            ToolParameter(
                name="exponent",
                type="number",
                description="Exponent",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    tool_registry.register(
        name="sqrt",
        description="Calculate square root of a number",
        function=sqrt,
        parameters=[
            ToolParameter(
                name="number",
                type="number",
                description="Number to take square root of",
                required=True
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="math"
    )
    
    logger.info("Calculator tools registered")


# Auto-register tools when module is imported
register_calculator_tools()


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Calculator Tool Demo")
    print("=" * 60)
    
    # Test basic calculations
    print("\n[Test 1] Basic Arithmetic")
    try:
        result = calculate("2 + 2")
        print(f"2 + 2 = {result['formatted_result']}")
        
        result = calculate("10 * 5")
        print(f"10 * 5 = {result['formatted_result']}")
        
        result = calculate("100 / 4")
        print(f"100 / 4 = {result['formatted_result']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test functions
    print("\n[Test 2] Mathematical Functions")
    try:
        result = calculate("sqrt(16)")
        print(f"sqrt(16) = {result['formatted_result']}")
        
        result = calculate("pow(2, 3)")
        print(f"pow(2, 3) = {result['formatted_result']}")
        
        result = calculate("sin(pi/2)")
        print(f"sin(pi/2) = {result['formatted_result']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test individual operations
    print("\n[Test 3] Individual Operations")
    print(f"add(5, 3) = {add(5, 3)}")
    print(f"multiply(4, 7) = {multiply(4, 7)}")
    print(f"power(2, 8) = {power(2, 8)}")
    print(f"sqrt(144) = {sqrt(144)}")
    
    # Test error handling
    print("\n[Test 4] Error Handling")
    try:
        calculate("10 / 0")
    except ValueError as e:
        print(f"[OK] Division by zero caught: {e}")
    
    try:
        calculate("sqrt(-1)")
    except ValueError as e:
        print(f"[OK] Negative sqrt caught: {e}")
    
    try:
        calculate("__import__('os')")
    except ValueError as e:
        print(f"[OK] Unsafe expression rejected: {e}")
    
    # Test tool registry integration
    print("\n[Test 5] Tool Registry Integration")
    from tools.registry import tool_registry
    from utils.schemas import ToolCall
    
    tool_call = ToolCall(
        id="test_1",
        name="calculate",
        arguments={"expression": "2 ** 10"}
    )
    
    result = tool_registry.execute_tool(tool_call)
    if result.is_success:
        print(f"Tool executed successfully!")
        print(f"Result: {result.result['formatted_result']}")
    else:
        print(f"Tool execution failed: {result.error}")
    
    print("\nCalculator tool demo complete!")



