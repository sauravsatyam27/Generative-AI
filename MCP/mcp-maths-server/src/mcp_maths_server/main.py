from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")


# -------------------------
# Calculator Tools
# -------------------------

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@mcp.tool
def power(a: float, b: float) -> float:
    """Calculate a raised to the power b."""
    return a ** b


@mcp.tool
def modulo(a: int, b: int) -> int:
    """Return the remainder when a is divided by b."""
    if b == 0:
        raise ValueError("Cannot modulo by zero.")
    return a % b


# -------------------------
# String Tools
# -------------------------

@mcp.tool
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@mcp.tool
def count_characters(text: str) -> int:
    """Count the number of characters in a string."""
    return len(text)


@mcp.tool
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@mcp.tool
def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


@mcp.tool
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())


# -------------------------
# List Tools
# -------------------------

@mcp.tool
def find_max(numbers: list[int]) -> int:
    """Find the maximum number from a list."""
    if not numbers:
        raise ValueError("List cannot be empty.")
    return max(numbers)


@mcp.tool
def find_min(numbers: list[int]) -> int:
    """Find the minimum number from a list."""
    if not numbers:
        raise ValueError("List cannot be empty.")
    return min(numbers)


@mcp.tool
def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of all numbers in a list."""
    return sum(numbers)


# -------------------------
# Utility Tools
# -------------------------

@mcp.tool
def is_even(number: int) -> bool:
    """Check whether a number is even."""
    return number % 2 == 0


@mcp.tool
def is_prime(number: int) -> bool:
    """Check whether a number is prime."""
    if number < 2:
        return False

    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False

    return True


@mcp.tool
def factorial(number: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    if number < 0:
        raise ValueError("Factorial is not defined for negative numbers.")

    result = 1

    for i in range(1, number + 1):
        result *= i

    return result


# -------------------------
# Server
# -------------------------

if __name__ == "__main__":
    mcp.run()