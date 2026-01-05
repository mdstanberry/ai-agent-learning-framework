"""
Golden Test Sets

This module contains golden test sets for all five design patterns.
Golden test sets are fixed, diverse scenarios that never change and are used
to track agent performance over time to detect drift.

Each pattern has 5 test cases covering different scenarios:
- Simple cases
- Complex cases
- Edge cases
- Different domains/topics
- Various difficulty levels

These test sets should NEVER be modified once established, as they serve
as a baseline for performance tracking.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class PatternType(str, Enum):
    """Enumeration of pattern types."""
    CHAINING = "chaining"
    ROUTING = "routing"
    PARALLELIZATION = "parallelization"
    ORCHESTRATOR = "orchestrator"
    EVALUATOR = "evaluator"


@dataclass
class GoldenTestCase:
    """
    Represents a single golden test case.
    
    Attributes:
        pattern: The pattern type this test case is for
        test_id: Unique identifier for this test case
        input: Input data for the test
        expected_output: Expected output (can be partial or criteria-based)
        description: Description of what this test case validates
        difficulty: Difficulty level (easy, medium, hard)
        domain: Domain/topic area (e.g., "technical", "creative", "business")
    """
    pattern: PatternType
    test_id: str
    input: Any
    expected_output: Any
    description: str
    difficulty: str = "medium"
    domain: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Prompt Chaining Golden Test Cases
# =============================================================================

CHAINING_GOLDEN_TESTS: List[GoldenTestCase] = [
    GoldenTestCase(
        pattern=PatternType.CHAINING,
        test_id="chaining_001",
        input="Python Programming Basics",
        expected_output={
            "has_outline": True,
            "min_sections": 3,
            "has_blog_post": True,
            "has_edited_post": True,
            "quality_indicators": ["clear_structure", "coherent_content"]
        },
        description="Simple technical topic - should generate outline, blog post, and edited version",
        difficulty="easy",
        domain="technical"
    ),
    GoldenTestCase(
        pattern=PatternType.CHAINING,
        test_id="chaining_002",
        input="The Future of Artificial Intelligence in Healthcare",
        expected_output={
            "has_outline": True,
            "min_sections": 4,
            "has_blog_post": True,
            "has_edited_post": True,
            "quality_indicators": ["comprehensive_coverage", "well_structured"]
        },
        description="Complex topic requiring multiple sections - tests outline quality",
        difficulty="hard",
        domain="technical"
    ),
    GoldenTestCase(
        pattern=PatternType.CHAINING,
        test_id="chaining_003",
        input="How to Bake Chocolate Chip Cookies",
        expected_output={
            "has_outline": True,
            "min_sections": 3,
            "has_blog_post": True,
            "has_edited_post": True,
            "quality_indicators": ["step_by_step", "practical"]
        },
        description="Creative/practical topic - tests pattern with non-technical content",
        difficulty="easy",
        domain="creative"
    ),
    GoldenTestCase(
        pattern=PatternType.CHAINING,
        test_id="chaining_004",
        input="A",
        expected_output={
            "has_outline": True,
            "min_sections": 2,  # Should handle minimal input gracefully
            "has_blog_post": True,
            "has_edited_post": True,
            "quality_indicators": ["handles_edge_case"]
        },
        description="Edge case - minimal input (single character) - tests robustness",
        difficulty="hard",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.CHAINING,
        test_id="chaining_005",
        input="Sustainable Business Practices: A Comprehensive Guide to Environmental Responsibility in Modern Corporations",
        expected_output={
            "has_outline": True,
            "min_sections": 5,
            "has_blog_post": True,
            "has_edited_post": True,
            "quality_indicators": ["comprehensive", "well_organized"]
        },
        description="Long, complex business topic - tests handling of verbose input",
        difficulty="hard",
        domain="business"
    )
]


# =============================================================================
# Routing Golden Test Cases
# =============================================================================

ROUTING_GOLDEN_TESTS: List[GoldenTestCase] = [
    GoldenTestCase(
        pattern=PatternType.ROUTING,
        test_id="routing_001",
        input="How do I reset my password?",
        expected_output={
            "category": "Tech Support",
            "confidence_threshold": 0.7,
            "has_response": True
        },
        description="Clear tech support question - should route to Tech Support handler",
        difficulty="easy",
        domain="technical"
    ),
    GoldenTestCase(
        pattern=PatternType.ROUTING,
        test_id="routing_002",
        input="What are your pricing plans and do you offer discounts for startups?",
        expected_output={
            "category": "Sales",
            "confidence_threshold": 0.7,
            "has_response": True
        },
        description="Sales inquiry - should route to Sales handler",
        difficulty="easy",
        domain="business"
    ),
    GoldenTestCase(
        pattern=PatternType.ROUTING,
        test_id="routing_003",
        input="I'm having trouble with my API returning 500 errors and also need to update my payment method",
        expected_output={
            "category": "Tech Support",  # Primary concern is tech issue
            "confidence_threshold": 0.6,  # Lower confidence due to mixed signals
            "has_response": True
        },
        description="Mixed query (tech + billing) - tests classification with multiple concerns",
        difficulty="hard",
        domain="mixed"
    ),
    GoldenTestCase(
        pattern=PatternType.ROUTING,
        test_id="routing_004",
        input="Hello",
        expected_output={
            "category": "General",  # Should fallback to general
            "confidence_threshold": 0.5,
            "has_response": True
        },
        description="Ambiguous greeting - tests fallback to General handler",
        difficulty="medium",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.ROUTING,
        test_id="routing_005",
        input="My subscription was charged twice this month and I need a refund. Also, can you explain your enterprise pricing?",
        expected_output={
            "category": "Billing",  # Primary concern is billing
            "confidence_threshold": 0.7,
            "has_response": True
        },
        description="Billing question with sales follow-up - should prioritize billing",
        difficulty="medium",
        domain="business"
    )
]


# =============================================================================
# Parallelization Golden Test Cases
# =============================================================================

PARALLELIZATION_GOLDEN_TESTS: List[GoldenTestCase] = [
    GoldenTestCase(
        pattern=PatternType.PARALLELIZATION,
        test_id="parallelization_001",
        input="This is a helpful article about Python programming.",
        expected_output={
            "num_votes": 3,
            "consensus": "safe",
            "all_votes_received": True
        },
        description="Clearly safe content - should get unanimous safe votes",
        difficulty="easy",
        domain="content_safety"
    ),
    GoldenTestCase(
        pattern=PatternType.PARALLELIZATION,
        test_id="parallelization_002",
        input="This content contains some potentially harmful information about security vulnerabilities.",
        expected_output={
            "num_votes": 3,
            "consensus": "unsafe",  # Or mixed, depending on severity
            "all_votes_received": True
        },
        description="Potentially unsafe content - tests voting on edge case",
        difficulty="hard",
        domain="content_safety"
    ),
    GoldenTestCase(
        pattern=PatternType.PARALLELIZATION,
        test_id="parallelization_003",
        input="A" * 1000,  # Very long input
        expected_output={
            "num_votes": 3,
            "all_votes_received": True,
            "handles_long_input": True
        },
        description="Very long input - tests handling of large content",
        difficulty="medium",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.PARALLELIZATION,
        test_id="parallelization_004",
        input="",
        expected_output={
            "num_votes": 3,
            "all_votes_received": True,
            "handles_empty_input": True
        },
        description="Empty input - tests edge case handling",
        difficulty="hard",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.PARALLELIZATION,
        test_id="parallelization_005",
        input="This is educational content about network security best practices for developers.",
        expected_output={
            "num_votes": 3,
            "consensus": "safe",
            "all_votes_received": True,
            "quality": "high"
        },
        description="Educational technical content - should be safe",
        difficulty="easy",
        domain="technical"
    )
]


# =============================================================================
# Orchestrator Golden Test Cases
# =============================================================================

ORCHESTRATOR_GOLDEN_TESTS: List[GoldenTestCase] = [
    GoldenTestCase(
        pattern=PatternType.ORCHESTRATOR,
        test_id="orchestrator_001",
        input="Create a simple Python script that prints 'Hello World'",
        expected_output={
            "has_breakdown": True,
            "num_tasks": 1,  # Simple task, minimal breakdown
            "has_result": True,
            "workers_used": ["coder"]
        },
        description="Simple task - should require minimal worker delegation",
        difficulty="easy",
        domain="technical"
    ),
    GoldenTestCase(
        pattern=PatternType.ORCHESTRATOR,
        test_id="orchestrator_002",
        input="Research the latest developments in quantum computing, write a comprehensive report, and create a presentation summarizing the findings",
        expected_output={
            "has_breakdown": True,
            "num_tasks": 3,  # Research, write, present
            "has_result": True,
            "workers_used": ["researcher", "writer", "analyst"]
        },
        description="Complex multi-step task - tests task breakdown and worker delegation",
        difficulty="hard",
        domain="research"
    ),
    GoldenTestCase(
        pattern=PatternType.ORCHESTRATOR,
        test_id="orchestrator_003",
        input="Build a web scraper that extracts product information from e-commerce sites",
        expected_output={
            "has_breakdown": True,
            "num_tasks": 2,  # Design + implement
            "has_result": True,
            "workers_used": ["researcher", "coder", "reviewer"]
        },
        description="Technical project - tests orchestrator with coding task",
        difficulty="medium",
        domain="technical"
    ),
    GoldenTestCase(
        pattern=PatternType.ORCHESTRATOR,
        test_id="orchestrator_004",
        input="x",
        expected_output={
            "has_breakdown": True,
            "has_result": True,
            "handles_minimal_input": True
        },
        description="Minimal input - tests robustness of task breakdown",
        difficulty="hard",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.ORCHESTRATOR,
        test_id="orchestrator_005",
        input="Analyze market trends for renewable energy, create a business plan, write marketing copy, and develop a financial model",
        expected_output={
            "has_breakdown": True,
            "num_tasks": 4,
            "has_result": True,
            "workers_used": ["researcher", "analyst", "writer"]
        },
        description="Complex business task requiring multiple specialists",
        difficulty="hard",
        domain="business"
    )
]


# =============================================================================
# Evaluator-Optimizer Golden Test Cases
# =============================================================================

EVALUATOR_GOLDEN_TESTS: List[GoldenTestCase] = [
    GoldenTestCase(
        pattern=PatternType.EVALUATOR,
        test_id="evaluator_001",
        input="Hello world",
        expected_output={
            "has_translation": True,
            "has_evaluation": True,
            "has_improvement": True,
            "quality_improved": True
        },
        description="Simple translation - tests basic evaluator-optimizer loop",
        difficulty="easy",
        domain="translation"
    ),
    GoldenTestCase(
        pattern=PatternType.EVALUATOR,
        test_id="evaluator_002",
        input="The quick brown fox jumps over the lazy dog. This is a test sentence.",
        expected_output={
            "has_translation": True,
            "has_evaluation": True,
            "has_improvement": True,
            "quality_improved": True,
            "iterations": "<=5"
        },
        description="Longer text - tests evaluator with more complex content",
        difficulty="medium",
        domain="translation"
    ),
    GoldenTestCase(
        pattern=PatternType.EVALUATOR,
        test_id="evaluator_003",
        input="A" * 500,  # Very repetitive input
        expected_output={
            "has_translation": True,
            "has_evaluation": True,
            "has_improvement": True,
            "handles_repetitive": True
        },
        description="Repetitive input - tests evaluator with low-quality input",
        difficulty="hard",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.EVALUATOR,
        test_id="evaluator_004",
        input="",
        expected_output={
            "has_translation": True,
            "has_evaluation": True,
            "handles_empty": True
        },
        description="Empty input - tests edge case handling",
        difficulty="hard",
        domain="edge_case"
    ),
    GoldenTestCase(
        pattern=PatternType.EVALUATOR,
        test_id="evaluator_005",
        input="The scientific method is a systematic approach to understanding the natural world through observation, hypothesis formation, experimentation, and analysis.",
        expected_output={
            "has_translation": True,
            "has_evaluation": True,
            "has_improvement": True,
            "quality_improved": True,
            "technical_accuracy": True
        },
        description="Technical/scientific text - tests evaluator with domain-specific content",
        difficulty="medium",
        domain="technical"
    )
]


# =============================================================================
# All Golden Test Sets
# =============================================================================

ALL_GOLDEN_TESTS: Dict[PatternType, List[GoldenTestCase]] = {
    PatternType.CHAINING: CHAINING_GOLDEN_TESTS,
    PatternType.ROUTING: ROUTING_GOLDEN_TESTS,
    PatternType.PARALLELIZATION: PARALLELIZATION_GOLDEN_TESTS,
    PatternType.ORCHESTRATOR: ORCHESTRATOR_GOLDEN_TESTS,
    PatternType.EVALUATOR: EVALUATOR_GOLDEN_TESTS
}


def get_golden_tests(pattern: PatternType) -> List[GoldenTestCase]:
    """
    Get all golden test cases for a specific pattern.
    
    Args:
        pattern: The pattern type to get tests for
        
    Returns:
        List of golden test cases for the pattern
    """
    return ALL_GOLDEN_TESTS.get(pattern, [])


def get_all_golden_tests() -> List[GoldenTestCase]:
    """
    Get all golden test cases across all patterns.
    
    Returns:
        List of all golden test cases
    """
    all_tests = []
    for tests in ALL_GOLDEN_TESTS.values():
        all_tests.extend(tests)
    return all_tests


def get_test_by_id(test_id: str) -> GoldenTestCase:
    """
    Get a specific golden test case by ID.
    
    Args:
        test_id: The test case ID (e.g., "chaining_001")
        
    Returns:
        The golden test case, or None if not found
    """
    for tests in ALL_GOLDEN_TESTS.values():
        for test in tests:
            if test.test_id == test_id:
                return test
    return None


if __name__ == "__main__":
    # Print summary of all golden test sets
    print("=" * 70)
    print("Golden Test Sets Summary")
    print("=" * 70)
    
    for pattern, tests in ALL_GOLDEN_TESTS.items():
        print(f"\n{pattern.value.upper()}: {len(tests)} test cases")
        print("-" * 70)
        for test in tests:
            print(f"  {test.test_id}: {test.description}")
            print(f"    Difficulty: {test.difficulty}, Domain: {test.domain}")
    
    print("\n" + "=" * 70)
    print(f"Total Test Cases: {len(get_all_golden_tests())}")
    print("=" * 70)

