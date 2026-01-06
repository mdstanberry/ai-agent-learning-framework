"""
Simple Test Runner for Evaluation Framework

This script provides a simple way to run evaluations with cost warnings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.evaluation import evaluate_pattern, PatternType, generate_evaluation_report
from tests.golden_sets import get_golden_tests, PatternType as PT


def main():
    """Main function to run evaluations."""
    print("=" * 80)
    print("AI Agent Learning Framework - Evaluation Test Runner")
    print("=" * 80)
    print("\nWARNING: Running evaluations will make API calls and incur costs.")
    print("Estimated cost: ~$0.20-0.50 per pattern, ~$1.00-2.50 for all patterns")
    print("\nAvailable patterns:")
    print("  1. CHAINING - Prompt Chaining pattern (5 test cases)")
    print("  2. ROUTING - Routing pattern (5 test cases)")
    print("  3. PARALLELIZATION - Parallelization pattern (5 test cases)")
    print("  4. ORCHESTRATOR - Orchestrator-Workers pattern (5 test cases)")
    print("  5. EVALUATOR - Evaluator-Optimizer pattern (5 test cases)")
    print("  6. ALL - All patterns (25 test cases total)")
    print("\n" + "-" * 80)
    
    # For now, just show what would be tested
    print("\n[DRY RUN] Showing test cases that would be evaluated:\n")
    
    for pattern in PatternType:
        tests = get_golden_tests(pattern)
        print(f"{pattern.value.upper()}: {len(tests)} test cases")
        for test in tests[:2]:  # Show first 2
            print(f"  - {test.test_id}: {test.description[:60]}...")
        if len(tests) > 2:
            print(f"  ... and {len(tests) - 2} more")
        print()
    
    print("=" * 80)
    print("\nTo actually run evaluations, uncomment the code below or use:")
    print("  python tests/evaluation.py")
    print("\nOr run a single pattern:")
    print("  python -c \"from tests.evaluation import evaluate_pattern, PatternType;")
    print("            report = evaluate_pattern(PatternType.ROUTING); print(report.summary)\"")
    print("=" * 80)
    
    # Uncomment below to actually run evaluations
    # print("\n[RUNNING EVALUATIONS]")
    # print("This will make API calls and incur costs...")
    # 
    # # Run single pattern as example
    # print("\nRunning ROUTING pattern evaluation...")
    # report = evaluate_pattern(PatternType.ROUTING)
    # print(report.summary)


if __name__ == "__main__":
    main()

