"""
LLM-as-Judge Evaluation Framework

This module implements an LLM-as-judge evaluation system that uses an LLM
to evaluate agent outputs against golden test sets.

The evaluator:
1. Runs agent patterns with golden test inputs
2. Collects outputs and performance metrics
3. Uses an LLM judge to evaluate outputs against expected criteria
4. Generates evaluation reports with scores and insights

Metrics Tracked:
- Accuracy: How well outputs match expected criteria
- Completeness: Whether all required components are present
- Style/Quality: Writing quality, coherence, structure
- Latency: Time taken to generate outputs
- Token Usage: Input/output tokens consumed
- Cost: Estimated cost based on token usage
"""

import sys
from pathlib import Path
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field
from utils.llm import call_llm
from utils.agent_logging import logger
from utils.config import config
from tests.golden_sets import (
    get_golden_tests,
    get_all_golden_tests,
    PatternType,
    GoldenTestCase
)
import sys
import io
import os

# Disable logger output during evaluation to avoid Unicode issues
# Set environment variable to disable Rich console
os.environ['NO_COLOR'] = '1'
os.environ['TERM'] = 'dumb'

# Safe print function for Windows compatibility
def safe_print(*args, **kwargs):
    """Print function that handles Unicode encoding errors on Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe printing
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode('ascii', 'replace').decode('ascii'))
            else:
                safe_args.append(str(arg).encode('ascii', 'replace').decode('ascii'))
        print(*safe_args, **kwargs)


# =============================================================================
# Evaluation Models
# =============================================================================

class EvaluationScore(BaseModel):
    """Score for a specific evaluation criterion."""
    criterion: str = Field(description="Name of the criterion")
    score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the score")


class PatternEvaluation(BaseModel):
    """Evaluation result for a single pattern test case."""
    test_id: str
    pattern: str
    input: Any
    actual_output: Any
    expected_output: Any
    
    # Evaluation scores
    accuracy_score: float = Field(description="Accuracy score (0.0-1.0)")
    completeness_score: float = Field(description="Completeness score (0.0-1.0)")
    quality_score: float = Field(description="Quality/style score (0.0-1.0)")
    overall_score: float = Field(description="Overall score (0.0-1.0)")
    
    # Detailed evaluations
    accuracy_evaluation: str = Field(description="Detailed accuracy evaluation")
    completeness_evaluation: str = Field(description="Detailed completeness evaluation")
    quality_evaluation: str = Field(description="Detailed quality evaluation")
    
    # Performance metrics
    latency_ms: float = Field(description="Latency in milliseconds")
    input_tokens: int = Field(description="Input tokens used")
    output_tokens: int = Field(description="Output tokens generated")
    estimated_cost: float = Field(description="Estimated cost in USD")
    
    # Metadata
    passed: bool = Field(description="Whether test passed (score >= 0.7)")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class EvaluationReport(BaseModel):
    """Complete evaluation report for all test cases."""
    timestamp: str
    pattern_type: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Aggregate scores
    avg_accuracy: float
    avg_completeness: float
    avg_quality: float
    avg_overall: float
    
    # Performance metrics
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    
    # Individual test results
    test_results: List[PatternEvaluation]
    
    # Summary
    summary: str = Field(description="Overall summary of evaluation")


# =============================================================================
# Pattern Executors
# =============================================================================

def execute_chaining_pattern(test_input: str) -> Dict[str, Any]:
    """Execute Prompt Chaining pattern with test input."""
    # Suppress logger output during execution
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    from patterns.chaining import PromptChain
    
    start_time = time.time()
    chain = PromptChain()
    
    # Step 1: Generate outline
    outline_result = chain.generate_outline(test_input)
    if not outline_result.is_success:
        return {"error": f"Outline generation failed: {outline_result.error}"}
    
    outline = outline_result.result
    
    # Step 2: Generate blog post
    blog_result = chain.generate_blog_post(outline)
    if not blog_result.is_success:
        return {"error": f"Blog post generation failed: {blog_result.error}"}
    
    blog_post = blog_result.result
    
    # Step 3: Edit blog post
    edit_result = chain.edit_blog_post(blog_post)
    if not edit_result.is_success:
        return {"error": f"Blog post editing failed: {edit_result.error}"}
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "outline": {
            "has_outline": True,
            "sections": len(outline.sections),
            "topic": outline.topic
        },
        "blog_post": {
            "has_blog_post": True,
            "title": blog_post.title,
            "word_count": len(blog_post.content.split())
        },
        "edited_post": {
            "has_edited_post": True,
            "title": edit_result.result.title,
            "improvements": edit_result.result.improvements
        },
        "latency_ms": latency_ms
    }


def execute_routing_pattern(test_input: str) -> Dict[str, Any]:
    """Execute Routing pattern with test input."""
    # Suppress logger output during execution
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    from patterns.routing import QueryRouter
    
    start_time = time.time()
    router = QueryRouter()
    
    # Route query
    route_result = router.route_query(test_input)
    if not route_result.is_success:
        return {"error": f"Routing failed: {route_result.error}"}
    
    classification = route_result.result
    
    # Get handler response
    handler_result = router.get_handler_response(classification)
    if not handler_result.is_success:
        return {"error": f"Handler response failed: {handler_result.error}"}
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "category": classification.category,
        "confidence": classification.confidence,
        "has_response": True,
        "handler_type": handler_result.result.handler_type if hasattr(handler_result.result, 'handler_type') else "unknown",
        "latency_ms": latency_ms
    }


def execute_parallelization_pattern(test_input: str) -> Dict[str, Any]:
    """Execute Parallelization pattern with test input."""
    # Suppress logger output during execution
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    import asyncio
    from patterns.parallelization import ParallelExecutor
    
    start_time = time.time()
    executor = ParallelExecutor()
    
    # Create tasks for voting
    tasks = [test_input] * 3  # 3 voters
    
    # Execute in parallel
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    votes = loop.run_until_complete(executor.execute_parallel(tasks))
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Aggregate votes
    safe_votes = sum(1 for vote in votes if vote.is_safe)
    consensus = "safe" if safe_votes >= 2 else "unsafe"
    
    return {
        "num_votes": len(votes),
        "consensus": consensus,
        "all_votes_received": len(votes) == 3,
        "latency_ms": latency_ms
    }


def execute_orchestrator_pattern(test_input: str) -> Dict[str, Any]:
    """Execute Orchestrator pattern with test input."""
    # Suppress logger output during execution
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    from patterns.orchestrator import Orchestrator
    
    start_time = time.time()
    orchestrator = Orchestrator()
    
    # Break down task
    breakdown_result = orchestrator.break_down_task(test_input)
    if not breakdown_result.is_success:
        return {"error": f"Task breakdown failed: {breakdown_result.error}"}
    
    breakdown = breakdown_result.result
    
    # Execute with workers
    execution_result = orchestrator.execute_with_workers(breakdown)
    if not execution_result.is_success:
        return {"error": f"Worker execution failed: {execution_result.error}"}
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Extract worker types used
    workers_used = []
    if hasattr(execution_result.result, 'worker_responses'):
        for response in execution_result.result.worker_responses:
            if hasattr(response, 'worker_type'):
                workers_used.append(response.worker_type)
    
    return {
        "has_breakdown": True,
        "num_tasks": len(breakdown.tasks),
        "has_result": True,
        "workers_used": list(set(workers_used)),
        "latency_ms": latency_ms
    }


def execute_evaluator_pattern(test_input: str) -> Dict[str, Any]:
    """Execute Evaluator-Optimizer pattern with test input."""
    # Suppress logger output during execution
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    from patterns.evaluator import EvaluatorOptimizer
    
    start_time = time.time()
    optimizer = EvaluatorOptimizer()
    
    # Refine translation
    refine_result = optimizer.refine(
        initial_output=test_input,
        max_iterations=5,
        quality_threshold=0.8
    )
    
    if not refine_result.is_success:
        return {"error": f"Refinement failed: {refine_result.error}"}
    
    result = refine_result.result
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "has_translation": True,
        "has_evaluation": True,
        "has_improvement": True,
        "quality_improved": result.final_score > result.initial_score if hasattr(result, 'final_score') else False,
        "iterations": result.iterations if hasattr(result, 'iterations') else 0,
        "latency_ms": latency_ms
    }


# Pattern executor mapping
PATTERN_EXECUTORS = {
    PatternType.CHAINING: execute_chaining_pattern,
    PatternType.ROUTING: execute_routing_pattern,
    PatternType.PARALLELIZATION: execute_parallelization_pattern,
    PatternType.ORCHESTRATOR: execute_orchestrator_pattern,
    PatternType.EVALUATOR: execute_evaluator_pattern
}


# =============================================================================
# LLM-as-Judge Evaluator
# =============================================================================

class LLMJudge(BaseModel):
    """LLM judge that evaluates pattern outputs."""
    accuracy: float = Field(description="Accuracy score (0.0-1.0)")
    completeness: float = Field(description="Completeness score (0.0-1.0)")
    quality: float = Field(description="Quality/style score (0.0-1.0)")
    accuracy_reasoning: str = Field(description="Reasoning for accuracy score")
    completeness_reasoning: str = Field(description="Reasoning for completeness score")
    quality_reasoning: str = Field(description="Reasoning for quality score")


def evaluate_with_llm_judge(
    test_case: GoldenTestCase,
    actual_output: Dict[str, Any],
    pattern_type: PatternType
) -> LLMJudge:
    """
    Use LLM as judge to evaluate actual output against expected output.
    
    Args:
        test_case: The golden test case
        actual_output: The actual output from the pattern
        pattern_type: The pattern type being evaluated
        
    Returns:
        LLMJudge with scores and reasoning
    """
    # Create evaluation prompt
    evaluation_prompt = f"""
You are an expert evaluator judging AI agent outputs. Evaluate the following:

**Test Case:**
- ID: {test_case.test_id}
- Description: {test_case.description}
- Input: {test_case.input}
- Expected Output Criteria: {json.dumps(test_case.expected_output, indent=2)}

**Actual Output:**
{json.dumps(actual_output, indent=2)}

**Pattern Type:** {pattern_type.value}

Evaluate on three criteria (score 0.0 to 1.0):

1. **Accuracy**: How well does the actual output match the expected criteria?
   - Check if required fields/components are present
   - Check if values meet expected thresholds
   - Check if the output makes sense for the input

2. **Completeness**: Are all required components present?
   - Check if all expected outputs are generated
   - Check if the output is complete (not partial)
   - Check if error handling worked correctly

3. **Quality/Style**: How good is the output quality?
   - Check coherence and structure
   - Check if output is well-formed
   - Check if output demonstrates good practices

Provide scores and brief reasoning for each criterion.
"""
    
    try:
        judge_result = call_llm(
            prompt=evaluation_prompt,
            response_model=LLMJudge,
            system_prompt="You are an expert evaluator. Be objective and thorough in your evaluation."
        )
        
        if judge_result.is_success:
            return judge_result.result
        else:
            # Fallback to default scores if LLM evaluation fails
            safe_print(f"Warning: LLM evaluation failed for {test_case.test_id}: {judge_result.error}")
            return LLMJudge(
                accuracy=0.5,
                completeness=0.5,
                quality=0.5,
                accuracy_reasoning="LLM evaluation failed, using default score",
                completeness_reasoning="LLM evaluation failed, using default score",
                quality_reasoning="LLM evaluation failed, using default score"
            )
    except Exception as e:
        safe_print(f"Error in LLM evaluation: {e}")
        return LLMJudge(
            accuracy=0.0,
            completeness=0.0,
            quality=0.0,
            accuracy_reasoning=f"Evaluation error: {str(e)}",
            completeness_reasoning=f"Evaluation error: {str(e)}",
            quality_reasoning=f"Evaluation error: {str(e)}"
        )


# =============================================================================
# Evaluation Runner
# =============================================================================

def evaluate_pattern(
    pattern_type: PatternType,
    track_tokens: bool = True
) -> EvaluationReport:
    """
    Evaluate a specific pattern using its golden test set.
    
    Args:
        pattern_type: The pattern to evaluate
        track_tokens: Whether to track token usage
        
    Returns:
        EvaluationReport with all test results
    """
    safe_print(f"Evaluating {pattern_type.value} pattern...")
    
    test_cases = get_golden_tests(pattern_type)
    test_results = []
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for test_case in test_cases:
        safe_print(f"Running test case: {test_case.test_id}")
        
        # Execute pattern
        executor = PATTERN_EXECUTORS[pattern_type]
        
        try:
            # Redirect stdout temporarily to avoid Unicode issues
            import contextlib
            from io import StringIO
            
            # Capture stdout to prevent Unicode errors from breaking execution
            stdout_capture = StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                try:
                    actual_output = executor(test_case.input)
                except UnicodeEncodeError:
                    # If Unicode error occurs, try again with ASCII-safe output
                    actual_output = executor(test_case.input)
            
            # Check for errors
            if "error" in actual_output:
                test_results.append(PatternEvaluation(
                    test_id=test_case.test_id,
                    pattern=pattern_type.value,
                    input=test_case.input,
                    actual_output=actual_output,
                    expected_output=test_case.expected_output,
                    accuracy_score=0.0,
                    completeness_score=0.0,
                    quality_score=0.0,
                    overall_score=0.0,
                    accuracy_evaluation="Execution failed",
                    completeness_evaluation="Execution failed",
                    quality_evaluation="Execution failed",
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost=0.0,
                    passed=False,
                    errors=[actual_output["error"]]
                ))
                continue
            
            # Evaluate with LLM judge
            judge_result = evaluate_with_llm_judge(test_case, actual_output, pattern_type)
            
            # Calculate overall score (weighted average)
            overall_score = (
                judge_result.accuracy * 0.4 +
                judge_result.completeness * 0.3 +
                judge_result.quality * 0.3
            )
            
            # Estimate token usage (simplified - would need actual tracking)
            input_tokens = len(str(test_case.input).split()) * 1.3  # Rough estimate
            output_tokens = len(str(actual_output).split()) * 1.3
            
            # Estimate cost (using config pricing)
            provider = config.get("llm.provider", "openai")
            model = config.get_nested(f"llm.{provider}.default_model", "gpt-4-turbo")
            
            # Simplified cost calculation (would need actual pricing lookup)
            cost_per_1k_input = 0.01  # Placeholder
            cost_per_1k_output = 0.03  # Placeholder
            estimated_cost = (input_tokens / 1000 * cost_per_1k_input) + (output_tokens / 1000 * cost_per_1k_output)
            
            total_input_tokens += int(input_tokens)
            total_output_tokens += int(output_tokens)
            
            test_results.append(PatternEvaluation(
                test_id=test_case.test_id,
                pattern=pattern_type.value,
                input=test_case.input,
                actual_output=actual_output,
                expected_output=test_case.expected_output,
                accuracy_score=judge_result.accuracy,
                completeness_score=judge_result.completeness,
                quality_score=judge_result.quality,
                overall_score=overall_score,
                accuracy_evaluation=judge_result.accuracy_reasoning,
                completeness_evaluation=judge_result.completeness_reasoning,
                quality_evaluation=judge_result.quality_reasoning,
                latency_ms=actual_output.get("latency_ms", 0.0),
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                estimated_cost=estimated_cost,
                passed=overall_score >= 0.7,
                errors=[]
            ))
            
        except Exception as e:
            safe_print(f"Error evaluating {test_case.test_id}: {e}")
            test_results.append(PatternEvaluation(
                test_id=test_case.test_id,
                pattern=pattern_type.value,
                input=test_case.input,
                actual_output={},
                expected_output=test_case.expected_output,
                accuracy_score=0.0,
                completeness_score=0.0,
                quality_score=0.0,
                overall_score=0.0,
                accuracy_evaluation=f"Error: {str(e)}",
                completeness_evaluation=f"Error: {str(e)}",
                quality_evaluation=f"Error: {str(e)}",
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                estimated_cost=0.0,
                passed=False,
                errors=[str(e)]
            ))
    
    # Calculate aggregate metrics
    passed_tests = sum(1 for r in test_results if r.passed)
    failed_tests = len(test_results) - passed_tests
    
    avg_accuracy = sum(r.accuracy_score for r in test_results) / len(test_results) if test_results else 0.0
    avg_completeness = sum(r.completeness_score for r in test_results) / len(test_results) if test_results else 0.0
    avg_quality = sum(r.quality_score for r in test_results) / len(test_results) if test_results else 0.0
    avg_overall = sum(r.overall_score for r in test_results) / len(test_results) if test_results else 0.0
    avg_latency = sum(r.latency_ms for r in test_results) / len(test_results) if test_results else 0.0
    total_cost = sum(r.estimated_cost for r in test_results)
    
    # Generate summary
    summary = f"""
Evaluation Summary for {pattern_type.value.upper()} Pattern:
- Total Tests: {len(test_results)}
- Passed: {passed_tests} ({passed_tests/len(test_results)*100:.1f}%)
- Failed: {failed_tests} ({failed_tests/len(test_results)*100:.1f}%)
- Average Overall Score: {avg_overall:.2%}
- Average Latency: {avg_latency:.0f}ms
- Total Cost: ${total_cost:.4f}
"""
    
    return EvaluationReport(
        timestamp=datetime.now().isoformat(),
        pattern_type=pattern_type.value,
        total_tests=len(test_results),
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        avg_accuracy=avg_accuracy,
        avg_completeness=avg_completeness,
        avg_quality=avg_quality,
        avg_overall=avg_overall,
        avg_latency_ms=avg_latency,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost=total_cost,
        test_results=test_results,
        summary=summary
    )


def evaluate_all_patterns() -> Dict[str, EvaluationReport]:
    """
    Evaluate all patterns and generate reports.
    
    Returns:
        Dictionary mapping pattern names to evaluation reports
    """
    reports = {}
    
    for pattern_type in PatternType:
        try:
            report = evaluate_pattern(pattern_type)
            reports[pattern_type.value] = report
        except Exception as e:
            logger.error(f"Error evaluating {pattern_type.value}: {e}")
    
    return reports


def generate_evaluation_report(reports: Dict[str, EvaluationReport], output_file: Optional[str] = None) -> str:
    """
    Generate a formatted evaluation report.
    
    Args:
        reports: Dictionary of evaluation reports
        output_file: Optional file path to save report
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("AI Agent Learning Framework - Evaluation Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall summary
    total_tests = sum(r.total_tests for r in reports.values())
    total_passed = sum(r.passed_tests for r in reports.values())
    total_failed = sum(r.failed_tests for r in reports.values())
    
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Tests: {total_tests}")
    report_lines.append(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    report_lines.append(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    report_lines.append("")
    
    # Per-pattern reports
    for pattern_name, report in reports.items():
        report_lines.append("=" * 80)
        report_lines.append(f"PATTERN: {pattern_name.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(report.summary)
        report_lines.append("")
        
        # Individual test results
        report_lines.append("Individual Test Results:")
        report_lines.append("-" * 80)
        for test_result in report.test_results:
            status = "PASS" if test_result.passed else "FAIL"
            report_lines.append(f"{status} | {test_result.test_id}")
            report_lines.append(f"  Overall Score: {test_result.overall_score:.2%}")
            report_lines.append(f"  Accuracy: {test_result.accuracy_score:.2%}")
            report_lines.append(f"  Completeness: {test_result.completeness_score:.2%}")
            report_lines.append(f"  Quality: {test_result.quality_score:.2%}")
            report_lines.append(f"  Latency: {test_result.latency_ms:.0f}ms")
            if test_result.errors:
                report_lines.append(f"  Errors: {', '.join(test_result.errors)}")
            report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        safe_print(f"Evaluation report saved to {output_file}")
    
    return report_text


if __name__ == "__main__":
    print("=" * 80)
    print("AI Agent Learning Framework - Evaluation")
    print("=" * 80)
    print("\nThis will evaluate all patterns using golden test sets.")
    print("This may take several minutes and will use API credits.\n")
    
    # Evaluate all patterns
    reports = evaluate_all_patterns()
    
    # Generate and print report
    report_text = generate_evaluation_report(reports, output_file="evaluation_report.txt")
    print(report_text)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

