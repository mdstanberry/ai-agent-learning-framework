# Performance Benchmarks and Expected Scores

This document defines expected performance benchmarks for the AI Agent Learning Framework patterns.

## Evaluation Criteria

All patterns are evaluated on three criteria:

1. **Accuracy** (40% weight): How well outputs match expected criteria
2. **Completeness** (30% weight): Whether all required components are present
3. **Quality/Style** (30% weight): Writing quality, coherence, structure

**Overall Score** = (Accuracy × 0.4) + (Completeness × 0.3) + (Quality × 0.3)

**Passing Threshold**: Overall score ≥ 0.7 (70%)

## Expected Performance Benchmarks

### Prompt Chaining Pattern

**Expected Scores:**
- Average Overall Score: **≥ 0.75** (75%)
- Accuracy: **≥ 0.80** (80%)
- Completeness: **≥ 0.70** (70%)
- Quality: **≥ 0.75** (75%)

**Performance Metrics:**
- Average Latency: **< 30 seconds** per test case
- Token Usage: **< 10,000 tokens** per test case
- Cost per Test: **< $0.10** (using GPT-4-turbo pricing)

**Test Cases:**
- 5 test cases covering simple, complex, creative, edge cases, and business topics
- Expected pass rate: **≥ 80%** (4/5 tests)

### Routing Pattern

**Expected Scores:**
- Average Overall Score: **≥ 0.80** (80%)
- Accuracy: **≥ 0.85** (85%) - Classification accuracy
- Completeness: **≥ 0.75** (75%)
- Quality: **≥ 0.80** (80%)

**Performance Metrics:**
- Average Latency: **< 10 seconds** per test case
- Token Usage: **< 5,000 tokens** per test case
- Cost per Test: **< $0.05**

**Test Cases:**
- 5 test cases covering tech support, sales, billing, general, and mixed queries
- Expected pass rate: **≥ 80%** (4/5 tests)

### Parallelization Pattern

**Expected Scores:**
- Average Overall Score: **≥ 0.75** (75%)
- Accuracy: **≥ 0.80** (80%)
- Completeness: **≥ 0.70** (70%) - All votes received
- Quality: **≥ 0.75** (75%)

**Performance Metrics:**
- Average Latency: **< 15 seconds** per test case (parallel execution)
- Token Usage: **< 8,000 tokens** per test case (3 parallel voters)
- Cost per Test: **< $0.08**

**Test Cases:**
- 5 test cases covering safe content, unsafe content, long input, empty input, and technical content
- Expected pass rate: **≥ 80%** (4/5 tests)

### Orchestrator-Workers Pattern

**Expected Scores:**
- Average Overall Score: **≥ 0.70** (70%)
- Accuracy: **≥ 0.75** (75%)
- Completeness: **≥ 0.65** (65%) - All tasks completed
- Quality: **≥ 0.70** (70%)

**Performance Metrics:**
- Average Latency: **< 45 seconds** per test case (multiple workers)
- Token Usage: **< 15,000 tokens** per test case
- Cost per Test: **< $0.15**

**Test Cases:**
- 5 test cases covering simple tasks, complex multi-step tasks, technical projects, edge cases, and business tasks
- Expected pass rate: **≥ 60%** (3/5 tests) - More complex, lower expected pass rate

### Evaluator-Optimizer Pattern

**Expected Scores:**
- Average Overall Score: **≥ 0.75** (75%)
- Accuracy: **≥ 0.80** (80%)
- Completeness: **≥ 0.70** (70%)
- Quality: **≥ 0.75** (75%) - Quality improvement demonstrated

**Performance Metrics:**
- Average Latency: **< 20 seconds** per test case (iterative refinement)
- Token Usage: **< 12,000 tokens** per test case (multiple iterations)
- Cost per Test: **< $0.12**

**Test Cases:**
- 5 test cases covering simple translations, complex text, repetitive input, empty input, and technical content
- Expected pass rate: **≥ 80%** (4/5 tests)

## Overall Framework Benchmarks

**Aggregate Metrics:**
- Total Test Cases: **25** (5 per pattern)
- Expected Overall Pass Rate: **≥ 75%** (19/25 tests)
- Average Overall Score Across All Patterns: **≥ 0.75** (75%)
- Total Evaluation Time: **< 10 minutes** (all patterns)
- Total Estimated Cost: **< $5.00** (all patterns)

## Performance Targets

### Latency Targets
- **Interactive Use**: < 3 seconds (for simple patterns like Routing)
- **Batch Processing**: < 60 seconds (for complex patterns like Orchestrator)
- **Evaluation Run**: < 10 minutes (all 25 test cases)

### Cost Targets
- **Per Pattern Evaluation**: < $0.50
- **Full Framework Evaluation**: < $5.00
- **Per Test Case Average**: < $0.20

### Quality Targets
- **Minimum Acceptable Score**: 0.70 (70%)
- **Target Score**: 0.75 (75%)
- **Excellent Score**: ≥ 0.85 (85%)

## Running Evaluations

### Run Single Pattern Evaluation

```bash
python -c "
from tests.evaluation import evaluate_pattern
from tests.golden_sets import PatternType

report = evaluate_pattern(PatternType.CHAINING)
print(report.summary)
"
```

### Run All Pattern Evaluations

```bash
python tests/evaluation.py
```

This will:
1. Run all 25 golden test cases
2. Evaluate outputs using LLM-as-judge
3. Generate performance metrics
4. Save report to `evaluation_report.txt`

### View Golden Test Sets

```bash
python tests/golden_sets.py
```

This will print a summary of all golden test cases.

## Interpreting Results

### Score Ranges

- **0.90 - 1.00**: Excellent - Output exceeds expectations
- **0.80 - 0.89**: Good - Output meets expectations
- **0.70 - 0.79**: Acceptable - Output passes but could improve
- **0.60 - 0.69**: Needs Improvement - Output partially meets requirements
- **0.00 - 0.59**: Failed - Output does not meet requirements

### Common Issues

**Low Accuracy Scores:**
- Output doesn't match expected structure
- Missing required fields/components
- Values don't meet thresholds

**Low Completeness Scores:**
- Missing steps in workflow
- Incomplete outputs
- Errors during execution

**Low Quality Scores:**
- Poor coherence or structure
- Inconsistent formatting
- Doesn't follow best practices

## Regression Detection

The golden test sets should **NEVER** be modified once established. They serve as a baseline for detecting performance regressions.

**Regression Indicators:**
- Overall score drops by >5% compared to baseline
- Pass rate drops below expected threshold
- Latency increases by >20%
- Cost increases by >20%

If regressions are detected:
1. Investigate the cause (code changes, model changes, etc.)
2. Document the regression
3. Fix the issue or update benchmarks if change is intentional

## Notes

- Benchmarks are based on using GPT-4-turbo or Claude-3.5-Sonnet
- Performance may vary with different models
- Costs are estimates and may vary with actual API pricing
- Latency includes network time and may vary based on connection speed
- Token usage is estimated and may not match actual API usage exactly

