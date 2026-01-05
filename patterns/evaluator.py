"""
Evaluator-Optimizer Pattern

This module demonstrates the Evaluator-Optimizer design pattern - an iterative
refinement loop that improves output quality through evaluation and optimization.

Pattern Overview:
1. Generate: Create initial output
2. Evaluate: Assess quality and identify issues
3. Optimize: Improve output based on evaluation feedback
4. Iterate: Repeat until quality threshold is met or max iterations reached

When to Use:
- When output quality is critical and needs refinement
- When you want iterative improvement of generated content
- When you need to meet specific quality criteria
- When initial outputs are good but can be improved
- When you want to optimize for multiple criteria (accuracy, style, clarity)

When NOT to Use:
- When initial output is already sufficient
- When speed is more important than quality
- When evaluation criteria are unclear or subjective
- When the improvement process is unpredictable
- When you need a single-pass solution (use Chaining instead)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from utils.llm import LLMClient, call_llm
from utils.agent_logging import logger
from utils.config import config
from utils.schemas import EvaluationScore


# =============================================================================
# Pydantic Models for Evaluator-Optimizer
# =============================================================================

class Translation(BaseModel):
    """
    Represents a translation that will be evaluated and improved.
    """
    source_text: str = Field(description="Original text in source language")
    target_language: str = Field(description="Target language for translation")
    translated_text: str = Field(description="Translated text")
    iteration: int = Field(default=1, description="Iteration number")


class Evaluation(BaseModel):
    """
    Evaluation of output quality.
    """
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1)"
    )
    accuracy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy score (0-1)"
    )
    style_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Style/naturalness score (0-1)"
    )
    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Clarity score (0-1)"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific issues found"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="List of strengths identified"
    )
    feedback: str = Field(description="Detailed feedback for improvement")
    meets_threshold: bool = Field(description="Whether quality threshold is met")


class ImprovedTranslation(BaseModel):
    """
    Improved version of the translation.
    """
    source_text: str = Field(description="Original source text")
    target_language: str = Field(description="Target language")
    improved_text: str = Field(description="Improved translation")
    iteration: int = Field(description="Iteration number")
    improvements_made: List[str] = Field(
        description="List of improvements made in this iteration"
    )
    previous_score: float = Field(description="Score from previous iteration")
    new_score: float = Field(description="Score after improvement")


class RefinementResult(BaseModel):
    """
    Final result after refinement iterations.
    """
    final_output: str = Field(description="Final refined output")
    initial_score: float = Field(description="Score of initial output")
    final_score: float = Field(description="Score of final output")
    iterations_completed: int = Field(description="Number of iterations")
    improvement_percentage: float = Field(description="Percentage improvement")
    iterations: List[Dict[str, Any]] = Field(
        description="History of all iterations"
    )


# =============================================================================
# Evaluator-Optimizer Implementation
# =============================================================================

class EvaluatorOptimizer:
    """
    Implements the Evaluator-Optimizer pattern for iterative refinement.
    
    This class:
    1. Generates initial output
    2. Evaluates output quality
    3. Optimizes based on evaluation feedback
    4. Iterates until quality threshold is met
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the evaluator-optimizer.
        
        Args:
            provider: LLM provider to use (defaults to config)
        """
        self.client = LLMClient(provider=provider)
        self.max_iterations = config.get_nested(
            "patterns.evaluator.max_iterations",
            5
        )
        self.quality_threshold = config.get_nested(
            "patterns.evaluator.quality_threshold",
            0.8
        )
        self.improvement_threshold = config.get_nested(
            "patterns.evaluator.improvement_threshold",
            0.1
        )
        self.iteration_history: List[Dict[str, Any]] = []
    
    def refine_translation(
        self,
        source_text: str,
        target_language: str
    ) -> RefinementResult:
        """
        Refine a translation through iterative evaluation and optimization.
        
        This demonstrates the evaluator-optimizer pattern using translation
        as an example use case.
        
        Args:
            source_text: Text to translate
            target_language: Target language
            
        Returns:
            Final refined result with improvement metrics
        """
        logger.section(
            "Evaluator-Optimizer Pattern",
            f"Refining translation to {target_language}"
        )
        
        # Step 1: Generate initial translation
        logger.info("Step 1: Generating initial translation...")
        current_translation = self._generate_initial_translation(
            source_text,
            target_language
        )
        
        # Evaluate initial translation
        initial_evaluation = self._evaluate_translation(current_translation)
        initial_score = initial_evaluation.overall_score
        
        logger.observation(
            f"Initial translation score: {initial_score:.2f}"
        )
        
        # Record initial iteration
        self.iteration_history.append({
            "iteration": 1,
            "translation": current_translation.translated_text,
            "score": initial_score,
            "evaluation": initial_evaluation.model_dump()
        })
        
        # Step 2: Iterative refinement loop
        logger.info("Step 2: Starting iterative refinement...")
        
        current_iteration = 1
        previous_score = initial_score
        
        while current_iteration < self.max_iterations:
            current_iteration += 1
            
            # Check if threshold is met
            if initial_evaluation.meets_threshold:
                logger.success(
                    f"Quality threshold ({self.quality_threshold}) met after "
                    f"{current_iteration - 1} iteration(s)"
                )
                break
            
            logger.action(
                f"Iteration {current_iteration}: Optimizing translation..."
            )
            
            # Optimize based on evaluation
            improved = self._optimize_translation(
                current_translation,
                initial_evaluation,
                previous_score
            )
            
            # Evaluate improved version
            new_evaluation = self._evaluate_translation(
                Translation(
                    source_text=source_text,
                    target_language=target_language,
                    translated_text=improved.improved_text,
                    iteration=current_iteration
                )
            )
            
            new_score = new_evaluation.overall_score
            improvement = new_score - previous_score
            
            logger.observation(
                f"Iteration {current_iteration} complete: "
                f"Score {previous_score:.2f} -> {new_score:.2f} "
                f"(+{improvement:.2f})"
            )
            
            # Record iteration
            self.iteration_history.append({
                "iteration": current_iteration,
                "translation": improved.improved_text,
                "score": new_score,
                "improvement": improvement,
                "improvements_made": improved.improvements_made,
                "evaluation": new_evaluation.model_dump()
            })
            
            # Check if improvement is significant
            if improvement < self.improvement_threshold:
                logger.warning(
                    f"Improvement ({improvement:.2f}) below threshold "
                    f"({self.improvement_threshold}). Stopping."
                )
                break
            
            # Update for next iteration
            current_translation = Translation(
                source_text=source_text,
                target_language=target_language,
                translated_text=improved.improved_text,
                iteration=current_iteration
            )
            initial_evaluation = new_evaluation
            previous_score = new_score
            
            # Check if threshold is now met
            if new_evaluation.meets_threshold:
                logger.success(
                    f"Quality threshold ({self.quality_threshold}) met!"
                )
                break
        
        # Calculate final metrics
        final_score = previous_score
        improvement_percentage = ((final_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        
        logger.success(
            f"Refinement complete: {initial_score:.2f} -> {final_score:.2f} "
            f"({improvement_percentage:+.1f}%)"
        )
        
        return RefinementResult(
            final_output=current_translation.translated_text,
            initial_score=initial_score,
            final_score=final_score,
            iterations_completed=current_iteration - 1,
            improvement_percentage=improvement_percentage,
            iterations=self.iteration_history
        )
    
    def _generate_initial_translation(
        self,
        source_text: str,
        target_language: str
    ) -> Translation:
        """
        Generate initial translation.
        
        Args:
            source_text: Text to translate
            target_language: Target language
            
        Returns:
            Initial translation
        """
        system_prompt = """You are an expert translator. Provide accurate,
natural translations that preserve meaning while sounding natural in the
target language."""
        
        user_prompt = f"""Translate the following text to {target_language}:

{source_text}

Provide a natural, accurate translation."""
        
        logger.thought(f"Generating initial translation to {target_language}")
        
        try:
            translated_text = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )
            
            return Translation(
                source_text=source_text,
                target_language=target_language,
                translated_text=translated_text,
                iteration=1
            )
            
        except Exception as e:
            logger.error(f"Initial translation failed: {e}", exception=e)
            raise
    
    def _evaluate_translation(self, translation: Translation) -> Evaluation:
        """
        Evaluate translation quality.
        
        Args:
            translation: Translation to evaluate
            
        Returns:
            Evaluation with scores and feedback
        """
        system_prompt = """You are an expert translation evaluator.
Evaluate translations for accuracy, style, and clarity. Provide specific,
actionable feedback."""
        
        user_prompt = f"""Evaluate this translation:

Source ({translation.target_language}): {translation.source_text}
Translation: {translation.translated_text}

Evaluate on:
1. Accuracy: Does it correctly convey the meaning?
2. Style: Does it sound natural in the target language?
3. Clarity: Is it clear and easy to understand?

Provide scores (0-1) for each criterion, overall score, specific issues,
strengths, and detailed feedback for improvement."""
        
        logger.thought(f"Evaluating translation quality (iteration {translation.iteration})")
        
        try:
            evaluation = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=Evaluation
            )
            
            # Determine if threshold is met
            evaluation.meets_threshold = (
                evaluation.overall_score >= self.quality_threshold
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exception=e)
            raise
    
    def _optimize_translation(
        self,
        translation: Translation,
        evaluation: Evaluation,
        previous_score: float
    ) -> ImprovedTranslation:
        """
        Optimize translation based on evaluation feedback.
        
        Args:
            translation: Current translation
            evaluation: Evaluation feedback
            previous_score: Score from previous iteration
            
        Returns:
            Improved translation
        """
        system_prompt = """You are an expert translator and editor.
Improve translations based on feedback while maintaining accuracy."""
        
        user_prompt = f"""Improve this translation based on the evaluation:

Source: {translation.source_text}
Current Translation: {translation.translated_text}

Evaluation Feedback:
- Overall Score: {evaluation.overall_score:.2f}
- Accuracy: {evaluation.accuracy_score:.2f}
- Style: {evaluation.style_score:.2f}
- Clarity: {evaluation.clarity_score:.2f}

Issues Found:
{chr(10).join(f'- {issue}' for issue in evaluation.issues)}

Strengths:
{chr(10).join(f'- {strength}' for strength in evaluation.strengths)}

Detailed Feedback: {evaluation.feedback}

Improve the translation addressing the issues while maintaining strengths.
List the specific improvements you made."""
        
        logger.thought(f"Optimizing translation based on feedback")
        
        try:
            improved_text = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )
            
            # Extract improvements made (simplified - in practice, you might parse this)
            improvements_made = evaluation.issues[:3]  # Use top issues as improvements
            
            # Re-evaluate to get new score
            new_evaluation = self._evaluate_translation(
                Translation(
                    source_text=translation.source_text,
                    target_language=translation.target_language,
                    translated_text=improved_text,
                    iteration=translation.iteration + 1
                )
            )
            
            return ImprovedTranslation(
                source_text=translation.source_text,
                target_language=translation.target_language,
                improved_text=improved_text,
                iteration=translation.iteration + 1,
                improvements_made=improvements_made,
                previous_score=previous_score,
                new_score=new_evaluation.overall_score
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exception=e)
            raise


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Evaluator-Optimizer Pattern Demo")
    print("=" * 60)
    
    evaluator = EvaluatorOptimizer()
    
    # Example translation
    source_text = "The quick brown fox jumps over the lazy dog."
    target_language = "Spanish"
    
    try:
        result = evaluator.refine_translation(source_text, target_language)
        
        print("\n" + "=" * 60)
        print("Refinement Results")
        print("=" * 60)
        print(f"\nSource: {source_text}")
        print(f"\nFinal Translation: {result.final_output}")
        print(f"\nScore Improvement:")
        print(f"  Initial: {result.initial_score:.2f}")
        print(f"  Final: {result.final_score:.2f}")
        print(f"  Improvement: {result.improvement_percentage:+.1f}%")
        print(f"\nIterations: {result.iterations_completed}")
        
        print(f"\nIteration History:")
        for iteration in result.iterations:
            print(f"\n  Iteration {iteration['iteration']}:")
            print(f"    Score: {iteration['score']:.2f}")
            if 'improvement' in iteration:
                print(f"    Improvement: {iteration['improvement']:+.2f}")
            if 'improvements_made' in iteration:
                print(f"    Improvements: {', '.join(iteration['improvements_made'][:2])}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your API key to .env")
        print("3. Installed all dependencies: pip install -r requirements.txt")



