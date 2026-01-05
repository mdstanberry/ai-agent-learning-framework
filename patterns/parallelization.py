"""
Parallelization Pattern

This module demonstrates the Parallelization design pattern - executing
multiple independent tasks concurrently to improve efficiency and gather
multiple perspectives.

Pattern Overview:
1. Define multiple independent tasks
2. Execute tasks in parallel (using asyncio or concurrent.futures)
3. Collect results from all tasks
4. Aggregate results into final decision

When to Use:
- When you have multiple independent tasks that don't depend on each other
- When you want to gather multiple opinions/perspectives (e.g., voting)
- When tasks can be executed simultaneously without conflicts
- When you want to improve performance by parallelizing work
- When you need redundancy or consensus (multiple agents verify something)

When NOT to Use:
- When tasks depend on each other (use Chaining pattern instead)
- When tasks need to be executed sequentially
- When parallel execution would cause conflicts or race conditions
- When the overhead of parallelization exceeds benefits
"""

import asyncio
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from utils.llm import LLMClient, call_llm
from utils.agent_logging import logger
from utils.config import config


# =============================================================================
# Pydantic Models for Parallelization
# =============================================================================

class SafetyVote(BaseModel):
    """
    Represents a single agent's vote on content safety.
    """
    agent_id: str = Field(description="Identifier for the voting agent")
    is_safe: bool = Field(description="Whether the content is considered safe")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the vote (0-1)"
    )
    reasoning: str = Field(description="Explanation for the vote")
    concerns: List[str] = Field(
        default_factory=list,
        description="List of specific concerns if not safe"
    )


class AggregatedResult(BaseModel):
    """
    Aggregated result from multiple parallel tasks.
    """
    final_decision: bool = Field(description="Final decision based on aggregation")
    vote_summary: Dict[str, int] = Field(
        description="Summary of votes (e.g., {'safe': 3, 'unsafe': 2})"
    )
    average_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Average confidence across all votes"
    )
    consensus_level: float = Field(
        ge=0.0,
        le=1.0,
        description="How much consensus there is (1.0 = unanimous)"
    )
    individual_votes: List[SafetyVote] = Field(
        description="All individual votes"
    )
    reasoning: str = Field(description="Explanation of the aggregation logic")


# =============================================================================
# Parallelization Implementation
# =============================================================================

class ParallelExecutor:
    """
    Implements the Parallelization pattern for concurrent task execution.
    
    This class handles:
    - Executing multiple independent tasks in parallel
    - Collecting results from all tasks
    - Aggregating results into a final decision
    
    Example: Content safety voting where multiple agents independently
    evaluate content and vote on its safety.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the parallel executor.
        
        Args:
            provider: LLM provider to use (defaults to config)
        """
        self.client = LLMClient(provider=provider)
        self.max_workers = config.get_nested(
            "patterns.parallelization.max_workers",
            5
        )
        self.timeout_per_task = config.get_nested(
            "patterns.parallelization.timeout_per_task",
            60
        )
    
    def content_safety_vote(
        self,
        content: str,
        num_voters: int = 3
    ) -> AggregatedResult:
        """
        Execute parallel content safety voting.
        
        Multiple agents independently evaluate content and vote on safety.
        Results are aggregated into a final decision.
        
        Args:
            content: The content to evaluate
            num_voters: Number of agents to vote
            
        Returns:
            Aggregated result with final decision and vote details
        """
        logger.section(
            "Parallel Content Safety Voting",
            f"Evaluating content with {num_voters} agents"
        )
        
        # Step 1: Create voting tasks
        logger.info(f"Step 1: Creating {num_voters} voting tasks...")
        tasks = [
            self._create_voting_task(content, agent_id=f"agent_{i+1}")
            for i in range(num_voters)
        ]
        
        # Step 2: Execute tasks in parallel
        logger.info("Step 2: Executing voting tasks in parallel...")
        votes = self._execute_parallel(tasks)
        
        # Step 3: Aggregate results
        logger.info("Step 3: Aggregating votes...")
        result = self._aggregate_votes(votes)
        
        logger.success(
            f"Voting complete. Final decision: {'SAFE' if result.final_decision else 'UNSAFE'}"
        )
        
        return result
    
    def _create_voting_task(self, content: str, agent_id: str) -> callable:
        """
        Create a voting task function for a single agent.
        
        Args:
            content: Content to evaluate
            agent_id: Unique identifier for this agent
            
        Returns:
            Callable function that returns a SafetyVote
        """
        def vote():
            """Execute the voting task."""
            logger.action(f"Agent {agent_id} evaluating content...")
            
            system_prompt = f"""You are Agent {agent_id}, a content safety evaluator.
Your job is to evaluate content for safety, appropriateness, and potential harm.
Be thorough and provide clear reasoning for your assessment."""
            
            user_prompt = f"""Evaluate this content for safety:

{content}

Consider:
- Is the content safe for general audiences?
- Are there any harmful, offensive, or inappropriate elements?
- What is your confidence level in this assessment?

Provide a clear vote (safe/unsafe), confidence level, and detailed reasoning."""
            
            try:
                vote_result = self.client.call(
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=system_prompt,
                    response_model=SafetyVote
                )
                
                # Ensure agent_id is set
                vote_result.agent_id = agent_id
                
                logger.observation(
                    f"Agent {agent_id} voted: {'SAFE' if vote_result.is_safe else 'UNSAFE'} "
                    f"(confidence: {vote_result.confidence:.2f})"
                )
                
                return vote_result
                
            except Exception as e:
                logger.error(f"Agent {agent_id} voting failed: {e}", exception=e)
                # Return a conservative vote on error
                return SafetyVote(
                    agent_id=agent_id,
                    is_safe=False,
                    confidence=0.5,
                    reasoning=f"Voting failed: {str(e)}",
                    concerns=["Unable to complete evaluation"]
                )
        
        return vote
    
    def _execute_parallel(self, tasks: List[callable]) -> List[SafetyVote]:
        """
        Execute multiple tasks in parallel using ThreadPoolExecutor.
        
        Args:
            tasks: List of callable tasks to execute
            
        Returns:
            List of results from all tasks
        """
        results = []
        
        logger.thought(f"Executing {len(tasks)} tasks in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task): i
                for i, task in enumerate(tasks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                try:
                    result = future.result(timeout=self.timeout_per_task)
                    results.append(result)
                    logger.info(f"Task {future_to_task[future] + 1} completed")
                except Exception as e:
                    logger.error(f"Task {future_to_task[future] + 1} failed: {e}", exception=e)
                    # Add a default vote on failure
                    results.append(SafetyVote(
                        agent_id=f"agent_failed_{future_to_task[future]}",
                        is_safe=False,
                        confidence=0.0,
                        reasoning=f"Task execution failed: {str(e)}",
                        concerns=["Task execution error"]
                    ))
        
        logger.success(f"All {len(results)} tasks completed")
        return results
    
    def _aggregate_votes(self, votes: List[SafetyVote]) -> AggregatedResult:
        """
        Aggregate multiple votes into a final decision.
        
        Uses majority voting with confidence weighting.
        
        Args:
            votes: List of individual votes
            
        Returns:
            Aggregated result with final decision
        """
        if not votes:
            raise ValueError("Cannot aggregate empty vote list")
        
        # Count votes
        safe_count = sum(1 for v in votes if v.is_safe)
        unsafe_count = len(votes) - safe_count
        
        # Calculate average confidence
        avg_confidence = sum(v.confidence for v in votes) / len(votes)
        
        # Calculate consensus level (how unanimous the votes are)
        # 1.0 = all agree, 0.0 = split evenly
        majority_size = max(safe_count, unsafe_count)
        consensus_level = (majority_size / len(votes)) * 2 - 1  # Scale to 0-1
        
        # Final decision: majority wins, but consider confidence
        # Weight votes by confidence
        weighted_safe = sum(
            v.confidence for v in votes if v.is_safe
        )
        weighted_unsafe = sum(
            v.confidence for v in votes if not v.is_safe
        )
        
        final_decision = weighted_safe > weighted_unsafe
        
        # Generate reasoning
        reasoning = (
            f"Majority vote: {safe_count} safe, {unsafe_count} unsafe. "
            f"Average confidence: {avg_confidence:.2f}. "
            f"Consensus level: {consensus_level:.2f}. "
            f"Final decision based on confidence-weighted voting."
        )
        
        logger.observation(
            f"Aggregation complete: {safe_count} safe, {unsafe_count} unsafe, "
            f"consensus: {consensus_level:.2f}"
        )
        
        return AggregatedResult(
            final_decision=final_decision,
            vote_summary={"safe": safe_count, "unsafe": unsafe_count},
            average_confidence=avg_confidence,
            consensus_level=consensus_level,
            individual_votes=votes,
            reasoning=reasoning
        )
    
    async def _execute_parallel_async(
        self,
        tasks: List[callable]
    ) -> List[Any]:
        """
        Alternative async implementation for parallel execution.
        
        This can be used if you prefer async/await syntax.
        
        Args:
            tasks: List of async callable tasks
            
        Returns:
            List of results
        """
        # Convert sync tasks to async if needed
        async def run_task(task):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, task)
        
        results = await asyncio.gather(*[run_task(task) for task in tasks])
        return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Parallelization Pattern Demo")
    print("=" * 60)
    
    executor = ParallelExecutor()
    
    # Test content to evaluate
    test_content = """
    This is a sample blog post about technology. It discusses
    the benefits of using AI agents for automation. The content
    is informative and appropriate for general audiences.
    """
    
    try:
        result = executor.content_safety_vote(test_content, num_voters=3)
        
        print("\n" + "=" * 60)
        print("Voting Results")
        print("=" * 60)
        print(f"\nFinal Decision: {'SAFE' if result.final_decision else 'UNSAFE'}")
        print(f"\nVote Summary:")
        print(f"  Safe votes: {result.vote_summary['safe']}")
        print(f"  Unsafe votes: {result.vote_summary['unsafe']}")
        print(f"\nAverage Confidence: {result.average_confidence:.2f}")
        print(f"Consensus Level: {result.consensus_level:.2f}")
        
        print(f"\nIndividual Votes:")
        for vote in result.individual_votes:
            print(f"\n  Agent {vote.agent_id}:")
            print(f"    Decision: {'SAFE' if vote.is_safe else 'UNSAFE'}")
            print(f"    Confidence: {vote.confidence:.2f}")
            print(f"    Reasoning: {vote.reasoning[:100]}...")
            if vote.concerns:
                print(f"    Concerns: {', '.join(vote.concerns)}")
        
        print(f"\nAggregation Reasoning: {result.reasoning}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your API key to .env")
        print("3. Installed all dependencies: pip install -r requirements.txt")



