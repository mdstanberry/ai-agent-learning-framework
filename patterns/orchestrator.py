"""
Orchestrator-Workers Pattern

This module demonstrates the Orchestrator-Workers design pattern - a dynamic
delegation system where an orchestrator breaks down complex tasks and assigns
them to specialized worker agents.

Pattern Overview:
1. Orchestrator receives a complex, high-level goal
2. Orchestrator analyzes the goal and breaks it into sub-tasks
3. Orchestrator delegates each sub-task to appropriate specialized workers
4. Workers execute their tasks independently
5. Orchestrator synthesizes worker outputs into final result

When to Use:
- When tasks are complex and require multiple types of expertise
- When you need dynamic task breakdown (can't predict all steps upfront)
- When different parts of a task need different specializations
- When you want to leverage specialized agents for different domains
- When tasks are too complex for a single agent to handle well

When NOT to Use:
- When tasks are simple and can be handled by one agent
- When the task breakdown is predictable (use Chaining pattern instead)
- When tasks are independent and don't need coordination (use Parallelization)
- When you need the agent to decide actions dynamically (use ReAct pattern)
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from utils.llm import LLMClient, call_llm
from utils.agent_logging import logger
from utils.config import config


# =============================================================================
# Pydantic Models for Orchestration
# =============================================================================

class Task(BaseModel):
    """
    Represents a single sub-task to be assigned to a worker.
    """
    task_id: str = Field(description="Unique identifier for this task")
    description: str = Field(description="What needs to be done")
    worker_type: Literal["researcher", "coder", "reviewer", "writer", "analyst"] = Field(
        description="Type of worker that should handle this task"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Task priority (1=lowest, 5=highest)"
    )


class WorkerResponse(BaseModel):
    """
    Response from a worker agent after completing a task.
    """
    task_id: str = Field(description="ID of the task that was completed")
    worker_type: str = Field(description="Type of worker that completed the task")
    result: str = Field(description="The result/output from the worker")
    status: Literal["success", "partial", "failed"] = Field(
        description="Status of task completion"
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional artifacts (code, data, etc.) produced"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Suggested next steps or follow-up tasks"
    )


class TaskBreakdown(BaseModel):
    """
    Orchestrator's breakdown of a complex goal into sub-tasks.
    """
    goal: str = Field(description="The original high-level goal")
    tasks: List[Task] = Field(description="List of sub-tasks to complete")
    estimated_complexity: str = Field(
        description="Estimated complexity (simple, moderate, complex)"
    )
    reasoning: str = Field(description="Why the goal was broken down this way")


class FinalResult(BaseModel):
    """
    Final synthesized result from the orchestrator.
    """
    goal: str = Field(description="Original goal that was accomplished")
    summary: str = Field(description="Summary of what was accomplished")
    components: List[Dict[str, Any]] = Field(
        description="Components contributed by each worker"
    )
    final_output: str = Field(description="Final synthesized output")
    tasks_completed: int = Field(description="Number of tasks completed")
    tasks_total: int = Field(description="Total number of tasks")
    success_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentage of tasks completed successfully"
    )


# =============================================================================
# Orchestrator-Workers Implementation
# =============================================================================

class Orchestrator:
    """
    Implements the Orchestrator-Workers pattern.
    
    The orchestrator:
    1. Breaks down complex goals into sub-tasks
    2. Assigns tasks to specialized workers
    3. Coordinates task execution
    4. Synthesizes results into final output
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            provider: LLM provider to use (defaults to config)
        """
        self.client = LLMClient(provider=provider)
        self.max_workers = config.get_nested(
            "patterns.orchestrator.max_workers",
            3
        )
        self.worker_timeout = config.get_nested(
            "patterns.orchestrator.worker_timeout",
            120
        )
        self.worker_responses: Dict[str, WorkerResponse] = {}
    
    def execute_goal(self, goal: str) -> FinalResult:
        """
        Execute a complex goal using the orchestrator-workers pattern.
        
        This is the main entry point that orchestrates the entire process.
        
        Args:
            goal: High-level goal to accomplish
            
        Returns:
            Final synthesized result
        """
        logger.section("Orchestrator-Workers Pattern", f"Goal: {goal}")
        
        # Step 1: Break down the goal
        logger.info("Step 1: Breaking down goal into sub-tasks...")
        breakdown = self._break_down_goal(goal)
        
        logger.observation(
            f"Goal broken down into {len(breakdown.tasks)} tasks: "
            f"{[t.worker_type for t in breakdown.tasks]}"
        )
        
        # Step 2: Execute tasks with workers
        logger.info("Step 2: Executing tasks with specialized workers...")
        self._execute_tasks(breakdown.tasks, goal)
        
        # Step 3: Synthesize results
        logger.info("Step 3: Synthesizing worker results...")
        final_result = self._synthesize_results(goal, breakdown.tasks)
        
        logger.success("Goal execution complete!")
        return final_result
    
    def _break_down_goal(self, goal: str) -> TaskBreakdown:
        """
        Break down a complex goal into manageable sub-tasks.
        
        Args:
            goal: The high-level goal
            
        Returns:
            Task breakdown with list of sub-tasks
        """
        system_prompt = """You are an expert project orchestrator.
Your job is to break down complex goals into smaller, manageable tasks
that can be assigned to specialized workers.

Available worker types:
- researcher: For gathering information, research, fact-checking
- coder: For writing code, implementing features, debugging
- reviewer: For reviewing code, content, or plans
- writer: For writing documentation, content, summaries
- analyst: For analyzing data, trends, patterns

Break down goals thoughtfully, considering dependencies and priorities."""
        
        user_prompt = f"""Break down this complex goal into sub-tasks:

Goal: {goal}

For each sub-task, specify:
1. A clear description of what needs to be done
2. Which worker type should handle it
3. Any dependencies on other tasks
4. Priority level (1-5)

Create a logical sequence of tasks that will accomplish the goal."""
        
        logger.thought(f"Analyzing goal to determine required tasks and workers")
        
        try:
            breakdown = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                response_model=TaskBreakdown
            )
            
            # Ensure tasks have IDs
            for i, task in enumerate(breakdown.tasks):
                if not task.task_id:
                    task.task_id = f"task_{i+1}"
            
            logger.success(f"Goal broken down into {len(breakdown.tasks)} tasks")
            return breakdown
            
        except Exception as e:
            logger.error(f"Goal breakdown failed: {e}", exception=e)
            raise
    
    def _execute_tasks(self, tasks: List[Task], goal: str) -> None:
        """
        Execute tasks by delegating to appropriate workers.
        
        Args:
            tasks: List of tasks to execute
            goal: Original goal (for context)
        """
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks(tasks)
        
        for task in sorted_tasks:
            logger.action(
                f"Delegating task to {task.worker_type} worker",
                arguments={"task_id": task.task_id, "description": task.description}
            )
            
            try:
                response = self._delegate_to_worker(task, goal)
                self.worker_responses[task.task_id] = response
                
                logger.observation(
                    f"Task {task.task_id} completed: {response.status}"
                )
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}", exception=e)
                # Create a failed response
                self.worker_responses[task.task_id] = WorkerResponse(
                    task_id=task.task_id,
                    worker_type=task.worker_type,
                    result=f"Task failed: {str(e)}",
                    status="failed"
                )
    
    def _sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """
        Sort tasks considering dependencies and priorities.
        
        Args:
            tasks: List of tasks to sort
            
        Returns:
            Sorted list of tasks
        """
        # Simple topological sort considering dependencies
        sorted_tasks = []
        completed_ids = set()
        
        # Sort by priority first
        tasks_by_priority = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        while tasks_by_priority:
            progress = False
            
            for task in tasks_by_priority[:]:
                # Check if all dependencies are completed
                if all(dep in completed_ids for dep in task.dependencies):
                    sorted_tasks.append(task)
                    completed_ids.add(task.task_id)
                    tasks_by_priority.remove(task)
                    progress = True
            
            if not progress:
                # If no progress, add remaining tasks (circular dependency or missing deps)
                sorted_tasks.extend(tasks_by_priority)
                break
        
        return sorted_tasks
    
    def _delegate_to_worker(self, task: Task, goal: str) -> WorkerResponse:
        """
        Delegate a task to the appropriate specialized worker.
        
        Args:
            task: The task to delegate
            goal: Original goal (for context)
            
        Returns:
            Worker's response
        """
        worker_handlers = {
            "researcher": self._worker_researcher,
            "coder": self._worker_coder,
            "reviewer": self._worker_reviewer,
            "writer": self._worker_writer,
            "analyst": self._worker_analyst
        }
        
        handler = worker_handlers.get(task.worker_type, self._worker_general)
        return handler(task, goal)
    
    def _worker_researcher(self, task: Task, goal: str) -> WorkerResponse:
        """Researcher worker - gathers information and facts."""
        system_prompt = """You are a research specialist. Your job is to gather
accurate information, conduct research, and provide well-sourced facts.
Be thorough and cite your sources when possible."""
        
        user_prompt = f"""Goal: {goal}

Task: {task.description}

Conduct research and provide:
1. Relevant information and facts
2. Key findings
3. Sources or references if applicable
4. Any important considerations"""
        
        logger.thought(f"Researcher worker processing task: {task.task_id}")
        
        result = self.client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        
        return WorkerResponse(
            task_id=task.task_id,
            worker_type="researcher",
            result=result,
            status="success"
        )
    
    def _worker_coder(self, task: Task, goal: str) -> WorkerResponse:
        """Coder worker - writes and implements code."""
        system_prompt = """You are a coding specialist. Your job is to write
clean, well-documented code that solves problems. Follow best practices
and include comments explaining your approach."""
        
        user_prompt = f"""Goal: {goal}

Task: {task.description}

Write code to accomplish this task. Include:
1. Complete, runnable code
2. Comments explaining the approach
3. Any necessary documentation
4. Test cases if applicable"""
        
        logger.thought(f"Coder worker processing task: {task.task_id}")
        
        result = self.client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        
        return WorkerResponse(
            task_id=task.task_id,
            worker_type="coder",
            result=result,
            status="success",
            artifacts={"code": result}
        )
    
    def _worker_reviewer(self, task: Task, goal: str) -> WorkerResponse:
        """Reviewer worker - reviews code, content, or plans."""
        system_prompt = """You are a review specialist. Your job is to
critically review work, identify issues, and suggest improvements.
Be constructive and specific in your feedback."""
        
        user_prompt = f"""Goal: {goal}

Task: {task.description}

Review the work and provide:
1. Overall assessment
2. Issues or problems found
3. Suggestions for improvement
4. Approval or recommendations"""
        
        logger.thought(f"Reviewer worker processing task: {task.task_id}")
        
        result = self.client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        
        return WorkerResponse(
            task_id=task.task_id,
            worker_type="reviewer",
            result=result,
            status="success"
        )
    
    def _worker_writer(self, task: Task, goal: str) -> WorkerResponse:
        """Writer worker - creates documentation and content."""
        system_prompt = """You are a writing specialist. Your job is to create
clear, well-structured written content. Write in an appropriate style
for the audience and purpose."""
        
        user_prompt = f"""Goal: {goal}

Task: {task.description}

Write content to accomplish this task. Ensure:
1. Clear, well-structured writing
2. Appropriate tone and style
3. Complete coverage of the topic
4. Good readability"""
        
        logger.thought(f"Writer worker processing task: {task.task_id}")
        
        result = self.client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        
        return WorkerResponse(
            task_id=task.task_id,
            worker_type="writer",
            result=result,
            status="success"
        )
    
    def _worker_analyst(self, task: Task, goal: str) -> WorkerResponse:
        """Analyst worker - analyzes data and patterns."""
        system_prompt = """You are an analysis specialist. Your job is to
analyze data, identify patterns, and provide insights. Be thorough
and data-driven in your analysis."""
        
        user_prompt = f"""Goal: {goal}

Task: {task.description}

Analyze and provide:
1. Key findings and patterns
2. Data-driven insights
3. Trends or correlations
4. Recommendations based on analysis"""
        
        logger.thought(f"Analyst worker processing task: {task.task_id}")
        
        result = self.client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        
        return WorkerResponse(
            task_id=task.task_id,
            worker_type="analyst",
            result=result,
            status="success"
        )
    
    def _worker_general(self, task: Task, goal: str) -> WorkerResponse:
        """General worker - fallback for unknown worker types."""
        system_prompt = """You are a general-purpose worker. Handle the task
to the best of your ability."""
        
        user_prompt = f"""Goal: {goal}

Task: {task.description}

Complete this task as best you can."""
        
        logger.thought(f"General worker processing task: {task.task_id}")
        
        result = self.client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        
        return WorkerResponse(
            task_id=task.task_id,
            worker_type="general",
            result=result,
            status="success"
        )
    
    def _synthesize_results(
        self,
        goal: str,
        tasks: List[Task]
    ) -> FinalResult:
        """
        Synthesize worker results into final output.
        
        Args:
            goal: Original goal
            tasks: List of tasks that were executed
            
        Returns:
            Final synthesized result
        """
        system_prompt = """You are a synthesis specialist. Your job is to
combine multiple worker outputs into a coherent final result that
accomplishes the original goal."""
        
        # Collect all worker results
        results_text = "\n\n".join([
            f"Task {resp.task_id} ({resp.worker_type}):\n{resp.result}"
            for resp in self.worker_responses.values()
        ])
        
        user_prompt = f"""Original Goal: {goal}

Worker Results:
{results_text}

Synthesize these results into a final output that accomplishes the goal.
Provide:
1. A summary of what was accomplished
2. The final synthesized output
3. How the different components work together"""
        
        logger.thought("Synthesizing worker results into final output")
        
        try:
            synthesis = self.client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt
            )
            
            # Calculate success metrics
            successful_tasks = sum(
                1 for resp in self.worker_responses.values()
                if resp.status == "success"
            )
            total_tasks = len(tasks)
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
            
            # Build components list
            components = [
                {
                    "task_id": resp.task_id,
                    "worker_type": resp.worker_type,
                    "status": resp.status,
                    "summary": resp.result[:200] + "..." if len(resp.result) > 200 else resp.result
                }
                for resp in self.worker_responses.values()
            ]
            
            return FinalResult(
                goal=goal,
                summary=f"Completed {successful_tasks} of {total_tasks} tasks",
                components=components,
                final_output=synthesis,
                tasks_completed=successful_tasks,
                tasks_total=total_tasks,
                success_rate=success_rate
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exception=e)
            raise


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Orchestrator-Workers Pattern Demo")
    print("=" * 60)
    
    orchestrator = Orchestrator()
    
    # Example goal
    goal = "Create a Python script that fetches weather data from an API and displays it in a formatted table"
    
    try:
        result = orchestrator.execute_goal(goal)
        
        print("\n" + "=" * 60)
        print("Final Result")
        print("=" * 60)
        print(f"\nGoal: {result.goal}")
        print(f"\nSummary: {result.summary}")
        print(f"\nTasks Completed: {result.tasks_completed}/{result.tasks_total}")
        print(f"Success Rate: {result.success_rate:.1%}")
        
        print(f"\nComponents:")
        for component in result.components:
            print(f"  - {component['task_id']} ({component['worker_type']}): {component['status']}")
        
        print(f"\nFinal Output:\n{result.final_output[:500]}...")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your API key to .env")
        print("3. Installed all dependencies: pip install -r requirements.txt")



