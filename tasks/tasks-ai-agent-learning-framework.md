# Task List: AI Agent Learning Framework

## Relevant Files

- `README.md` - Main project documentation with setup instructions and architecture overview
- `requirements.txt` - Python dependencies list
- `.env.example` - Template for environment variables (API keys)
- `config.yaml` - Configuration file for model selection and parameters
- `utils/__init__.py` - Utils module initialization
- `utils/llm.py` - Generic LLM interface supporting OpenAI and Anthropic
- `utils/agent_logging.py` - Rich-formatted logging system
- `utils/schemas.py` - Common Pydantic models for structured outputs
- `utils/config.py` - Configuration management
- `patterns/__init__.py` - Patterns module initialization
- `patterns/chaining.py` - Prompt Chaining pattern implementation
- `patterns/routing.py` - Routing pattern implementation
- `patterns/parallelization.py` - Parallelization pattern implementation
- `patterns/orchestrator.py` - Orchestrator-Workers pattern implementation
- `patterns/evaluator.py` - Evaluator-Optimizer pattern implementation
- `memory/__init__.py` - Memory module initialization
- `memory/short_term.py` - Short-term memory (context window management)
- `memory/episodic.py` - Episodic memory (vector database for past events)
- `memory/semantic.py` - Semantic memory (knowledge graph)
- `memory/procedural.py` - Procedural memory (templates/snippets)
- `tools/__init__.py` - Tools module initialization
- `tools/registry.py` - Tool registration and management system
- `tools/search.py` - Web search tool example
- `tools/file_ops.py` - File operations tool example
- `tools/calculator.py` - Calculator tool example
- `agents/__init__.py` - Agents module initialization
- `agents/react_agent.py` - Main ReAct loop implementation
- `examples/blog_generator.py` - Blog post generator example (demonstrates Prompt Chaining)
- `examples/support_router.py` - Customer support router example (demonstrates Routing)
- `examples/research_assistant.py` - Research assistant example (demonstrates ReAct loop with memory)
- `tests/__init__.py` - Tests module initialization
- `tests/golden_sets.py` - Golden test sets for each pattern
- `tests/evaluation.py` - LLM-as-judge evaluation script

### Notes

- This is a pure Python project (no web framework needed)
- Focus on clear documentation and comments throughout the code
- Each module should be runnable independently where possible
- Use type hints consistently throughout the codebase

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.1 Create and checkout a new branch for this feature (e.g., `git checkout -b feature/initial-framework`)
  
- [x] 1.0 Set up project structure and core dependencies
  - [x] 1.1 Create all necessary directories (patterns/, memory/, tools/, utils/, agents/, examples/, tests/)
  - [x] 1.2 Create `__init__.py` files in each module directory
  - [x] 1.3 Create `requirements.txt` with all dependencies (anthropic, openai, pydantic, rich, python-dotenv, chromadb, networkx)
  - [x] 1.4 Create `.env.example` template file with placeholder API keys and configuration
  - [x] 1.5 Create `config.yaml` with default model settings and parameters
  - [x] 1.6 Test that the project structure is correct by running `python -m patterns` (should import without errors)
  
- [x] 2.0 Implement core utilities (LLM interface, logging, configuration)
  - [x] 2.1 Implement `utils/config.py` - Load configuration from .env and config.yaml files
  - [x] 2.2 Implement `utils/schemas.py` - Define common Pydantic models (Message, ToolCall, AgentResponse, etc.)
  - [x] 2.3 Implement `utils/agent_logging.py` - Create Rich-formatted logger with color-coded output (Thought=Blue, Action=Yellow, Observation=Green)
  - [x] 2.4 Implement `utils/llm.py` - Create generic `call_llm()` function that supports both OpenAI and Anthropic APIs
  - [x] 2.5 Add error handling and retry logic with exponential backoff to `call_llm()`
  - [x] 2.6 Add token tracking and logging to `call_llm()`
  - [x] 2.7 Test the LLM interface with a simple call to ensure API connectivity
  
- [x] 3.0 Implement Prompt Chaining pattern
  - [x] 3.1 Create `patterns/chaining.py` file
  - [x] 3.2 Define Pydantic models for chain steps (Outline, BlogPost, EditedPost)
  - [x] 3.3 Implement Step 1: Generate outline from topic
  - [x] 3.4 Implement validation gate between steps (check outline has minimum sections)
  - [x] 3.5 Implement Step 2: Generate full blog post from outline
  - [x] 3.6 Implement Step 3: Edit and refine the blog post
  - [x] 3.7 Add comprehensive docstrings explaining the chaining pattern
  - [x] 3.8 Test the chaining pattern with a sample topic
  
- [x] 4.0 Implement Routing pattern
  - [x] 4.1 Create `patterns/routing.py` file
  - [x] 4.2 Define Pydantic models for routing (QueryClassification, TechSupportResponse, SalesResponse)
  - [x] 4.3 Implement router function that classifies incoming queries
  - [x] 4.4 Implement tech support handler (specialized agent for technical questions)
  - [x] 4.5 Implement sales handler (specialized agent for sales inquiries)
  - [x] 4.6 Implement general handler (fallback for unclassified queries)
  - [x] 4.7 Add comprehensive docstrings explaining the routing pattern
  - [x] 4.8 Test with sample queries from different categories
  
- [x] 5.0 Implement Parallelization pattern
  - [x] 5.1 Create `patterns/parallelization.py` file
  - [x] 5.2 Define Pydantic models for parallel tasks (SafetyVote, AggregatedResult)
  - [x] 5.3 Implement parallel task execution using asyncio or concurrent.futures
  - [x] 5.4 Implement content safety voting system (multiple agents vote on content safety)
  - [x] 5.5 Implement result aggregation logic (combine votes into final decision)
  - [x] 5.6 Add comprehensive docstrings explaining the parallelization pattern
  - [x] 5.7 Test with sample content that should trigger different votes
  
- [x] 6.0 Implement Orchestrator-Workers pattern
  - [x] 6.1 Create `patterns/orchestrator.py` file
  - [x] 6.2 Define Pydantic models for orchestration (Task, WorkerResponse, FinalResult)
  - [x] 6.3 Implement orchestrator agent that breaks down complex tasks
  - [x] 6.4 Implement worker agents with different specializations (e.g., researcher, coder, reviewer)
  - [x] 6.5 Implement task delegation logic (orchestrator assigns tasks to appropriate workers)
  - [x] 6.6 Implement result synthesis (orchestrator combines worker outputs)
  - [x] 6.7 Add comprehensive docstrings explaining the orchestrator-workers pattern
  - [x] 6.8 Test with a complex task that requires multiple worker types
  
- [x] 7.0 Implement Evaluator-Optimizer pattern
  - [x] 7.1 Create `patterns/evaluator.py` file
  - [x] 7.2 Define Pydantic models for evaluation (Translation, Evaluation, ImprovedTranslation)
  - [x] 7.3 Implement generator function that creates initial output
  - [x] 7.4 Implement evaluator function that scores output quality
  - [x] 7.5 Implement optimizer function that improves output based on evaluation
  - [x] 7.6 Implement iterative loop with max iterations and quality threshold
  - [x] 7.7 Add comprehensive docstrings explaining the evaluator-optimizer pattern
  - [x] 7.8 Test with sample text that needs quality improvement
  
- [x] 8.0 Implement tool system
  - [x] 8.1 Create `tools/registry.py` - Tool registration and management system
  - [x] 8.2 Define tool schema format (name, description, parameters, function)
  - [x] 8.3 Implement `tools/search.py` - Mock web search tool
  - [x] 8.4 Implement `tools/file_ops.py` - File reading tool (with safety checks)
  - [x] 8.5 Implement `tools/calculator.py` - Calculator tool for mathematical operations
  - [x] 8.6 Implement tool JSON schema generation for LLM tool calling
  - [x] 8.7 Implement safety wrapper for destructive tools (human-in-the-loop confirmation)
  - [x] 8.8 Add comprehensive docstrings explaining how to add custom tools
  - [x] 8.9 Test each tool individually to ensure correct functionality
  
- [x] 9.0 Implement ReAct agent loop
  - [x] 9.1 Create `agents/react_agent.py` file
  - [x] 9.2 Define Pydantic models for agent state (AgentState, ThoughtAction, Observation)
  - [x] 9.3 Implement conversation history management
  - [x] 9.4 Implement Think step - Agent decides what to do next
  - [x] 9.5 Implement Act step - Agent calls a tool or responds directly
  - [x] 9.6 Implement Observe step - Process tool results and update context
  - [x] 9.7 Implement loop control (max iterations, stopping conditions)
  - [x] 9.8 Integrate logging to show agent's thought process at each step
  - [x] 9.9 Add comprehensive docstrings explaining the ReAct loop pattern
  - [x] 9.10 Test with a simple goal that requires multiple tool calls
  
- [ ] 10.0 Implement memory stack (short-term, episodic, semantic, procedural)
  - [x] 10.1 Create `memory/short_term.py` - Context window management
  - [ ] 10.2 Implement message summarization for when context gets too long
  - [ ] 10.3 Create `memory/episodic.py` - Vector database integration (ChromaDB)
  - [ ] 10.4 Implement semantic search over past events/conversations
  - [ ] 10.5 Create `memory/semantic.py` - Simple knowledge graph using NetworkX
  - [ ] 10.6 Implement methods to store and query facts about users/entities
  - [ ] 10.7 Create `memory/procedural.py` - Template and snippet storage
  - [ ] 10.8 Implement methods to store and retrieve standard operating procedures
  - [ ] 10.9 Add comprehensive docstrings explaining each memory type and when to use it
  - [ ] 10.10 Test each memory type with sample data
  
- [ ] 11.0 Create example applications
  - [ ] 11.1 Create `examples/blog_generator.py` - Demonstrates Prompt Chaining pattern
  - [ ] 11.2 Add clear comments and print statements showing each step of blog generation
  - [ ] 11.3 Make blog_generator runnable with: `python examples/blog_generator.py`
  - [ ] 11.4 Create `examples/support_router.py` - Demonstrates Routing pattern
  - [ ] 11.5 Add clear comments and example queries for different route types
  - [ ] 11.6 Make support_router runnable with: `python examples/support_router.py`
  - [ ] 11.7 Create `examples/research_assistant.py` - Demonstrates ReAct loop with memory
  - [ ] 11.8 Add integration of tools and memory systems
  - [ ] 11.9 Make research_assistant runnable with: `python examples/research_assistant.py`
  - [ ] 11.10 Test all three examples end-to-end to ensure they work correctly
  
- [ ] 12.0 Create documentation
  - [ ] 12.1 Create main `README.md` with project overview and purpose
  - [ ] 12.2 Add "Quick Start" section with setup instructions (pip install, .env setup)
  - [ ] 12.3 Add "Architecture Overview" section explaining the cognitive architecture
  - [ ] 12.4 Add "Design Patterns" section with descriptions of when to use each
  - [ ] 12.5 Create decision tree diagram (ASCII art) for "Workflow vs. Agent"
  - [ ] 12.6 Add "Memory Stack" section explaining each memory type
  - [ ] 12.7 Add "Adding Custom Tools" section with step-by-step guide
  - [ ] 12.8 Add "Examples" section with descriptions and expected outputs
  - [ ] 12.9 Add "Troubleshooting" section for common issues (API keys, dependencies, etc.)
  - [ ] 12.10 Review all inline code comments for clarity and completeness
  
- [ ] 13.0 Create evaluation and testing framework
  - [ ] 13.1 Create `tests/golden_sets.py` file
  - [ ] 13.2 Define golden test cases for Prompt Chaining pattern (5 test inputs)
  - [ ] 13.3 Define golden test cases for Routing pattern (5 test inputs)
  - [ ] 13.4 Define golden test cases for Parallelization pattern (5 test inputs)
  - [ ] 13.5 Define golden test cases for Orchestrator pattern (5 test inputs)
  - [ ] 13.6 Define golden test cases for Evaluator pattern (5 test inputs)
  - [ ] 13.7 Create `tests/evaluation.py` - LLM-as-judge evaluation script
  - [ ] 13.8 Implement evaluation criteria (accuracy, style, completeness) for each pattern
  - [ ] 13.9 Implement performance metrics tracking (latency, token usage, cost)
  - [ ] 13.10 Run all golden test sets and generate evaluation report
  - [ ] 13.11 Document expected scores and performance benchmarks

