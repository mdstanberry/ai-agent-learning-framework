# Task List: AI Agent Learning Framework

## Relevant Files

- `README.md` - Main project documentation with setup instructions and architecture overview
- `requirements.txt` - Python dependencies list
- `.env.example` - Template for environment variables (API keys)
- `config.yaml` - Configuration file for model selection and parameters
- `utils/__init__.py` - Utils module initialization
- `utils/llm.py` - Generic LLM interface supporting OpenAI and Anthropic
- `utils/logging.py` - Rich-formatted logging system
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

- [ ] 0.0 Create feature branch
- [ ] 1.0 Set up project structure and core dependencies
- [ ] 2.0 Implement core utilities (LLM interface, logging, configuration)
- [ ] 3.0 Implement Prompt Chaining pattern
- [ ] 4.0 Implement Routing pattern
- [ ] 5.0 Implement Parallelization pattern
- [ ] 6.0 Implement Orchestrator-Workers pattern
- [ ] 7.0 Implement Evaluator-Optimizer pattern
- [ ] 8.0 Implement tool system
- [ ] 9.0 Implement ReAct agent loop
- [ ] 10.0 Implement memory stack (short-term, episodic, semantic, procedural)
- [ ] 11.0 Create example applications
- [ ] 12.0 Create documentation
- [ ] 13.0 Create evaluation and testing framework

