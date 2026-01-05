y# Product Requirements Document: AI Agent Learning Framework

## Introduction/Overview

This project creates a **complete, educational Python framework** for building AI agents based on industry best practices from Anthropic and OpenAI. The framework demonstrates the five core design patterns (Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer) and implements a full "cognitive architecture" including memory systems, tool integration, and observability.

**Problem it solves:** Developers learning AI agent development currently face fragmented resources and lack hands-on, production-quality examples. This framework provides a single, well-documented codebase that teaches concepts through working code.

**Goal:** Create a learning tool that is both educational (clear explanations, beginner-friendly) and functional (production-ready patterns you can adapt for real projects).

---

## Goals

1. **Educational Clarity:** Every pattern and component should include clear documentation and comments explaining *why* it works this way
2. **Complete Coverage:** Implement all 5 design patterns plus the full memory stack (episodic, semantic, procedural)
3. **Production-Ready Patterns:** Code should follow best practices (structured outputs, error handling, logging)
4. **Easy Experimentation:** Users should be able to run examples immediately with minimal setup
5. **Extensibility:** Framework should be modular so users can easily add their own tools and patterns

---

## User Stories

1. **As a beginner developer**, I want to see working examples of each design pattern so I can understand when to use workflows vs. agents.

2. **As a learner**, I want clear logging that shows me the agent's "thought process" so I can debug and understand what's happening at each step.

3. **As a Python developer**, I want to use the framework as a starting point for my own agent projects without needing to learn heavy frameworks like LangChain.

4. **As someone new to AI agents**, I want documentation that explains concepts in plain language with visual diagrams where helpful.

5. **As an experimenter**, I want to swap between different LLM providers (OpenAI, Anthropic) without rewriting my code.

---

## Functional Requirements

### Core Framework (Required)

1. **Project Structure:** The framework must be organized into clear modules:
   - `patterns/` - One file per design pattern (chaining, routing, etc.)
   - `memory/` - Memory system implementations (vector DB, knowledge graph)
   - `tools/` - Example tools agents can use
   - `utils/` - Shared utilities (LLM calls, logging, schemas)
   - `examples/` - Runnable demonstrations of each pattern
   - `tests/` - Evaluation scripts

2. **Generic LLM Interface:** The system must provide a single `call_llm()` function that:
   - Works with both OpenAI and Anthropic APIs
   - Enforces structured outputs using Pydantic models
   - Handles errors gracefully with retry logic
   - Logs all requests/responses for debugging

3. **Pattern Implementations:** Each of the 5 patterns must be implemented as standalone, runnable examples:
   - **Prompt Chaining:** Linear multi-step workflow (e.g., Outline → Draft → Edit)
   - **Routing:** Classify input and route to specialized handlers (e.g., Tech Support vs. Sales)
   - **Parallelization:** Execute independent tasks concurrently (e.g., content safety voting)
   - **Orchestrator-Workers:** Dynamic delegation to specialized sub-agents (e.g., coding assistant)
   - **Evaluator-Optimizer:** Iterative refinement loop (e.g., translation quality improvement)

4. **Memory Stack:** The framework must implement a simplified version of each memory type:
   - **Short-Term Memory:** Context window management (show how to summarize old messages)
   - **Episodic Memory:** Vector database for semantic search of past events (use ChromaDB or FAISS)
   - **Semantic Memory:** Simple knowledge graph (use NetworkX or dictionaries)
   - **Procedural Memory:** Template/snippet storage system

5. **Tool System:** Provide a clear pattern for defining and registering tools:
   - Example tools: `search_web()`, `read_file()`, `calculate()`, `get_weather()`
   - JSON schema generation for tool definitions
   - Safety wrappers (e.g., "human-in-the-loop" for destructive actions)

6. **ReAct Agent Loop:** Implement a complete Think → Act → Observe loop that:
   - Maintains conversation history
   - Decides when to use tools vs. respond directly
   - Has a configurable max iteration limit
   - Returns structured final outputs

7. **Observability/Logging:** Every agent step must be logged with:
   - Color-coded terminal output (using `rich` library)
   - Timestamps and execution duration
   - Tool calls and their results
   - Agent reasoning/thoughts
   - Export logs to JSON for later analysis

8. **Configuration Management:** Use a simple config file (YAML or `.env`) for:
   - API keys (with clear instructions on how to get them)
   - Model selection (GPT-4, Claude, etc.)
   - Default parameters (temperature, max tokens, etc.)

9. **Example Use Cases:** Provide at least 3 complete, runnable examples:
   - Example 1: Blog post generator (demonstrates Prompt Chaining)
   - Example 2: Customer support router (demonstrates Routing)
   - Example 3: Research assistant (demonstrates ReAct loop with memory)

10. **Documentation:** Include:
    - Main `README.md` with setup instructions and architecture overview
    - Inline code comments explaining key concepts
    - Architecture diagrams (can be simple ASCII art or links to diagrams)
    - A "When to use which pattern" decision tree

11. **Testing/Evaluation:** Provide evaluation tools:
    - Golden test set for each pattern
    - LLM-as-judge evaluation script
    - Performance metrics (latency, token usage)

### Quality Requirements

12. **Error Handling:** All LLM calls must handle network errors, rate limits, and invalid responses gracefully

13. **Type Safety:** Use Python type hints throughout and Pydantic for all structured data

14. **Dependencies:** Keep dependencies minimal (anthropic, openai, pydantic, rich, python-dotenv, chromadb/faiss, networkx)

15. **Python Version:** Target Python 3.10+ for modern type hint support

---

## Non-Goals (Out of Scope)

1. **Production Deployment:** This is a learning framework, not a production deployment system (no Docker, Kubernetes, API servers, etc.)

2. **Heavy Frameworks:** Will NOT use LangChain, LlamaIndex, or other high-level frameworks - the goal is to understand fundamentals

3. **Web UI:** No web interface - this is a command-line/programmatic framework (users can build UIs on top if needed)

4. **Multi-Agent Communication:** Will not implement agent-to-agent protocols or swarm intelligence

5. **Fine-Tuning:** Will not include model training or fine-tuning - only prompt engineering and architecture

6. **Enterprise Features:** No authentication, authorization, multi-tenancy, or billing systems

7. **Every Possible Tool:** Only provide 4-5 example tools - users will add their own

---

## Design Considerations

### File Structure
```
ai-agent-framework/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── patterns/
│   ├── __init__.py
│   ├── chaining.py
│   ├── routing.py
│   ├── parallelization.py
│   ├── orchestrator.py
│   └── evaluator.py
├── memory/
│   ├── __init__.py
│   ├── short_term.py
│   ├── episodic.py
│   ├── semantic.py
│   └── procedural.py
├── tools/
│   ├── __init__.py
│   ├── registry.py
│   ├── search.py
│   ├── file_ops.py
│   └── calculator.py
├── utils/
│   ├── __init__.py
│   ├── llm.py         # Generic LLM interface
│   ├── logging.py     # Rich-formatted logging
│   ├── schemas.py     # Common Pydantic models
│   └── config.py      # Config management
├── agents/
│   ├── __init__.py
│   └── react_agent.py # Main ReAct loop implementation
├── examples/
│   ├── blog_generator.py
│   ├── support_router.py
│   └── research_assistant.py
└── tests/
    ├── __init__.py
    ├── golden_sets.py
    └── evaluation.py
```

### Code Style
- Follow PEP 8 conventions
- Use descriptive variable names (prefer clarity over brevity)
- Add docstrings to all public functions
- Include type hints for all function signatures

### User Experience
- Every example should run with a single command: `python examples/blog_generator.py`
- Setup should be simple: `pip install -r requirements.txt` and add API key to `.env`
- Errors should give clear guidance (e.g., "API key not found. Copy .env.example to .env and add your key")

---

## Technical Considerations

1. **LLM Provider Abstraction:** Create a unified interface so users can switch between OpenAI and Anthropic with a config change

2. **Rate Limiting:** Implement exponential backoff for API errors

3. **Token Management:** Track and log token usage for cost awareness

4. **Vector Database Choice:** Use ChromaDB (easier setup) or FAISS (lighter weight) for episodic memory - make this configurable

5. **Async Support:** Consider making the framework async-compatible for parallel operations, but provide sync wrappers for simplicity

6. **Cross-Platform:** Ensure logging and file paths work on Windows, Mac, and Linux

---

## Success Metrics

1. **Completeness:** All 5 patterns implemented with working examples
2. **Clarity:** A beginner should be able to run the first example within 10 minutes of cloning the repo
3. **Reusability:** Users should be able to copy a pattern file and adapt it for their use case with minimal changes
4. **Learning Outcomes:** After using the framework, users should be able to:
   - Explain the difference between workflows and agents
   - Implement a basic ReAct loop from scratch
   - Choose the appropriate pattern for a given problem
   - Add custom tools to an agent

---

## Open Questions

1. **Vector Database:** Should we use ChromaDB (full-featured, heavier) or FAISS (lightweight, less features) for episodic memory?

2. **Visualization:** Would a simple web dashboard for viewing logs/traces add value, or keep it CLI-only?

3. **Model Defaults:** Should we default to GPT-4 Turbo (faster, cheaper) or Claude 3.5 Sonnet (better reasoning)?

4. **Community:** Should this include a way for users to share their custom patterns/tools (e.g., a `contrib/` folder)?

5. **Jupyter Notebooks:** Would Jupyter notebook versions of examples help with learning, or is Python scripts sufficient?

---

## Implementation Priority

**Phase 1 (MVP):**
- Core LLM interface (`utils/llm.py`)
- Prompt Chaining pattern
- Basic logging
- One complete example (blog generator)

**Phase 2:**
- Remaining 4 patterns
- Tool system
- ReAct agent loop

**Phase 3:**
- Memory stack (all 4 types)
- Evaluation framework
- Complete documentation

**Phase 4:**
- Polish, testing, examples refinement

