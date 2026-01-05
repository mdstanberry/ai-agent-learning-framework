# AI Agent Learning Framework

A comprehensive, educational Python framework for building AI agents based on industry best practices from Anthropic and OpenAI. This framework demonstrates the five core design patterns (Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer) and implements a full "cognitive architecture" including memory systems, tool integration, and observability.

## Purpose

This framework is designed to help developers learn how to build AI agents by providing:
- **Working examples** of all major agent design patterns
- **Complete memory stack** implementation (short-term, episodic, semantic, procedural)
- **Tool system** with safety wrappers and registry
- **ReAct agent loop** implementation
- **Comprehensive documentation** and examples

**Target Audience**: Beginners and intermediate developers who want to understand AI agent architecture without heavy framework abstractions.

## Features

- ✅ **Five Design Patterns**: Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer
- ✅ **Memory Stack**: Four types of memory (short-term, episodic, semantic, procedural)
- ✅ **Tool System**: Registry, safety wrappers, and example tools (search, calculator, file ops)
- ✅ **ReAct Agent**: Think → Act → Observe loop with tool calling
- ✅ **LLM Support**: OpenAI and Anthropic APIs with structured outputs
- ✅ **Observability**: Rich-formatted logging with color-coded agent thoughts
- ✅ **Examples**: Three complete example applications

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI or Anthropic API key

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/mdstanberry/ai-agent-learning-framework.git
cd ai-agent-learning-framework
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Get OpenAI API key: https://platform.openai.com/api-keys
# Get Anthropic API key: https://console.anthropic.com/settings/keys
```

4. **Configure your provider** (in `.env`):
```bash
LLM_PROVIDER=anthropic  # or "openai"
DEFAULT_MODEL=claude-3-5-sonnet-20241022  # or "gpt-4-turbo"
```

### Running Examples

```bash
# Blog Generator (Prompt Chaining)
python examples/blog_generator.py

# Support Router (Routing)
python examples/support_router.py

# Research Assistant (ReAct Loop)
python examples/research_assistant.py
```

## Architecture Overview

### Cognitive Architecture

AI agents need a "cognitive architecture" - a structured way to think, remember, and act. This framework implements:

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Profile                        │
│  (System prompt, role, constraints, capabilities)      │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Memory     │  │   Planning   │  │    Tools     │
│              │  │              │  │              │
│ - Short-term │  │ - Goal       │  │ - Search     │
│ - Episodic   │  │ - Steps      │  │ - Calculator │
│ - Semantic   │  │ - Validation │  │ - File Ops   │
│ - Procedural │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Workflow vs. Agent Decision Tree

```
                    Start: What problem are you solving?
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            Can you predict                    Cannot predict
            all steps?                         all steps?
                    │                             │
                    ▼                             ▼
        ┌───────────────────┐         ┌──────────────────┐
        │   Use Workflow    │         │   Use Agent      │
        │   (Patterns)      │         │   (ReAct Loop)   │
        └───────────────────┘         └──────────────────┘
                │                             │
        ┌───────┴────────┐                    │
        │                │                    │
        ▼                ▼                    ▼
┌─────────────┐  ┌─────────────┐      ┌──────────────┐
│   Simple    │  │  Complex    │      │  Open-ended  │
│   Linear    │  │  Multi-step │      │  Problems     │
│             │  │             │      │               │
│ Prompt      │  │ Orchestrator│      │ ReAct Agent  │
│ Chaining    │  │ - Workers   │      │ with Tools    │
│             │  │             │      │               │
│ Routing     │  │ Evaluator-  │      │               │
│             │  │ Optimizer   │      │               │
│ Parallel-   │  │             │      │               │
│ ization     │  │             │      │               │
└─────────────┘  └─────────────┘      └──────────────┘
```

**Decision Guidelines:**
- **Workflow (Patterns)**: Use when you know the steps ahead of time
  - Simple linear tasks → Prompt Chaining
  - Classification tasks → Routing
  - Independent parallel tasks → Parallelization
  - Complex multi-step → Orchestrator-Workers
  - Quality improvement → Evaluator-Optimizer

- **Agent (ReAct Loop)**: Use when steps cannot be predicted
  - Open-ended research questions
  - Problems requiring exploration
  - Tasks needing tool discovery
  - Dynamic problem-solving

## Design Patterns

### 1. Prompt Chaining

**When to use**: Linear multi-step workflows where each step builds on the previous one.

**Example**: Blog post generation (Outline → Draft → Edit)

```python
from patterns.chaining import PromptChain

chain = PromptChain()
outline = chain.generate_outline("AI Agents")
blog_post = chain.generate_blog_post(outline)
edited_post = chain.edit_blog_post(blog_post)
```

**Key Features**:
- Sequential steps with validation gates
- Each step uses structured outputs (Pydantic models)
- Automatic validation between steps

### 2. Routing

**When to use**: Classifying inputs and routing to specialized handlers.

**Example**: Customer support router (Tech Support, Sales, Billing, General)

```python
from patterns.routing import QueryRouter

router = QueryRouter()
classification = router.route_query("How do I reset my password?")
response = router.get_handler_response(classification)
```

**Key Features**:
- LLM-based classification with confidence scores
- Specialized handlers for each category
- Fallback to general handler

### 3. Parallelization

**When to use**: Executing independent tasks concurrently.

**Example**: Content safety voting (multiple agents vote independently)

```python
from patterns.parallelization import ParallelExecutor

executor = ParallelExecutor()
votes = await executor.execute_parallel(tasks)
result = executor.aggregate_results(votes)
```

**Key Features**:
- Concurrent task execution
- Result aggregation with confidence weighting
- Async/await support

### 4. Orchestrator-Workers

**When to use**: Breaking down complex tasks and delegating to specialized workers.

**Example**: Research project (researcher, coder, reviewer, writer)

```python
from patterns.orchestrator import Orchestrator

orchestrator = Orchestrator()
breakdown = orchestrator.break_down_task("Build a web scraper")
result = orchestrator.execute_with_workers(breakdown)
```

**Key Features**:
- Dynamic task breakdown
- Worker specialization (researcher, coder, reviewer, etc.)
- Result synthesis

### 5. Evaluator-Optimizer

**When to use**: Iterative refinement to improve output quality.

**Example**: Translation improvement loop

```python
from patterns.evaluator import EvaluatorOptimizer

optimizer = EvaluatorOptimizer()
result = optimizer.refine(
    initial_output="Hello world",
    max_iterations=5,
    quality_threshold=0.8
)
```

**Key Features**:
- Quality scoring
- Iterative improvement
- Loop control (max iterations, quality threshold)

## Memory Stack

The framework implements four types of memory, each serving a distinct purpose:

### 1. Short-Term Memory

**Purpose**: Manage immediate conversation context and context window

**Use for**: Current conversation, session state, working memory

**Example**:
```python
from memory.short_term import ShortTermMemory

memory = ShortTermMemory()
memory.add_message("user", "What is Python?")
memory.add_message("assistant", "Python is a programming language...")
context = memory.get_context_for_llm(system_prompt, retrieved_context)
```

**Features**:
- Conversation history management
- Token budget tracking (80% context, 20% output)
- Automatic summarization when context gets too long
- Message prioritization (system > retrieved context > conversation)

### 2. Episodic Memory

**Purpose**: Store and retrieve past events and conversations

**Use for**: Historical interactions, past conversations, event history

**Example**:
```python
from memory.episodic import EpisodicMemory

memory = EpisodicMemory()
memory.store_event(
    content="User asked about Python file operations",
    event_type="conversation",
    metadata={"topic": "python"}
)
results = memory.search("Python programming help", limit=5)
```

**Features**:
- Vector database (ChromaDB) for semantic search
- Timestamp-based filtering
- Metadata storage

### 3. Semantic Memory

**Purpose**: Store facts and relationships about entities

**Use for**: User profiles, entity facts, knowledge base, relationships

**Example**:
```python
from memory.semantic import SemanticMemory

memory = SemanticMemory()
memory.add_fact("user123", "name", "Alice", entity_type="user")
memory.add_relationship("user123", "product456", "purchased")
related = memory.find_related_entities("user123", max_depth=2)
```

**Features**:
- Knowledge graph (NetworkX)
- Fact storage (entity properties)
- Relationship storage (entity connections)
- Graph traversal for finding related entities

### 4. Procedural Memory

**Purpose**: Store reusable templates, snippets, and procedures

**Use for**: Code templates, snippets, SOPs, learned patterns

**Example**:
```python
from memory.procedural import ProceduralMemory

memory = ProceduralMemory()
memory.store_template(
    name="email_greeting",
    content="Hello {{name}}, thank you!",
    category="email"
)
memory.store_snippet(
    name="json_parser",
    content="import json\ndata = json.loads(json_string)",
    language="python"
)
results = memory.search("json", item_type="snippet")
```

**Features**:
- Template storage with placeholders
- Code snippet storage by language
- Standard operating procedures (SOPs)
- Search by content, tags, category

### Memory Decision Guide

**What are you storing?**
- Current conversation → **Short-Term Memory**
- Past events/conversations → **Episodic Memory**
- Facts about entities → **Semantic Memory**
- Reusable templates/procedures → **Procedural Memory**

## Adding Custom Tools

Tools allow agents to interact with the world. Here's how to add your own:

### Step 1: Create Your Tool Function

```python
# tools/my_tool.py
from typing import Dict, Any
from tools.registry import tool_registry, ToolSafetyLevel, ToolParameter
from utils.agent_logging import logger

def my_custom_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Description of what your tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Dictionary with results
    """
    logger.info(f"Executing my_custom_tool with {param1}, {param2}")
    
    # Your tool logic here
    result = {"status": "success", "data": f"Processed {param1}"}
    
    logger.observation(f"Tool completed: {result}")
    return result
```

### Step 2: Register Your Tool

```python
# At the end of tools/my_tool.py
def register_my_tool():
    tool_registry.register(
        name="my_custom_tool",
        description="What your tool does and when to use it",
        function=my_custom_tool,
        parameters=[
            ToolParameter(
                name="param1",
                type="string",
                description="Description of param1",
                required=True
            ),
            ToolParameter(
                name="param2",
                type="number",
                description="Description of param2",
                required=False
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,  # or MODERATE, DESTRUCTIVE
        category="custom"
    )

# Auto-register when module is imported
register_my_tool()
```

### Step 3: Use Your Tool in an Agent

```python
from agents.react_agent import ReActAgent

agent = ReActAgent(
    system_prompt="You are a helpful assistant with access to custom tools.",
    max_iterations=5
)

# The tool will be automatically available to the agent
result = agent.run("Use my_custom_tool to process something")
```

### Safety Levels

- **SAFE**: No side effects, read-only operations (e.g., search, calculator)
- **MODERATE**: Some side effects, but reversible (e.g., create draft)
- **DESTRUCTIVE**: Permanent changes (e.g., delete file, send email)

Destructive tools require human confirmation by default (see `tools/safety.py`).

## Examples

### Blog Generator (`examples/blog_generator.py`)

Demonstrates **Prompt Chaining** pattern with a 3-step workflow:
1. Generate outline from topic
2. Generate full blog post from outline
3. Edit and refine the blog post

**Run**:
```bash
python examples/blog_generator.py
```

**Expected Output**:
- Outline with sections and key points
- Full blog post generated from outline
- Edited and refined final post

### Support Router (`examples/support_router.py`)

Demonstrates **Routing** pattern with query classification:
- Classifies queries into categories (Tech Support, Sales, Billing, General)
- Routes to specialized handlers
- Returns category-specific responses

**Run**:
```bash
python examples/support_router.py
```

**Expected Output**:
- Query classifications with confidence scores
- Specialized handler responses for each category
- Multiple example queries demonstrating different routes

### Research Assistant (`examples/research_assistant.py`)

Demonstrates **ReAct loop** with full memory integration:
- Uses all 4 memory types
- Calls tools (search, calculator, file operations)
- Shows Think → Act → Observe cycle

**Run**:
```bash
python examples/research_assistant.py
```

**Expected Output**:
- Agent thoughts and reasoning
- Tool calls and observations
- Final answers with memory statistics

## Troubleshooting

### API Key Issues

**Problem**: `ValueError: OPENAI_API_KEY not found`

**Solution**:
1. Copy `.env.example` to `.env`
2. Add your API key to `.env`:
   ```
   OPENAI_API_KEY=sk-your-key-here
   # or
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
3. Set `LLM_PROVIDER` in `.env` to match your API key

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'patterns'`

**Solution**:
1. Make sure you're in the project root directory
2. Install dependencies: `pip install -r requirements.txt`
3. Check that all `__init__.py` files exist in module directories

### ChromaDB Download Issues

**Problem**: Long download time or errors when initializing Episodic Memory

**Solution**:
- ChromaDB downloads embedding models on first use (~79MB)
- This is a one-time download
- Subsequent runs will be faster
- If download fails, check your internet connection

### Unicode Encoding Errors (Windows)

**Problem**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution**:
- The framework uses ASCII-friendly characters by default
- If you see encoding errors, ensure your terminal supports UTF-8
- Or set environment variable: `PYTHONIOENCODING=utf-8`

### Memory Not Persisting

**Problem**: Episodic or Semantic memory data disappears between runs

**Solution**:
- Check that `chroma_db/` directory exists and is writable
- Check that `knowledge_graph.json` file exists and is writable
- Ensure `memory.episodic.enabled` and `memory.semantic.enabled` are `true` in `config.yaml`

### Tool Not Found

**Problem**: `ToolNotFoundError: Tool 'my_tool' not found`

**Solution**:
1. Make sure the tool module is imported before using the agent
2. Check that `register_my_tool()` is called
3. Verify the tool name matches exactly (case-sensitive)

## Project Structure

```
ai-agent-learning-framework/
├── agents/              # ReAct agent implementation
│   └── react_agent.py
├── examples/            # Example applications
│   ├── blog_generator.py
│   ├── support_router.py
│   └── research_assistant.py
├── memory/              # Memory stack implementation
│   ├── short_term.py
│   ├── episodic.py
│   ├── semantic.py
│   └── procedural.py
├── patterns/            # Design pattern implementations
│   ├── chaining.py
│   ├── routing.py
│   ├── parallelization.py
│   ├── orchestrator.py
│   └── evaluator.py
├── tools/               # Tool system
│   ├── registry.py
│   ├── safety.py
│   ├── search.py
│   ├── calculator.py
│   └── file_ops.py
├── utils/               # Core utilities
│   ├── llm.py
│   ├── agent_logging.py
│   ├── config.py
│   └── schemas.py
├── tests/               # Tests and evaluation
│   └── test_memory_stack.py
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md            # This file
```

## Contributing

This is an educational framework. Contributions that improve clarity, add examples, or fix bugs are welcome!

## License

MIT License - See LICENSE file for details

## Resources

- [Anthropic's Building Effective Agents Guide](https://docs.anthropic.com/claude/docs/agents)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.

