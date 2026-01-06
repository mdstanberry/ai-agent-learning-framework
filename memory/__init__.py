"""
Memory Stack Module

This module implements the four types of memory for AI agents, each serving
a distinct purpose in the agent's cognitive architecture.

## Memory Types Overview

### 1. Short-Term Memory (`short_term.py`)
**Purpose**: Manage immediate conversation context and context window
- **Use for**: Current conversation, session state, working memory
- **Storage**: In-memory conversation history
- **Retrieval**: Direct access to recent messages
- **Lifespan**: Single session

### 2. Episodic Memory (`episodic.py`)
**Purpose**: Store and retrieve past events and conversations
- **Use for**: Historical interactions, past conversations, event history
- **Storage**: Vector database (ChromaDB) with embeddings
- **Retrieval**: Semantic search over past events
- **Lifespan**: Long-term, persists across sessions

### 3. Semantic Memory (`semantic.py`)
**Purpose**: Store facts and relationships about entities
- **Use for**: User profiles, entity facts, knowledge base, relationships
- **Storage**: Knowledge graph (NetworkX)
- **Retrieval**: Graph traversal and fact queries
- **Lifespan**: Long-term, permanent facts

### 4. Procedural Memory (`procedural.py`)
**Purpose**: Store reusable templates, snippets, and procedures
- **Use for**: Code templates, snippets, SOPs, learned patterns
- **Storage**: File-based JSON storage
- **Retrieval**: Search by content, tags, category
- **Lifespan**: Long-term, reusable patterns

## Quick Decision Guide

**What are you storing?**
- Current conversation → **Short-Term Memory**
- Past events/conversations → **Episodic Memory**
- Facts about entities → **Semantic Memory**
- Reusable templates/procedures → **Procedural Memory**

**When does it matter?**
- Only this session → **Short-Term Memory**
- Historical/temporal → **Episodic Memory**
- Permanent facts → **Semantic Memory**
- Reusable patterns → **Procedural Memory**

## Memory Retrieval Order

According to .cursorrules, when retrieving information:
1. **Vector search** (Episodic Memory) for semantic matches
2. **Graph traversal** (Semantic Memory) for relationships
3. **Event log validation** for consistency

## Import Examples

```python
from memory.short_term import ShortTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory

# Initialize all memory types
short_term = ShortTermMemory()
episodic = EpisodicMemory()
semantic = SemanticMemory()
procedural = ProceduralMemory()
```
"""

