"""
Procedural Memory Module

This module implements procedural memory for AI agents - storing and retrieving
templates, code snippets, and standard operating procedures.

## When to Use Procedural Memory

Procedural memory is used for:
- **Reusable templates**: Text or code patterns with placeholders
- **Code snippets**: Reusable code blocks for common tasks
- **Standard operating procedures**: Step-by-step workflows and processes
- **Learned patterns**: Procedures that the agent has learned work well
- **Best practices**: Templates and patterns that represent good solutions

**Use procedural memory when:**
- You have reusable templates that can be filled in with different data
- You want to store code snippets for common programming tasks
- You need to remember step-by-step procedures for complex tasks
- You want to store patterns that have worked well in the past
- You need to quickly retrieve standard workflows or templates

**Do NOT use procedural memory for:**
- Current conversation context (use Short-Term Memory)
- Past events or conversations (use Episodic Memory)
- Facts about entities (use Semantic Memory)
- Information that changes frequently or is session-specific

## How It Works

According to .cursorrules:
- Store user profiles as: preferences (key-value), episodic interactions
  (timestamped events), and learned patterns (procedural memory).
  Never mix these stores. [file:3][file:5]

The module stores:
- **Templates**: Reusable text/code patterns (e.g., email templates, prompt templates)
- **Snippets**: Code snippets organized by language and category
- **Procedures**: Step-by-step workflows (SOPs) for complex tasks

All items are:
- Organized by category and tags for easy searching
- Versioned to track changes over time
- Persisted to disk for long-term storage
- Searchable by content, tags, category, or name

## Example Usage

```python
from memory.procedural import ProceduralMemory

memory = ProceduralMemory()

# Store a template
memory.store_template(
    name="email_greeting",
    content="Hello {{name}}, thank you for your interest!",
    category="email",
    tags=["greeting", "customer-service"]
)

# Store a code snippet
memory.store_snippet(
    name="json_parser",
    content="import json\ndata = json.loads(json_string)",
    language="python",
    category="utilities"
)

# Store a procedure
memory.store_procedure(
    name="user_onboarding",
    steps=["1. Create account", "2. Send email", "3. Set preferences"],
    category="workflow"
)

# Search for items
results = memory.search("json", item_type="snippet")
```

## Comparison with Other Memory Types

- **vs. Short-Term Memory**: Procedural stores reusable patterns; Short-term is current context
- **vs. Episodic Memory**: Procedural stores how to do things; Episodic stores what happened
- **vs. Semantic Memory**: Procedural stores processes; Semantic stores facts
"""

from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from pathlib import Path
import json
from utils.agent_logging import logger
from utils.config import config


def safe_log(level: str, message: str, **kwargs) -> None:
    """
    Safely log a message, handling Unicode encoding errors.
    
    Args:
        level: Log level (info, error, warning, etc.)
        message: Message to log
        **kwargs: Additional arguments for logger methods
    """
    try:
        log_method = getattr(logger, level, logger.info)
        log_method(message, **kwargs)
    except (UnicodeEncodeError, Exception):
        # Fallback to simple print for Windows console issues
        print(f"[{level.upper()}] {message}")


class ProceduralMemory:
    """
    Procedural memory for storing templates, snippets, and standard operating procedures.
    
    This stores:
    - Code templates and snippets
    - Standard operating procedures (SOPs)
    - Learned patterns and workflows
    - Reusable code blocks
    
    All stored with metadata for easy retrieval.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize procedural memory.
        
        Args:
            storage_path: Path to storage directory (defaults to config)
            enabled: Whether procedural memory is enabled (defaults to config)
        """
        self.storage_path = Path(storage_path or config.get_nested(
            "memory.procedural.templates_path",
            "./templates"
        ))
        self.enabled = enabled if enabled is not None else config.get_nested(
            "memory.procedural.enabled",
            True
        )
        
        # In-memory storage
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.snippets: Dict[str, Dict[str, Any]] = {}
        self.procedures: Dict[str, Dict[str, Any]] = {}
        
        if not self.enabled:
            safe_log("info", "Procedural memory is disabled")
            return
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing templates and snippets
        self._load_all()
        
        total_items = len(self.templates) + len(self.snippets) + len(self.procedures)
        safe_log("info", f"Procedural memory initialized: {total_items} items loaded")
    
    def store_template(
        self,
        name: str,
        content: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a template.
        
        Templates are reusable code/text patterns with placeholders.
        
        Args:
            name: Template name/ID
            content: Template content
            category: Optional category (e.g., "email", "code", "prompt")
            tags: Optional list of tags for searching
            description: Optional description
            metadata: Optional additional metadata
            
        Returns:
            Template ID
        """
        if not self.enabled:
            return ""
        
        template = {
            "name": name,
            "content": content,
            "category": category or "general",
            "tags": tags or [],
            "description": description or "",
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1
        }
        
        # Check if template exists (for versioning)
        if name in self.templates:
            template["version"] = self.templates[name].get("version", 1) + 1
        
        self.templates[name] = template
        self._save_template(name, template)
        
        safe_log("info", f"Stored template: {name} (version {template['version']})")
        return name
    
    def store_snippet(
        self,
        name: str,
        content: str,
        language: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a code snippet.
        
        Snippets are reusable code blocks.
        
        Args:
            name: Snippet name/ID
            content: Snippet content
            language: Optional programming language (e.g., "python", "javascript")
            category: Optional category
            tags: Optional list of tags for searching
            description: Optional description
            metadata: Optional additional metadata
            
        Returns:
            Snippet ID
        """
        if not self.enabled:
            return ""
        
        snippet = {
            "name": name,
            "content": content,
            "language": language or "text",
            "category": category or "general",
            "tags": tags or [],
            "description": description or "",
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1
        }
        
        # Check if snippet exists (for versioning)
        if name in self.snippets:
            snippet["version"] = self.snippets[name].get("version", 1) + 1
        
        self.snippets[name] = snippet
        self._save_snippet(name, snippet)
        
        safe_log("info", f"Stored snippet: {name} (version {snippet['version']})")
        return name
    
    def store_procedure(
        self,
        name: str,
        steps: List[str],
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a standard operating procedure (SOP).
        
        Procedures are step-by-step workflows.
        
        Args:
            name: Procedure name/ID
            steps: List of procedure steps
            category: Optional category
            tags: Optional list of tags for searching
            description: Optional description
            metadata: Optional additional metadata
            
        Returns:
            Procedure ID
        """
        if not self.enabled:
            return ""
        
        procedure = {
            "name": name,
            "steps": steps,
            "category": category or "general",
            "tags": tags or [],
            "description": description or "",
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1
        }
        
        # Check if procedure exists (for versioning)
        if name in self.procedures:
            procedure["version"] = self.procedures[name].get("version", 1) + 1
        
        self.procedures[name] = procedure
        self._save_procedure(name, procedure)
        
        safe_log("info", f"Stored procedure: {name} (version {procedure['version']})")
        return name
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a template by name.
        
        Args:
            name: Template name/ID
            
        Returns:
            Template dictionary or None if not found
        """
        if not self.enabled:
            return None
        return self.templates.get(name)
    
    def get_snippet(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a snippet by name.
        
        Args:
            name: Snippet name/ID
            
        Returns:
            Snippet dictionary or None if not found
        """
        if not self.enabled:
            return None
        return self.snippets.get(name)
    
    def get_procedure(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a procedure by name.
        
        Args:
            name: Procedure name/ID
            
        Returns:
            Procedure dictionary or None if not found
        """
        if not self.enabled:
            return None
        return self.procedures.get(name)
    
    def search(
        self,
        query: str,
        item_type: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search templates, snippets, and procedures.
        
        Args:
            query: Search query (searches in name, content, description)
            item_type: Optional filter by type ("template", "snippet", "procedure")
            category: Optional filter by category
            tags: Optional filter by tags (all tags must match)
            language: Optional filter by language (for snippets)
            
        Returns:
            List of matching items
        """
        if not self.enabled:
            return []
        
        query_lower = query.lower()
        results = []
        
        # Search templates
        if item_type is None or item_type == "template":
            for name, template in self.templates.items():
                if self._matches_search(template, query_lower, category, tags):
                    results.append({**template, "item_type": "template"})
        
        # Search snippets
        if item_type is None or item_type == "snippet":
            for name, snippet in self.snippets.items():
                if self._matches_search(snippet, query_lower, category, tags):
                    if language and snippet.get("language") != language:
                        continue
                    results.append({**snippet, "item_type": "snippet"})
        
        # Search procedures
        if item_type is None or item_type == "procedure":
            for name, procedure in self.procedures.items():
                if self._matches_search(procedure, query_lower, category, tags):
                    results.append({**procedure, "item_type": "procedure"})
        
        return results
    
    def _matches_search(
        self,
        item: Dict[str, Any],
        query: str,
        category: Optional[str],
        tags: Optional[List[str]]
    ) -> bool:
        """Check if an item matches search criteria."""
        # Category filter
        if category and item.get("category") != category:
            return False
        
        # Tags filter (all tags must be present)
        if tags:
            item_tags = set(item.get("tags", []))
            if not all(tag in item_tags for tag in tags):
                return False
        
        # Query search (in name, content, description)
        if query:
            query_matches = (
                query in item.get("name", "").lower() or
                query in item.get("content", "").lower() or
                query in item.get("description", "").lower()
            )
            if not query_matches:
                return False
        
        return True
    
    def list_all(
        self,
        item_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all stored items.
        
        Args:
            item_type: Optional filter by type ("template", "snippet", "procedure")
            category: Optional filter by category
            
        Returns:
            List of items (summary info only)
        """
        if not self.enabled:
            return []
        
        results = []
        
        if item_type is None or item_type == "template":
            for name, template in self.templates.items():
                if not category or template.get("category") == category:
                    results.append({
                        "name": name,
                        "item_type": "template",
                        "category": template.get("category"),
                        "description": template.get("description"),
                        "tags": template.get("tags", [])
                    })
        
        if item_type is None or item_type == "snippet":
            for name, snippet in self.snippets.items():
                if not category or snippet.get("category") == category:
                    results.append({
                        "name": name,
                        "item_type": "snippet",
                        "language": snippet.get("language"),
                        "category": snippet.get("category"),
                        "description": snippet.get("description"),
                        "tags": snippet.get("tags", [])
                    })
        
        if item_type is None or item_type == "procedure":
            for name, procedure in self.procedures.items():
                if not category or procedure.get("category") == category:
                    results.append({
                        "name": name,
                        "item_type": "procedure",
                        "category": procedure.get("category"),
                        "description": procedure.get("description"),
                        "steps_count": len(procedure.get("steps", [])),
                        "tags": procedure.get("tags", [])
                    })
        
        return results
    
    def delete(self, name: str, item_type: Optional[str] = None) -> bool:
        """
        Delete a template, snippet, or procedure.
        
        Args:
            name: Item name/ID
            item_type: Optional item type. If None, searches all types.
            
        Returns:
            True if deleted, False if not found
        """
        if not self.enabled:
            return False
        
        deleted = False
        
        if item_type is None or item_type == "template":
            if name in self.templates:
                del self.templates[name]
                self._delete_file("templates", name)
                deleted = True
        
        if item_type is None or item_type == "snippet":
            if name in self.snippets:
                del self.snippets[name]
                self._delete_file("snippets", name)
                deleted = True
        
        if item_type is None or item_type == "procedure":
            if name in self.procedures:
                del self.procedures[name]
                self._delete_file("procedures", name)
                deleted = True
        
        if deleted:
            safe_log("info", f"Deleted {item_type or 'item'}: {name}")
        
        return deleted
    
    def _load_all(self) -> None:
        """Load all templates, snippets, and procedures from disk."""
        # Load templates
        templates_dir = self.storage_path / "templates"
        if templates_dir.exists():
            for file_path in templates_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template = json.load(f)
                        self.templates[template["name"]] = template
                except Exception as e:
                    safe_log("error", f"Failed to load template {file_path}: {e}", exception=e)
        
        # Load snippets
        snippets_dir = self.storage_path / "snippets"
        if snippets_dir.exists():
            for file_path in snippets_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        snippet = json.load(f)
                        self.snippets[snippet["name"]] = snippet
                except Exception as e:
                    safe_log("error", f"Failed to load snippet {file_path}: {e}", exception=e)
        
        # Load procedures
        procedures_dir = self.storage_path / "procedures"
        if procedures_dir.exists():
            for file_path in procedures_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        procedure = json.load(f)
                        self.procedures[procedure["name"]] = procedure
                except Exception as e:
                    safe_log("error", f"Failed to load procedure {file_path}: {e}", exception=e)
    
    def _save_template(self, name: str, template: Dict[str, Any]) -> None:
        """Save a template to disk."""
        templates_dir = self.storage_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = templates_dir / f"{name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, default=str)
    
    def _save_snippet(self, name: str, snippet: Dict[str, Any]) -> None:
        """Save a snippet to disk."""
        snippets_dir = self.storage_path / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = snippets_dir / f"{name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(snippet, f, indent=2, default=str)
    
    def _save_procedure(self, name: str, procedure: Dict[str, Any]) -> None:
        """Save a procedure to disk."""
        procedures_dir = self.storage_path / "procedures"
        procedures_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = procedures_dir / f"{name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(procedure, f, indent=2, default=str)
    
    def _delete_file(self, subdir: str, name: str) -> None:
        """Delete a file from disk."""
        file_path = self.storage_path / subdir / f"{name}.json"
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                safe_log("error", f"Failed to delete file {file_path}: {e}", exception=e)
    
    def clear(self) -> None:
        """Clear all procedural memory."""
        self.templates.clear()
        self.snippets.clear()
        self.procedures.clear()
        
        # Delete all files
        for subdir in ["templates", "snippets", "procedures"]:
            dir_path = self.storage_path / subdir
            if dir_path.exists():
                for file_path in dir_path.glob("*.json"):
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
        
        safe_log("info", "Procedural memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about procedural memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.enabled:
            return {"enabled": False}
        
        # Count by category
        template_categories = {}
        snippet_categories = {}
        procedure_categories = {}
        
        for template in self.templates.values():
            cat = template.get("category", "general")
            template_categories[cat] = template_categories.get(cat, 0) + 1
        
        for snippet in self.snippets.values():
            cat = snippet.get("category", "general")
            snippet_categories[cat] = snippet_categories.get(cat, 0) + 1
        
        for procedure in self.procedures.values():
            cat = procedure.get("category", "general")
            procedure_categories[cat] = procedure_categories.get(cat, 0) + 1
        
        return {
            "enabled": True,
            "total_templates": len(self.templates),
            "total_snippets": len(self.snippets),
            "total_procedures": len(self.procedures),
            "total_items": len(self.templates) + len(self.snippets) + len(self.procedures),
            "template_categories": template_categories,
            "snippet_categories": snippet_categories,
            "procedure_categories": procedure_categories,
            "storage_path": str(self.storage_path)
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Procedural Memory Demo")
    print("=" * 60)
    
    memory = ProceduralMemory()
    
    # Store templates
    print("\n[Test 1] Storing Templates")
    memory.store_template(
        name="email_greeting",
        content="Hello {{name}},\n\nThank you for your interest in {{product}}.",
        category="email",
        tags=["greeting", "customer-service"],
        description="Standard email greeting template"
    )
    
    # Store snippets
    print("\n[Test 2] Storing Snippets")
    memory.store_snippet(
        name="python_hello",
        content="print('Hello, World!')",
        language="python",
        category="examples",
        tags=["hello-world", "beginner"],
        description="Simple Python hello world"
    )
    
    memory.store_snippet(
        name="json_parser",
        content="import json\ndata = json.loads(json_string)",
        language="python",
        category="utilities",
        tags=["json", "parsing"],
        description="JSON parsing snippet"
    )
    
    # Store procedures
    print("\n[Test 3] Storing Procedures")
    memory.store_procedure(
        name="user_onboarding",
        steps=[
            "1. Create user account",
            "2. Send welcome email",
            "3. Set up initial preferences",
            "4. Log first activity"
        ],
        category="workflow",
        tags=["onboarding", "user-management"],
        description="Standard user onboarding procedure"
    )
    
    # Search
    print("\n[Test 4] Searching")
    python_results = memory.search("python", item_type="snippet")
    print(f"Found {len(python_results)} Python snippets:")
    for result in python_results:
        print(f"  - {result['name']}: {result['description']}")
    
    # Get specific item
    print("\n[Test 5] Retrieving Specific Items")
    template = memory.get_template("email_greeting")
    if template:
        print(f"Template: {template['name']}")
        print(f"Content: {template['content'][:50]}...")
    
    # List all
    print("\n[Test 6] Listing All Items")
    all_items = memory.list_all()
    print(f"Total items: {len(all_items)}")
    for item in all_items:
        print(f"  - {item['item_type']}: {item['name']} ({item.get('category', 'N/A')})")
    
    # Get stats
    print("\n[Test 7] Memory Statistics")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nProcedural memory demo complete!")

