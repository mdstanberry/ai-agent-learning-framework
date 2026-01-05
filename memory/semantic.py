"""
Semantic Memory Module

This module implements semantic memory for AI agents - storing and retrieving
facts and relationships using a knowledge graph.

## When to Use Semantic Memory

Semantic memory is used for:
- **Permanent facts**: Information about entities that doesn't change
- **Entity relationships**: How entities are connected to each other
- **User profiles**: Static information about users (name, preferences, etc.)
- **Knowledge base**: Facts that persist across sessions
- **Structured data**: Information that fits into a graph structure

**Use semantic memory when:**
- You need to store permanent facts about users, products, or concepts
- You want to model relationships between entities (e.g., "user likes product")
- You need to query facts about specific entities
- You want to traverse relationships to find related information
- You're building a knowledge base that persists long-term

**Do NOT use semantic memory for:**
- Current conversation context (use Short-Term Memory)
- Past events or conversations (use Episodic Memory)
- Reusable code templates (use Procedural Memory)
- Temporary information that changes frequently

## How It Works

According to .cursorrules:
- Memory retrieval order: (1) vector search for semantic matches,
  (2) graph traversal for relationships, (3) event log validation for consistency.
  [file:3][file:4]
- Store user profiles as: preferences (key-value), episodic interactions
  (timestamped events), and learned patterns (procedural memory).
  Never mix these stores. [file:3][file:5]

The module uses NetworkX to:
- Store entities as nodes with properties (facts)
- Store relationships as edges with labels
- Traverse the graph to find related entities
- Query entities by type, properties, or relationships

## Example Usage

```python
from memory.semantic import SemanticMemory

memory = SemanticMemory()

# Store facts about an entity
memory.add_fact("user123", "name", "Alice", entity_type="user")
memory.add_fact("user123", "age", 30, entity_type="user")

# Store relationships
memory.add_relationship("user123", "product456", "purchased")

# Query facts
user_name = memory.get_fact("user123", "name")

# Find related entities
related = memory.find_related_entities("user123", max_depth=2)
```

## Comparison with Other Memory Types

- **vs. Short-Term Memory**: Semantic stores permanent facts; Short-term is temporary context
- **vs. Episodic Memory**: Semantic is fact-based; Episodic is event-based
- **vs. Procedural Memory**: Semantic stores what things are; Procedural stores how to do things
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime
from pathlib import Path
import json
import networkx as nx
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


class SemanticMemory:
    """
    Semantic memory using a knowledge graph (NetworkX) for storing facts and relationships.
    
    This stores:
    - Facts about entities (nodes with attributes)
    - Relationships between entities (edges with labels)
    - Entity properties and metadata
    
    All stored as a graph structure for relationship traversal.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize semantic memory.
        
        Args:
            storage_path: Path to JSON file for persistence (defaults to config)
            enabled: Whether semantic memory is enabled (defaults to config)
        """
        self.storage_path = storage_path or config.get_nested(
            "memory.semantic.storage_path",
            "./knowledge_graph.json"
        )
        self.enabled = enabled if enabled is not None else config.get_nested(
            "memory.semantic.enabled",
            True
        )
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()  # Directed graph for relationships
        
        if not self.enabled:
            safe_log("info", "Semantic memory is disabled")
            return
        
        # Load existing graph if it exists
        self._load_graph()
        
        safe_log("info", f"Semantic memory initialized: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def add_fact(
        self,
        entity: str,
        property_name: str,
        value: Any,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a fact about an entity.
        
        Facts are stored as node attributes in the graph.
        
        Args:
            entity: Entity name/ID
            property_name: Property name (e.g., "age", "location", "preference")
            value: Property value
            entity_type: Optional entity type (e.g., "user", "product", "concept")
            metadata: Optional metadata (timestamp, source, etc.)
        """
        if not self.enabled:
            return
        
        # Ensure entity node exists
        if entity not in self.graph:
            self.graph.add_node(entity, type=entity_type or "entity")
        
        # Add property to entity
        if "properties" not in self.graph.nodes[entity]:
            self.graph.nodes[entity]["properties"] = {}
        
        self.graph.nodes[entity]["properties"][property_name] = value
        
        # Add metadata if provided
        if metadata:
            if "metadata" not in self.graph.nodes[entity]:
                self.graph.nodes[entity]["metadata"] = []
            self.graph.nodes[entity]["metadata"].append({
                **metadata,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update entity type if provided
        if entity_type:
            self.graph.nodes[entity]["type"] = entity_type
        
        # Persist to disk
        self._save_graph()
        
        safe_log("info", f"Added fact: {entity}.{property_name} = {value}")
    
    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between two entities.
        
        Relationships are stored as edges in the graph.
        
        Args:
            source: Source entity
            target: Target entity
            relationship_type: Type of relationship (e.g., "knows", "likes", "works_at")
            metadata: Optional metadata (strength, timestamp, etc.)
        """
        if not self.enabled:
            return
        
        # Ensure both nodes exist
        if source not in self.graph:
            self.graph.add_node(source)
        if target not in self.graph:
            self.graph.add_node(target)
        
        # Add edge with relationship type
        edge_data = {"type": relationship_type}
        if metadata:
            edge_data["metadata"] = {
                **metadata,
                "timestamp": datetime.now().isoformat()
            }
        
        self.graph.add_edge(source, target, **edge_data)
        
        # Persist to disk
        self._save_graph()
        
        safe_log("info", f"Added relationship: {source} --[{relationship_type}]--> {target}")
    
    def get_fact(
        self,
        entity: str,
        property_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get a fact about an entity.
        
        Args:
            entity: Entity name/ID
            property_name: Optional property name. If None, returns all properties.
            
        Returns:
            Property value(s) or None if not found
        """
        if not self.enabled or entity not in self.graph:
            return None
        
        properties = self.graph.nodes[entity].get("properties", {})
        
        if property_name is None:
            return properties
        
        return properties.get(property_name)
    
    def get_relationships(
        self,
        entity: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for an entity.
        
        According to .cursorrules, this is step (2) in memory retrieval:
        graph traversal for relationships.
        
        Args:
            entity: Entity name/ID
            relationship_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both" (default: "both")
            
        Returns:
            List of relationships, each containing:
            - source: Source entity
            - target: Target entity
            - type: Relationship type
            - metadata: Optional metadata
        """
        if not self.enabled or entity not in self.graph:
            return []
        
        relationships = []
        
        # Get outgoing edges
        if direction in ("outgoing", "both"):
            for target, edge_data in self.graph[entity].items():
                rel_type = edge_data.get("type", "related")
                if relationship_type is None or rel_type == relationship_type:
                    relationships.append({
                        "source": entity,
                        "target": target,
                        "type": rel_type,
                        "metadata": edge_data.get("metadata", {}),
                        "direction": "outgoing"
                    })
        
        # Get incoming edges
        if direction in ("incoming", "both"):
            for source in self.graph.predecessors(entity):
                edge_data = self.graph[source][entity]
                rel_type = edge_data.get("type", "related")
                if relationship_type is None or rel_type == relationship_type:
                    relationships.append({
                        "source": source,
                        "target": entity,
                        "type": rel_type,
                        "metadata": edge_data.get("metadata", {}),
                        "direction": "incoming"
                    })
        
        return relationships
    
    def find_related_entities(
        self,
        entity: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity through graph traversal.
        
        Uses breadth-first search to find related entities up to max_depth.
        
        Args:
            entity: Starting entity
            max_depth: Maximum traversal depth (default: 2)
            relationship_types: Optional list of relationship types to follow
            
        Returns:
            List of related entities with their paths
        """
        if not self.enabled or entity not in self.graph:
            return []
        
        related = []
        visited = {entity}
        queue = [(entity, 0, [entity])]  # (node, depth, path)
        
        while queue:
            current, depth, path = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Explore neighbors
            for neighbor in self.graph.successors(current):
                if neighbor in visited:
                    continue
                
                edge_data = self.graph[current][neighbor]
                rel_type = edge_data.get("type", "related")
                
                # Filter by relationship type if specified
                if relationship_types and rel_type not in relationship_types:
                    continue
                
                visited.add(neighbor)
                new_path = path + [neighbor]
                
                # Get entity info
                entity_info = {
                    "entity": neighbor,
                    "type": self.graph.nodes[neighbor].get("type", "entity"),
                    "depth": depth + 1,
                    "path": new_path,
                    "relationship": rel_type
                }
                
                related.append(entity_info)
                queue.append((neighbor, depth + 1, new_path))
        
        return related
    
    def query(
        self,
        entity: Optional[str] = None,
        entity_type: Optional[str] = None,
        property_name: Optional[str] = None,
        property_value: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for entities matching criteria.
        
        Args:
            entity: Optional entity name/ID
            entity_type: Optional filter by entity type
            property_name: Optional filter by property name
            property_value: Optional filter by property value
            
        Returns:
            List of matching entities with their properties
        """
        if not self.enabled:
            return []
        
        results = []
        
        for node in self.graph.nodes():
            # Filter by entity name
            if entity and node != entity:
                continue
            
            # Filter by entity type
            node_type = self.graph.nodes[node].get("type")
            if entity_type and node_type != entity_type:
                continue
            
            # Filter by property
            properties = self.graph.nodes[node].get("properties", {})
            if property_name:
                if property_name not in properties:
                    continue
                if property_value is not None and properties[property_name] != property_value:
                    continue
            
            results.append({
                "entity": node,
                "type": node_type,
                "properties": properties.copy(),
                "relationships_count": len(list(self.graph.successors(node))) + len(list(self.graph.predecessors(node)))
            })
        
        return results
    
    def _load_graph(self) -> None:
        """Load graph from JSON file."""
        if not Path(self.storage_path).exists():
            safe_log("info", f"Knowledge graph file not found, starting fresh")
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct graph from JSON
            self.graph = nx.node_link_graph(data)
            
            safe_log("info", f"Loaded knowledge graph from {self.storage_path}")
        except Exception as e:
            safe_log("error", f"Failed to load knowledge graph: {e}", exception=e)
            self.graph = nx.DiGraph()
    
    def _save_graph(self) -> None:
        """Save graph to JSON file."""
        try:
            # Convert graph to JSON-serializable format
            data = nx.node_link_data(self.graph)
            
            # Ensure directory exists
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            safe_log("error", f"Failed to save knowledge graph: {e}", exception=e)
    
    def clear(self) -> None:
        """Clear all semantic memory."""
        self.graph = nx.DiGraph()
        self._save_graph()
        safe_log("info", "Semantic memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about semantic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.enabled:
            return {"enabled": False}
        
        # Count entity types
        entity_types = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "unknown")
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
        
        # Count relationship types
        relationship_types = {}
        for source, target, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get("type", "unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "enabled": True,
            "total_entities": len(self.graph.nodes),
            "total_relationships": len(self.graph.edges),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "storage_path": self.storage_path
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Semantic Memory Demo")
    print("=" * 60)
    
    memory = SemanticMemory()
    
    # Add facts about entities
    print("\n[Test 1] Adding Facts")
    memory.add_fact("user123", "name", "Alice", entity_type="user")
    memory.add_fact("user123", "age", 30, entity_type="user")
    memory.add_fact("user123", "location", "San Francisco", entity_type="user")
    memory.add_fact("product456", "name", "Widget Pro", entity_type="product")
    memory.add_fact("product456", "price", 99.99, entity_type="product")
    
    # Add relationships
    print("\n[Test 2] Adding Relationships")
    memory.add_relationship("user123", "product456", "purchased")
    memory.add_relationship("user123", "user789", "knows")
    memory.add_relationship("user789", "product456", "recommended")
    
    # Query facts
    print("\n[Test 3] Querying Facts")
    user_facts = memory.get_fact("user123")
    print(f"User123 facts: {user_facts}")
    
    user_age = memory.get_fact("user123", "age")
    print(f"User123 age: {user_age}")
    
    # Get relationships
    print("\n[Test 4] Getting Relationships")
    user_relationships = memory.get_relationships("user123")
    print(f"User123 relationships: {len(user_relationships)}")
    for rel in user_relationships:
        print(f"  {rel['source']} --[{rel['type']}]--> {rel['target']}")
    
    # Find related entities
    print("\n[Test 5] Finding Related Entities")
    related = memory.find_related_entities("user123", max_depth=2)
    print(f"Entities related to user123: {len(related)}")
    for entity_info in related:
        print(f"  {entity_info['entity']} (type: {entity_info['type']}, depth: {entity_info['depth']})")
    
    # Query by criteria
    print("\n[Test 6] Querying by Criteria")
    users = memory.query(entity_type="user")
    print(f"Found {len(users)} users:")
    for user in users:
        print(f"  {user['entity']}: {user['properties']}")
    
    # Get stats
    print("\n[Test 7] Memory Statistics")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSemantic memory demo complete!")

