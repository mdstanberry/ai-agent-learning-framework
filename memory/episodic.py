"""
Episodic Memory Module

This module implements episodic memory for AI agents - storing and retrieving
past events and conversations using vector search.

## When to Use Episodic Memory

Episodic memory is used for:
- **Past conversations**: Historical interactions with users
- **Event history**: Timestamped events and experiences
- **Conversation context**: What happened in previous sessions
- **Temporal information**: Things that happened at specific times

**Use episodic memory when:**
- You need to recall what was discussed in past conversations
- You want to search for similar past interactions
- You need to remember events that happened at specific times
- You're building a conversation history that spans multiple sessions
- You need semantic search over past events (e.g., "find conversations about Python")

**Do NOT use episodic memory for:**
- Current conversation context (use Short-Term Memory)
- Permanent facts about entities (use Semantic Memory)
- Reusable templates or code (use Procedural Memory)
- Information that doesn't have a temporal component

## How It Works

According to .cursorrules:
- Store user profiles as: preferences (key-value), episodic interactions
  (timestamped events), and learned patterns (procedural memory).
  Never mix these stores. [file:3][file:5]
- Memory retrieval order: (1) vector search for semantic matches,
  (2) graph traversal for relationships, (3) event log validation for consistency.
  [file:3][file:4]

The module uses ChromaDB (vector database) to:
- Store events as embeddings for semantic search
- Retrieve similar past events based on content similarity
- Filter by metadata (timestamp, type, etc.)
- Maintain a searchable history of past interactions

## Example Usage

```python
from memory.episodic import EpisodicMemory

memory = EpisodicMemory()

# Store a past event
memory.store_event(
    content="User asked about Python file operations",
    event_type="conversation",
    metadata={"user_id": "user123", "topic": "python"}
)

# Search for similar past events
results = memory.search("Python programming help", limit=5)
```

## Comparison with Other Memory Types

- **vs. Short-Term Memory**: Episodic stores past events; Short-term is current session
- **vs. Semantic Memory**: Episodic is temporal/event-based; Semantic is fact-based
- **vs. Procedural Memory**: Episodic stores what happened; Procedural stores how to do things
"""

from typing import List, Optional, Dict, Any
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


# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Use print instead of logger during import to avoid encoding issues
    print("[WARNING] ChromaDB not installed. Episodic memory will use fallback storage.")


class EpisodicMemory:
    """
    Episodic memory using vector database for semantic search of past events.
    
    This stores:
    - Past conversations
    - User interactions
    - Events and experiences
    
    All stored as embeddings for semantic retrieval.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize episodic memory.
        
        Args:
            db_path: Path to ChromaDB database (defaults to config)
            collection_name: Name of the collection (defaults to config)
            embedding_model: Embedding model to use (defaults to config)
        """
        self.db_path = db_path or config.get_nested(
            "memory.episodic.db_path",
            "./chroma_db"
        )
        self.collection_name = collection_name or config.get_nested(
            "memory.episodic.collection_name",
            "agent_episodes"
        )
        self.embedding_model = embedding_model or config.get_nested(
            "memory.episodic.embedding_model",
            "text-embedding-3-small"
        )
        
        self.enabled = config.get_nested("memory.episodic.enabled", True)
        
        if not self.enabled:
            safe_log("info", "Episodic memory is disabled")
            self.client = None
            self.collection = None
            return
        
        if not CHROMADB_AVAILABLE:
            safe_log("warning", "ChromaDB not available, using fallback storage")
            self.client = None
            self.collection = None
            self._fallback_storage: List[Dict[str, Any]] = []
            return
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Agent episodic memory"}
            )
            
            safe_log("info", f"Episodic memory initialized: {self.collection_name}")
            safe_log("info", f"Stored episodes: {self.collection.count()}")
            
        except Exception as e:
            safe_log("error", f"Failed to initialize ChromaDB: {e}", exception=e)
            self.client = None
            self.collection = None
            self._fallback_storage: List[Dict[str, Any]] = []
    
    def store_event(
        self,
        content: str,
        event_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Store an event in episodic memory.
        
        Args:
            content: The event content (will be embedded)
            event_type: Type of event (conversation, interaction, etc.)
            metadata: Additional metadata to store
            timestamp: Event timestamp (defaults to now)
            
        Returns:
            Event ID
        """
        if not self.enabled:
            return ""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Prepare metadata
        event_metadata = {
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            **(metadata or {})
        }
        
        # Generate unique ID
        event_id = f"episode_{timestamp.timestamp()}_{hash(content) % 10000}"
        
        if self.collection:
            try:
                # Store in ChromaDB
                self.collection.add(
                    documents=[content],
                    metadatas=[event_metadata],
                    ids=[event_id]
                )
                safe_log("info", f"Stored event: {event_type} (ID: {event_id})")
            except Exception as e:
                safe_log("error", f"Failed to store event in ChromaDB: {e}", exception=e)
        else:
            # Fallback storage
            self._fallback_storage.append({
                "id": event_id,
                "content": content,
                "metadata": event_metadata
            })
            safe_log("info", f"Stored event in fallback storage: {event_type}")
        
        return event_id
    
    def search(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memory using semantic similarity.
        
        According to .cursorrules, this is step (1) in memory retrieval:
        vector search for semantic matches.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching events with scores
        """
        if not self.enabled:
            return []
        
        if self.collection:
            try:
                # Perform semantic search
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=filter_metadata
                )
                
                # Format results
                episodes = []
                if results["ids"] and len(results["ids"][0]) > 0:
                    for i, event_id in enumerate(results["ids"][0]):
                        episodes.append({
                            "id": event_id,
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i] if "distances" in results else None
                        })
                
                safe_log("observation", f"Found {len(episodes)} episodes matching query")
                return episodes
                
            except Exception as e:
                safe_log("error", f"Search failed: {e}", exception=e)
                return []
        else:
            # Fallback: simple text matching
            safe_log("warning", "Using fallback search (not semantic)")
            matches = []
            query_lower = query.lower()
            
            for event in self._fallback_storage:
                if query_lower in event["content"].lower():
                    matches.append(event)
            
            return matches[:limit]
    
    def get_recent_events(
        self,
        limit: int = 10,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent events, optionally filtered by type.
        
        Args:
            limit: Maximum number of events
            event_type: Optional event type filter
            
        Returns:
            List of recent events
        """
        if not self.enabled:
            return []
        
        if self.collection:
            try:
                # Get all events (ChromaDB doesn't have built-in timestamp ordering)
                # So we'll get more and sort manually
                all_results = self.collection.get()
                
                events = []
                for i, event_id in enumerate(all_results["ids"]):
                    metadata = all_results["metadatas"][i]
                    
                    # Filter by type if specified
                    if event_type and metadata.get("event_type") != event_type:
                        continue
                    
                    events.append({
                        "id": event_id,
                        "content": all_results["documents"][i],
                        "metadata": metadata
                    })
                
                # Sort by timestamp (most recent first)
                events.sort(
                    key=lambda x: x["metadata"].get("timestamp", ""),
                    reverse=True
                )
                
                return events[:limit]
                
            except Exception as e:
                safe_log("error", f"Failed to get recent events: {e}", exception=e)
                return []
        else:
            # Fallback: return from storage
            events = self._fallback_storage.copy()
            if event_type:
                events = [e for e in events if e["metadata"].get("event_type") == event_type]
            events.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
            return events[:limit]
    
    def delete_event(self, event_id: str) -> bool:
        """
        Delete an event from episodic memory.
        
        Args:
            event_id: ID of event to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not self.enabled:
            return False
        
        if self.collection:
            try:
                self.collection.delete(ids=[event_id])
                safe_log("info", f"Deleted event: {event_id}")
                return True
            except Exception as e:
                safe_log("error", f"Failed to delete event: {e}", exception=e)
                return False
        else:
            # Fallback: remove from storage
            self._fallback_storage = [
                e for e in self._fallback_storage if e["id"] != event_id
            ]
            return True
    
    def clear(self) -> None:
        """Clear all episodic memory."""
        if self.collection:
            try:
                # Delete and recreate collection
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Agent episodic memory"}
                )
                safe_log("info", "Episodic memory cleared")
            except Exception as e:
                safe_log("error", f"Failed to clear episodic memory: {e}", exception=e)
        else:
            self._fallback_storage.clear()
            safe_log("info", "Fallback episodic memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about episodic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.enabled:
            return {"enabled": False}
        
        if self.collection:
            count = self.collection.count()
            return {
                "enabled": True,
                "storage_type": "chromadb",
                "total_events": count,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        else:
            return {
                "enabled": True,
                "storage_type": "fallback",
                "total_events": len(self._fallback_storage),
                "collection_name": self.collection_name
            }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Episodic Memory Demo")
    print("=" * 60)
    
    memory = EpisodicMemory()
    
    # Store some events
    print("\n[Test 1] Storing Events")
    memory.store_event(
        content="User asked about Python file operations",
        event_type="conversation",
        metadata={"topic": "python", "user_id": "user123"}
    )
    
    memory.store_event(
        content="User requested help with API integration",
        event_type="conversation",
        metadata={"topic": "api", "user_id": "user123"}
    )
    
    memory.store_event(
        content="User completed a task successfully",
        event_type="interaction",
        metadata={"action": "task_completion", "user_id": "user123"}
    )
    
    print(f"Stored events")
    
    # Search for events
    print("\n[Test 2] Semantic Search")
    results = memory.search("Python programming help", limit=3)
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  - {result['content'][:60]}...")
        print(f"    Type: {result['metadata'].get('event_type')}")
    
    # Get recent events
    print("\n[Test 3] Recent Events")
    recent = memory.get_recent_events(limit=5)
    print(f"Recent events: {len(recent)}")
    for event in recent:
        print(f"  - {event['content'][:50]}...")
    
    # Get stats
    print("\n[Test 4] Memory Statistics")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nEpisodic memory demo complete!")

