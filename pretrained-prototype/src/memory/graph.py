"""Neo4j-backed memory graph for conversation context."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from neo4j import AsyncGraphDatabase, AsyncDriver


@dataclass
class MemoryNode:
    """A node in the memory graph."""

    id: str
    type: str  # entity, fact, event, preference, task
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "relevance_score": self.relevance_score,
        }


@dataclass
class MemoryEdge:
    """An edge in the memory graph."""

    id: str
    source_id: str
    target_id: str
    type: str  # related_to, causes, part_of, references, etc.
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class MemoryGraph:
    """Neo4j-backed memory graph manager."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Connect to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        # Verify connection
        async with self._driver.session(database=self.database) as session:
            await session.run("RETURN 1")

        # Create indexes
        await self._create_indexes()

    async def close(self) -> None:
        """Close the connection."""
        if self._driver:
            await self._driver.close()

    async def _create_indexes(self) -> None:
        """Create necessary indexes."""
        async with self._driver.session(database=self.database) as session:
            # Index on node ID
            await session.run(
                "CREATE INDEX memory_node_id IF NOT EXISTS FOR (n:Memory) ON (n.id)"
            )
            # Index on node type
            await session.run(
                "CREATE INDEX memory_node_type IF NOT EXISTS FOR (n:Memory) ON (n.type)"
            )
            # Index on relevance for pruning
            await session.run(
                "CREATE INDEX memory_relevance IF NOT EXISTS FOR (n:Memory) ON (n.relevance_score)"
            )

    async def add_node(self, node: MemoryNode) -> str:
        """Add a memory node."""
        query = """
        CREATE (n:Memory {
            id: $id,
            type: $type,
            content: $content,
            metadata: $metadata,
            created_at: datetime($created_at),
            last_accessed: datetime($last_accessed),
            access_count: $access_count,
            relevance_score: $relevance_score
        })
        RETURN n.id
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                query,
                id=node.id,
                type=node.type,
                content=node.content,
                metadata=str(node.metadata),  # Neo4j doesn't support nested dicts
                created_at=node.created_at.isoformat(),
                last_accessed=node.last_accessed.isoformat(),
                access_count=node.access_count,
                relevance_score=node.relevance_score,
            )
            record = await result.single()
            return record["n.id"]

    async def add_edge(self, edge: MemoryEdge) -> str:
        """Add an edge between nodes."""
        query = """
        MATCH (a:Memory {id: $source_id})
        MATCH (b:Memory {id: $target_id})
        CREATE (a)-[r:RELATES {
            id: $id,
            type: $type,
            weight: $weight,
            metadata: $metadata,
            created_at: datetime($created_at)
        }]->(b)
        RETURN r.id
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                query,
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                type=edge.type,
                weight=edge.weight,
                metadata=str(edge.metadata),
                created_at=edge.created_at.isoformat(),
            )
            record = await result.single()
            return record["r.id"] if record else edge.id

    async def get_node(self, node_id: str) -> MemoryNode | None:
        """Get a node by ID."""
        query = """
        MATCH (n:Memory {id: $id})
        RETURN n
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                n = record["n"]
                return MemoryNode(
                    id=n["id"],
                    type=n["type"],
                    content=n["content"],
                    metadata=eval(n["metadata"]) if n["metadata"] else {},
                    created_at=n["created_at"].to_native() if n.get("created_at") else datetime.utcnow(),
                    last_accessed=n["last_accessed"].to_native() if n.get("last_accessed") else datetime.utcnow(),
                    access_count=n.get("access_count", 0),
                    relevance_score=n.get("relevance_score", 1.0),
                )
            return None

    async def search_by_content(
        self,
        query: str,
        limit: int = 10,
        node_types: list[str] | None = None,
    ) -> list[MemoryNode]:
        """Search nodes by content (simple text matching)."""
        cypher = """
        MATCH (n:Memory)
        WHERE n.content CONTAINS $query
        """
        if node_types:
            cypher += " AND n.type IN $types"
        cypher += """
        RETURN n
        ORDER BY n.relevance_score DESC, n.last_accessed DESC
        LIMIT $limit
        """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                cypher,
                query=query,
                types=node_types or [],
                limit=limit,
            )
            nodes = []
            async for record in result:
                n = record["n"]
                nodes.append(
                    MemoryNode(
                        id=n["id"],
                        type=n["type"],
                        content=n["content"],
                        metadata=eval(n["metadata"]) if n["metadata"] else {},
                        relevance_score=n.get("relevance_score", 1.0),
                    )
                )
            return nodes

    async def get_related_nodes(
        self,
        node_id: str,
        depth: int = 2,
        limit: int = 20,
    ) -> list[MemoryNode]:
        """Get nodes related to a given node up to N hops."""
        query = f"""
        MATCH (start:Memory {{id: $id}})
        MATCH path = (start)-[*1..{depth}]-(related:Memory)
        WHERE related.id <> start.id
        RETURN DISTINCT related
        ORDER BY related.relevance_score DESC
        LIMIT $limit
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, id=node_id, limit=limit)
            nodes = []
            async for record in result:
                n = record["related"]
                nodes.append(
                    MemoryNode(
                        id=n["id"],
                        type=n["type"],
                        content=n["content"],
                        metadata=eval(n["metadata"]) if n["metadata"] else {},
                        relevance_score=n.get("relevance_score", 1.0),
                    )
                )
            return nodes

    async def get_recent_nodes(self, limit: int = 50) -> list[MemoryNode]:
        """Get most recently accessed nodes."""
        query = """
        MATCH (n:Memory)
        RETURN n
        ORDER BY n.last_accessed DESC
        LIMIT $limit
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, limit=limit)
            nodes = []
            async for record in result:
                n = record["n"]
                nodes.append(
                    MemoryNode(
                        id=n["id"],
                        type=n["type"],
                        content=n["content"],
                        metadata=eval(n["metadata"]) if n["metadata"] else {},
                        relevance_score=n.get("relevance_score", 1.0),
                    )
                )
            return nodes

    async def update_access(self, node_id: str) -> None:
        """Update last accessed time and increment access count."""
        query = """
        MATCH (n:Memory {id: $id})
        SET n.last_accessed = datetime(),
            n.access_count = n.access_count + 1
        """
        async with self._driver.session(database=self.database) as session:
            await session.run(query, id=node_id)

    async def update_relevance(self, node_id: str, score: float) -> None:
        """Update relevance score."""
        query = """
        MATCH (n:Memory {id: $id})
        SET n.relevance_score = $score
        """
        async with self._driver.session(database=self.database) as session:
            await session.run(query, id=node_id, score=score)

    async def prune_low_relevance(
        self,
        threshold: float = 0.3,
        keep_min: int = 100,
    ) -> int:
        """Prune nodes with low relevance, keeping at least keep_min nodes."""
        # First check how many nodes we have
        count_query = "MATCH (n:Memory) RETURN count(n) as count"
        async with self._driver.session(database=self.database) as session:
            result = await session.run(count_query)
            record = await result.single()
            total = record["count"]

            if total <= keep_min:
                return 0

            # Delete low relevance nodes, but keep at least keep_min
            delete_query = """
            MATCH (n:Memory)
            WHERE n.relevance_score < $threshold
            WITH n ORDER BY n.relevance_score ASC
            LIMIT $limit
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            max_delete = total - keep_min
            result = await session.run(
                delete_query,
                threshold=threshold,
                limit=max_delete,
            )
            record = await result.single()
            return record["deleted"] if record else 0

    async def decay_relevance(self, factor: float = 0.95) -> None:
        """Apply decay to all relevance scores."""
        query = """
        MATCH (n:Memory)
        SET n.relevance_score = n.relevance_score * $factor
        """
        async with self._driver.session(database=self.database) as session:
            await session.run(query, factor=factor)

    async def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        query = """
        MATCH (n:Memory)
        WITH count(n) as node_count,
             avg(n.relevance_score) as avg_relevance,
             min(n.relevance_score) as min_relevance,
             max(n.relevance_score) as max_relevance
        MATCH ()-[r:RELATES]->()
        RETURN node_count, count(r) as edge_count,
               avg_relevance, min_relevance, max_relevance
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                return {
                    "node_count": record["node_count"],
                    "edge_count": record["edge_count"],
                    "avg_relevance": record["avg_relevance"],
                    "min_relevance": record["min_relevance"],
                    "max_relevance": record["max_relevance"],
                }
            return {"node_count": 0, "edge_count": 0}

    async def export_context(self, node_ids: list[str]) -> str:
        """Export nodes as context string for LLM."""
        if not node_ids:
            return ""

        query = """
        MATCH (n:Memory)
        WHERE n.id IN $ids
        OPTIONAL MATCH (n)-[r:RELATES]->(related:Memory)
        WHERE related.id IN $ids
        RETURN n, collect({rel: r, target: related}) as relationships
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, ids=node_ids)

            context_parts = []
            async for record in result:
                n = record["n"]
                rels = record["relationships"]

                part = f"[{n['type'].upper()}] {n['content']}"

                # Add relationships
                rel_strs = []
                for rel in rels:
                    if rel["rel"] and rel["target"]:
                        rel_strs.append(
                            f"  → {rel['rel']['type']}: {rel['target']['content'][:100]}"
                        )
                if rel_strs:
                    part += "\n" + "\n".join(rel_strs)

                context_parts.append(part)

            return "\n\n".join(context_parts)


def create_memory_node(
    type: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> MemoryNode:
    """Helper to create a memory node with auto-generated ID."""
    return MemoryNode(
        id=str(uuid4()),
        type=type,
        content=content,
        metadata=metadata or {},
    )


def create_memory_edge(
    source_id: str,
    target_id: str,
    type: str,
    weight: float = 1.0,
) -> MemoryEdge:
    """Helper to create a memory edge with auto-generated ID."""
    return MemoryEdge(
        id=str(uuid4()),
        source_id=source_id,
        target_id=target_id,
        type=type,
        weight=weight,
    )
