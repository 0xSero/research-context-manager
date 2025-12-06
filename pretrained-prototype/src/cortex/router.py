"""Core Cortex Router - orchestrates Model A (context manager) and Model B (worker)."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from cortex.config import CortexConfig
from memory.graph import MemoryGraph, MemoryNode, create_memory_node, create_memory_edge
from providers.base import LLMProvider, LLMResponse, Message, ToolCall
from providers.factory import create_provider
from tools.registry import ToolRegistry, get_registry
from tools.executor import ToolExecutor, get_executor
from mcp.client import MCPClient


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    id: str
    role: str  # user, assistant
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens: int = 0


@dataclass
class ContextDecision:
    """Decision from Model A about what context to inject."""

    selected_tools: list[str]
    memory_nodes: list[str]
    context_summary: str
    additional_instructions: str
    priority: str  # low, medium, high, critical
    reasoning: str


# System prompt for Model A (the context manager)
MODEL_A_SYSTEM_PROMPT = """You are Cortex, an intelligent context manager. Your job is to analyze incoming user messages and decide what context should be provided to the worker model (Model B).

You have access to:
1. A memory graph containing facts, entities, preferences, and past interactions
2. A registry of available tools (including MCP servers)
3. The recent conversation history

For each user message, you must decide:
1. Which tools (if any) Model B will need
2. Which memories are relevant
3. What additional context or instructions to provide
4. The priority/urgency of the request

IMPORTANT RULES:
- Only include tools that are actually needed. Don't overload Model B.
- Prioritize recent and highly relevant memories.
- Be concise in your instructions.
- If the user's request is simple, minimal context is fine.

You will respond with a JSON object in this exact format:
{
    "selected_tools": ["tool_name_1", "tool_name_2"],
    "memory_query": "search query for relevant memories",
    "additional_context": "any additional context or instructions for Model B",
    "priority": "low|medium|high|critical",
    "reasoning": "brief explanation of your decisions"
}

Only output valid JSON, nothing else."""


class CortexRouter:
    """Main router that orchestrates Model A and Model B."""

    def __init__(self, config: CortexConfig):
        self.config = config

        # Initialize providers
        self.model_a: LLMProvider = create_provider(config.model_a)
        self.model_b: LLMProvider = create_provider(config.model_b)

        # Initialize components
        self.memory = MemoryGraph(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            database=config.neo4j.database,
        )
        self.tool_registry: ToolRegistry = get_registry()
        self.tool_executor: ToolExecutor = get_executor()
        self.mcp_client: MCPClient | None = None

        # Conversation state
        self.conversation: list[ConversationTurn] = []
        self.system_prompt: str = ""

        # Background tasks
        self._pruning_task: asyncio.Task | None = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize all components."""
        # Connect to Neo4j
        try:
            await self.memory.connect()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Memory features will be disabled.")

        # Load system prompt
        if self.config.system_prompt_path:
            path = Path(self.config.system_prompt_path)
            if path.exists():
                self.system_prompt = path.read_text()

        # Initialize MCP if configured
        if self.config.mcp_config_path:
            self.mcp_client = MCPClient(self.tool_registry)
            self.mcp_client.load_config(self.config.mcp_config_path)
            await self.mcp_client.connect_all()
            self.tool_executor.set_mcp_client(self.mcp_client)

        # Start background pruning
        if self.config.pruning.enabled:
            self._running = True
            self._pruning_task = asyncio.create_task(self._background_pruning())

    async def shutdown(self) -> None:
        """Shutdown all components."""
        self._running = False

        if self._pruning_task:
            self._pruning_task.cancel()
            try:
                await self._pruning_task
            except asyncio.CancelledError:
                pass

        if self.mcp_client:
            await self.mcp_client.disconnect_all()

        await self.memory.close()

    async def process_message(self, user_message: str) -> AsyncIterator[str]:
        """Process a user message and stream the response."""
        # Record the user turn
        user_turn = ConversationTurn(
            id=str(uuid4()),
            role="user",
            content=user_message,
            tokens=self.model_a.count_tokens(user_message),
        )
        self.conversation.append(user_turn)

        # Step 1: Get context decision from Model A
        context_decision = await self._get_context_decision(user_message)

        if self.config.debug:
            print(f"\n[DEBUG] Model A Decision: {context_decision}")

        # Step 2: Build context for Model B
        model_b_context = await self._build_model_b_context(
            user_message,
            context_decision,
        )

        # Step 3: Stream response from Model B
        assistant_content = ""
        async for chunk in self._stream_model_b_response(model_b_context):
            assistant_content += chunk
            yield chunk

        # Step 4: Handle any tool calls from Model B
        # (This happens after streaming completes, tool results shown separately)

        # Record assistant turn
        assistant_turn = ConversationTurn(
            id=str(uuid4()),
            role="assistant",
            content=assistant_content,
            tokens=self.model_b.count_tokens(assistant_content),
        )
        self.conversation.append(assistant_turn)

        # Step 5: Extract and store memories (async, don't block response)
        asyncio.create_task(
            self._extract_and_store_memories(user_message, assistant_content)
        )

    async def _get_context_decision(self, user_message: str) -> ContextDecision:
        """Ask Model A to decide what context Model B needs."""
        # Build context for Model A
        model_a_context = self._build_model_a_context(user_message)

        messages = [Message(role="user", content=model_a_context)]

        response = await self.model_a.complete(
            messages=messages,
            system=MODEL_A_SYSTEM_PROMPT,
        )

        # Parse the JSON response
        try:
            decision_json = json.loads(response.content)
            return ContextDecision(
                selected_tools=decision_json.get("selected_tools", []),
                memory_nodes=[],  # Will be filled by memory query
                context_summary=decision_json.get("additional_context", ""),
                additional_instructions="",
                priority=decision_json.get("priority", "medium"),
                reasoning=decision_json.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            # Fallback if Model A doesn't return valid JSON
            return ContextDecision(
                selected_tools=[],
                memory_nodes=[],
                context_summary="",
                additional_instructions="",
                priority="medium",
                reasoning="Failed to parse Model A response",
            )

    def _build_model_a_context(self, user_message: str) -> str:
        """Build the context that Model A sees."""
        parts = []

        # 1. Available tools summary
        tools_summary = self.tool_registry.get_summary_list()
        parts.append(f"# AVAILABLE TOOLS\n{tools_summary}")

        # 2. Recent conversation (truncated to config limit)
        recent_turns = self._get_recent_conversation(
            max_tokens=self.config.context.model_a_context_window
        )
        if recent_turns:
            conv_text = "\n".join(
                f"[{t.role.upper()}]: {t.content[:500]}..." if len(t.content) > 500 else f"[{t.role.upper()}]: {t.content}"
                for t in recent_turns
            )
            parts.append(f"# RECENT CONVERSATION\n{conv_text}")

        # 3. Current user message
        parts.append(f"# CURRENT USER MESSAGE\n{user_message}")

        # 4. Instructions
        parts.append("""
# YOUR TASK
Analyze the user message and decide what context Model B needs.
Respond with a JSON object specifying tools, memory query, and additional context.""")

        return "\n\n".join(parts)

    async def _build_model_b_context(
        self,
        user_message: str,
        decision: ContextDecision,
    ) -> dict[str, Any]:
        """Build the full context for Model B based on Model A's decision."""
        context = {
            "system_prompt": self.system_prompt,
            "tools": [],
            "memories": "",
            "additional_context": decision.context_summary,
            "messages": [],
        }

        # Add selected tools
        if decision.selected_tools:
            context["tools"] = self.tool_registry.get_definitions(decision.selected_tools)
            tools_context = self.tool_registry.export_for_context(decision.selected_tools)
            context["additional_context"] = f"{tools_context}\n\n{decision.context_summary}"

        # Query relevant memories
        try:
            memory_nodes = await self.memory.search_by_content(
                user_message,
                limit=self.config.context.max_memory_nodes,
            )
            if memory_nodes:
                node_ids = [n.id for n in memory_nodes]
                context["memories"] = await self.memory.export_context(node_ids)
        except Exception:
            pass  # Memory not available

        # Add conversation messages (limited)
        recent = self._get_recent_conversation(
            max_tokens=self.config.context.model_b_max_context // 2
        )
        context["messages"] = [
            Message(role=t.role, content=t.content)
            for t in recent
        ]

        # Add current user message
        context["messages"].append(Message(role="user", content=user_message))

        return context

    async def _stream_model_b_response(
        self,
        context: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Stream response from Model B."""
        # Build system prompt with additional context
        system = context["system_prompt"]
        if context["memories"]:
            system += f"\n\n# RELEVANT MEMORIES\n{context['memories']}"
        if context["additional_context"]:
            system += f"\n\n{context['additional_context']}"

        # Stream the response
        async for chunk in self.model_b.stream(
            messages=context["messages"],
            tools=context["tools"] if context["tools"] else None,
            system=system,
        ):
            yield chunk

    def _get_recent_conversation(
        self,
        max_tokens: int,
    ) -> list[ConversationTurn]:
        """Get recent conversation turns up to token limit."""
        if not self.conversation:
            return []

        turns = []
        total_tokens = 0

        # Work backwards from most recent
        for turn in reversed(self.conversation):
            if total_tokens + turn.tokens > max_tokens:
                break
            turns.insert(0, turn)
            total_tokens += turn.tokens

        return turns

    async def _extract_and_store_memories(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Extract entities and facts from the conversation and store in memory."""
        try:
            # Simple extraction - in production, use NER/entity extraction
            # For now, store the turn itself as a memory

            # Create a memory node for significant turns
            if len(user_message) > 50 or len(assistant_response) > 100:
                node = create_memory_node(
                    type="conversation",
                    content=f"User: {user_message[:500]}\nAssistant: {assistant_response[:500]}",
                    metadata={
                        "timestamp": datetime.utcnow().isoformat(),
                        "user_tokens": self.model_b.count_tokens(user_message),
                        "assistant_tokens": self.model_b.count_tokens(assistant_response),
                    },
                )
                await self.memory.add_node(node)

        except Exception as e:
            if self.config.debug:
                print(f"[DEBUG] Memory extraction error: {e}")

    async def _background_pruning(self) -> None:
        """Background task to prune the memory graph."""
        while self._running:
            try:
                await asyncio.sleep(self.config.pruning.interval_seconds)

                if not self._running:
                    break

                # Decay relevance scores
                await self.memory.decay_relevance(
                    factor=self.config.pruning.age_decay_factor
                )

                # Prune low relevance nodes
                pruned = await self.memory.prune_low_relevance(
                    threshold=self.config.pruning.relevance_threshold,
                    keep_min=100,
                )

                if self.config.debug and pruned > 0:
                    print(f"[DEBUG] Pruned {pruned} memory nodes")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.debug:
                    print(f"[DEBUG] Pruning error: {e}")

    async def compact_conversation(self) -> str:
        """Compact the conversation history into a summary."""
        if len(self.conversation) < 5:
            return ""

        # Get all conversation content
        full_text = "\n".join(
            f"[{t.role}]: {t.content}" for t in self.conversation
        )

        # Ask Model A to summarize
        messages = [
            Message(
                role="user",
                content=f"""Summarize this conversation into key facts and context that should be preserved for continuation:

{full_text}

Provide a concise summary that captures:
1. Key facts and decisions
2. User preferences expressed
3. Current task state
4. Important context for continuation

Format as bullet points.""",
            )
        ]

        response = await self.model_a.complete(
            messages=messages,
            system="You are a conversation summarizer. Be concise but thorough.",
        )

        # Store summary as a high-relevance memory
        summary_node = create_memory_node(
            type="summary",
            content=response.content,
            metadata={"turn_count": len(self.conversation)},
        )
        summary_node.relevance_score = 2.0  # Higher than normal
        await self.memory.add_node(summary_node)

        # Clear old turns, keep last few
        keep_count = 3
        self.conversation = self.conversation[-keep_count:]

        return response.content

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "conversation_turns": len(self.conversation),
            "total_tokens": sum(t.tokens for t in self.conversation),
            "tool_count": len(self.tool_registry.get_all()),
            "mcp_servers": (
                self.mcp_client.list_connected_servers()
                if self.mcp_client
                else []
            ),
        }
