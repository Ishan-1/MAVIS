"""
benchmarks/harness/mock_channel.py
In-memory async channel connecting simulated user agents with MAVIS.
"""
from __future__ import annotations

import asyncio
from typing import Any


class MockChannel:
    """
    Fast in-memory async channel for passing messages and replies between
    simulated user agents and MAVIS in headless benchmark runs.
    """

    def __init__(self, default_timeout: float = 60.0) -> None:
        self.default_timeout = default_timeout
        self._user_to_agent: asyncio.Queue[str] = asyncio.Queue()
        self._agent_to_user: asyncio.Queue[str] = asyncio.Queue()
        self._history: list[dict[str, Any]] = []
        self._sender_id: str = "user"
        self._chat_id: str = "default_task"

    def set_runtime_identity(self, sender_id: str, chat_id: str) -> None:
        self._sender_id = sender_id
        self._chat_id = chat_id

    async def send_user_message(self, message: str) -> None:
        """Called by SimulatedUser to send a prompt to MAVIS."""
        self._history.append({
            "role": "user",
            "content": message,
            "sender_id": self._sender_id,
            "chat_id": self._chat_id,
        })
        await self._user_to_agent.put(message)

    async def receive_user_message(self, timeout: float | None = None) -> str:
        """Called by MAVIS test harness to read the next user prompt."""
        t = timeout or self.default_timeout
        return await asyncio.wait_for(self._user_to_agent.get(), timeout=t)

    async def send_agent_reply(self, reply: str, metadata: dict[str, Any] | None = None) -> None:
        """Called by MAVIS test harness to send reply back to user."""
        self._history.append({
            "role": "assistant",
            "content": reply,
            "sender_id": "mavis",
            "chat_id": self._chat_id,
            "metadata": metadata or {},
        })
        await self._agent_to_user.put(reply)

    async def wait_for_reply(self, timeout: float | None = None) -> str:
        """Called by benchmark runner to wait for MAVIS's response."""
        t = timeout or self.default_timeout
        return await asyncio.wait_for(self._agent_to_user.get(), timeout=t)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        """Clear queues and history for the next task."""
        while not self._user_to_agent.empty():
            try:
                self._user_to_agent.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._agent_to_user.empty():
            try:
                self._agent_to_user.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._history.clear()
