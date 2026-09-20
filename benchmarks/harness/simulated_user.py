"""
benchmarks/harness/simulated_user.py
Simulated Persona User Agent that guides tasks, progressively reveals hidden
intents upon clarification, and evaluates task termination.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from benchmarks.harness.models import PersonaProfile, TaskSpec


@dataclass
class UserAction:
    action_type: str  # "message" | "terminate"
    message: str = ""
    reason: str = ""


USER_SIMULATOR_PROMPT = """You are a simulated human user in an automated benchmark evaluating MAVIS, an AI assistant.
Your persona: {persona_name} ({persona_description})
Your personal preferences: {preferences}

Task Title: {task_title}
Initial Goal: {initial_input}

Ground-Truth Hidden Requirements (You know these, but the assistant was NOT given them upfront):
{hidden_intents_text}

The assistant just sent you this response:
\"\"\"{agent_reply}\"\"\"

Your behavior rules:
1. If the assistant has satisfied the requirements or declared the task complete and your verification looks satisfied, reply with ACTION: TERMINATE.
2. If the assistant asked a clarifying question, answer naturally from your persona's perspective, revealing only the relevant hidden requirement without copy-pasting the benchmark prompt verbatim.
3. If the assistant stopped prematurely or missed an obvious step, provide a concise realistic nudge (max 2-3 sentences).
4. Do NOT output markdown code blocks. Output a single valid JSON object with:
{{
  "action": "message" | "terminate",
  "message": "your reply text if action is message, else empty",
  "reason": "brief rationale for continuing or terminating"
}}
"""


class SimulatedUser:
    """
    Simulates a human user interacting with MAVIS across multi-turn workflows.
    Can operate via an active LLM client or rule-based heuristic fallback.
    """

    def __init__(
        self,
        task: TaskSpec,
        profile: PersonaProfile | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.task = task
        self.profile = profile or PersonaProfile(
            persona_id=task.persona,
            name=task.persona.replace("_", " ").title(),
            description=f"User persona for {task.persona}",
            default_workspace="workspace",
        )
        self.llm_client = llm_client
        self.current_turn: int = 0
        self.revealed_intent_ids: set[str] = set()

    def get_initial_message(self) -> str:
        """Returns the initial, intentionally underspecified user prompt."""
        self.current_turn = 1
        return self.task.initial_input

    async def next_action(self, agent_reply: str) -> UserAction:
        """
        Analyze MAVIS's reply and decide whether to send a follow-up message
        or terminate the task.
        """
        self.current_turn += 1

        # If LLM client is available, use semantic evaluation
        if self.llm_client is not None:
            try:
                action = await self._evaluate_with_llm(agent_reply)
                if action:
                    return action
            except Exception:
                pass  # fallback to heuristics on error

        # 2. Heuristic fallback evaluation
        # Check if assistant asks a question or asks for confirmation
        reply_lower = agent_reply.lower()
        is_question = "?" in agent_reply or "confirm" in reply_lower or "proceed?" in reply_lower

        # Find any unrevealed intent to offer as a hint/clarification
        unrevealed = [h for h in self.task.hidden_intents if h.id not in self.revealed_intent_ids]

        if is_question and unrevealed:
            next_hint = unrevealed[0]
            self.revealed_intent_ids.add(next_hint.id)
            return UserAction(
                action_type="message",
                message=f"Yes, please proceed. Also, make sure to {next_hint.content}",
                reason=f"Answered query with hidden intent {next_hint.id}",
            )

        if not unrevealed or self.current_turn >= 25:
            # All hints revealed or max horizon reached
            return UserAction(
                action_type="terminate",
                reason="All relevant requirements communicated or turn horizon reached.",
            )

        # Mild nudge with next intent
        next_hint = unrevealed[0]
        self.revealed_intent_ids.add(next_hint.id)
        return UserAction(
            action_type="message",
            message=f"Thanks. Please also ensure: {next_hint.content}",
            reason=f"Follow-up nudge with intent {next_hint.id}",
        )

    async def _evaluate_with_llm(self, agent_reply: str) -> UserAction | None:
        """Call LLM client to act as user persona."""
        intents_text = "\n".join(
            f"- [{h.id}]: {h.content}" for h in self.task.hidden_intents
        )
        prompt = USER_SIMULATOR_PROMPT.format(
            persona_name=self.profile.name,
            persona_description=self.profile.description,
            preferences=", ".join(self.profile.preferences) or "standard",
            task_title=self.task.title,
            initial_input=self.task.initial_input,
            hidden_intents_text=intents_text,
            agent_reply=agent_reply[:3000],
        )

        resp = self.llm_client.generate(prompt, json_mode=True)
        data = json.loads(resp)
        action_type = data.get("action", "message")
        return UserAction(
            action_type="terminate" if action_type == "terminate" else "message",
            message=str(data.get("message", "")),
            reason=str(data.get("reason", "")),
        )
