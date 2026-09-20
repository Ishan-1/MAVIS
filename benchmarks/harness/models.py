"""
benchmarks/harness/models.py
Data models and schemas for MAVIS-Bench tasks, episodes, and evaluation results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HiddenIntent:
    id: str
    content: str
    weight: float = 1.0
    satisfied_at_turn: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiddenIntent:
        return cls(
            id=str(data.get("id", "")),
            content=str(data.get("content", "")),
            weight=float(data.get("weight", 1.0)),
            satisfied_at_turn=data.get("satisfied_at_turn"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InjectedFault:
    trigger_step: str
    fault_type: str
    target_module: str | None = None
    expected_debugger_action: str | None = None
    applied: bool = False
    repaired: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InjectedFault:
        return cls(
            trigger_step=str(data.get("trigger_step", "")),
            fault_type=str(data.get("fault_type", "runtime_error")),
            target_module=data.get("target_module"),
            expected_debugger_action=data.get("expected_debugger_action"),
            applied=bool(data.get("applied", False)),
            repaired=bool(data.get("repaired", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationArtifact:
    path: str
    must_contain: list[str] = field(default_factory=list)
    min_lines: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationArtifact:
        return cls(
            path=str(data.get("path", "")),
            must_contain=list(data.get("must_contain", [])),
            min_lines=data.get("min_lines"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationSpec:
    script: str = "verify.py"
    artifacts: list[VerificationArtifact] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VerificationSpec:
        if not data:
            return cls()
        artifacts = [VerificationArtifact.from_dict(a) for a in data.get("artifacts", [])]
        return cls(
            script=str(data.get("script", "verify.py")),
            artifacts=artifacts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "script": self.script,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


@dataclass
class TaskSpec:
    task_id: str
    persona: str
    title: str
    initial_input: str
    schema_version: str = "mavis_v1"
    difficulty: str = "medium"
    hidden_intents: list[HiddenIntent] = field(default_factory=list)
    injected_faults: list[InjectedFault] = field(default_factory=list)
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_dir: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], task_dir: Path | None = None) -> TaskSpec:
        trigger = data.get("trigger", {})
        initial_input = (
            trigger.get("initial_input")
            or data.get("initial_input")
            or data.get("goal_prompt")
            or ""
        )

        intents = [HiddenIntent.from_dict(h) for h in data.get("hidden_intents", [])]
        faults = [InjectedFault.from_dict(f) for f in data.get("injected_faults", [])]
        verification = VerificationSpec.from_dict(data.get("verification"))

        return cls(
            schema_version=str(data.get("schema_version", "mavis_v1")),
            task_id=str(data.get("task_id", "")),
            persona=str(data.get("persona", "")),
            title=str(data.get("title", "")),
            difficulty=str(data.get("difficulty", "medium")),
            initial_input=str(initial_input),
            hidden_intents=intents,
            injected_faults=faults,
            verification=verification,
            depends_on=list(data.get("depends_on", [])),
            metadata=dict(data.get("metadata", {})),
            task_dir=task_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "persona": self.persona,
            "title": self.title,
            "difficulty": self.difficulty,
            "initial_input": self.initial_input,
            "hidden_intents": [h.to_dict() for h in self.hidden_intents],
            "injected_faults": [f.to_dict() for f in self.injected_faults],
            "verification": self.verification.to_dict(),
            "depends_on": list(self.depends_on),
            "metadata": dict(self.metadata),
        }


@dataclass
class EpisodeSpec:
    episode_id: str
    persona: str
    schema_version: str = "mavis_v1"
    tasks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeSpec:
        return cls(
            schema_version=str(data.get("schema_version", "mavis_v1")),
            episode_id=str(data.get("episode_id", "")),
            persona=str(data.get("persona", data.get("user_id", ""))),
            tasks=list(data.get("tasks", [])),
        )


@dataclass
class PersonaProfile:
    persona_id: str
    name: str
    description: str
    default_workspace: str
    preferences: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaProfile:
        return cls(
            persona_id=str(data.get("persona_id", data.get("user_id", ""))),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            default_workspace=str(data.get("default_workspace", "workspace")),
            preferences=list(data.get("preferences", [])),
        )


@dataclass
class TaskResult:
    task_id: str
    persona: str
    status: str  # SUCCESS, FAILED, TIMEOUT, MAX_TURNS, ERROR
    turns_taken: int
    waves_taken: int
    proc_score: float = 0.0
    comp_score: float = 0.0
    heal_score: float = 1.0
    safe_score: float = 1.0
    satisfied_intents: list[str] = field(default_factory=list)
    verification_passed: bool = False
    verification_output: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    episode_id: str
    persona: str
    task_results: list[TaskResult] = field(default_factory=list)
    overall_proc: float = 0.0
    overall_comp: float = 0.0
    overall_heal: float = 0.0
    overall_safe: float = 0.0
    overall_score: float = 0.0
    duration_seconds: float = 0.0

    def calculate_averages(self) -> None:
        if not self.task_results:
            return
        n = len(self.task_results)
        self.overall_proc = round(sum(t.proc_score for t in self.task_results) / n, 4)
        self.overall_comp = round(sum(t.comp_score for t in self.task_results) / n, 4)
        self.overall_heal = round(sum(t.heal_score for t in self.task_results) / n, 4)
        self.overall_safe = round(sum(t.safe_score for t in self.task_results) / n, 4)
        # Weighted composite score: 35% COMP, 35% PROC, 15% SAFE, 15% HEAL
        self.overall_score = round(
            0.35 * self.overall_comp + 0.35 * self.overall_proc + 0.15 * self.overall_safe + 0.15 * self.overall_heal,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "persona": self.persona,
            "overall_score": self.overall_score,
            "overall_proc": self.overall_proc,
            "overall_comp": self.overall_comp,
            "overall_heal": self.overall_heal,
            "overall_safe": self.overall_safe,
            "duration_seconds": self.duration_seconds,
            "task_count": len(self.task_results),
            "tasks": [t.to_dict() for t in self.task_results],
        }
