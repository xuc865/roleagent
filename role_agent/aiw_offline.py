"""
Structured records for Agent-In-World (AIW) workflows (e.g. LLM retrieval over a failure library).

For training-time curriculum when ``algorithm.role_agent.enable_aiw`` is enabled, see
``AIWCurriculum`` in ``aiw_curriculum.py`` and ``docs/role_agent_alignment.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FailureHistoryEntry:
    """One row in an offline failure library used for retrieval-style workflows."""

    index: int
    task_description: str
    failure_analysis: str
    trajectory_summary: str = ""

    def as_candidate_block(self) -> str:
        parts = [
            f"[{self.index}] TASK: {self.task_description}",
            f"FAILURE_ANALYSIS: {self.failure_analysis}",
        ]
        if self.trajectory_summary:
            parts.append(f"TRAJECTORY: {self.trajectory_summary}")
        return "\n".join(parts)


def format_candidate_catalog(entries: List[FailureHistoryEntry]) -> str:
    """Build `candidates_text` for PROMPT_RETRIEVE_SIMILAR_FAILURES."""
    return "\n\n".join(e.as_candidate_block() for e in entries)


@dataclass
class FailureModeLibrary:
    """In-memory failure library for AIW-style candidate formatting."""

    entries: List[FailureHistoryEntry] = field(default_factory=list)

    def append(self, entry: FailureHistoryEntry) -> None:
        entry.index = len(self.entries)
        self.entries.append(entry)

    def candidates_text(self) -> str:
        return format_candidate_catalog(self.entries)
