from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import uuid4

from app.models.schemas import ReasoningStep

LOGGER = logging.getLogger(__name__)


@dataclass
class StepTracer:
    """Collects reasoning steps for API responses and persistent logs."""

    query: str
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    steps: list[ReasoningStep] = field(default_factory=list)

    def add(
        self,
        action: str,
        detail: str,
        tool_input: dict | None = None,
        tool_output: dict | None = None,
    ) -> None:
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            trace_id=self.trace_id,
            action=action,
            detail=detail,
            tool_input=tool_input or {},
            tool_output=tool_output or {},
        )
        self.steps.append(step)

        LOGGER.info(
            "agent_step",
            extra={
                "trace_id": self.trace_id,
                "step_number": step.step_number,
                "action": action,
                "detail": detail,
                "tool_input": step.tool_input,
                "tool_output": step.tool_output,
            },
        )
