"""
LLM client wrappers used by the Week 6 experiment matrix runner.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
from alpha.clients.llm_client import LLMClient
from alpha.core.entities import TaskSpec
from alpha.schemas.action_plan import ActionPlan


class RateLimitedLLMClient(LLMClient):
    """Retry LLM calls with backoff when provider rate limits are hit."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        min_interval_seconds: float = 0.5,
        max_retries: int = 6,
        base_backoff_seconds: float = 2.0,
    ) -> None:
        self._inner = inner
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.name = inner.name
        self._last_call_at = 0.0

    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:
        for attempt in range(self.max_retries):
            self._respect_min_interval()
            try:
                return self._inner.generate_action_plan(task, scene_info)
            except Exception as exc:
                if attempt >= self.max_retries - 1 or not _is_rate_limit_error(exc):
                    raise
                delay = self.base_backoff_seconds * (2**attempt)
                time.sleep(delay)
        raise RuntimeError("RateLimitedLLMClient exhausted retries")

    def _respect_min_interval(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_call_at = time.monotonic()


class ChaosPlannerLLMClient(LLMClient):
    """Apply planner-output corruption faults without modifying the Brain service."""

    def __init__(self, inner: LLMClient, chaos_wrapper: ChaosRoboticsWrapper) -> None:
        self._inner = inner
        self._chaos = chaos_wrapper
        self.name = inner.name

    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:
        plan = self._inner.generate_action_plan(task, scene_info)
        raw = json.dumps(plan.model_dump(mode="json"), default=str)
        corrupted = self._chaos.intercept_planner_output(raw)
        parsed = json.loads(corrupted)
        return ActionPlan.validate_raw_plan(parsed, agent_name=self.name)


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "rate limit",
        "429",
        "too many requests",
        "quota",
        "capacity",
        "overloaded",
        "retry later",
    )
    if any(marker in message for marker in markers):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code == 429 or bool(re.search(r"\b429\b", message))
