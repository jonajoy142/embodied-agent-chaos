"""
Provider-agnostic LLM client boundary for the Week 2 Brain.

Supported providers:
- mock   : deterministic local implementation
- openai : OpenAI Responses API
- ollama : local Ollama server

All providers MUST return the same validated ActionPlan schema.

Architecture:

    TaskSpec
        ↓
    LLMClient
        ↓
    ActionPlan
        ↓
    AgentService
        ↓
    ControlService
        ↓
    PyBullet
"""

from __future__ import annotations

import json
import os
import re
import importlib.util
from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from alpha.config import settings
from alpha.core.entities import ActionType, TaskSpec
from alpha.models.agent_config import AgentConfig
from alpha.schemas.action_plan import ActionPlan


class LLMClient(ABC):
    """Common interface for every Brain/LLM provider."""

    name: str

    @abstractmethod
    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:
        """Return a validated structured action plan."""
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    Deterministic local Brain.

    This is intentionally not an LLM.
    It exists so the complete Brain → Body → PyBullet
    pipeline can be tested without an external model.
    """

    name = "mock"

    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:

        spawn_pos = settings.BLOCK_SPAWN_POSITIONS[task.block]

        above_block = (
            spawn_pos[0],
            spawn_pos[1],
            spawn_pos[2] + 0.15,
        )

        grasp_pos = (
            spawn_pos[0],
            spawn_pos[1],
            spawn_pos[2] + 0.02,
        )

        above_place = (
            task.place_zone[0],
            task.place_zone[1],
            task.place_zone[2] + 0.15,
        )

        release_pos = (
            task.place_zone[0],
            task.place_zone[1],
            task.place_zone[2]
            + settings.PLACE_RELEASE_HEIGHT_OFFSET,
        )

        return ActionPlan(
            agent_name=self.name,
            actions=[
                {
                    "action": ActionType.MOVE_TO,
                    "target": above_block,
                },
                {
                    "action": ActionType.MOVE_TO,
                    "target": grasp_pos,
                },
                {
                    "action": ActionType.GRIP,
                    "block": task.block,
                },
                {
                    "action": ActionType.MOVE_TO,
                    "target": above_block,
                },
                {
                    "action": ActionType.MOVE_TO,
                    "target": above_place,
                },
                {
                    "action": ActionType.MOVE_TO,
                    "target": release_pos,
                },
                {
                    "action": ActionType.RELEASE,
                },
            ],
        )


class OpenAIActionPlanClient(LLMClient):
    """
    OpenAI-backed Brain.

    The model is only responsible for generating structured JSON.
    It does NOT execute Python or interact directly with PyBullet.
    """

    name = "openai"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        self.api_key = os.getenv(config.api_key_env_var)

        if not self.api_key:
            raise RuntimeError(
                f"{config.api_key_env_var} is not set"
            )

        if importlib.util.find_spec("openai") is None:
            raise RuntimeError(
                "Install the optional 'openai' package "
                "to use --agent openai."
            )

    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'openai' package "
                "to use --agent openai."
            ) from exc

        client = OpenAI(api_key=self.api_key)

        prompt = _build_robot_prompt(
            task=task,
            scene_info=scene_info,
            agent_name=self.name,
        )

        response = client.responses.create(
            model=self.config.model_name,
            input=prompt,
        )
        self._record_usage(response)

        raw_text = response.output_text

        raw_plan = _parse_json_response(raw_text)

        plan = ActionPlan.validate_raw_plan(
            raw_plan,
            agent_name=self.name,
        )

        return plan.model_copy(
            update={"agent_name": self.name}
        )

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            return

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", input_tokens)
            output_tokens = usage.get("output_tokens", output_tokens)

        self.last_prompt_tokens = int(input_tokens or 0)
        self.last_completion_tokens = int(output_tokens or 0)


class OllamaActionPlanClient(LLMClient):
    """
    Local Ollama-backed Brain.

    Ollama normally runs locally at:

        http://localhost:11434

    No API key is required.

    The model receives a constrained robot planning prompt and must
    return JSON matching the ActionPlan schema.
    """

    name = "ollama"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        # Allow explicit environment override.
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            config.ollama_base_url,
        ).rstrip("/")

        # Model can be configured through OLLAMA_MODEL.
        #
        # If that is not present, use AgentConfig.model_name.
        configured_model = os.getenv("OLLAMA_MODEL")

        if configured_model:
            self.model_name = configured_model
        else:
            self.model_name = config.model_name

        if not self.model_name:
            raise RuntimeError(
                "No Ollama model configured. "
                "Set OLLAMA_MODEL, for example:\n\n"
                "export OLLAMA_MODEL=llama3.2"
            )

    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:

        prompt = _build_robot_prompt(
            task=task,
            scene_info=scene_info,
            agent_name=self.name,
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Brain of a robot manipulation "
                        "system. You produce structured robot "
                        "action plans. Never output Python code. "
                        "Never execute tools. Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,

            # Ask Ollama for JSON rather than Markdown prose.
            "format": "json",

            "options": {
                "temperature": 0,
            },
        }

        response_data = self._request_ollama(payload)
        self._record_usage(response_data)

        raw_text = _extract_ollama_text(response_data)

        raw_plan = _parse_json_response(raw_text)

        plan = ActionPlan.validate_raw_plan(
            raw_plan,
            agent_name=self.name,
        )

        return plan.model_copy(
            update={"agent_name": self.name}
        )

    def _record_usage(self, response_data: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(response_data.get("prompt_eval_count") or 0)
        self.last_completion_tokens = int(response_data.get("eval_count") or 0)

    def _request_ollama(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        url = f"{self.base_url}/api/chat"

        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")

        except HTTPError as exc:
            error_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"Ollama request failed with HTTP "
                f"{exc.code}: {error_body}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.base_url}.\n\n"
                "Make sure Ollama is running and try:\n"
                "  ollama list\n"
                "  ollama serve"
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                "Ollama request timed out after 120 seconds."
            ) from exc

        try:
            return json.loads(body)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc


def _build_robot_prompt(
    task: TaskSpec,
    scene_info: dict[str, Any] | None = None,
    agent_name: str = "llm",
) -> str:
    """
    Build the same planning prompt for OpenAI and Ollama.

    The model is intentionally restricted to three action types:
    MOVE_TO, GRIP, RELEASE.
    """
    block_position = _block_position_from_scene(task, scene_info)
    above_block = (
        block_position[0],
        block_position[1],
        block_position[2] + 0.15,
    )
    grasp_position = (
        block_position[0],
        block_position[1],
        block_position[2] + 0.02,
    )
    above_place = (
        task.place_zone[0],
        task.place_zone[1],
        task.place_zone[2] + 0.15,
    )
    release_position = (
        task.place_zone[0],
        task.place_zone[1],
        task.place_zone[2] + settings.PLACE_RELEASE_HEIGHT_OFFSET,
    )

    return f"""
You are the Brain for a robot pick-and-place task.

Your job is to create a safe, minimal action plan.

Task:
- block: {task.block.value}
- place_zone: {task.place_zone}
- description: {task.description}

Scene information:
{scene_info or {}}

Use this canonical pick-and-place sequence:

1. Move above the {task.block.value} block at {above_block}.
2. Move down to grasp the {task.block.value} block at {grasp_position}.
3. Grip the {task.block.value} block.
4. Lift back above the {task.block.value} block at {above_block}.
5. Move above the place zone at {above_place}.
6. Move to the release position at {release_position}.
7. Release the block.

Allowed actions ONLY:

1. MOVE_TO
   Required field:
   target: [x, y, z]

2. GRIP
   Required field:
   block: one of the available block colors

3. RELEASE
   No additional fields.

Return ONLY a JSON object with this structure:

{{
  "agent_name": "{agent_name}",
  "actions": [
    {{
      "action": "MOVE_TO",
      "target": {list(above_block)}
    }},
    {{
      "action": "MOVE_TO",
      "target": {list(grasp_position)}
    }},
    {{
      "action": "GRIP",
      "block": "{task.block.value}"
    }},
    {{
      "action": "MOVE_TO",
      "target": {list(above_block)}
    }},
    {{
      "action": "MOVE_TO",
      "target": {list(above_place)}
    }},
    {{
      "action": "MOVE_TO",
      "target": {list(release_position)}
    }},
    {{
      "action": "RELEASE"
    }}
  ]
}}

Rules:

- Do not output Python.
- Do not output Markdown.
- Do not explain your reasoning.
- Do not invent new action types.
- Only use MOVE_TO, GRIP, and RELEASE.
- Use numeric x, y, z coordinates.
- The final action should release the block.
""".strip()


def _block_position_from_scene(
    task: TaskSpec,
    scene_info: dict[str, Any] | None,
) -> tuple[float, float, float]:
    blocks = (scene_info or {}).get("blocks", {})
    if isinstance(blocks, dict):
        position = blocks.get(task.block.value) or blocks.get(task.block)
        if isinstance(position, (list, tuple)) and len(position) == 3:
            return (float(position[0]), float(position[1]), float(position[2]))
    return settings.BLOCK_SPAWN_POSITIONS[task.block]


def _extract_ollama_text(
    response_data: dict[str, Any],
) -> str:
    """
    Extract the assistant's text from Ollama /api/chat response.
    """

    message = response_data.get("message")

    if not isinstance(message, dict):
        raise RuntimeError(
            "Ollama response did not contain a valid "
            "'message' object."
        )

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "Ollama returned an empty model response."
        )

    return content.strip()


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    """
    Parse JSON returned by an LLM.

    Supports:
    - normal JSON
    - ```json ... ``` fenced JSON
    - accidental surrounding whitespace/text

    Validation against ActionPlan happens separately.
    """

    text = raw_text.strip()

    # Remove Markdown code fences if a model ignored
    # the "JSON only" instruction.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    try:
        parsed = json.loads(text)

    except json.JSONDecodeError:
        # Try to recover a JSON object surrounded by text.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(
                "LLM did not return valid JSON.\n\n"
                f"Raw response:\n{text}"
            )

        candidate = text[start : end + 1]

        try:
            parsed = json.loads(candidate)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "LLM returned malformed JSON.\n\n"
                f"Raw response:\n{text}"
            ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "LLM JSON response must be a JSON object."
        )

    return parsed


def build_llm_client(
    config: AgentConfig,
) -> LLMClient:
    """
    Construct the configured Brain provider.
    """

    if config.provider == "mock":
        return MockLLMClient()

    if config.provider == "openai":
        return OpenAIActionPlanClient(config)

    if config.provider == "ollama":
        return OllamaActionPlanClient(config)

    raise ValueError(
        f"Unsupported agent provider: {config.provider}"
    )


def validate_llm_runtime(config: AgentConfig) -> None:
    """Fail fast when the configured LLM provider cannot run."""
    build_llm_client(config)


# Backward-compatible alias.
LLMPlannerClient = LLMClient
