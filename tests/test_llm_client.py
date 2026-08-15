import json
import sys
import types

import pytest

from alpha.clients.experiment_llm_clients import RateLimitedLLMClient, TelemetryLLMClient
from alpha.clients.llm_client import OllamaActionPlanClient, OpenAIActionPlanClient, build_llm_client
from alpha.config import settings
from alpha.core.entities import BlockColor, TaskSpec
from alpha.models.agent_config import AgentConfig
from alpha.services.matrix_episode_runner import _format_exception_failure


def test_openai_client_fails_at_construction_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "openai" else object())

    with pytest.raises(RuntimeError, match="Install the optional 'openai' package"):
        build_llm_client(AgentConfig(provider="openai", model_name="gpt-4o"))


def test_build_openai_client_still_constructs_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "openai" else None)

    client = build_llm_client(AgentConfig(provider="openai", model_name="gpt-4o"))

    assert isinstance(client, OpenAIActionPlanClient)
    assert client.name == "openai"


def test_unknown_provider_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported agent provider: anthropic"):
        build_llm_client(AgentConfig(provider="anthropic", model_name="claude"))


def test_openai_client_records_response_token_usage(monkeypatch):
    class FakeResponses:
        def create(self, *, model, input):
            assert model == "gpt-4o"
            assert "pick-and-place" in input
            return types.SimpleNamespace(
                output_text=json.dumps(
                    {
                        "agent_name": "openai",
                        "actions": [
                            {"action": "MOVE_TO", "target": [0.5, -0.15, 0.83]},
                            {"action": "GRIP", "block": "red"},
                            {"action": "MOVE_TO", "target": [0.6, 0.35, 0.71]},
                            {"action": "RELEASE"},
                        ],
                    }
                ),
                usage=types.SimpleNamespace(input_tokens=123, output_tokens=45),
            )

    class FakeOpenAI:
        def __init__(self, *, api_key):
            assert api_key == "dummy"
            self.responses = FakeResponses()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "openai" else None)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = OpenAIActionPlanClient(AgentConfig(provider="openai", model_name="gpt-4o"))
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)

    plan = client.generate_action_plan(task)

    assert plan.agent_name == "openai"
    assert client.last_prompt_tokens == 123
    assert client.last_completion_tokens == 45


def test_build_ollama_client_does_not_require_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = build_llm_client(
        AgentConfig(
            provider="ollama",
            model_name="llama3.2",
            ollama_base_url="http://ollama.test:11434",
        )
    )

    assert isinstance(client, OllamaActionPlanClient)
    assert client.name == "ollama"
    assert client.model_name == "llama3.2"
    assert client.base_url == "http://ollama.test:11434"


def test_ollama_response_is_converted_to_action_plan(monkeypatch):
    def fake_request(self, payload):
        assert payload["model"] == "llama3.2"
        assert payload["format"] == "json"
        assert payload["stream"] is False
        return {
            "message": {
                "content": json.dumps(
                    {
                        "agent_name": "ollama",
                        "actions": [
                            {"action": "MOVE_TO", "target": [0.5, -0.15, 0.83]},
                            {"action": "GRIP", "block": "red"},
                            {"action": "MOVE_TO", "target": [0.6, 0.35, 0.71]},
                            {"action": "RELEASE"},
                        ],
                    }
                )
            },
            "prompt_eval_count": 88,
            "eval_count": 31,
        }

    monkeypatch.setattr(OllamaActionPlanClient, "_request_ollama", fake_request)
    client = OllamaActionPlanClient(AgentConfig(provider="ollama", model_name="llama3.2"))
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)

    plan = client.generate_action_plan(task)

    assert plan.agent_name == "ollama"
    assert len(plan.actions) == 4
    assert client.last_prompt_tokens == 88
    assert client.last_completion_tokens == 31


def test_invalid_ollama_output_is_rejected_safely(monkeypatch):
    def fake_request(self, payload):
        return {"message": {"content": "not json"}}

    monkeypatch.setattr(OllamaActionPlanClient, "_request_ollama", fake_request)
    client = OllamaActionPlanClient(AgentConfig(provider="ollama", model_name="llama3.2"))
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)

    with pytest.raises(RuntimeError, match="LLM did not return valid JSON"):
        client.generate_action_plan(task)


def test_openai_failure_does_not_invoke_ollama(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ollama_calls = []

    def fake_ollama_init(self, config):
        ollama_calls.append(config)

    monkeypatch.setattr(OllamaActionPlanClient, "__init__", fake_ollama_init)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        build_llm_client(AgentConfig(provider="openai", model_name="gpt-4o"))

    assert ollama_calls == []


def test_telemetry_llm_client_records_usage_through_wrappers():
    class FakeClient:
        name = "fake"
        last_prompt_tokens = 11
        last_completion_tokens = 7

        def generate_action_plan(self, task, scene_info=None):
            return []

    class FakeTelemetry:
        def __init__(self):
            self.calls = []

        def record_llm_usage(self, *, prompt_tokens=0, completion_tokens=0):
            self.calls.append((prompt_tokens, completion_tokens))

    telemetry = FakeTelemetry()
    client = TelemetryLLMClient(
        RateLimitedLLMClient(FakeClient(), min_interval_seconds=0.0),
        telemetry,
    )

    client.generate_action_plan(TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION))

    assert telemetry.calls == [(11, 7)]


def test_exception_failure_includes_type_message_and_source_line():
    def raise_nested():
        raise RuntimeError("Install the optional 'openai' package to use --agent openai.")

    try:
        raise_nested()
    except RuntimeError as exc:
        failure = _format_exception_failure(exc)

    assert failure.startswith("exception:RuntimeError: Install the optional 'openai' package")
    assert "test_llm_client.py:" in failure
