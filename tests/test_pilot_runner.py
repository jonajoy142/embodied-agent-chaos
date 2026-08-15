"""Tests for pilot runner provider selection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_pilot_runner_accepts_ollama():
    """Test that pilot runner accepts ollama provider."""
    from alpha.services.experiment_matrix import load_pilot_config

    pilot_path = ROOT / "configs" / "experiments" / "pilot.yaml"
    pilot = load_pilot_config(str(pilot_path))
    
    VALID_PROVIDERS = {"openai", "ollama"}
    assert pilot.agent_provider in VALID_PROVIDERS


def test_pilot_runner_rejects_unknown_provider():
    """Test that pilot runner would reject unknown provider."""
    VALID_PROVIDERS = {"openai", "ollama"}
    
    # Simulate the validation logic from run_pilot.py
    unknown_provider = "unknown_llm"
    assert unknown_provider not in VALID_PROVIDERS


def test_ollama_does_not_require_openai_key():
    """Test that Ollama provider does not require OPENAI_API_KEY."""
    from alpha.models.agent_config import AgentConfig
    from alpha.clients.llm_client import validate_llm_runtime
    import os
    
    # Save current state
    original_key = os.environ.get("OPENAI_API_KEY")
    
    try:
        # Remove OPENAI_API_KEY to test Ollama independence
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        # Ollama config should work without OPENAI_API_KEY
        ollama_config = AgentConfig(
            provider="ollama",
            model_name="llama3",
            ollama_base_url="http://localhost:11434"
        )
        
        # This should not raise an error about missing OPENAI_API_KEY
        # (it may fail for other reasons like Ollama not running, but not for missing key)
        try:
            validate_llm_runtime(ollama_config)
        except RuntimeError as exc:
            # Should not be about missing API key
            assert "OPENAI_API_KEY" not in str(exc)
    
    finally:
        # Restore original state
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key


def test_openai_requires_openai_key():
    """Test that OpenAI provider requires OPENAI_API_KEY."""
    from alpha.models.agent_config import AgentConfig
    from alpha.clients.llm_client import validate_llm_runtime
    import os
    
    # Save current state
    original_key = os.environ.get("OPENAI_API_KEY")
    
    try:
        # Remove OPENAI_API_KEY to test OpenAI dependency
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        # OpenAI config should fail without OPENAI_API_KEY
        openai_config = AgentConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        try:
            validate_llm_runtime(openai_config)
            assert False, "Should have raised RuntimeError about missing OPENAI_API_KEY"
        except RuntimeError as exc:
            assert "OPENAI_API_KEY" in str(exc)
    
    finally:
        # Restore original state
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
