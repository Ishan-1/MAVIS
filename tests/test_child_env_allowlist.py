"""
Unit tests for child tool subprocess environment allowlist isolation (main.py).
"""
import os
from unittest.mock import MagicMock, patch
import pytest

from main import call_command, _TOOL_ENV_ALLOWLIST


def test_child_env_allowlist_strips_sensitive_credentials():
    sensitive_keys = {
        "GEMINI_API_KEY": "secret-gemini-key",
        "OPENAI_API_KEY": "secret-openai-key",
        "ANTHROPIC_API_KEY": "secret-anthropic-key",
        "NEO4J_PASSWORD": "secret-neo4j-password",
        "DATABASE_URL": "postgres://user:pass@localhost:5432/db",
        "AWS_SECRET_ACCESS_KEY": "secret-aws-key",
    }
    allowed_keys = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/mavis_user",
        "NEWS_API_KEY": "test-news-key",
        "MAVIS_CUSTOM_FEATURE": "enabled",
        "MAVIS_TOOL_WEATHER_API_KEY": "weather-api-secret-123",
    }

    test_env = {**os.environ, **sensitive_keys, **allowed_keys}

    captured_env = None

    def mock_popen(args, **kwargs):
        nonlocal captured_env
        captured_env = kwargs.get("env")

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.poll.return_value = 0
        return mock_proc

    with patch.dict(os.environ, test_env, clear=True):
        with patch("subprocess.Popen", side_effect=mock_popen):
            with patch("selectors.DefaultSelector") as mock_sel_cls:
                mock_sel = MagicMock()
                mock_sel.select.return_value = []
                mock_sel_cls.return_value = mock_sel

                with patch("main._oni") as mock_oni:
                    mock_oni._effective_trust.return_value = "ask"
                    mock_oni.get_active_workspace.return_value = "/tmp/workspace"
                    mock_oni.session_allowances = set()
                    mock_oni.config.approved_fs_write_paths = ["data/scratch/"]

                    # Execute command
                    try:
                        call_command("get_current_datetime", {})
                    except Exception:
                        pass

    assert captured_env is not None, "Popen was not called with an env dictionary"

    # Verify sensitive keys were stripped
    for key in sensitive_keys:
        assert key not in captured_env, f"Sensitive credential '{key}' leaked to tool subprocess!"

    # Verify allowed and MAVIS_* keys were retained
    assert captured_env.get("PATH") == "/usr/bin:/bin"
    assert captured_env.get("HOME") == "/home/mavis_user"
    assert captured_env.get("NEWS_API_KEY") == "test-news-key"
    assert captured_env.get("MAVIS_CUSTOM_FEATURE") == "enabled"
    assert captured_env.get("MAVIS_TOOL_WEATHER_API_KEY") == "weather-api-secret-123"
    assert captured_env.get("MAVIS_TOOL_SUBPROCESS") == "1"
    assert captured_env.get("MAVIS_TRUST_LEVEL") == "ask"
    assert captured_env.get("MAVIS_ACTIVE_WORKSPACE") == "/tmp/workspace"
