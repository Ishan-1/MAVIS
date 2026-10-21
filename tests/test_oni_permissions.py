"""
Unit tests for ONI filesystem permission gate and protected path enforcement.
"""
import os
import pytest

from types import SimpleNamespace
from oni.permissions import classify_fs_write, _is_protected_path


@pytest.fixture
def base_config():
    return SimpleNamespace(
        trust_level="ask",
        approved_fs_write_paths=[
            "tools/",
            "tests/",
            "memories/",
            "logs/",
            "data/scratch/",
        ],
        approved_fs_files=["requirements.txt"],
    )


def test_protected_paths_unconditional_denial_in_yolo(base_config):
    """Protected paths (.env, mavis_config.json, oni/) MUST be denied even in yolo mode."""
    protected_cases = [
        ".env",
        "./.env",
        "data/../.env",
        "/custom/workspace/.env",
        "data/mavis_config.json",
        "./data/mavis_config.json",
        "data/../data/mavis_config.json",
        "oni/",
        "oni/permissions.py",
        "oni/config.py",
        "data/../oni/permissions.py",
        "tools/../oni/config.py",
    ]

    for path in protected_cases:
        decision = classify_fs_write(path, base_config, trust_level="yolo")
        assert decision == "deny", f"Protected path '{path}' was not denied in yolo mode!"


def test_protected_paths_unconditional_denial_in_ask_and_whitelist(base_config):
    """Protected paths must also be denied in ask and whitelist_only modes."""
    for mode in ("ask", "whitelist_only"):
        assert classify_fs_write(".env", base_config, trust_level=mode) == "deny"
        assert classify_fs_write("data/mavis_config.json", base_config, trust_level=mode) == "deny"
        assert classify_fs_write("oni/permissions.py", base_config, trust_level=mode) == "deny"


def test_narrowed_data_write_paths(base_config):
    """data/scratch/ is approved, but arbitrary files in data/ are greylisted in ask mode."""
    # Under data/scratch/ -> allowed
    assert classify_fs_write("data/scratch/output.txt", base_config, trust_level="ask") == "allow"
    assert classify_fs_write("data/scratch/nested/dir/test.json", base_config, trust_level="ask") == "allow"

    # In data/ directly (not in scratch/) -> greylist (requires user prompt in ask mode)
    assert classify_fs_write("data/arbitrary.txt", base_config, trust_level="ask") == "greylist"
    assert classify_fs_write("data/new_dataset.csv", base_config, trust_level="ask") == "greylist"


def test_standard_approved_paths_and_files(base_config):
    """Standard approved directories and exact files remain allowed."""
    assert classify_fs_write("tools/my_tool.py", base_config, trust_level="ask") == "allow"
    assert classify_fs_write("tests/test_something.py", base_config, trust_level="ask") == "allow"
    assert classify_fs_write("requirements.txt", base_config, trust_level="ask") == "allow"


def test_unapproved_paths_behavior(base_config):
    """Unapproved external paths follow trust level resolution."""
    unapproved = "/tmp/some_random_file.txt"
    assert classify_fs_write(unapproved, base_config, trust_level="yolo") == "allow"
    assert classify_fs_write(unapproved, base_config, trust_level="ask") == "greylist"
    assert classify_fs_write(unapproved, base_config, trust_level="whitelist_only") == "deny"
