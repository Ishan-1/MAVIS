"""
Unit tests for GoalRunner scratchpad / blackboard management.
"""
import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.goal_runner import GoalRunner


@pytest.fixture(autouse=True)
def clean_test_scratchpad():
    scratch_goals = Path("data/scratch/goals")
    yield
    # Cleanup any test goal dirs
    for test_dir in scratch_goals.glob("test_goal_*"):
        shutil.rmtree(test_dir, ignore_errors=True)


def test_scratchpad_init_and_update():
    runner = GoalRunner(llm=MagicMock())
    goal_id = "test_goal_123"
    goal = "Refactor module and verify"

    sp = runner._init_scratchpad(goal_id=goal_id, goal=goal)
    assert sp["goal_id"] == goal_id
    assert sp["goal"] == goal
    assert sp["iteration_state"] == {}
    assert sp["artifacts"] == {}
    assert sp["notes_and_findings"] == []

    sp_file = runner._get_scratchpad_path(goal_id)
    assert sp_file.exists()

    # Update scratchpad
    updates = {
        "iteration_state": {"current_index": 5, "backlog": ["a.py", "b.py"]},
        "notes_and_findings": ["Found deprecation warning in a.py"],
        "artifacts": {"artifact_1": {"path": "data/scratch/dummy.txt"}},
    }
    updated_sp = runner._update_scratchpad(goal_id, sp, updates)
    assert updated_sp["iteration_state"]["current_index"] == 5
    assert len(updated_sp["iteration_state"]["backlog"]) == 2
    assert "Found deprecation warning in a.py" in updated_sp["notes_and_findings"]
    assert "artifact_1" in updated_sp["artifacts"]

    # Verify disk content matches
    with open(sp_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["iteration_state"]["current_index"] == 5


def test_register_artifacts_from_results():
    runner = GoalRunner(llm=MagicMock())
    goal_id = "test_goal_artifacts"
    sp = runner._init_scratchpad(goal_id=goal_id, goal="Test artifacts")

    # Create dummy spilled scratch file
    dummy_file = Path("data/scratch/turn1_step1.txt")
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.write_text("Spilled output content", encoding="utf-8")

    try:
        wave_results = {
            "step1": "Digest of output ... [250 bytes offloaded to file: data/scratch/turn1_step1.txt] ...",
            "step2": "Short output with no offload",
        }

        runner._register_artifacts_from_results(
            goal_id=goal_id,
            scratchpad=sp,
            wave=1,
            wave_results=wave_results,
        )

        assert "wave_1_step1" in sp["artifacts"]
        art = sp["artifacts"]["wave_1_step1"]
        assert art["path"] == "data/scratch/turn1_step1.txt"
        assert art["size_bytes"] > 0
        assert art["wave"] == 1
    finally:
        if dummy_file.exists():
            dummy_file.unlink()


def test_multi_wave_scratchpad_flow():
    mock_llm = MagicMock()

    wave_1_plan = {
        "status": "in_progress",
        "reasoning": "Wave 1: scan workspace",
        "scratchpad_updates": {
            "iteration_state": {"current_index": 1, "remaining_items": 3},
            "notes_and_findings": ["Discovered 4 source files"],
        },
        "pipeline": [
            {"id": "n1", "type": "tool", "function_name": "run_shell_command", "params": {"command": "ls"}}
        ],
    }

    wave_2_plan = {
        "status": "completed",
        "reasoning": "Wave 2: finish processing",
        "scratchpad_updates": {
            "iteration_state": {"current_index": 4, "remaining_items": 0},
            "notes_and_findings": ["All 4 files processed"],
        },
        "pipeline": [],
        "final_summary": "All items verified.",
    }

    mock_llm.generate.side_effect = [
        json.dumps(wave_1_plan),
        json.dumps(wave_2_plan),
    ]

    def dummy_executor(pipeline, query, turn_id):
        return {"n1": "file1.py\nfile2.py"}

    runner = GoalRunner(
        llm=mock_llm,
        execute_pipeline_fn=dummy_executor,
        memory_store=MagicMock(),
    )

    with patch("core.goal_runner._oni") as mock_oni:
        mock_oni.acquire_task_lease.return_value = "ask"
        res = runner.run_goal("Process backlog", turn_id="test_goal_flow", max_iterations=3)

        assert res["status"] == "completed"
        sp = res["scratchpad"]
        assert sp["iteration_state"]["current_index"] == 4
        assert sp["iteration_state"]["remaining_items"] == 0
        assert len(sp["notes_and_findings"]) == 2
        assert "Discovered 4 source files" in sp["notes_and_findings"]
        assert "All 4 files processed" in sp["notes_and_findings"]
