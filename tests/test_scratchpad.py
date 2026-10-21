"""
Unit tests for core/scratchpad.py
"""
import os
import shutil
from pathlib import Path
from core.scratchpad import spill_if_large, read_scratch_file, SCRATCH_DIR


def test_scratchpad_small_payload_no_spill():
    small_data = "Short output 123"
    result, path = spill_if_large("turn_test", "node_1", small_data, threshold_bytes=100)
    assert result == small_data
    assert path is None


def test_scratchpad_large_payload_spill():
    large_data = "Line of log output data\n" * 200
    byte_size = len(large_data.encode("utf-8"))
    assert byte_size > 1000

    result, path = spill_if_large("turn_test_large", "node_large", large_data, threshold_bytes=500)
    assert path is not None
    assert Path(path).exists()
    assert "offloaded to file" in result

    # Read back full content
    recovered = read_scratch_file(path)
    assert recovered == large_data

    # Cleanup test file
    if Path(path).exists():
        Path(path).unlink()


def test_scratchpad_large_json_spill():
    large_dict = {"data": [{"id": i, "val": f"value_{i}" * 10} for i in range(100)]}
    result, path = spill_if_large("turn_json", "node_json", large_dict, threshold_bytes=500)
    assert path is not None
    assert path.endswith(".json")
    assert Path(path).exists()
    assert "offloaded" in result

    # Cleanup
    if Path(path).exists():
        Path(path).unlink()
