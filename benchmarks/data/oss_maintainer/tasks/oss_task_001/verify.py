"""
Verification script for oss_task_001.
Runs pytest on tests/test_math_lib.py and verifies pyproject.toml version is 1.1.0.
"""
import subprocess
import sys
from pathlib import Path


def verify(workspace_path: str) -> tuple[bool, str]:
    ws = Path(workspace_path)

    # 1. Run tests
    test_file = ws / "tests" / "test_math_lib.py"
    if not test_file.exists():
        return False, "test_math_lib.py not found in workspace."

    res = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file)],
        capture_output=True,
        text=True,
        cwd=str(ws),
    )
    if res.returncode != 0:
        return False, f"Tests failed:\n{res.stdout}\n{res.stderr}"

    # 2. Check pyproject.toml version
    pyproject = ws / "pyproject.toml"
    if not pyproject.exists():
        return False, "pyproject.toml missing."

    content = pyproject.read_text(encoding="utf-8")
    if 'version = "1.1.0"' not in content and "version = '1.1.0'" not in content:
        return False, "pyproject.toml version was not bumped to 1.1.0."

    return True, "All tests pass and version bumped to 1.1.0."


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msg = verify(path)
    print(msg)
    sys.exit(0 if ok else 1)
