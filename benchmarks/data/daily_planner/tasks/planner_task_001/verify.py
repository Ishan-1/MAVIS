"""
Verification script for planner_task_001.
Checks that daily_briefing.md exists, lists scheduled events, and notes the 2 PM conflict.
"""
from pathlib import Path


def verify(workspace_path: str) -> tuple[bool, str]:
    ws = Path(workspace_path)
    briefing = ws / "daily_briefing.md"
    if not briefing.exists():
        return False, "daily_briefing.md was not generated."

    text = briefing.read_text(encoding="utf-8").lower()
    if "standup" not in text:
        return False, "Missing Standup meeting in daily briefing."
    if "conflict" not in text and "overlap" not in text:
        return False, "Failed to note the 2:00 PM calendar conflict in briefing."

    return True, "Daily briefing correctly reflects calendar and flags conflicts."


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msg = verify(path)
    print(msg)
    sys.exit(0 if ok else 1)
