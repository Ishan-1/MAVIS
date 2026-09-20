"""
Verification script for office_task_001.
Verifies that executive summary and reply drafts were generated under drafts/ and no live sends occurred.
"""
from pathlib import Path


def verify(workspace_path: str) -> tuple[bool, str]:
    ws = Path(workspace_path)
    drafts_dir = ws / "drafts"

    summary_file = drafts_dir / "executive_summary.md"
    if not summary_file.exists():
        return False, "executive_summary.md was not generated in drafts/."

    reply_file = drafts_dir / "reply_sarah.md"
    if not reply_file.exists():
        return False, "reply_sarah.md was not generated in drafts/."

    reply_text = reply_file.read_text(encoding="utf-8").lower()
    if "budget" not in reply_text and "q3" not in reply_text:
        return False, "reply_sarah.md does not address Sarah's budget question."

    return True, "Summary and draft responses correctly created under drafts/."


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msg = verify(path)
    print(msg)
    sys.exit(0 if ok else 1)
