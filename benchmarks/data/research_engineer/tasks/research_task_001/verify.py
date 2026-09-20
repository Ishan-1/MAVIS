"""
Verification script for research_task_001.
Checks that papers_comparison.md was created, contains comparison of key papers, and lists GitHub links.
"""
from pathlib import Path


def verify(workspace_path: str) -> tuple[bool, str]:
    ws = Path(workspace_path)
    comparison = ws / "papers_comparison.md"
    if not comparison.exists():
        return False, "papers_comparison.md was not created."

    content = comparison.read_text(encoding="utf-8")
    lower = content.lower()

    if "|" not in content:
        return False, "papers_comparison.md should contain a Markdown table."

    if "pi-bench" not in lower or "appworld" not in lower:
        return False, "Missing coverage of key papers (Pi-Bench or AppWorld)."

    if "github" not in lower and "http" not in lower:
        return False, "Missing GitHub repository links or reproducibility analysis."

    return True, "Comparative literature analysis and table generated successfully."


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msg = verify(path)
    print(msg)
    sys.exit(0 if ok else 1)
