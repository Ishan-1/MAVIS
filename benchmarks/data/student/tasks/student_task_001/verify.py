"""
Verification script for student_task_001.
Checks that rubric feedback was created and flashcards.tsv contains valid questions.
"""
from pathlib import Path


def verify(workspace_path: str) -> tuple[bool, str]:
    ws = Path(workspace_path)

    # 1. Flashcards TSV
    fc = ws / "notes" / "flashcards.tsv"
    if not fc.exists():
        return False, "notes/flashcards.tsv missing."

    lines = [line.strip() for line in fc.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        return False, f"Expected at least 3 flashcard entries, found {len(lines)}."

    # 2. Rubric feedback file
    fb = ws / "feedback" / "bft_review.md"
    if not fb.exists():
        # Check root of workspace as fallback
        fb = ws / "bft_review.md"
        if not fb.exists():
            return False, "Review feedback file (bft_review.md) not found."

    text = fb.read_text(encoding="utf-8").lower()
    if "3f" not in text and "quorum" not in text:
        return False, "Review does not critique Byzantine quorum or 3f+1 bound."

    return True, "Review feedback and flashcards successfully created."


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    ok, msg = verify(path)
    print(msg)
    sys.exit(0 if ok else 1)
