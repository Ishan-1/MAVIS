import os
import json
from oni import call_fs


def read_emails(
    query: str = None,
    sender: str = None,
    subject: str = None,
    start_date: str = None,
    end_date: str = None,
    unread_only: bool = False,
) -> tuple[int, list[dict]]:
    """
    Reads emails from the inbox with optional filtering by sender, subject, keywords, date range, or unread status.
    Returns a tuple of (status_code, list_of_email_dicts).
    """
    try:
        candidate_dirs = [
            "inbox",
            "workspace/inbox",
            "./inbox",
            "./workspace/inbox",
            "data/inbox",
        ]
        target_dir = None
        for d in candidate_dirs:
            if os.path.isdir(d):
                target_dir = d
                break

        if not target_dir:
            from oni import oni as _oni
            ws = _oni.get_active_workspace()
            if ws and os.path.isdir(os.path.join(ws, "inbox")):
                target_dir = os.path.join(ws, "inbox")

        raw_emails = []

        # 1. Try reading individual JSON email files from inbox/ directory
        if target_dir and os.path.isdir(target_dir):
            for fname in sorted(os.listdir(target_dir)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(target_dir, fname)
                try:
                    status, content = call_fs("read", fpath)
                    if status == 0 and content:
                        data = json.loads(content) if isinstance(content, str) else content
                        if isinstance(data, dict):
                            raw_emails.append(data)
                        elif isinstance(data, list):
                            raw_emails.extend(data)
                except Exception:
                    pass

        # 2. Try reading data/emails.json fallback if no inbox files found
        if not raw_emails:
            for ef in ("data/emails.json", "emails.json", "workspace/data/emails.json"):
                if os.path.exists(ef):
                    try:
                        status, content = call_fs("read", ef)
                        if status == 0 and content:
                            data = json.loads(content) if isinstance(content, str) else content
                            if isinstance(data, list):
                                raw_emails.extend(data)
                                break
                    except Exception:
                        pass

        # 3. Apply filters
        filtered = []
        for em in raw_emails:
            if not isinstance(em, dict):
                continue

            em_sender = str(em.get("sender") or em.get("from") or "")
            em_subject = str(em.get("subject") or "")
            em_body = str(em.get("body") or "")
            em_date = str(em.get("date") or "")
            em_read = bool(em.get("read", False))

            if unread_only and em_read:
                continue
            if sender and sender.lower() not in em_sender.lower():
                continue
            if subject and subject.lower() not in em_subject.lower():
                continue
            if query:
                q = query.lower()
                if q not in em_sender.lower() and q not in em_subject.lower() and q not in em_body.lower():
                    continue
            if start_date and em_date and em_date < start_date:
                continue
            if end_date and em_date and em_date > end_date:
                continue

            filtered.append({
                "id": em.get("id", ""),
                "sender": em_sender,
                "from": em_sender,
                "subject": em_subject,
                "body": em_body,
                "date": em_date,
                "read": em_read,
            })

        return 0, filtered
    except Exception as e:
        return -1, str(e)
