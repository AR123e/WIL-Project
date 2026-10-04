"""
Live usage log: every question asked in the app (and thumbs-up/down feedback) is appended to logs/usage.jsonl.
The dashboard reads it to show real-time stats. Questions stay on your machine (logs/ is git-ignored).
Set CHESS_LOG_PATH to write somewhere else (used by the tests).
"""
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from src.config import ROOT


def log_path():
    return Path(os.getenv("CHESS_LOG_PATH", ROOT / "logs" / "usage.jsonl"))


def _append(event):
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **event}, ensure_ascii=False) + "\n")


def log_question(question, res):
    """Record one answered/declined question and return its id (used to attach feedback)."""
    qid = uuid.uuid4().hex[:8]
    _append({"type": "ask", "id": qid, "question": question, "routed": res.get("routed"),
             "declined": bool(res["declined"]), "reason": res.get("reason"), "latency": round(res["latency"], 2),
             "uncited": bool(res.get("uncited")), "invalid_citations": len(res.get("invalid_citations", [])),
             "top_chunk": res["chunks"][0]["chunk_id"] if res.get("chunks") else None})
    return qid


def log_feedback(qid, value):
    """value: 'up' or 'down'. A later vote on the same question replaces the earlier one."""
    _append({"type": "feedback", "id": qid, "value": value})


def read_events(path=None):
    p = Path(path) if path else log_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:      # a half-written last line must not break the dashboard
            continue
    return out


def usage_summary(events):
    asks = [e for e in events if e.get("type") == "ask"]
    votes = {}
    for e in events:
        if e.get("type") == "feedback":
            votes[e["id"]] = e["value"]
    answered = [a for a in asks if not a["declined"]]
    lat = sorted(a["latency"] for a in asks)
    return {
        "n_questions": len(asks),
        "decline_rate": 100 * sum(a["declined"] for a in asks) / len(asks) if asks else float("nan"),
        "uncited_rate": 100 * sum(a["uncited"] for a in answered) / len(answered) if answered else float("nan"),
        "latency_mean": sum(lat) / len(lat) if lat else float("nan"),
        "latency_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else float("nan"),
        "thumbs_up": sum(v == "up" for v in votes.values()),
        "thumbs_down": sum(v == "down" for v in votes.values()),
        "reasons": Counter(a["reason"] for a in asks if a["declined"]),
        "routes": Counter(a["routed"] or "keyword" for a in asks),
    }
