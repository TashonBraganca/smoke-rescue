"""What the two robots have said, proposed, agreed and set out to do, kept for one recording.

Each decision step used to start from nothing: a robot's agent saw the radio texts that arrived since its last step and
forgot them after it (Codex design review, 22 Oct: no proposal ids, no acceptance, no expiry, nothing tying a received
finding to the next action). This module keeps, per robot, the proposals it sent and received with their status, the
agreements in force, the partner's recent findings and the robot's own last intent. Code assigns every id, time and
status; the model only chooses what to propose, accept or reject and why.

Messages travel as JSON over the simulated radio and may be late or lost:
    {"message_id", "from", "t", "kind", "text", "proposal_id"?, "reply_to"?, "expires_t"?, "my_part"?, "your_part"?}
kind is finding, outcome, proposal, accept or reject. Messages without a kind (older senders, the commander) are findings.
"""

from __future__ import annotations

import itertools
import json
from collections import deque
from pathlib import Path

_CONFIG = json.loads((Path(__file__).resolve().parents[3] / "config" / "team.json").read_text())
PROPOSAL_TTL_S = float(_CONFIG["proposal_ttl_s"])
AGREEMENT_TTL_S = float(_CONFIG["agreement_ttl_s"])
FINDINGS_KEPT = int(_CONFIG["findings_kept"])
TEXT_MAX = int(_CONFIG["text_max_chars"])

KINDS = ("finding", "outcome", "proposal", "accept", "reject")


class TeamState:
    """One robot's view of the team conversation. Class-level stores are per recording; reset() clears them."""

    _robots: dict[str, "TeamState"] = {}
    _ids = itertools.count(1)

    def __init__(self, robot: str):
        self.robot = robot
        self.seen: set[str] = set()
        self.sent: dict[str, dict] = {}          # proposal_id -> proposal this robot made, with status
        self.received: dict[str, dict] = {}      # proposal_id -> proposal from the partner, with status
        self.findings: deque = deque(maxlen=FINDINGS_KEPT)
        self.intent: dict | None = None

    @classmethod
    def of(cls, robot: str) -> "TeamState":
        if robot not in cls._robots:
            cls._robots[robot] = TeamState(robot)
        return cls._robots[robot]

    @classmethod
    def reset(cls) -> None:
        cls._robots.clear()

    # ----- incoming ------------------------------------------------------------------------------------------------
    def ingest(self, inbox: list, t: float) -> list[str]:
        """Record radio messages not seen before; returns their ids. A repeated delivery changes nothing."""
        new = []
        for m in inbox or []:
            mid = str(m.get("message_id") or f"{m.get('from')}@{m.get('t')}:{str(m.get('text'))[:40]}")
            if mid in self.seen:
                continue
            self.seen.add(mid); new.append(mid)
            kind = m.get("kind") if m.get("kind") in KINDS else "finding"
            sender = str(m.get("from", "?"))
            if kind == "proposal" and m.get("proposal_id"):
                self.received[str(m["proposal_id"])] = {
                    "proposal_id": str(m["proposal_id"]), "from": sender, "t": float(m.get("t", t)),
                    "expires_t": float(m.get("expires_t", float(m.get("t", t)) + PROPOSAL_TTL_S)),
                    # Swapped: what the partner will do, and what it asks of this robot.
                    "partner_part": m.get("my_part"), "your_part": m.get("your_part"), "why": m.get("text"), "status": "pending",
                    "target": m.get("target")}
            elif kind in ("accept", "reject") and str(m.get("reply_to")) in self.sent:
                p = self.sent[str(m["reply_to"])]
                if p["status"] == "pending" and t <= p["expires_t"]:
                    p["status"] = "accepted" if kind == "accept" else "rejected"
                    p["answered_t"] = float(m.get("t", t)); p["answer"] = m.get("text")
            else:
                self.findings.append({"from": sender, "t": m.get("t"), "kind": kind, "text": m.get("text"),
                                      "position": m.get("position")})
        return new

    # ----- outgoing ------------------------------------------------------------------------------------------------
    def message(self, kind: str, text: str, t: float, **fields) -> dict:
        """A radio message with a code-assigned id; fields with None values are left out."""
        msg = {"message_id": f"{self.robot}-{next(self._ids)}", "kind": kind, "text": str(text or "").strip()[:TEXT_MAX], "t": t}
        msg.update({k: v for k, v in fields.items() if v is not None})
        return msg

    def propose(self, my_part: str, your_part: str, why: str, t: float, target: dict | None = None) -> dict:
        pid = f"{self.robot}-p{next(self._ids)}"
        self.sent[pid] = {"proposal_id": pid, "t": t, "expires_t": t + PROPOSAL_TTL_S, "my_part": my_part,
                          "your_part": your_part, "status": "pending", "target": target}
        return self.message("proposal", why, t, proposal_id=pid, expires_t=round(t + PROPOSAL_TTL_S, 2),
                            my_part=str(my_part)[:TEXT_MAX], your_part=str(your_part)[:TEXT_MAX], target=target)

    def answer_refusal(self, proposal_id: str, t: float) -> str | None:
        p = self.received.get(str(proposal_id))
        if p is None:
            return "no proposal with that id has reached this robot"
        if p["status"] != "pending":
            return f"that proposal was already {p['status']}"
        if t > p["expires_t"]:
            return "that proposal has expired; propose a new plan instead"
        return None

    def answer(self, proposal_id: str, accept: bool, reason: str, t: float) -> dict:
        p = self.received[str(proposal_id)]
        p["status"] = "accepted" if accept else "rejected"; p["answered_t"] = t
        return self.message("accept" if accept else "reject", reason, t, reply_to=str(proposal_id))

    # ----- what the model sees -------------------------------------------------------------------------------------
    def agreements(self, t: float) -> list[dict]:
        """Accepted plans in force: the newest per direction, within AGREEMENT_TTL_S of acceptance."""
        out = []
        for p in self.sent.values():
            if p["status"] == "accepted" and t - p.get("answered_t", p["t"]) <= AGREEMENT_TTL_S:
                out.append({"proposal_id": p["proposal_id"], "proposed_by": "you", "your_part": p["my_part"],
                            "partner_part": p["your_part"], "target": p.get("target"), "since_t": p.get("answered_t")})
        for p in self.received.values():
            if p["status"] == "accepted" and t - p.get("answered_t", p["t"]) <= AGREEMENT_TTL_S:
                out.append({"proposal_id": p["proposal_id"], "proposed_by": p["from"], "your_part": p["your_part"],
                            "partner_part": p["partner_part"], "target": p.get("target"), "since_t": p.get("answered_t")})
        return sorted(out, key=lambda a: a["since_t"] or 0)[-2:]

    def context(self, t: float) -> dict:
        awaiting = [{k: p[k] for k in ("proposal_id", "from", "your_part", "partner_part", "why", "target")} |
                    {"expires_in_s": round(p["expires_t"] - t, 1)}
                    for p in self.received.values() if p["status"] == "pending" and t <= p["expires_t"]]
        mine = [{"proposal_id": p["proposal_id"], "status": p["status"] if p["status"] != "pending" or t <= p["expires_t"] else "expired",
                 "my_part": p["my_part"], "your_part": p["your_part"]} for p in list(self.sent.values())[-3:]]
        # Age on every finding and the time since the partner was last heard: a stale report must read as stale (agents
        # under message loss do worst when old facts look current; AgentComm-Bench, arXiv 2603.20285).
        findings = [{**f, "age_s": round(t - float(f["t"]), 1) if f.get("t") is not None else None} for f in self.findings]
        heard = [float(f["t"]) for f in self.findings if f.get("t") is not None and str(f.get("from", "")).startswith("robot")]
        return {"awaiting_your_answer": awaiting, "your_proposals": mine, "agreements_in_force": self.agreements(t),
                "partner_findings": findings, "partner_last_heard_s": round(t - max(heard), 1) if heard else None,
                "your_last_intent": self.intent}

    def note_intent(self, t: float, summary: str | None, orders: list) -> None:
        """What this robot set out to do at the end of a step, from the orders the tools accepted (not the model's words)."""
        moves = [o for o in orders if o.get("type") in ("go_to", "explore_frontier", "approach_candidate", "turn_to_face")]
        if not moves and summary is None:
            return
        self.intent = {"t": round(t, 2), "summary": (summary or "")[:TEXT_MAX],
                       "orders": [{k: o[k] for k in ("type", "x", "y", "frontier_id") if k in o} for o in moves],
                       "agreements": [a["proposal_id"] for a in self.agreements(t)]}
