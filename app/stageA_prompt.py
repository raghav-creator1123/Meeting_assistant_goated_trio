"""
Stage A prompt (extraction) + Stage B filter for the meeting assistant.

Prompt format: the survey's prompting paradigm (Rennard et al., arXiv 2208.04163,
Sec. 6): "Transcript: {{transcript}}" followed by an instruction, with few-shot
examples in the same format built from AMI by build_stageA_prompts_from_ami.py.

Use in the app:
    from stageA_prompt import extract, stage_b_filter
    raw = extract(lines, client, model)          # Stage A (LLM, Gemini)
    record = stage_b_filter(raw, lines)          # Stage B (code)

`lines` = ["[1] SPEAKER_00: ...", "[2] SPEAKER_01: ...", ...]  (speaker optional)
Prompt variants for tuning: examples="ami+synthetic" (default) | "ami" | "none".
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

INSTRUCTION = (
    "Extract the topics, decisions and tasks from the above transcript. "
    "Return only JSON that follows the schema."
)

SYSTEM = """You extract a structured meeting record from a numbered meeting transcript.
Lines look like "[12] SPEAKER_01: text"; the speaker label may be missing.
Cite line numbers for everything.

Rules:
- topics: the main discussion segments in order, each with a short title and the first and last line.
- Decision or task? Use these rules:
  * Work assigned to a person, especially with a deadline, is a TASK, not a decision
    ("Priya, can you fix the bug?" "Yes, by Thursday." -> task, owner Priya, deadline Thursday).
  * A suggestion that others agree to is a DECISION with status "agreed".
  * A suggestion that is parked, postponed, rejected or left open is a PROPOSAL:
    a decision with status "proposed_only".
  * "Someone should ..." is a TASK with owner null.
  * A statement of fact (a number, a status update, a budget) is neither, unless the group
    decides to change something.
- decisions: something the group settled on.
  * status "agreed" only if a line shows acceptance or confirmation (e.g. "yes, let's do that",
    "okay, we'll go with that", nobody objects and the chair confirms, or an announced decision).
    Put the line(s) where it was proposed in proposal_lines and the acceptance in confirmation_lines.
  * status "proposed_only" if it was suggested but not accepted, rejected, or left open.
    confirmation_lines must then be [].
  * Keep negation exactly ("will not use teletext" is not "will use teletext").
- tasks: work someone must do after the meeting.
  * owner: the name, role or speaker label exactly as the transcript states it; else null.
    "Someone should look into X" has owner null. Never guess an owner from context.
  * deadline: only if a time or date is said ("by Friday", "next meeting"); else null.
  * owner_lines / deadline_lines: the lines that state them; [] when null.
  * evidence_lines: the lines where the task is given or accepted.
- Do not invent anything that is not in the transcript. Write decisions and tasks as short
  third-person sentences. Return empty lists when there is nothing to report."""

SCHEMA = {
    "name": "meeting_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["topics", "decisions", "tasks"],
        "properties": {
            "topics": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["title", "start_line", "end_line"],
                "properties": {"title": {"type": "string"},
                               "start_line": {"type": "integer"},
                               "end_line": {"type": "integer"}}}},
            "decisions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "status", "proposal_lines", "confirmation_lines"],
                "properties": {"text": {"type": "string"},
                               "status": {"type": "string", "enum": ["agreed", "proposed_only"]},
                               "proposal_lines": {"type": "array", "items": {"type": "integer"}},
                               "confirmation_lines": {"type": "array", "items": {"type": "integer"}}}}},
            "tasks": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "owner", "owner_lines", "deadline", "deadline_lines",
                             "evidence_lines"],
                "properties": {"text": {"type": "string"},
                               "owner": {"type": ["string", "null"]},
                               "owner_lines": {"type": "array", "items": {"type": "integer"}},
                               "deadline": {"type": ["string", "null"]},
                               "deadline_lines": {"type": "array", "items": {"type": "integer"}},
                               "evidence_lines": {"type": "array", "items": {"type": "integer"}}}}},
        },
    },
}

# Hand-written example (synthetic, not from AMI): AMI never labels proposals that
# were NOT agreed or tasks with NO stated owner, which the task rubric checks.
SYNTHETIC_EXAMPLE = {
    "meeting": "synthetic",
    "lines": [
        "[1] SPEAKER_00: Okay, first thing, the login bug. It's still failing on Android.",
        "[2] SPEAKER_01: Yeah, it's the token refresh. Priya, can you take that?",
        "[3] SPEAKER_02: Sure, I'll fix it by Thursday.",
        "[4] SPEAKER_00: Great. Next, should we move the release to Redis caching first?",
        "[5] SPEAKER_01: Maybe we could switch the whole thing to GraphQL as well.",
        "[6] SPEAKER_02: Hmm, I'm not sure. That's a big change.",
        "[7] SPEAKER_00: Let's park GraphQL for now. But yes, we'll add Redis caching before release.",
        "[8] SPEAKER_01: Agreed.",
        "[9] SPEAKER_00: And we are not supporting Android 9 anymore, everyone okay with that?",
        "[10] SPEAKER_02: Yes.",
        "[11] SPEAKER_01: Someone should probably update the docs too.",
        "[12] SPEAKER_00: Right. Okay, that's it, thanks everyone.",
    ],
    "target": {
        "topics": [
            {"title": "Android login bug", "start_line": 1, "end_line": 3},
            {"title": "Release architecture", "start_line": 4, "end_line": 10},
            {"title": "Documentation and close", "start_line": 11, "end_line": 12},
        ],
        "decisions": [
            {"text": "Redis caching will be added before the release.", "status": "agreed",
             "proposal_lines": [4], "confirmation_lines": [7, 8]},
            {"text": "Switching to GraphQL was suggested but parked.", "status": "proposed_only",
             "proposal_lines": [5], "confirmation_lines": []},
            {"text": "Android 9 will no longer be supported.", "status": "agreed",
             "proposal_lines": [9], "confirmation_lines": [10]},
        ],
        "tasks": [
            {"text": "Fix the Android login token refresh bug.", "owner": "Priya",
             "owner_lines": [2, 3], "deadline": "Thursday", "deadline_lines": [3],
             "evidence_lines": [2, 3]},
            {"text": "Update the documentation.", "owner": None, "owner_lines": [],
             "deadline": None, "deadline_lines": [], "evidence_lines": [11, 12]},
        ],
    },
}


def user_turn(lines):
    """Survey format: 'Transcript: {{transcript}}' + instruction."""
    return "Transcript:\n" + "\n".join(lines) + "\n\n" + INSTRUCTION


def load_examples(path=None):
    path = path or os.getenv("STAGEA_FEWSHOT") or os.path.join(ROOT, "stageA", "stageA_fewshot.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def build_messages(lines, examples=None, include_synthetic=True):
    examples = load_examples() if examples is None else examples
    shots = list(examples) + ([SYNTHETIC_EXAMPLE] if include_synthetic else [])
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        msgs.append({"role": "user", "content": user_turn(ex["lines"])})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["target"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_turn(lines)})
    return msgs


def extract(lines, client, model, examples=None, include_synthetic=True):
    """Stage A: one LLM call. Returns the parsed JSON dict."""
    from llm import chat_json
    return chat_json(client, model, build_messages(lines, examples, include_synthetic),
                     schema=SCHEMA, temperature=0)


def messages_for_variant(lines, variant="ami+synthetic"):
    """Prompt variants compared in the notebook."""
    if variant == "none":
        return build_messages(lines, [], include_synthetic=False)
    if variant == "ami":
        return build_messages(lines, None, include_synthetic=False)
    if variant == "synthetic":      # small prompt for models with low token limits
        return build_messages(lines, [], include_synthetic=True)
    return build_messages(lines, None, include_synthetic=True)


def _words(t):
    return set(re.findall(r"[a-z0-9']+", (t or "").lower()))


def _similar(a, b, threshold=0.6):
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def merge_extractions(parts):
    """Combine Stage A outputs from overlapping transcript chunks (absolute line numbers)."""
    out = {"topics": [], "decisions": [], "tasks": []}
    for p in parts:
        p = p if isinstance(p, dict) else {}
        out["topics"] += [t for t in p.get("topics", []) or [] if isinstance(t, dict)]
        for kind in ("decisions", "tasks"):
            for item in p.get(kind, []) or []:
                if not isinstance(item, dict):
                    continue
                twin = next((x for x in out[kind] if _similar(x.get("text"), item.get("text"))), None)
                if twin is None:
                    out[kind].append(dict(item))
                    continue
                for key, val in item.items():        # same item seen twice (overlap): merge
                    if key.endswith("_lines"):
                        twin[key] = sorted(set(twin.get(key) or []) | set(val or []))
                    elif not twin.get(key) and val:
                        twin[key] = val
                if item.get("status") == "agreed":
                    twin["status"] = "agreed"
    topics = sorted(out["topics"], key=lambda t: t.get("start_line") or 0)
    merged = []
    for t in topics:                                  # same topic continuing across chunks
        if merged and (t.get("title") or "").lower() == (merged[-1].get("title") or "").lower():
            merged[-1]["end_line"] = max(merged[-1].get("end_line") or 0, t.get("end_line") or 0)
        else:
            merged.append(dict(t))
    out["topics"] = merged
    return out


LINE_RE = re.compile(r"^\[(\d+)\]\s*(?:(SPEAKER_\d+|[A-Za-z][\w .-]{0,30}?):\s)?(.*)$")


def parse_line(line):
    m = LINE_RE.match(line)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else (None, None, line)


def _mentioned(value, cited, by_no):
    """Is `value` (an owner or deadline) actually stated on one of the cited lines?"""
    v = value.lower().strip()
    toks = [t for t in re.findall(r"[a-z0-9']+", v) if len(t) > 2 and t not in
            ("the", "and", "for", "next", "end", "will")]
    for n in cited:
        _, spk, text = by_no.get(n, (None, None, ""))
        t = text.lower()
        if v in t or (spk and v == spk.lower()) or (toks and all(x in t for x in toks)):
            return True
    return False


def stage_b_filter(raw, lines):
    """Stage B: deterministic rules the LLM cannot override.

    `lines` is the numbered transcript (or just its length, for a quick check).
    Returns the record with decisions, proposals, tasks, topics and flags.
    """
    n_lines = lines if isinstance(lines, int) else len(lines)
    by_no = {} if isinstance(lines, int) else \
        {no: (no, spk, text) for no, spk, text in map(parse_line, lines) if no}
    ok = lambda ids: sorted({i for i in ids or [] if isinstance(i, int) and 1 <= i <= n_lines})
    rec = {"topics": [], "decisions": [], "proposals": [], "tasks": [], "flags": []}
    flag = lambda kind, item, why: rec["flags"].append({"stage": "B", "type": kind,
                                                        "item": item, "reason": why})
    raw = raw if isinstance(raw, dict) else {}

    for t in raw.get("topics", []) or []:
        s, e = t.get("start_line"), t.get("end_line")
        if ok([s]) and ok([e]) and s <= e and t.get("title"):
            rec["topics"].append({"title": t["title"], "start_line": s, "end_line": e})

    for d in raw.get("decisions", []) or []:
        text = (d.get("text") or "").strip()
        prop, conf = ok(d.get("proposal_lines")), ok(d.get("confirmation_lines"))
        if not text or (not prop and not conf):
            flag("dropped_decision", text, "no valid transcript lines cited")
            continue
        item = {"text": text, "proposal_lines": prop, "confirmation_lines": conf}
        if d.get("status") == "agreed" and conf:
            rec["decisions"].append(item)
        else:
            if d.get("status") == "agreed":
                flag("decision_to_proposal", text, "marked agreed but no confirmation line")
            rec["proposals"].append(item)

    for t in raw.get("tasks", []) or []:
        text = (t.get("text") or "").strip()
        ev = ok(t.get("evidence_lines"))
        if not text or not ev:
            flag("dropped_task", text, "no valid transcript lines cited")
            continue
        task = {"text": text, "owner": "unspecified", "owner_lines": [],
                "deadline": "unspecified", "deadline_lines": [], "evidence_lines": ev}
        for field in ("owner", "deadline"):
            value, cited = t.get(field), ok(t.get(f"{field}_lines"))
            if not value or str(value).lower() in ("null", "none", "unspecified"):
                continue
            if not cited:
                flag(f"{field}_unsupported", text, f"{field} '{value}' has no supporting line")
            elif by_no and not _mentioned(str(value), cited, by_no):
                flag(f"{field}_not_in_text", text,
                     f"{field} '{value}' is not stated on lines {cited}")
            else:
                task[field], task[f"{field}_lines"] = str(value), cited
        rec["tasks"].append(task)
    return rec
