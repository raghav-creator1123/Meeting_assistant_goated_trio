"""
Documentation stages after the Stage B filter (language model 2, Gemini).

Stage D (verify)  - every decision, proposal and task is checked against the
                    transcript lines it cites. When the verifier rejects an item,
                    code checks whether the item's key words appear on its cited
                    lines: if they do, the item is kept and flagged for review
                    (a small verifier is often too strict); if not, it is removed
                    and flagged. Runs BEFORE Stage C so the minutes never mention
                    something that was removed.
Stage C (write)   - writes the concise summary and organised minutes from the
                    verified record and the transcript lines of each topic.
                    Decisions and tasks are NOT rewritten by the LLM; they are
                    copied from the verified record, so the Markdown and JSON
                    outputs always contain exactly the same items.
"""

import json
import re

from llm import chat_json
from stageA_prompt import parse_line

VERIFY_SYSTEM = """You check a meeting record against the transcript.
For each item you get its text and the exact transcript lines it cites.
Answer "supported": true only if those lines clearly state it.
- A decision needs the lines to show it was actually agreed or announced, not just suggested.
- A proposal needs the lines to show it was suggested.
- A task needs the lines to show the work was assigned, volunteered or said to be needed;
  if an owner or deadline is given, the lines must state that owner or deadline.
- A task with owner "unspecified" is valid: "someone should look into X" supports the task
  "look into X". Do not reject a task because it has no owner or no deadline.
- Judge the meaning, not the exact wording: a paraphrase of what the lines say is supported.
Return JSON: {"results": [{"id": "<id>", "supported": true|false, "reason": "<short>"}]}"""

VERIFY_SCHEMA = {"name": "verification", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["results"],
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "supported", "reason"],
        "properties": {"id": {"type": "string"}, "supported": {"type": "boolean"},
                       "reason": {"type": "string"}}}}}}}

WRITE_SYSTEM = """You write meeting documentation from a verified meeting record.
Use only the information given. Do not add decisions, tasks, owners, deadlines,
numbers or names that are not in the record or the quoted transcript lines.
- Say something was agreed or decided ONLY if it is in the record's "decisions" list.
- Describe items in "proposals" as suggested but not agreed. Describe tasks as work assigned,
  with the owner and deadline from the record.
- Describe everything else neutrally as discussed or mentioned (e.g. "The travel budget
  is 2,000 euros; members were asked to stay within it", not "the team cut the budget"). Never turn a
  remark or a fact into a decision, a change or an action.
- Never add facts, causes or outcomes that are not in the transcript lines.
- Each topic's paragraph covers only that topic's lines; do not repeat other topics.
Return JSON:
{"summary": "<3-5 sentence overview of the meeting>",
 "minutes": [{"topic": "<topic title>", "text": "<one paragraph: what was discussed,
   what was decided, what was proposed but not agreed>"}]}
Write one minutes entry per topic, in order. Keep owners and deadlines exactly as given
and write "unspecified" where the record says unspecified."""

WRITE_SCHEMA = {"name": "documentation", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["summary", "minutes"],
    "properties": {"summary": {"type": "string"}, "minutes": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["topic", "text"],
        "properties": {"topic": {"type": "string"}, "text": {"type": "string"}}}}}}}


def _cited(lines, ids):
    by_no = {parse_line(l)[0]: l for l in lines}
    return [by_no[i] for i in ids if i in by_no]


STOP = set("""a an the and or but of to in on at by for with from into is are was were be been
will would shall should can could may might must do does did it its this that these those
there their they them we our us you your he she his her i me my not no yes so as up out
about over than then also just only all any some""".split())


def _stems(text):
    return {w[:5] for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in STOP}


def _words_on_lines(it, cited, min_share=0.5):
    """Do the item's key words actually appear on the lines it cites?
    Words are compared by their first 5 letters, so "fixes" matches "fix", "decided" "decide"."""
    key = _stems(it["text"])
    if not key:
        return False
    found = key & _stems(" ".join(parse_line(l)[2] for l in cited))
    return len(found) / len(key) >= min_share


def _items(rec):
    for kind in ("decisions", "proposals", "tasks"):
        for i, it in enumerate(rec[kind]):
            ids = sorted(set(it.get("proposal_lines", []) + it.get("confirmation_lines", []) +
                             it.get("evidence_lines", []) + it.get("owner_lines", []) +
                             it.get("deadline_lines", [])))
            yield f"{kind[0]}{i}", kind, it, ids


def stage_d_verify(rec, lines, client, model, batch_size=15):
    """Checks every item (rec is changed in place). An item the verifier rejects is kept and
    flagged if its key words appear on its cited lines, and removed and flagged otherwise."""
    items = list(_items(rec))
    verdicts = {}
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        blocks = []
        for iid, kind, it, ids in chunk:
            extra = ""
            if kind == "tasks":
                extra = f"\nowner: {it['owner']}\ndeadline: {it['deadline']}"
            blocks.append(f"id: {iid}\ntype: {kind[:-1]}\ntext: {it['text']}{extra}\n"
                          f"cited lines:\n" + "\n".join(_cited(lines, ids)))
        data = chat_json(client, model, [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": "\n\n---\n\n".join(blocks)}],
            schema=VERIFY_SCHEMA, temperature=0)
        for r in data.get("results", []) if isinstance(data, dict) else []:
            verdicts[str(r.get("id"))] = r
    removed = set()
    for iid, kind, it, ids in items:
        v = verdicts.get(iid)
        if v is None:      # the verifier skipped it: keep, but flag for human review
            rec["flags"].append({"stage": "D", "type": "not_verified", "item": it["text"],
                                 "reason": "verifier returned no verdict"})
        elif not v.get("supported", True):
            reason = v.get("reason", "")
            if _words_on_lines(it, _cited(lines, ids)):     # verifier likely too strict: keep
                rec["flags"].append({"stage": "D", "type": f"review_{kind[:-1]}",
                                     "item": it["text"],
                                     "reason": f"kept: its words appear on lines {ids}, but the "
                                               f"verifier said: {reason}"})
            else:
                removed.add(id(it))
                rec["flags"].append({"stage": "D", "type": f"unsupported_{kind[:-1]}",
                                     "item": it["text"],
                                     "reason": f"removed: {reason} (its words are not on lines "
                                               f"{ids})"})
    for kind in ("decisions", "proposals", "tasks"):
        rec[kind] = [it for it in rec[kind] if id(it) not in removed]
    return rec


def stage_c_write(rec, lines, client, model, max_chars=60000):
    topics = rec["topics"] or [{"title": "Meeting discussion", "start_line": 1,
                                "end_line": len(lines)}]
    budget = max_chars // max(1, len(topics))
    blocks = []
    for t in topics:
        seg = "\n".join(lines[t["start_line"] - 1:t["end_line"]])
        if len(seg) > budget:
            seg = seg[:budget] + "\n[...]"
        blocks.append(f"## Topic: {t['title']} (lines {t['start_line']}-{t['end_line']})\n{seg}")
    record = {k: rec[k] for k in ("decisions", "proposals", "tasks")}
    user = ("Verified record:\n" + json.dumps(record, ensure_ascii=False, indent=1) +
            "\n\nTranscript by topic:\n\n" + "\n\n".join(blocks))
    data = chat_json(client, model, [{"role": "system", "content": WRITE_SYSTEM},
                                     {"role": "user", "content": user}],
                     schema=WRITE_SCHEMA, temperature=0.2)
    summary = (data.get("summary") or "").strip() if isinstance(data, dict) else ""
    minutes = [m for m in (data.get("minutes") or []) if isinstance(m, dict) and m.get("text")] \
        if isinstance(data, dict) else []
    if not summary:
        rec["flags"].append({"stage": "C", "type": "empty_summary", "item": "",
                             "reason": "the writing model returned no summary"})
    return summary, minutes


def render_markdown(result):
    r = result["record"]
    out = [f"# Meeting record: {result['source']}", "",
           f"_Generated {result['generated_at']} · STT: {result['models']['stt']} · "
           f"refinement: {result['models']['refinement']} · "
           f"documentation: {result['models']['documentation']}_", "",
           "## Summary", "", result["summary"] or "_No summary generated._", "",
           "## Minutes", ""]
    for m in result["minutes"] or [{"topic": "Discussion", "text": "_No minutes generated._"}]:
        out += [f"### {m['topic']}", "", m["text"], ""]
    out += ["## Key decisions", ""]
    out += [f"{i}. {d['text']} _(lines {', '.join(map(str, d['proposal_lines'] + d['confirmation_lines']))})_"
            for i, d in enumerate(r["decisions"], 1)] or ["_No agreed decisions._"]
    out += ["", "## Proposals discussed (not agreed)", ""]
    out += [f"- {p['text']} _(lines {', '.join(map(str, p['proposal_lines']))})_"
            for p in r["proposals"]] or ["_None._"]
    out += ["", "## Action items", "", "| # | Task | Owner | Deadline | Lines |",
            "|---|---|---|---|---|"]
    out += [f"| {i} | {t['text']} | {t['owner']} | {t['deadline']} | "
            f"{', '.join(map(str, t['evidence_lines']))} |"
            for i, t in enumerate(r["tasks"], 1)] or ["| – | _No action items._ | | | |"]
    if result["flags"]:
        out += ["", "## Flags for review", ""]
        out += [f"- **{f['stage']} · {f['type']}**: {f.get('item') or ''} — {f.get('reason', '')}"
                for f in result["flags"]]
    return "\n".join(out) + "\n"


def to_json(result):
    """Machine-readable record; same decisions and tasks as the Markdown."""
    r = result["record"]
    return {
        "source": result["source"], "generated_at": result["generated_at"],
        "models": result["models"], "summary": result["summary"], "minutes": result["minutes"],
        "topics": r["topics"], "decisions": r["decisions"], "proposals": r["proposals"],
        "action_items": r["tasks"], "flags": result["flags"],
    }
