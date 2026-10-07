"""
Transcript refinement (language model 1, Groq).

Input: segments with up to 5 Whisper hypotheses each.
Output: one refined line per segment, plus flags.

The prompt follows the HyPoradise idea (the LLM sees the N-best list and writes
the true transcription), adapted for meetings:
  * several lines per call (free-tier friendly) with JSON output keyed by line id,
  * a domain glossary,
  * rules that protect names, numbers, negation and commitments,
  * normal casing and punctuation (HyPoradise references are lowercase, so its
    examples are used for evaluation only, not inside this prompt).

Code-side guards (the LLM cannot override them):
  * a missing or empty line falls back to hypothesis 1             -> flag refine_missing
  * a line that drifts far from every hypothesis falls back         -> flag refine_drift
  * a number or negation word not present in any hypothesis is
    reverted to hypothesis 1                                        -> flag meaning_guard
"""

import json
import re

from llm import chat_json

SYSTEM = """You correct speech-recognition errors in a meeting transcript.
For each line you get up to 5 ASR hypotheses of the same audio, best first.
Write the most accurate transcription of each line.

Rules:
- Fix only words that were likely misheard: sound-alike errors, technical terms, acronyms,
  product names and people's names. Use the hypotheses and the glossary as evidence.
- Do not rephrase, summarise, translate or correct grammar. Keep fillers, repetitions and
  false starts ("so um we we should").
- Never change numbers, dates, names, negations (not, n't, never, no) or commitments
  (I'll, we will) unless the hypotheses disagree and one is clearly right.
- If you are unsure, keep hypothesis 1.
- Use normal capitalisation and punctuation.
- Return JSON: {"lines": [{"id": <id>, "text": "<corrected line>"}]} with every id exactly once."""

DEMO_USER = """Glossary: Kubernetes, PostgreSQL, Redis, CI/CD, OKR

Lines to correct:
[1]
 1. we'll deploy the new build to cube and eighties after the post grass migration
 2. we'll deploy the new build to kubernetes after the post grass migration
 3. will deploy the new build to cuban eighties after the postgres migration
[2]
 1. i don't think we should ship it on friday
 2. i do think we should ship it on friday
[3]
 1. the budget is fifty thousand dollars for the first phase
 2. the budget is fifteen thousand dollars for the first phase
 3. the budget is fifteen thousand dollars for the first faze
[4]
 1. so um we we should probably revisit the the pricing page
 2. so um we should probably revisit the pricing page
[5]
 1. pria will review the pull request before on air been merges it
 2. priya will review the pull request before anirban merges it"""

DEMO_ASSISTANT = json.dumps({"lines": [
    {"id": 1, "text": "We'll deploy the new build to Kubernetes after the Postgres migration."},
    {"id": 2, "text": "I don't think we should ship it on Friday."},
    {"id": 3, "text": "The budget is fifteen thousand dollars for the first phase."},
    {"id": 4, "text": "So um, we we should probably revisit the the pricing page."},
    {"id": 5, "text": "Priya will review the pull request before Anirban merges it."},
]})

SCHEMA = {"name": "refined_lines", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["lines"],
    "properties": {"lines": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "text"],
        "properties": {"id": {"type": "integer"}, "text": {"type": "string"}}}}}}}

NUM_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand million billion first second third fourth fifth half""".split())
NEG_WORDS = {"not", "no", "never", "nobody", "nothing", "none", "neither", "nor", "cannot", "without"}


def words(s):
    s = s.lower().replace("’", "'")
    return re.findall(r"[a-z0-9']+", s)


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def critical_tokens(s):
    toks = set()
    for w in words(s):
        if w.isdigit() or w in NUM_WORDS or w in NEG_WORDS:
            toks.add(w)
        if w.endswith("n't"):
            toks.add("n't")
    return toks


def negations(s):
    return {t for t in critical_tokens(s) if t in NEG_WORDS or t == "n't"}


def guard(hyps, text, max_drift=0.5):
    """Return (final_text, flag or None).

    - empty output                       -> hypothesis 1, refine_missing
    - far from every hypothesis          -> hypothesis 1, refine_drift
    - adds a number/negation that no hypothesis contains,
      or drops one that all hypotheses agree on      -> hypothesis 1, meaning_guard
    - negation differs from hypothesis 1 (e.g. "won't" -> "will"): flipping
      negation is too risky to automate             -> hypothesis 1, negation_guard
    """
    if not text or not text.strip():
        return hyps[0], "refine_missing"
    if len(words(hyps[0])) <= 3:
        # one- to three-word lines ("Yes.", "'Kay.") are too short to judge by WER: accept the
        # model's line only if it is one of the hypotheses (e.g. a casing fix), else keep Whisper's
        if not any(words(text) == words(h) for h in hyps):
            return hyps[0], None
        if negations(text) != negations(hyps[0]):
            return hyps[0], "negation_guard"
        return text.strip(), None
    if min(wer(h, text) for h in hyps) > max_drift:
        return hyps[0], "refine_drift"
    per_hyp = [critical_tokens(h) for h in hyps]
    out = critical_tokens(text)
    added = out - set().union(*per_hyp)          # number/negation no hypothesis has
    lost = set.intersection(*per_hyp) - out      # one every hypothesis agrees on, dropped
    if added or lost:
        return hyps[0], "meaning_guard"
    if negations(text) != negations(hyps[0]):
        return hyps[0], "negation_guard"
    return text.strip(), None


def format_lines(items, glossary=None, context=None):
    parts = []
    if glossary:
        parts.append("Glossary: " + ", ".join(glossary))
    if context:
        parts.append("Context (already corrected, do not return these):\n" + "\n".join(context))
    body = []
    for it in items:
        body.append(f"[{it['id']}]")
        body += [f" {k}. {h}" for k, h in enumerate(it["hypotheses"], 1)]
    parts.append("Lines to correct:\n" + "\n".join(body))
    return "\n\n".join(parts)


def build_messages(items, glossary=None, context=None, use_demo=True):
    msgs = [{"role": "system", "content": SYSTEM}]
    if use_demo:
        msgs += [{"role": "user", "content": DEMO_USER},
                 {"role": "assistant", "content": DEMO_ASSISTANT}]
    msgs.append({"role": "user", "content": format_lines(items, glossary, context)})
    return msgs


def refine_items(items, client, model, glossary=None, batch_size=20, use_demo=True,
                 context_size=3, progress=None):
    """items: [{"id": int, "hypotheses": [...]}]. Returns [{"id", "text", "flag"}]."""
    results, context = [], []
    for b in range(0, len(items), batch_size):
        batch = items[b:b + batch_size]
        data = chat_json(client, model,
                         build_messages(batch, glossary, context[-context_size:], use_demo),
                         schema=SCHEMA, temperature=0, max_tokens=4000)
        got = {}
        for ln in data.get("lines", []) if isinstance(data, dict) else []:
            try:
                got[int(ln.get("id"))] = str(ln.get("text", ""))
            except (TypeError, ValueError):
                continue
        for it in batch:
            text, flag = guard(it["hypotheses"], got.get(it["id"], ""))
            results.append({"id": it["id"], "text": text, "flag": flag})
            context.append(f"[{it['id']}] {text}")
        if progress:
            progress(f"Refined {min(b + batch_size, len(items))}/{len(items)} lines")
    return results


def refine_segments(segments, client, model, glossary=None, batch_size=20, use_demo=True,
                    progress=None):
    """Adds 'refined' and refinement flags to each segment, in place."""
    out = refine_items([{"id": s["id"], "hypotheses": s["hypotheses"]} for s in segments],
                       client, model, glossary, batch_size, use_demo, progress=progress)
    by_id = {o["id"]: o for o in out}
    for s in segments:
        o = by_id[s["id"]]
        s["refined"] = o["text"]
        s["changed"] = words(o["text"]) != words(s["hypotheses"][0])   # ignore case/punctuation
        if o["flag"]:
            s.setdefault("flags", []).append(o["flag"])
    return segments
