"""
Build Stage A (extraction) few-shot prompts from the AMI Meeting Corpus.

Prompt format follows the prompting paradigm in Rennard et al. (2023),
"Abstractive Meeting Summarization: A Survey" (arXiv 2208.04163), Sec. 6:
    "Transcript: {{transcript}}"  +  an instruction.
The instruction asks for the Stage A JSON instead of "N sentences", and
few-shot examples in that same format are built from AMI annotations:

    AMI annotation (as described in the survey)      -> Stage A field
    abstractive summary, type "decisions"             -> decisions[].text
    abstractive summary, type "actions"               -> tasks[].text
    abstractive communities (summlink)                -> evidence / proposal lines
    adjacency pairs of type POS                       -> confirmation_lines
    topic segmentation                                -> topics[] with line ranges

Steps
 1. Convert the raw AMI XML to JSON with the survey authors' scripts
    (https://github.com/guokan-shang/ami-and-icsi-corpora): run dialogueActs.py,
    abstractive.py, extractive.py, summlink.py, adjacencyPairs.py, topics.py.
    This gives an  output/  folder with one sub-folder per annotation.
 2. python build_stageA_prompts_from_ami.py --ami-json path/to/output

Outputs (in --out-dir, default ./stageA)
    stageA_fewshot.json      the chosen few-shot examples (transcript + gold JSON)
    stageA_candidates.jsonl  every candidate excerpt, if you want to pick others
    stageA_eval.jsonl        full held-out AMI test meetings + gold, for scoring
    stageA_example_prompt.txt  one complete prompt, to read what the model sees
The prompt itself is assembled at run time by stageA_prompt.py.

AMI's official test meetings (ES2004, ES2014, IS1009, TS3003, TS3007, EN2002)
are never used as few-shot examples, so you can evaluate on them fairly.
"""

import argparse
import json
import os
import re
from collections import defaultdict

TEST_PREFIXES = ("ES2004", "ES2014", "IS1009", "TS3003", "TS3007", "EN2002")
ROLE_WORDS = {   # longest first; matched in the transcript as words
    "PM": ["project manager"],
    "ME": ["marketing expert", "marketing"],
    "UI": ["user interface designer", "interface designer", "interface specialist",
           "user interface", "uid"],
    "ID": ["industrial designer", "industrial design"],
}
DECISIVE_RE = re.compile(
    r"\b(decided|we'll|we will|we're gonna|we are going to|gonna|going to|let's|"
    r"agreed|stick (?:to|with)|go with|so we|okay so|fine)\b", re.I)
DEADLINE_RE = re.compile(
    r"\b(by (?:the )?(?:next meeting|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"end of (?:the )?(?:day|week))|for (?:the )?next meeting|next meeting|tomorrow|"
    r"in (?:the )?next (?:\w+ )?minutes|in (?:\w+ )?(?:thirty|forty|twenty|ten|five) minutes)\b",
    re.I)
SELF_COMMIT_RE = re.compile(r"\b(i'll|i will|i'm gonna|i'm going to|i can do|i'll do)\b", re.I)


# --------------------------------------------------------------------------
# text clean-up: make AMI look like Whisper output
# --------------------------------------------------------------------------
def clean(text):
    t = re.sub(r"<disfmarker>", "—", text)
    t = re.sub(r"<[^>]+>", " ", t)                       # vocalsound, gap, ...
    t = re.sub(r"\b((?:[A-Za-z]_){2,})", lambda m: m.group(1).replace("_", ""), t)  # V_C_R_ -> VCR
    t = re.sub(r"\b([A-Za-z])_\b", r"\1", t)
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"(—\s*)+", "— ", t)
    t = " ".join(t.split()).strip(" —")
    if t:
        t = t[0].upper() + t[1:]
    return t


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# one meeting -> numbered lines + gold items
# --------------------------------------------------------------------------
def build_meeting(mid, root):
    das = load(os.path.join(root, "dialogueActs", f"{mid}.json"))
    das.sort(key=lambda d: float(d.get("starttime") or 0))

    # merge consecutive dialogue acts by the same speaker into one line
    lines, da2line = [], {}
    for d in das:
        txt = clean(d.get("text", ""))
        if not txt:
            continue
        if lines and lines[-1]["spk"] == d["speaker"] and len(lines[-1]["text"]) < 400:
            lines[-1]["text"] += " " + txt
            lines[-1]["labels"].append(d.get("label", ""))
        else:
            lines.append({"spk": d["speaker"], "role": d.get("attributes", {}).get("role", ""),
                          "text": txt, "labels": [d.get("label", "")]})
        da2line[d["id"]] = len(lines) - 1

    def L(da_id):
        return da2line.get(da_id)

    # agreement: POS adjacency pair  source -> target
    pos_targets = defaultdict(set)
    ap_path = os.path.join(root, "adjacencyPairs", f"{mid}.json")
    if os.path.exists(ap_path):
        for ap in load(ap_path):
            src, tgt = ap.get("source"), ap.get("target")
            sid = src.get("id") if isinstance(src, dict) else src   # a few are bare ids
            tid = tgt.get("id") if isinstance(tgt, dict) else tgt
            if ap.get("type") == "POS" and sid and tid:
                s, t = L(sid), L(tid)
                if s is not None and t is not None and t != s:
                    pos_targets[s].add(t)

    decisions, tasks = [], []
    sl_path = os.path.join(root, "summlink", f"{mid}.json")
    for item in (load(sl_path) if os.path.exists(sl_path) else []):
        typ, text = item["abstractive"]["type"], item["abstractive"]["text"].strip()
        ev = sorted({L(e["id"]) for e in item["extractive"] if L(e["id"]) is not None})
        if not ev:
            continue
        if typ == "decisions":
            conf = set()
            for e in ev:
                conf |= pos_targets.get(e, set())           # someone agreed to it
            conf |= {e for e in ev if any(lb in ("ass", "be.pos") for lb in lines[e]["labels"])
                     and lines[e]["spk"] != lines[ev[0]]["spk"]}
            external = bool(conf)
            if not conf:  # announced decision: decisive wording by the speaker
                conf = {e for e in ev if DECISIVE_RE.search(lines[e]["text"])}
            prop = [e for e in ev if e not in conf] or ev[:1]
            decisions.append({"text": text, "status": "agreed",
                              "proposal_lines": prop, "confirmation_lines": sorted(conf),
                              "_span": sorted(set(ev) | conf),
                              # prefer examples where someone else visibly agrees
                              "_clear": external})
        elif typ == "actions":
            role = next((r for r, ws in ROLE_WORDS.items()
                         if any(w in text.lower() for w in ws)), None)
            owner, owner_lines = None, []
            if role:
                pat = re.compile(r"\b(" + "|".join(map(re.escape, ROLE_WORDS[role])) + r")\b", re.I)
                named = [e for e in ev if pat.search(lines[e]["text"])]
                selfc = [e for e in ev if lines[e]["role"] == role
                         and SELF_COMMIT_RE.search(lines[e]["text"])]
                if named:      # "our Industrial Designer, you're gonna ..." -> as said
                    said = pat.search(lines[named[0]]["text"]).group(1)
                    owner = said.upper() if said.lower() == "uid" else said.title()
                    owner_lines = named
                elif selfc:    # owner says "I'll ..."; resolved to a speaker label later
                    owner, owner_lines = f"@SPK:{lines[selfc[0]]['spk']}", selfc
            deadline, dl_lines = None, []
            for e in ev:
                m = DEADLINE_RE.search(lines[e]["text"])
                if m:
                    deadline, dl_lines = m.group(0), [e]
                    break
            tasks.append({"text": text, "owner": owner, "owner_lines": owner_lines,
                          "deadline": deadline, "deadline_lines": dl_lines,
                          "evidence_lines": ev, "_span": ev, "_clear": True})

    topics = []
    tp_path = os.path.join(root, "topics", f"{mid}.json")
    if os.path.exists(tp_path):
        def as_list(x):            # xmltodict collapses single items to a dict
            return x if isinstance(x, list) else ([x] if x else [])

        def collect(t):
            if not isinstance(t, dict):
                return []
            ids = [d["id"] if isinstance(d, dict) else d
                   for d in as_list(t.get("dialogueacts"))]
            for s in as_list(t.get("subtopics")):
                ids += collect(s)
            return [i for i in ids if isinstance(i, str)]
        for t in load(tp_path):
            ls = sorted({L(i) for i in collect(t) if L(i) is not None})
            if ls and t.get("topic") not in (None, "None"):
                topics.append({"title": t["topic"], "start": ls[0], "end": ls[-1]})
    return lines, decisions, tasks, topics


# --------------------------------------------------------------------------
# excerpt (window) -> one few-shot example
# --------------------------------------------------------------------------
def make_example(mid, lines, decisions, tasks, topics, a, b):
    """Lines a..b-1, renumbered from 1, speakers relabelled SPEAKER_00..."""
    spk_map = {}
    for i in range(a, b):
        spk_map.setdefault(lines[i]["spk"], f"SPEAKER_{len(spk_map):02d}")
    n = lambda i: i - a + 1
    inside = lambda span: all(a <= x < b for x in span)

    out_lines = [f"[{n(i)}] {spk_map[lines[i]['spk']]}: {lines[i]['text']}" for i in range(a, b)]
    gold = {"topics": [], "decisions": [], "tasks": []}
    for t in topics:
        s, e = max(t["start"], a), min(t["end"], b - 1)
        if e - s >= 2:
            gold["topics"].append({"title": t["title"].capitalize(), "start_line": n(s), "end_line": n(e)})
    for d in decisions:
        if inside(d["_span"]):
            gold["decisions"].append({"text": d["text"], "status": d["status"],
                                      "proposal_lines": [n(x) for x in d["proposal_lines"]],
                                      "confirmation_lines": [n(x) for x in d["confirmation_lines"]]})
    for t in tasks:
        if inside(t["_span"]):
            owner = t["owner"]
            if owner and owner.startswith("@SPK:"):
                owner = spk_map.get(owner[5:])
            gold["tasks"].append({"text": t["text"], "owner": owner,
                                  "owner_lines": [n(x) for x in t["owner_lines"]] if owner else [],
                                  "deadline": t["deadline"],
                                  "deadline_lines": [n(x) for x in t["deadline_lines"]],
                                  "evidence_lines": [n(x) for x in t["evidence_lines"]]})
    return {"meeting": mid, "window": [a, b], "lines": out_lines, "target": gold}


def best_windows(mid, lines, decisions, tasks, topics, size, step):
    items = decisions + tasks
    cands = []
    for a in range(0, max(1, len(lines) - size + 1), step):
        b = min(a + size, len(lines))
        full = [it for it in items if all(a <= x < b for x in it["_span"])]
        partial = [it for it in items if any(a <= x < b for x in it["_span"]) and it not in full]
        unclear = [it for it in full if not it["_clear"]]
        if not full:
            continue
        n_dec = sum(1 for it in full if it in decisions)
        n_task = len(full) - n_dec
        # reward complete, clearly-confirmed items; punish cut-off items
        score = 2 * n_dec + 3 * n_task - 4 * len(partial) - 3 * len(unclear)
        cands.append((score, a, b, n_dec, n_task, len(partial)))
    return sorted(cands, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami-json", required=True, help="the converters' output/ folder")
    ap.add_argument("--out-dir", default="stageA")
    ap.add_argument("--window", type=int, default=40,
                    help="lines per excerpt (40 keeps 3 examples + a transcript chunk within a T4)")
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meetings = sorted(f[:-5] for f in os.listdir(os.path.join(args.ami_json, "summlink"))
                      if f.endswith(".json"))
    cands, evals = [], []
    for mid in meetings:
        if not os.path.exists(os.path.join(args.ami_json, "dialogueActs", f"{mid}.json")):
            continue
        lines, dec, tasks, topics = build_meeting(mid, args.ami_json)
        if mid.startswith(TEST_PREFIXES):
            ex = make_example(mid, lines, dec, tasks, topics, 0, len(lines))
            evals.append(ex)
            continue
        for score, a, b, nd, nt, npart in best_windows(mid, lines, dec, tasks, topics,
                                                       args.window, step=10)[:2]:
            if npart == 0:
                ex = make_example(mid, lines, dec, tasks, topics, a, b)
                ex["score"] = score
                cands.append(ex)

    cands.sort(key=lambda e: -e["score"])
    # pick a varied set: at least one with tasks that have an owner, one decision-heavy,
    # from different meetings and scenario types
    chosen, used = [], set()

    def pick(pred):
        for ex in cands:
            series = ex["meeting"][:2]          # ES / IS / TS scenario families
            if series not in used and ex not in chosen and pred(ex):
                chosen.append(ex)
                used.add(series)
                return

    pick(lambda e: any(t["owner"] for t in e["target"]["tasks"]) and e["target"]["decisions"])
    pick(lambda e: len(e["target"]["decisions"]) >= 2 and
         any(d["confirmation_lines"] for d in e["target"]["decisions"]))
    pick(lambda e: any(t["deadline"] or (t["owner"] or "").startswith("SPEAKER")
                       for t in e["target"]["tasks"]))
    while len(chosen) < args.n_examples:
        used.clear() if len(used) >= 3 else None
        before = len(chosen)
        pick(lambda e: True)
        if len(chosen) == before:
            break
    chosen = chosen[:args.n_examples]

    with open(os.path.join(args.out_dir, "stageA_fewshot.json"), "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "stageA_candidates.jsonl"), "w", encoding="utf-8") as f:
        for ex in cands:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "stageA_eval.jsonl"), "w", encoding="utf-8") as f:
        for ex in evals:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"{len(meetings)} meetings, {len(cands)} candidate excerpts, "
          f"{len(evals)} held-out test meetings")
    for ex in chosen:
        g = ex["target"]
        print(f"  example {ex['meeting']} lines {ex['window']}: {len(g['topics'])} topics, "
              f"{len(g['decisions'])} decisions, {len(g['tasks'])} tasks")

    try:
        from stageA_prompt import build_messages
        msgs = build_messages(["[1] SPEAKER_00: (your meeting transcript goes here)"], chosen)
        with open(os.path.join(args.out_dir, "stageA_example_prompt.txt"), "w", encoding="utf-8") as f:
            for m in msgs:
                f.write(f"===== {m['role'].upper()} =====\n{m['content']}\n\n")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
