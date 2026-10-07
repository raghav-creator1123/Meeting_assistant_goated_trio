"""Scoring helpers used by prompt_tuning.ipynb."""

import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from refine import refine_items
from stageA_prompt import stage_b_filter


# ---------------------------------------------------------------- refinement
def norm(s):
    s = (s or "").lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).split())


def corpus_wer(refs, hyps):
    import jiwer
    pairs = [(norm(r), norm(h) or "<empty>") for r, h in zip(refs, hyps) if norm(r)]
    return 100 * jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")


def extract_answer(text):
    """Pull the transcription out of a chatty answer to the paper template."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ""
    cand = lines[-1]
    for l in lines:
        m = re.search(r"(?:transcription|ground[- ]truth|output)[^:]*:\s*(.+)", l, re.I)
        if m:
            cand = m.group(1)
    return cand.strip().strip('"“”\'').strip()


def hp_paper_answers(rows, client, model, workers=2, system=None):
    """Variant A: the HyPoradise one-shot template, one call per utterance."""
    def one(row):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": row["prompt"]}]
        try:
            from llm import create_with_wait
            r = create_with_wait(client, model=model, temperature=0, max_tokens=300,
                                 messages=msgs)
            return extract_answer(r.choices[0].message.content)
        except Exception as e:                       # count as a failure, keep going
            return f"__ERROR__ {type(e).__name__}: {' '.join(str(e).split())[:300]}"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


def hp_batched_answers(rows, client, model, use_demo=True, batch_size=20, glossary=None):
    """Variants B/C: the pipeline's refinement prompt + guards, many utterances per call."""
    items = [{"id": i + 1, "hypotheses": r["hypotheses"] or [""]} for i, r in enumerate(rows)]
    out = []
    for b in range(0, len(items), batch_size):     # no cross-utterance context for HP
        try:
            out += refine_items(items[b:b + batch_size], client, model, glossary,
                                batch_size=batch_size, use_demo=use_demo, context_size=0)
        except Exception as e:                       # record as failures, not as empty answers
            print(f"[batch {b // batch_size + 1}] failed: {type(e).__name__}: {str(e)[:300]}")
            out += [{"id": it["id"], "text": f"__ERROR__ {type(e).__name__}", "flag": "error"}
                    for it in items[b:b + batch_size]]
    by_id = {o["id"]: o["text"] for o in out}
    return [by_id.get(i + 1, "") for i in range(len(rows))]


def hp_table(rows, answers_by_variant):
    """WER per group for the 1-best baseline, the oracle and each variant."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups["ALL"].append(i)
        groups[f"domain:{r['domain']}"].append(i)
        for t in r.get("tags", []):
            groups[f"tag:{t}"].append(i)
    import jiwer
    table = []
    for g in ["ALL"] + sorted(k for k in groups if k != "ALL"):
        idx = groups[g]
        refs = [rows[i]["reference"] for i in idx]
        oracle = [min(rows[i]["hypotheses"] or [""],
                      key=lambda h: jiwer.wer(norm(rows[i]["reference"]) or "x", norm(h) or "x"))
                  for i in idx]
        row = {"group": g, "n": len(idx),
               "1-best": round(corpus_wer(refs, [(rows[i]["hypotheses"] or [""])[0] for i in idx]), 2),
               "oracle": round(corpus_wer(refs, oracle), 2)}
        for name, ans in answers_by_variant.items():
            keep = [i for i in idx if ans[i] is not None and not str(ans[i]).startswith("__ERROR__")]
            row[name] = round(corpus_wer([rows[i]["reference"] for i in keep],
                                         [ans[i] for i in keep]), 2) if keep else None
            row[f"{name}_n"] = len(keep)
        table.append(row)
    return table


# ------------------------------------------------------------------- Stage A
def tok_f1(a, b):
    stop = {"the", "a", "an", "will", "be", "to", "of", "and", "for", "on", "in", "they",
            "team", "group", "is", "it", "that", "with"}
    ta = [w for w in norm(a).split() if w not in stop]
    tb = [w for w in norm(b).split() if w not in stop]
    if not ta or not tb:
        return 0.0
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def _lines(it):
    return set(it.get("proposal_lines", []) + it.get("confirmation_lines", []) +
               it.get("evidence_lines", []))


def _match(pred, gold):
    """Greedy one-to-one matching on text similarity and nearby evidence lines."""
    used, hits = set(), 0
    for p in pred:
        best, best_s = None, 0.0
        for j, g in enumerate(gold):
            if j in used:
                continue
            s = tok_f1(p["text"], g["text"])
            near = any(abs(a - b) <= 2 for a in _lines(p) for b in _lines(g))
            score = s + (0.2 if near else 0)
            if (s >= 0.4 or (near and s >= 0.2)) and score > best_s:
                best, best_s = j, score
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def stageA_metrics(raw, gold, lines):
    """Compare one Stage A output (after Stage B) with AMI gold for a meeting."""
    rec = stage_b_filter(raw, lines)
    pd_, gd = rec["decisions"], [d for d in gold["decisions"] if d["status"] == "agreed"]
    pt, gt = rec["tasks"], gold["tasks"]
    dh, th = _match(pd_, gd), _match(pt, gt)
    raw_tasks = raw.get("tasks", []) if isinstance(raw, dict) else []
    owners = [t for t in raw_tasks if t.get("owner")]
    bad_owner = sum(1 for f in rec["flags"] if f["type"].startswith("owner_"))
    cited = [i for k in ("decisions", "tasks") for it in (raw.get(k, []) if isinstance(raw, dict) else [])
             for i in _lines(it)]
    valid = sum(1 for i in cited if isinstance(i, int) and 1 <= i <= len(lines))
    safe = lambda a, b: round(a / b, 3) if b else None
    return {
        "dec_precision": safe(dh, len(pd_)), "dec_recall": safe(dh, len(gd)),
        "task_precision": safe(th, len(pt)), "task_recall": safe(th, len(gt)),
        "owner_invented_rate": safe(bad_owner, len(owners)),
        "valid_citations": safe(valid, len(cited)),
        "n_pred_decisions": len(pd_), "n_gold_decisions": len(gd),
        "n_pred_tasks": len(pt), "n_gold_tasks": len(gt),
        "proposals": len(rec["proposals"]),
    }


def average(dicts):
    keys = [k for k in dicts[0] if not k.startswith("n_") and k != "proposals"]
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d[k] is not None]
        out[k] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def polite_sleep(seconds):
    if seconds:
        time.sleep(seconds)
