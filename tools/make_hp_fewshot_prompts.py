"""
Build HyPoradise-style one-shot prompts (the GPT-3.5 in-context template from
Chen et al., NeurIPS 2023), selected to cover the requirements of the
meeting-assistant task:

  - spontaneous, multi-speaker conversation      -> swbd, coraal
  - semi-spontaneous talks with technical terms  -> td3 (TED-LIUM 3)
  - noisy / far-field audio                      -> chime4
  - accented English (incl. Indian English)      -> cv (CommonVoice), coraal
  - broadcast / overlapping speakers             -> lrs2
  - business vocabulary, numbers, amounts        -> wsj_score
  - names, numbers, dates, times                 -> atis
  - meeting jargon, acronyms, names, negation,
    commitments, disfluencies (NOT in HyPoradise) -> meeting_synth (built in below)

LibriSpeech (read audiobooks) is excluded by default as least meeting-like;
add it with --domains all.

Within every domain, test examples are sampled with quotas so the set checks
each behaviour the task demands:
  - no_change  (~25%): the 1-best is already right -> tests over-correction
  - number     (~25%): reference contains numbers  -> "preserve numbers"
  - negation   (~15%): reference contains negation -> "preserve negation"
  - the rest: other recognition errors

Usage (Colab or local; needs internet access to huggingface.co):
    pip install huggingface_hub
    python make_hp_fewshot_prompts.py --per-domain 200 --out hp_prompts.jsonl

    python make_hp_fewshot_prompts.py --domains all --per-domain all
    python make_hp_fewshot_prompts.py --local-dir ./HyPoradise-v0   # offline copy
    python make_hp_fewshot_prompts.py --dump-meeting meeting_demos.json

Output (one JSON object per line):
    domain, test_id, demo_id, tags, category, prompt, reference, hypotheses
Then score a model on them with run_hp_refinement_eval.py.
A coverage table (domain x tag) is printed and saved to <out>.coverage.json.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

REPO = "PeacefulData/HyPoradise-v0"

DOMAIN_NAMES = {
    "swbd": "Switchboard (spontaneous telephone conversations)",
    "coraal": "CORAAL (spontaneous conversational interviews)",
    "td3": "TED-LIUM 3 (TED talks)",
    "chime4": "CHiME-4 (noisy speech)",
    "cv": "CommonVoice (accented read speech)",
    "lrs2": "LRS2 (BBC broadcast speech)",
    "wsj_score": "WSJ (Wall Street Journal business news)",
    "atis": "ATIS (airline travel information queries)",
    "ls_clean": "LibriSpeech test-clean (audiobooks)",
    "ls_other": "LibriSpeech test-other (audiobooks)",
    "meeting_synth": "business and technical team meetings",
}
DEFAULT_DOMAINS = ["swbd", "coraal", "td3", "chime4", "cv", "lrs2",
                   "wsj_score", "atis", "meeting_synth"]
# test files with no train file of the same name -> train file to draw demos from
TRAIN_FALLBACK = {"ls_clean": "other_500", "ls_other": "other_500"}

# ---------------------------------------------------------------------------
# Meeting-domain examples. HyPoradise has no meeting speech, so these cover the
# task-specific requirements. They are SYNTHETIC: hand-written N-best lists that
# imitate typical Whisper/ASR errors. Say so in your report.
# Each correct transcription is recoverable from the hypotheses.
# ---------------------------------------------------------------------------
MEETING_SYNTH = [
    # technical terms
    {"id": "m01", "category": "technical_term",
     "output": "we'll deploy the new build to kubernetes after the postgres migration",
     "input": ["we'll deploy the new build to cube and eighties after the post grass migration",
               "we'll deploy the new build to kubernetes after the post grass migration",
               "will deploy the new build to cuban eighties after the postgres migration",
               "we'll deploy the new bill to kubernetes after the post gress migration",
               "we'll deploy the new build to kubernetes after the post cross migration"]},
    {"id": "m02", "category": "technical_term",
     "output": "the latency spike was caused by the redis cache eviction policy",
     "input": ["the latency spike was caused by the red is cash eviction policy",
               "the latency spike was caused by the redis cash eviction policy",
               "the late and see spike was caused by the redis cache eviction policy",
               "the latency spike was cause by the reddish cache eviction policy",
               "the latency spike was caused by the red is cache eviction policy"]},
    {"id": "m03", "category": "technical_term",
     "output": "let's fine tune the model with lora instead of full training",
     "input": ["let's fine tune the model with laura instead of full training",
               "let's find tune the model with lora instead of full training",
               "let's fine tune the model with lower instead of full training",
               "lets fine tune the model with laura instead of full training",
               "let's fine tune the modal with lora instead of full training"]},
    {"id": "m04", "category": "technical_term",
     "output": "we should cache the embeddings in the vector database",
     "input": ["we should cash the embeddings in the vector database",
               "we should cache the embedding in the vector database",
               "we should cash the and beddings in the vector database",
               "we should cache the embeddings in the vector data base",
               "we should cache the embeddings in the vector database"]},
    {"id": "m05", "category": "technical_term",
     "output": "the api returns a four oh four when the token expires",
     "input": ["the a pie returns a four oh four when the token expires",
               "the api returns a for oh for when the token expires",
               "the api returns a four oh four when the toke and expires",
               "the api returns of four oh four when the token expires",
               "the api returns a four oh four when the token expire"]},
    # acronyms
    {"id": "m06", "category": "acronym",
     "output": "we need the s o w signed before the q three kickoff",
     "input": ["we need the so signed before the cue three kickoff",
               "we need the s o w signed before the queue three kick off",
               "we need the s o w sign before the q three kickoff",
               "we need the sow signed before the q three kickoff",
               "we need the s o w signed before the q three kickoff"]},
    {"id": "m07", "category": "acronym",
     "output": "the c i c d pipeline failed on the staging branch",
     "input": ["the see icy d pipeline failed on the staging branch",
               "the c i c d pipe line failed on the staging branch",
               "the c i c d pipeline failed on the stage in branch",
               "the sea i see d pipeline failed on the staging branch",
               "the c i c d pipeline fail on the staging branch"]},
    {"id": "m08", "category": "acronym",
     "output": "our o k r for this quarter is to cut churn by ten percent",
     "input": ["our okay are for this quarter is to cut churn by ten percent",
               "our o k r for this quarter is to cut turn by ten percent",
               "our o k r for this quarter is to cut church by ten percent",
               "are o k r for this quarter is to cut churn by ten percent",
               "our o k r for this quarter is to cut churn by ten per cent"]},
    {"id": "m09", "category": "acronym",
     "output": "the g d p r review is still pending with legal",
     "input": ["the g d p r review is still pending with legal",
               "the gdp are review is still pending with legal",
               "the g d p r review is still spending with legal",
               "the g dpr review is still pending with legal",
               "the g d p r reviews still pending with legal"]},
    # names
    {"id": "m10", "category": "name",
     "output": "priya will review the pull request before anirban merges it",
     "input": ["pria will review the pull request before on air been merges it",
               "priya will review the pull request before anirban merges it",
               "priya will review the pool request before an urban merges it",
               "pre a will review the pull request before anirban merges it",
               "priya will review the pull request before honor ban merges it"]},
    {"id": "m11", "category": "name",
     "output": "send the deck to mister okafor and the team in bengaluru",
     "input": ["send the deck to mister okay for and the team in bengaluru",
               "send the deck to mister okafor and the team in bangalore you",
               "send the deck to mister okafor and the team in bengaluru",
               "send the dec to mister o cafe or and the team in bengaluru",
               "send the deck to mister okafor and the team and bengaluru"]},
    {"id": "m12", "category": "name",
     "output": "rahul is going to handle the client demo next week",
     "input": ["raul is going to handle the client demo next week",
               "rahul is going to handle the client demo next week",
               "rahul is going to handle the client demo next weak",
               "rahul's going to handle the client demo next week",
               "rahul is going to handle the clients demo next week"]},
    # numbers, dates, amounts
    {"id": "m13", "category": "number",
     "output": "the budget is fifteen thousand dollars for the first phase",
     "input": ["the budget is fifty thousand dollars for the first phase",
               "the budget is fifteen thousand dollars for the first phase",
               "the budget is fifteen thousand dollars for the first faze",
               "the budget is fifty thousand dollars for the first phase",
               "the budget is fifteen thousand dollars for the first base"]},
    {"id": "m14", "category": "number",
     "output": "let's move the release to the twenty third of october",
     "input": ["let's move the release to the twenty third of october",
               "let's move the release to the twenty-third of october",
               "let's move the release to the twenty third of october",
               "lets move the release to the twenty third of october",
               "let's move the release to the twenty third october"]},
    {"id": "m15", "category": "number",
     "output": "we hit ninety nine point nine percent uptime last month",
     "input": ["we hit ninety nine point nine percent up time last month",
               "we hit ninety nine point nine percent uptime last month",
               "we hit ninety-nine point nine percent uptime last month",
               "we hit ninety nine point nine per cent uptime last month",
               "we hit ninety nine point nine percent uptime lost month"]},
    {"id": "m16", "category": "number",
     "output": "the meeting moved from two thirty to four p m",
     "input": ["the meeting moved from two thirty to four p m",
               "the meeting moved from two thirty two four p m",
               "the meeting moved from too thirty to four p m",
               "the meeting moved from two thirty to for p m",
               "the meeting move from two thirty to four p m"]},
    # negation
    {"id": "m17", "category": "negation",
     "output": "i don't think we should ship it on friday",
     "input": ["i don't think we should ship it on friday",
               "i do think we should ship it on friday",
               "i don't think we should shift it on friday",
               "i don't think we should ship it on friday",
               "i don't think we should ship it friday"]},
    {"id": "m18", "category": "negation",
     "output": "we can't support the legacy android version anymore",
     "input": ["we can support the legacy android version anymore",
               "we can't support the legacy android version any more",
               "we can't support the legacy android version anymore",
               "we can't support the legacy and droid version anymore",
               "we can't support the legacy android version anyway"]},
    {"id": "m19", "category": "negation",
     "output": "nobody objected to dropping the old dashboard",
     "input": ["nobody objected to dropping the old dash board",
               "nobody object to dropping the old dashboard",
               "nobody objected to dropping the old dashboard",
               "nobody objected to dropping the gold dashboard",
               "no body objected to dropping the old dashboard"]},
    # commitments / assignments (wording must survive refinement intact)
    {"id": "m20", "category": "commitment",
     "output": "i'll have the test report ready by monday",
     "input": ["i'll have the test report ready by monday",
               "i have the test report ready by monday",
               "all have the test report ready by monday",
               "i'll have the test report ready by sunday",
               "i'll have the test reports ready by monday"]},
    {"id": "m21", "category": "commitment",
     "output": "can someone look into the memory leak",
     "input": ["can someone look into the memory leek",
               "can some one look into the memory leak",
               "can someone look into the memory leak",
               "can someone look in to the memory leak",
               "can someone look into the memory lake"]},
    {"id": "m22", "category": "commitment",
     "output": "okay that's agreed then we go with option b",
     "input": ["okay that's agreed then we go with option b",
               "okay that's agreed then we go with option be",
               "ok that's agreed then we go with option b",
               "okay that's a greed then we go with option b",
               "okay that's agreed and we go with option b"]},
    # disfluencies must be kept (no grammar clean-up)
    {"id": "m23", "category": "disfluency",
     "output": "so um we we should probably revisit the the pricing page",
     "input": ["so um we we should probably revisit the the pricing page",
               "so um we should probably revisit the pricing page",
               "so we we should probably revisit the the pricing page",
               "so um we we should probably revisit the the price in page",
               "so um we we should probably revisit the pricing page"]},
    {"id": "m24", "category": "disfluency",
     "output": "yeah i mean the the onboarding flow is kind of broken right now",
     "input": ["yeah i mean the the on boarding flow is kind of broken right now",
               "yeah i mean the onboarding flow is kind of broken right now",
               "yeah i mean the the onboarding flow is kind of broken right now",
               "yeah i mean the the onboarding flow is kinda broken right now",
               "yeah i mean the the onboarding flow is kind of broke in right now"]},
]

TEMPLATE = (
    "Q: Are you familiar with speech recognition? "
    "R: Yes, I am familiar with speech recognition. Speech recognition, also known as "
    "automatic speech recognition (ASR) or speech-to-text, is the process of converting "
    "spoken language into text. This technology involves using algorithms and machine "
    "learning models to analyze and transcribe the acoustic features of spoken words and "
    "phrases. Speech recognition has many applications, including voice-controlled "
    "assistants, automated phone systems, and transcription services. "
    "Q: Are you familiar with language model rescoring in ASR? "
    "R: Yes, I am familiar with language model rescoring for speech recognition. Language "
    "model rescoring is a technique used to improve the accuracy of speech recognition "
    "systems. It involves using a separate language model to evaluate the likelihood of a "
    "given hypothese list. This separate model is typically more complex and powerful than "
    "the initial language model used for the transcription, and it is used to re-score the "
    "transcription based on the probability of the words occurring in the given context. "
    "The rescoring process involves taking the output of the initial language model, which "
    "is usually based on statistical methods such as Hidden Markov Models, and then "
    "applying a more advanced language model, such as a neural network-based language "
    "model, to generate a more accurate transcription. This is accomplished by re-ranking "
    "the possible transcriptions based on the probabilities assigned by the more advanced "
    "language model. Language model rescoring has been shown to significantly improve the "
    "accuracy of speech recognition systems, particularly in noisy or challenging "
    "environments where the initial language model may not perform well. "
    "Q: Can you give a possible example on language model rescoring with 5-best hypotheses? "
    "R: Sure, here is an example of language model rescoring for ASR with 5-best "
    "hypotheses: 1. I want to go to the store. 2. I want to go to the storm. 3. I want to "
    "go to the stove. 4. I want to go to the star. 5. I want to go to the storage. After "
    "rescoring, I think the ground-truth of this speech should be: I want to go to the store. "
    "Q: Nice job, i will give you a real example as a demonstration from {domain}. "
    "The 5- best hypothesis is: {demo_hyps}, and I expect your output is: {demo_ref}. "
    "Following this example, can you report the true transcription from the following "
    "5-best hypotheses:? {test_hyps}"
)

NUMBER_RE = re.compile(
    r"\d|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"first|second|third|fourth|fifth|percent|dollars?|o'clock)\b")
NEGATION_RE = re.compile(
    r"n't\b|\b(not|no|never|nobody|nothing|none|neither|nor|cannot|without)\b")


def norm(s):
    return " ".join(s.lower().split())


def fmt_hyps(hyps):
    return " ".join(f"{i}. {h.strip()}" for i, h in enumerate(hyps, 1))


def get_hyps(rec, n=5):
    hyps = rec.get("input")
    if isinstance(hyps, str):
        hyps = [hyps]
    if not hyps:
        hyps = [rec[k] for k in ("input1", "input2") if rec.get(k)]
    hyps = [h for h in hyps if isinstance(h, str) and h.strip() and h.strip() != "<UNK>"]
    return hyps[:n]


def tags_for(rec):
    ref, hyps = rec["output"], get_hyps(rec)
    t = []
    t.append("no_change" if hyps and norm(hyps[0]) == norm(ref) else "error")
    if NUMBER_RE.search(ref.lower()):
        t.append("number")
    if NEGATION_RE.search(ref.lower()):
        t.append("negation")
    if rec.get("category"):
        t.append(rec["category"])
    return t


def quota_sample(recs, n, rng):
    """Sample n records with quotas: 25% no_change, 25% number, 15% negation."""
    if n >= len(recs):
        return list(recs)
    pool = list(recs)
    rng.shuffle(pool)
    chosen, used = [], set()

    def take(pred, k):
        for i, r in enumerate(pool):
            if k <= 0:
                break
            if i not in used and pred(r):
                chosen.append(r); used.add(i); k -= 1

    take(lambda r: "no_change" in r["_tags"], round(n * 0.25))
    take(lambda r: "number" in r["_tags"] and "error" in r["_tags"], round(n * 0.25))
    take(lambda r: "negation" in r["_tags"], round(n * 0.15))
    take(lambda r: True, n - len(chosen))
    return chosen


def load_json(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def find_files(local_dir):
    if local_dir:
        paths = []
        for root, _, names in os.walk(local_dir):
            paths += [os.path.relpath(os.path.join(root, n), local_dir) for n in names]
    else:
        from huggingface_hub import list_repo_files
        paths = list_repo_files(REPO, repo_type="dataset")
    found = defaultdict(dict)
    for p in paths:
        m = re.match(r"(?:.*/)?(train|test)_(.+)\.json$", p)
        if m:
            found[m.group(2)][m.group(1)] = p
    pairs = {}
    for dom, s in found.items():
        if "test" not in s:
            continue
        train = s.get("train") or found.get(TRAIN_FALLBACK.get(dom, ""), {}).get("train")
        if train:
            pairs[dom] = {"train": train, "test": s["test"]}
    return pairs


def fetch(path, local_dir):
    if local_dir:
        return load_json(os.path.join(local_dir, path))
    from huggingface_hub import hf_hub_download
    return load_json(hf_hub_download(REPO, path, repo_type="dataset"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hp_prompts.jsonl")
    ap.add_argument("--per-domain", default="200", help="test examples per domain, or 'all'")
    ap.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS,
                    help="domain keys, or 'all' (adds LibriSpeech)")
    ap.add_argument("--local-dir", help="folder with already-downloaded HP files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-meeting", help="also save the meeting examples to this JSON file")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.dump_meeting:
        with open(args.dump_meeting, "w", encoding="utf-8") as f:
            json.dump(MEETING_SYNTH, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(MEETING_SYNTH)} meeting examples to {args.dump_meeting}")

    want = None if args.domains == ["all"] else set(args.domains)
    hp_doms = [d for d in (want or {"x"}) if d != "meeting_synth"]
    files = find_files(args.local_dir) if hp_doms else {}
    if want:
        missing = want - set(files) - {"meeting_synth"}
        if missing:
            print("Warning: not found in the dataset:", ", ".join(sorted(missing)))
        files = {d: s for d, s in files.items() if d in want}

    coverage = defaultdict(Counter)
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:

        def emit(dom, rec, demo):
            nonlocal total
            out.write(json.dumps({
                "domain": dom,
                "test_id": rec.get("id", ""),
                "demo_id": demo.get("id", ""),
                "tags": rec["_tags"],
                "category": rec.get("category", ""),
                "prompt": TEMPLATE.format(
                    domain=DOMAIN_NAMES.get(dom, dom),
                    demo_hyps=fmt_hyps(get_hyps(demo)),
                    demo_ref=demo["output"].strip(),
                    test_hyps=fmt_hyps(get_hyps(rec))),
                "reference": rec["output"].strip(),
                "hypotheses": get_hyps(rec),
            }, ensure_ascii=False) + "\n")
            coverage[dom]["total"] += 1
            for t in rec["_tags"]:
                coverage[dom][t] += 1
            total += 1

        for dom in sorted(files):
            skipped = 0
            train, test = [], []
            for split, bucket in (("train", train), ("test", test)):
                for r in fetch(files[dom][split], args.local_dir):
                    if r.get("output") and get_hyps(r):
                        r["_tags"] = tags_for(r)
                        bucket.append(r)
                    else:
                        skipped += 1
            train = [r for r in train if len(get_hyps(r)) == 5]
            if not train or not test:
                print(f"  skip {dom}: no usable records")
                continue
            hard = [r for r in train if "error" in r["_tags"]]
            pool = hard if len(hard) >= 20 else train   # demos that show a fix
            if args.per_domain != "all":
                test = quota_sample(test, int(args.per_domain), rng)
            for rec in test:
                emit(dom, rec, rng.choice(pool))
            print(f"  {dom}: {len(test)} prompts, demo pool {len(pool)}"
                  + (f", skipped {skipped} malformed records" if skipped else ""))

        if want is None or "meeting_synth" in want:
            # Every meeting example is a test item; its demo is another meeting
            # example (never itself) that shows an actual correction.
            recs = [dict(r, _tags=None) for r in MEETING_SYNTH]
            for r in recs:
                r["_tags"] = tags_for(r)
            for rec in recs:
                pool = [d for d in recs if d["id"] != rec["id"] and "error" in d["_tags"]]
                emit("meeting_synth", rec, rng.choice(pool))
            print(f"  meeting_synth: {len(recs)} prompts (synthetic)")

    cols = ["total", "no_change", "error", "number", "negation"]
    print("\nCoverage")
    print(f"{'domain':<15}" + "".join(f"{c:>11}" for c in cols))
    for dom in sorted(coverage):
        print(f"{dom:<15}" + "".join(f"{coverage[dom][c]:>11}" for c in cols))
    with open(args.out + ".coverage.json", "w") as f:
        json.dump({d: dict(c) for d, c in coverage.items()}, f, indent=2)
    print(f"\nWrote {total} prompts to {args.out}")


if __name__ == "__main__":
    main()
