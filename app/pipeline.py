"""
End-to-end meeting assistant pipeline.

  audio ─► Whisper 5-best ─► refinement (Groq) ─► Stage A extract (Gemini)
        ─► Stage B filter (code) ─► Stage D verify (Gemini) ─► Stage C write (Gemini)
        ─► Markdown + JSON record

Usage
    python pipeline.py meeting.mp3 --out outputs --glossary glossary.txt [--diarize]
    python pipeline.py --segments nbest.json --out outputs     # skip speech-to-text

Prompt choices made in prompt_tuning.ipynb are read from prompts/*.json.
Every failure is raised as PipelineError with a message that can be shown in the UI.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from asr import AudioError                                  # noqa: E402
from llm import LLMError, RequestTooLarge, make_client      # noqa: E402


def _now():
    tz = os.getenv("MEETING_TZ", "")
    if tz:
        try:
            from zoneinfo import ZoneInfo
            now = dt.datetime.now(ZoneInfo(tz))
            return now.strftime("%Y-%m-%d %H:%M ") + (now.tzname() or tz)
        except Exception:
            pass
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


class PipelineError(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(f"{stage}: {message}")
        self.stage, self.message = stage, message


def load_config(name, default):
    path = os.path.join(HERE, "prompts", name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {**default, **json.load(f)}
    return default


def load_glossary(path):
    if not path:
        path = os.path.join(HERE, "prompts", "glossary.txt")
        if not os.path.exists(path):
            return []
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f.read().replace(",", "\n").splitlines() if w.strip()]


def numbered(segments, key):
    return [f"[{s['id']}] {s['speaker'] + ': ' if s.get('speaker') else ''}{s[key]}"
            for s in segments]


def run(audio_path=None, out_dir="outputs", glossary=None, diarize=False,
        segments=None, progress=print, transcriber=None, clients=None):
    """Run the whole pipeline. Returns the result dict and writes the output files."""
    t0 = time.time()
    say = progress or (lambda *_: None)
    refine_cfg = load_config("refine_config.json", {"use_demo": True, "batch_size": 20})
    stageA_cfg = load_config("stageA_config.json", {"variant": "ami+synthetic"})
    os.makedirs(out_dir, exist_ok=True)
    flags, notes = [], []

    # ---- models ---------------------------------------------------------
    try:
        (r_client, r_model), (d_client, d_model) = clients or \
            (make_client("refine"), make_client("doc"))
    except LLMError as e:
        raise PipelineError("Setup", str(e))
    stt_name = os.getenv("WHISPER_MODEL", "openai/whisper-small")

    # ---- 1. speech-to-text ------------------------------------------------
    if segments is None:
        say("1/6 Transcribing audio (Whisper, 5-best)...")
        try:
            from asr import Transcriber, add_speakers, load_audio
            audio = load_audio(audio_path)          # check the file before loading any model
            own_stt = transcriber is None
            transcriber = transcriber or Transcriber(stt_name)
            segments = transcriber.transcribe(audio_path, progress=say, audio=audio)
            if own_stt:                     # free the GPU for the language models
                del transcriber
                from llm import unload_local
                unload_local(keep="__none__")
            if diarize:
                segments, note = add_speakers(audio_path, segments)
                notes.append(note)
        except AudioError as e:
            raise PipelineError("Audio", str(e))
        except Exception as e:
            raise PipelineError("Speech-to-text", f"{type(e).__name__}: {e}")
    else:
        stt_name = "provided n-best"
    for s in segments:
        s.setdefault("flags", [])
        s.setdefault("speaker", None)
    raw_lines = numbered([{**s, "raw": s["hypotheses"][0]} for s in segments], "raw")

    # ---- 2. refinement ----------------------------------------------------
    say(f"2/6 Refining transcript ({r_model})...")
    try:
        from refine import refine_segments
        refine_segments(segments, r_client, r_model, glossary=load_glossary(glossary),
                        batch_size=refine_cfg["batch_size"], use_demo=refine_cfg["use_demo"],
                        progress=say)
    except LLMError as e:
        raise PipelineError("Refinement", str(e))
    lines = numbered(segments, "refined")
    for s in segments:
        for f in s["flags"]:
            flags.append({"stage": "STT" if "asr" in f or "silent" in f else "Refinement",
                          "type": f, "item": f"line {s['id']}", "reason": s["refined"][:120]})

    # ---- 3. Stage A: extract ----------------------------------------------
    say(f"3/6 Extracting topics, decisions and tasks ({d_model})...")
    from stageA_prompt import messages_for_variant, SCHEMA, stage_b_filter, merge_extractions
    from llm import chat_json
    variants = list(dict.fromkeys([stageA_cfg["variant"], "synthetic", "none"]))
    state = {"v": 0}                     # once a smaller prompt was needed, keep using it

    def extract(chunk_lines):
        while True:
            variant = variants[state["v"]]
            try:
                return chat_json(d_client, d_model, messages_for_variant(chunk_lines, variant),
                                 schema=SCHEMA, temperature=0)
            except RequestTooLarge as e:
                if state["v"] == len(variants) - 1:
                    raise PipelineError("Stage A (extraction)",
                                        f"{e} Even the smallest prompt does not fit; lower "
                                        f"STAGEA_CHUNK_LINES.")
                state["v"] += 1
                flags.append({"stage": "A", "type": "smaller_prompt_used",
                              "item": variants[state["v"]], "reason": str(e)[:300]})
                say(f"   prompt too large ({e}); retrying with fewer examples "
                    f"({variants[state['v']]})...")
            except LLMError as e:
                raise PipelineError("Stage A (extraction)", str(e))

    # long transcripts are processed in overlapping chunks (local models have less memory)
    local = os.getenv("LLM_BACKEND", "api").lower() == "local"
    size = int(os.getenv("STAGEA_CHUNK_LINES", "100" if local else "100000"))
    overlap = 10
    if len(lines) <= size:
        raw = extract(lines)
    else:
        parts, start = [], 0
        while True:
            say(f"   Stage A on lines {start + 1}-{min(start + size, len(lines))} of {len(lines)}...")
            parts.append(extract(lines[start:start + size]))
            if start + size >= len(lines):
                break
            start += size - overlap
        raw = merge_extractions(parts)

    # ---- 4. Stage B: filter -----------------------------------------------
    say("4/6 Applying evidence rules (Stage B)...")
    record = stage_b_filter(raw, lines)

    # ---- 5. Stage D: verify -----------------------------------------------
    from document import stage_d_verify, stage_c_write, render_markdown, to_json
    say("5/6 Verifying every item against the transcript (Stage D)...")
    try:
        try:
            stage_d_verify(record, lines, d_client, d_model)
        except RequestTooLarge:
            say("   request too large; verifying in smaller batches...")
            stage_d_verify(record, lines, d_client, d_model, batch_size=4)
    except LLMError as e:          # verification is a safety net: keep going, but say so
        record["flags"].append({"stage": "D", "type": "verification_failed", "item": "",
                                "reason": str(e)})

    # ---- 6. Stage C: write ------------------------------------------------
    say("6/6 Writing summary and minutes (Stage C)...")
    summary, minutes = "", []
    for max_chars in (60000, 15000, 5000):
        try:
            summary, minutes = stage_c_write(record, lines, d_client, d_model, max_chars=max_chars)
            if max_chars < 60000:
                flags.append({"stage": "C", "type": "smaller_prompt_used", "item": str(max_chars),
                              "reason": "transcript excerpts shortened to fit the model's limit"})
            break
        except RequestTooLarge as e:
            if max_chars == 5000:
                raise PipelineError("Stage C (writing)", str(e))
            say("   request too large; writing from shorter transcript excerpts...")
        except LLMError as e:
            raise PipelineError("Stage C (writing)", str(e))

    def used(client, default):
        names = getattr(client, "used", None) or [default]
        for a, b, why in getattr(client, "switches", []):
            flags.append({"stage": "Models", "type": "model_switched", "item": f"{a} -> {b}",
                          "reason": why})
        return " + ".join(names)
    refine_name, doc_name = used(r_client, r_model), used(d_client, d_model)
    result = {
        "source": os.path.basename(audio_path) if audio_path else "provided transcript",
        "generated_at": _now(),
        "models": {"stt": stt_name, "refinement": refine_name, "documentation": doc_name},
        "summary": summary, "minutes": minutes, "record": record,
        "flags": flags + record["flags"], "notes": notes,
        "raw_transcript": raw_lines, "refined_transcript": lines, "segments": segments,
        "seconds": round(time.time() - t0, 1),
    }
    write_outputs(result, out_dir)
    say(f"Done in {result['seconds']} s. {len(record['decisions'])} decisions, "
        f"{len(record['tasks'])} tasks, {len(result['flags'])} flags.")
    return result


def write_outputs(result, out_dir):
    from document import render_markdown, to_json
    files = {
        "raw_transcript.txt": "\n".join(result["raw_transcript"]) + "\n",
        "refined_transcript.txt": "\n".join(result["refined_transcript"]) + "\n",
        "meeting_record.md": render_markdown(result),
        "meeting_record.json": json.dumps(to_json(result), indent=2, ensure_ascii=False),
        "nbest_segments.json": json.dumps(result["segments"], indent=1, ensure_ascii=False),
    }
    result["files"] = {}
    for name, content in files.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        result["files"][name] = path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("audio", nargs="?", help="meeting recording")
    ap.add_argument("--segments", help="JSON list of {id, hypotheses[, speaker]} (skips STT)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--glossary", help="text file of domain terms, one per line")
    ap.add_argument("--diarize", action="store_true", help="speaker labels (needs HF_TOKEN)")
    args = ap.parse_args()
    if not args.audio and not args.segments:
        ap.error("give an audio file or --segments")
    segs = None
    if args.segments:
        with open(args.segments, encoding="utf-8") as f:
            segs = json.load(f)
    try:
        res = run(args.audio, args.out, args.glossary, args.diarize, segments=segs)
    except PipelineError as e:
        print(f"\nERROR in {e.stage}: {e.message}", file=sys.stderr)
        sys.exit(1)
    print("\nOutputs:")
    for name, path in res["files"].items():
        print(f"  {path}")


if __name__ == "__main__":
    main()
