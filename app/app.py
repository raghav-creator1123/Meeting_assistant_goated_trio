"""
Meeting Assistant — Interactive Gradio Application (Local GPU edition).

Runs entirely on your GPU — no API keys required.
  • Whisper (openai/whisper-small by default) for speech-to-text
  • Qwen2.5-3B-Instruct (16-bit) for transcript refinement
  • Qwen2.5-7B-Instruct (4-bit, bitsandbytes) for documentation

Usage:
    python app/app.py             # local URL at http://localhost:7860
    python app.py --share        # also create a public Gradio link
    python app.py --port 8080    # custom port
    python app.py --api          # switch to Groq + Gemini API backend
    (or set DOC_BACKEND=api in .env: refinement on the GPU, documentation on Gemini)

From a notebook:
    from app import build_app
    build_app().queue().launch(share=True)
"""

import argparse
import datetime as dt
import os
import queue
import shutil
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                # repository root: .env, prompts/, outputs/
sys.path.insert(0, HERE)

# Load .env before importing pipeline modules (sets LLM_BACKEND=local etc.)
env_path = os.path.join(ROOT, ".env")
if os.path.isfile(env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        pass

from asr import SUPPORTED                               # noqa: E402
from pipeline import PipelineError, run                # noqa: E402

GLOSSARY = os.path.join(ROOT, "prompts", "glossary.txt")
JOBS = os.path.join(ROOT, "outputs", "app_runs")

SUPPORTED_FMT = ", ".join(SUPPORTED)

# ------------------------------------------------------------------ UI text

def model_names():
    """(refinement, documentation) as 'model (where)' labels, from the .env settings."""
    from llm import LOCAL_DEFAULTS, backend
    out = []
    for role, prefix in (("refine", "REFINE"), ("doc", "DOC")):
        if backend(role) == "local":
            name = os.getenv(f"{prefix}_LOCAL_MODEL") or LOCAL_DEFAULTS[role][0]
            out.append((name.split("/")[-1], "GPU"))
        elif role == "refine":
            out.append((os.getenv("REFINE_MODEL") or "Groq model", "Groq API"))
        else:
            out.append((os.getenv("DOC_MODEL") or "Gemini Flash", "Gemini API"))
    return out


def intro():
    (r_name, r_where), (d_name, d_where) = model_names()
    return f"""# 🎙️ Meeting Assistant

Upload (or record) an **English meeting recording** and press **▶ Run**.

| Step | What runs | Where |
|---|---|---|
| 1 · Transcription | Whisper 5-best hypotheses | GPU |
| 2 · Refinement | {r_name} — fixes domain terms, names, numbers | {r_where} |
| 3 · Extraction | {d_name} — topics, decisions, action items | {d_where} |
| 4 · Verification | {d_name} checks every item against the transcript | {d_where} |
| 5 · Writing | {d_name} — summary + minutes from the verified record | {d_where} |

Owners and deadlines are filled in **only when the recording states them**.  
Everything the pipeline is uncertain about appears under **⚠ Flags**.
"""

CSS = """
.status-box { font-family: monospace; font-size: 0.85em; }
footer { display: none !important; }
"""

# ------------------------------------------------------------------ helpers

def _default_glossary():
    try:
        with open(GLOSSARY, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _empty(status):
    """Return the 6-tuple Gradio outputs expect when results are not ready."""
    return status, "", "", "", "", None


def _fmt_flags(flags):
    if not flags:
        return "_No flags — the pipeline is confident in all items._ ✅"
    lines = []
    for f in flags:
        stage = f.get("stage", "?")
        kind = f.get("type", "?")
        item = f.get("item") or ""
        reason = f.get("reason", "")
        icon = "🔴" if any(x in kind for x in ("unsupported", "error", "failed")) else "🟡"
        parts = [f"{icon} **{stage} · {kind}**"]
        if item:
            parts.append(f"*{item}*")
        if reason:
            parts.append(f"— {reason}")
        lines.append(" ".join(parts))
    return "\n\n".join(lines)


def _save_glossary(text, job_dir):
    if not text or not text.strip():
        return None
    path = os.path.join(job_dir, "glossary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return path


# ------------------------------------------------------------------ pipeline runner

def process(audio_file, audio_path_input, audio_mic, glossary_text, diarize, whisper_model):
    """
    Generator used by the Run button.
    Accepts:
      - audio_file: uploaded file path from gr.File
      - audio_path_input: direct local file path typed/pasted by user
      - audio_mic: microphone recording from gr.Audio
    Yields (status, raw, refined, record_md, flags_md, files) tuples.
    """
    # Prefer direct path if provided, else uploaded file, else microphone
    audio_path = None
    if audio_path_input and audio_path_input.strip():
        audio_path = audio_path_input.strip().strip('"').strip("'")
    elif audio_file:
        audio_path = audio_file
    elif audio_mic:
        audio_path = audio_mic

    if not audio_path:
        yield _empty("❌ Please upload an audio/video file, enter a local file path, or record from microphone.")
        return

    if not os.path.isfile(audio_path):
        yield _empty(f"❌ File not found: `{audio_path}`. Please verify the path.")
        return

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED:
        yield _empty(
            f"❌ Unsupported file type **'{ext or 'none'}'**.\n\n"
            f"Supported formats: {SUPPORTED_FMT}"
        )
        return

    # Override Whisper model if the user changed it
    if whisper_model:
        os.environ["WHISPER_MODEL"] = whisper_model

    # Create a timestamped output folder for this run
    os.makedirs(JOBS, exist_ok=True)
    job = os.path.join(JOBS, dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(job, exist_ok=True)
    gpath = _save_glossary(glossary_text, job)

    # Run the pipeline in a background thread so we can stream progress
    msgs: queue.Queue = queue.Queue()
    box: dict = {}

    def worker():
        try:
            box["result"] = run(
                audio_path,
                out_dir=job,
                glossary=gpath,
                diarize=bool(diarize),
                progress=msgs.put,
            )
        except PipelineError as e:
            box["error"] = f"**{e.stage}:** {e.message}"
        except Exception as e:
            box["error"] = f"**Unexpected error** ({type(e).__name__}): {e}"
        finally:
            msgs.put(None)  # sentinel — tells the loop to stop

    threading.Thread(target=worker, daemon=True).start()

    log = [f"▶ {os.path.basename(audio_path)}"]
    yield _empty("⏳ Starting pipeline…")

    # Stream progress messages to the status box
    while True:
        m = msgs.get()
        if m is None:
            break
        log.append(str(m))
        yield _empty("⏳ Processing…\n```\n" + "\n".join(log[-14:]) + "\n```")

    # --- Error path ---
    if "error" in box:
        yield _empty(
            f"❌ {box['error']}\n\n"
            "```\n" + "\n".join(log[-10:]) + "\n```"
        )
        return

    # --- Success path ---
    res = box["result"]
    r = res["record"]

    raw_text = "\n".join(res["raw_transcript"])
    refined_text = "\n".join(res["refined_transcript"])

    with open(res["files"]["meeting_record.md"], encoding="utf-8") as f:
        record_md = f.read()

    flags_md = _fmt_flags(res["flags"])

    changed = sum(1 for s in res["segments"] if s.get("changed"))
    n_dec = len(r["decisions"])
    n_prop = len(r["proposals"])
    n_task = len(r["tasks"])
    n_flag = len(res["flags"])

    status_md = (
        f"✅ **Done in {res['seconds']:.1f} s**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Transcript segments | {len(res['segments'])} ({changed} refined) |\n"
        f"| Decisions (agreed) | {n_dec} |\n"
        f"| Proposals (not agreed) | {n_prop} |\n"
        f"| Action items | {n_task} |\n"
        f"| Flags for review | {n_flag} |\n\n"
        f"**Models:** `{res['models']['stt']}` → "
        f"`{res['models']['refinement']}` → "
        f"`{res['models']['documentation']}`"
    )

    # ZIP all outputs so user can download everything at once
    zip_path = shutil.make_archive(job + "_all_outputs", "zip", job)
    all_files = list(res["files"].values()) + [zip_path]

    yield status_md, raw_text, refined_text, record_md, flags_md, all_files


# ------------------------------------------------------------------ Gradio UI

def build_app():
    import gradio as gr

    with gr.Blocks(title="Meeting Assistant") as demo:
        gr.Markdown(intro())

        with gr.Row(equal_height=False):

            # ── Left panel: controls ──────────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                gr.Markdown("### 📁 Meeting Recording")
                gr.Markdown(f"<small>Supported: {SUPPORTED_FMT}</small>")
                audio_file = gr.File(
                    label="Upload file (drag & drop or browse)",
                    type="filepath",
                )
                audio_path_input = gr.Textbox(
                    label="Or enter local file path directly",
                    placeholder=r"e.g. D:\Downloads\sample_meeting.mp4",
                    lines=1,
                )
                with gr.Accordion("Or record from microphone", open=False):
                    audio_mic = gr.Audio(
                        label="Record from microphone",
                        sources=["microphone"],
                        type="filepath",
                    )

                gr.Markdown("### 🔧 Options")
                whisper_model = gr.Dropdown(
                    label="Whisper model",
                    info="Larger = more accurate, but slower to load",
                    choices=[
                        "openai/whisper-tiny",
                        "openai/whisper-base",
                        "openai/whisper-small",
                        "openai/whisper-medium",
                        "openai/whisper-large-v3",
                    ],
                    value=os.getenv("WHISPER_MODEL", "openai/whisper-small"),
                )
                diarize = gr.Checkbox(
                    label="Speaker labels (needs HF_TOKEN in .env + pyannote; slower)",
                    value=False,
                )

                gr.Markdown("### 📖 Domain glossary")
                glossary = gr.Textbox(
                    label="One term per line — helps recognise domain-specific words",
                    placeholder="Kubernetes\nPostgreSQL\nCI/CD\n…",
                    value=_default_glossary(),
                    lines=7,
                )

                run_btn = gr.Button("▶  Run", variant="primary", size="lg")

            # ── Right panel: status + tabbed results ──────────────────────
            with gr.Column(scale=2):

                status = gr.Markdown(
                    value=(
                        "_Upload a recording (or use the microphone) and press **▶ Run**._\n\n"
                        "> ⏱ First run downloads Whisper + Qwen model weights "
                        "(~7 GB total) and may take a few minutes."
                    ),
                    elem_classes=["status-box"],
                )

                with gr.Tabs():

                    with gr.Tab("📄 Meeting record"):
                        record = gr.Markdown(
                            value="_Results appear here after processing._"
                        )

                    with gr.Tab("📝 Raw transcript"):
                        raw = gr.Textbox(
                            label="Whisper top hypothesis — before refinement",
                            lines=25,
                            interactive=False,
                        )

                    with gr.Tab("✨ Refined transcript"):
                        refined = gr.Textbox(
                            label="After Qwen2.5-3B correction (domain terms, names, numbers)",
                            lines=25,
                            interactive=False,
                        )

                    with gr.Tab("⚠️ Flags"):
                        flags = gr.Markdown(
                            value="_Flags appear after processing._"
                        )

                    with gr.Tab("📥 Downloads"):
                        gr.Markdown(
                            "All output files are listed below. "
                            "The **_all_outputs.zip** contains everything in one archive."
                        )
                        files = gr.File(
                            label="Output files",
                            file_count="multiple",
                            interactive=False,
                        )

        # Show the picked file path in the textbox when a file is selected
        audio_file.change(
            fn=lambda p: p or "",
            inputs=[audio_file],
            outputs=[audio_path_input],
        )

        run_btn.click(
            fn=process,
            inputs=[audio_file, audio_path_input, audio_mic, glossary, diarize, whisper_model],
            outputs=[status, raw, refined, record, flags, files],
        )

    return demo


# ------------------------------------------------------------------ CLI entry point

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860,
                    help="Local port (default: 7860)")
    ap.add_argument("--api", action="store_true",
                    help="Use Groq + Gemini API backend instead of local GPU")
    args = ap.parse_args()

    if args.api:
        os.environ["LLM_BACKEND"] = "api"
    else:
        os.environ.setdefault("LLM_BACKEND", "local")
    (r_name, r_where), (d_name, d_where) = model_names()
    print(f"[app] Refinement: {r_name} ({r_where}); documentation: {d_name} ({d_where}).")

    build_app().queue().launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
        show_error=True,
        css=CSS,
    )
