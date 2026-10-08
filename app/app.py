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
import html
import inspect
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


# ---------------------------------------------------------------------------------------
# Look and feel: black background, one accent colour per section.
#   1 Upload = violet   2 Status = cyan   3 Results = green
# ---------------------------------------------------------------------------------------
CSS = """
:root, .dark, .light {
  --ma-violet:#8b5cf6; --ma-blue:#3b82f6; --ma-cyan:#22d3ee; --ma-green:#34d399;
  --ma-amber:#fbbf24; --ma-rose:#fb7185; --ma-line:#22222e; --ma-dim:#8d8da6; --ma-text:#ececf4;
  --body-background-fill:#000 !important;
  --background-fill-primary:#07070b !important;
  --background-fill-secondary:#0d0d14 !important;
  --block-background-fill:#07070b !important;
  --panel-background-fill:#07070b !important;
  --block-border-color:#1f1f2b !important;
  --border-color-primary:#262634 !important;
  --body-text-color:#ececf4 !important;
  --body-text-color-subdued:#9a9ab2 !important;
  --block-label-background-fill:#0d0d14 !important;
  --block-label-text-color:#b9b9d0 !important;
  --block-title-text-color:#d6d6e6 !important;
  --input-background-fill:#0d0d14 !important;
  --input-background-fill-focus:#11111b !important;
  --input-border-color:#2a2a3a !important;
  --checkbox-background-color:#0d0d14 !important;
  --checkbox-label-background-fill:#0d0d14 !important;
  --color-accent:#8b5cf6 !important;
  --link-text-color:#67e8f9 !important;
}
html { scroll-behavior:smooth; }
html, body, gradio-app, .gradio-container { background:#000 !important; color:var(--ma-text); }
.gradio-container { width:100% !important; max-width:1280px !important; margin:0 auto !important; box-sizing:border-box; }
footer { opacity:.55; }

/* ---------- header (centred) and index ---------- */
.ma-hero { text-align:center; padding:1.6rem 1rem .4rem; }
.ma-hero h1 { margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-.5px;
  background:linear-gradient(90deg,#a78bfa,#60a5fa 45%,#22d3ee); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.ma-hero p { margin:.55rem auto 0; max-width:760px; color:var(--ma-dim); line-height:1.6; font-size:.97rem; }
.ma-hero .ma-tablewrap { max-width:760px; margin:.9rem auto 0; }
.ma-index { display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:.55rem;
  margin:.9rem auto 1.1rem; padding:.55rem .8rem; border:1px solid var(--ma-line);
  border-radius:999px; background:#07070b; width:fit-content; max-width:100%; }
.ma-index .ma-index-title { color:var(--ma-dim); font-size:.72rem; text-transform:uppercase;
  letter-spacing:.14em; padding:0 .4rem; }
.ma-index a { display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .8rem .3rem .35rem;
  border-radius:999px; color:var(--ma-text) !important; text-decoration:none !important;
  font-size:.9rem; transition:background .15s; }
.ma-index a:hover { background:#14141f; }
.ma-index a b, .ma-num { display:inline-grid; place-items:center; min-width:1.55rem; height:1.55rem;
  border-radius:50%; font-size:.82rem; font-weight:800; color:#000; }
.ma-index a.c1 b, .ma-sec.c1 .ma-num { background:linear-gradient(135deg,#a78bfa,#6366f1); color:#fff; }
.ma-index a.c2 b, .ma-sec.c2 .ma-num { background:linear-gradient(135deg,#22d3ee,#3b82f6); }
.ma-index a.c3 b, .ma-sec.c3 .ma-num { background:linear-gradient(135deg,#34d399,#22d3ee); }

/* ---------- numbered section headings (left aligned) ---------- */
.ma-sec { display:flex; align-items:center; gap:.75rem; text-align:left; padding:.2rem 0 .1rem;
  scroll-margin-top:1rem; }
.ma-sec .ma-num { min-width:2.1rem; height:2.1rem; font-size:1rem; flex:none; }
.ma-sec h2 { margin:0; font-size:1.18rem; font-weight:700; }
.ma-sec p { margin:.1rem 0 0; color:var(--ma-dim); font-size:.86rem; }
#sec-input, #sec-status, #sec-results { scroll-margin-top:1rem; }

/* ---------- buttons ---------- */
button.ma-run, .ma-run > button {
  background:linear-gradient(135deg,#7c3aed 0%,#2563eb 100%) !important; color:#fff !important;
  border:0 !important; font-weight:700 !important; font-size:1rem !important; letter-spacing:.2px;
  box-shadow:0 8px 24px rgba(99,102,241,.38); transition:filter .15s, transform .15s; }
button.ma-clear, .ma-clear > button {
  background:linear-gradient(135deg,#f43f5e 0%,#f97316 100%) !important; color:#fff !important;
  border:0 !important; font-weight:700 !important; box-shadow:0 8px 22px rgba(244,63,94,.28);
  transition:filter .15s, transform .15s; }
button.ma-run:hover, .ma-run > button:hover, button.ma-clear:hover, .ma-clear > button:hover {
  filter:brightness(1.14); transform:translateY(-1px); }
button.ma-run:active, button.ma-clear:active { transform:translateY(0); filter:brightness(.95); }

/* ---------- tabs: one colour each ---------- */
button[role="tab"] { color:var(--ma-dim) !important; font-weight:600 !important; border-radius:10px 10px 0 0 !important; }
button[role="tab"][aria-selected="true"], .tab-nav button.selected { color:#fff !important;
  border-bottom:3px solid var(--ma-violet) !important;
  background:linear-gradient(180deg,rgba(139,92,246,.16),transparent) !important; }
[role="tablist"] button[role="tab"]:nth-child(2)[aria-selected="true"] { border-bottom-color:var(--ma-amber) !important; background:linear-gradient(180deg,rgba(251,191,36,.14),transparent) !important; }
[role="tablist"] button[role="tab"]:nth-child(3)[aria-selected="true"] { border-bottom-color:var(--ma-green) !important; background:linear-gradient(180deg,rgba(52,211,153,.14),transparent) !important; }
[role="tablist"] button[role="tab"]:nth-child(4)[aria-selected="true"] { border-bottom-color:var(--ma-rose) !important;  background:linear-gradient(180deg,rgba(251,113,133,.14),transparent) !important; }
[role="tablist"] button[role="tab"]:nth-child(5)[aria-selected="true"] { border-bottom-color:var(--ma-cyan) !important;  background:linear-gradient(180deg,rgba(34,211,238,.14),transparent) !important; }

/* ---------- code-like text: left aligned, monospace ---------- */
.ma-mono textarea { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace !important;
  font-size:.86rem !important; line-height:1.55 !important; text-align:left !important; }
.ma-foot { text-align:center; color:var(--ma-dim); font-size:.8rem; padding:.8rem 0 1.2rem; }

/* ---------- results (HTML) ---------- */
.ma-rec { text-align:left; padding:.2rem .1rem 1rem; }
.ma-meta { text-align:center; color:var(--ma-dim); font-size:.84rem; margin:.1rem 0 1rem; }
.ma-stats { display:grid; grid-template-columns:repeat(6,1fr); gap:.7rem; margin:0 0 1.1rem; }
.ma-stat { text-align:center; border:1px solid var(--ma-line); border-radius:14px; padding:.8rem .4rem;
  background:linear-gradient(180deg,#0e0e16,#07070b); }
.ma-stat b { display:block; font-size:1.7rem; line-height:1.1; font-variant-numeric:tabular-nums; }
.ma-stat span { color:var(--ma-dim); font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }
.ma-stat.s1 b { color:#34d399; } .ma-stat.s2 b { color:#fbbf24; } .ma-stat.s3 b { color:#60a5fa; }
.ma-stat.s4 b { color:#fb7185; } .ma-stat.s5 b { color:#a78bfa; } .ma-stat.s6 b { color:#22d3ee; }
.ma-toc { display:flex; flex-wrap:wrap; gap:.45rem; margin:0 0 1.2rem; padding:.5rem; border:1px solid var(--ma-line);
  border-radius:12px; background:#07070b; }
.ma-toc a { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem .28rem .3rem; border-radius:999px;
  color:var(--ma-text) !important; text-decoration:none !important; font-size:.86rem; }
.ma-toc a:hover { background:#14141f; }
.ma-toc a b { display:inline-grid; place-items:center; min-width:1.4rem; height:1.4rem; border-radius:50%;
  background:#1c1c2a; color:#c4b5fd; font-size:.75rem; }
.ma-toc a i { font-style:normal; font-size:.72rem; color:var(--ma-dim); background:#12121c; border-radius:999px; padding:0 .45rem; }
.ma-rec h3 { margin:1.5rem 0 .6rem; padding-bottom:.35rem; border-bottom:1px solid var(--ma-line); font-size:1.08rem;
  display:flex; align-items:center; gap:.55rem; scroll-margin-top:1rem; }
.ma-rec h3 em { font-style:normal; color:#c4b5fd; font-variant-numeric:tabular-nums; }
.ma-rec h4 { margin:.9rem 0 .25rem; font-size:.98rem; color:#e0e7ff; }
.ma-rec h4 em { font-style:normal; color:var(--ma-dim); margin-right:.35rem; font-variant-numeric:tabular-nums; }
.ma-prose { max-width:78ch; line-height:1.7; margin:.2rem 0 .6rem; color:#dcdcea; text-align:left; }
.ma-rows { display:flex; flex-direction:column; gap:.45rem; margin:0; padding:0; list-style:none; }
.ma-rows li { display:grid; grid-template-columns:2rem 1fr auto; gap:.8rem; align-items:start;
  padding:.6rem .8rem; border:1px solid var(--ma-line); border-radius:12px; background:#09090f; }
.ma-rows li .n { text-align:center; font-weight:800; color:#34d399; font-variant-numeric:tabular-nums; }
.ma-rows.prop li .n { color:#fbbf24; }
.ma-rows li .t { text-align:left; line-height:1.5; }
.ma-rows li .r { text-align:right; white-space:nowrap; }
.ma-ref { display:inline-block; min-width:1.7rem; padding:.05rem .5rem; margin-left:.25rem; border-radius:999px;
  background:rgba(139,92,246,.18); color:#c4b5fd; font-size:.78rem; text-align:center; font-variant-numeric:tabular-nums; }
.ma-tablewrap { overflow-x:auto; border:1px solid var(--ma-line); border-radius:12px; }
.ma-table { width:100%; border-collapse:collapse; font-size:.93rem; }
.ma-table th { text-align:left; padding:.6rem .8rem; color:var(--ma-dim); font-size:.72rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.09em; background:#0d0d14; border-bottom:1px solid var(--ma-line); }
.ma-table td { padding:.65rem .8rem; border-bottom:1px solid #15151f; vertical-align:top; text-align:left; line-height:1.5; }
.ma-table tr:last-child td { border-bottom:0; }
.ma-table th.c, .ma-table td.c { text-align:center; }
.ma-table td.num { color:#60a5fa; font-weight:700; font-variant-numeric:tabular-nums; }
.ma-rec table, .ma-rec thead, .ma-rec tbody, .ma-rec tr, .ma-rec th, .ma-rec td { border:0 !important; }
.ma-rec .ma-table th { border-bottom:1px solid var(--ma-line) !important; }
.ma-rec .ma-table td { border-bottom:1px solid #15151f !important; }
.ma-rec .ma-table tr:last-child td { border-bottom:0 !important; }
.ma-dim { color:var(--ma-dim); font-style:italic; }
.ma-badge { display:inline-block; padding:.08rem .55rem; border-radius:999px; font-size:.75rem; font-weight:700;
  background:rgba(251,113,133,.15); color:#fda4af; white-space:nowrap; }
.ma-badge.b { background:rgba(96,165,250,.15); color:#93c5fd; }
.ma-badge.a { background:rgba(251,191,36,.15); color:#fcd34d; }
.ma-empty { text-align:center; color:var(--ma-dim); padding:2.6rem 1rem; border:1px dashed #2a2a3a; border-radius:14px; }
@media (max-width:820px) {
  .ma-stats { grid-template-columns:repeat(3,1fr); }
  .ma-rows li { grid-template-columns:1.6rem 1fr; } .ma-rows li .r { grid-column:2; text-align:left; }
}
"""


def intro():
    (r_name, r_where), (d_name, d_where) = model_names()
    e = html.escape
    steps = [("1 · Transcription", "Whisper 5-best hypotheses", "GPU"),
             ("2 · Refinement", f"{r_name} — fixes domain terms, names, numbers", r_where),
             ("3 · Extraction", f"{d_name} — topics, decisions, action items", d_where),
             ("4 · Verification", f"{d_name} checks every item against the transcript", d_where),
             ("5 · Writing", f"{d_name} — summary + minutes from the verified record", d_where)]
    rows = "".join(f'<tr><td>{e(s)}</td><td>{e(w)}</td><td class="c">{e(p)}</td></tr>'
                   for s, w, p in steps)
    return f"""
<div class="ma-hero">
  <h1>Meeting assistant</h1>
  <p>Upload an English meeting recording (or record one) and press <b>Run</b>. It is transcribed
  (Whisper, 5-best), cleaned up (language model 1) and turned into a summary, minutes, decisions and
  action items (language model 2). Owners and deadlines are filled in only when the recording states
  them; anything that needs a human check is listed under <b>Flags</b>.</p>
  <div class="ma-tablewrap"><table class="ma-table"><thead><tr><th>Step</th><th>What runs</th>
  <th class="c">Where</th></tr></thead><tbody>{rows}</tbody></table></div>
</div>
<nav class="ma-index" aria-label="Index">
  <span class="ma-index-title">Index</span>
  <a class="c1" href="#sec-input"><b>1</b>Upload &amp; settings</a>
  <a class="c2" href="#sec-status"><b>2</b>Run &amp; status</a>
  <a class="c3" href="#sec-results"><b>3</b>Results</a>
</nav>
"""


FOOT = ("A gradio.live link (--share) is public while the app is running, "
        "so share it only with people you trust.")

EMPTY_RECORD = ('<div class="ma-empty">The meeting record appears here once a run finishes.<br>'
                'Start with <b>1 · Upload &amp; settings</b>.</div>')
RUNNING_RECORD = ('<div class="ma-empty">Working on it. The meeting record appears here when the run '
                  'finishes (progress is shown in <b>2 · Run &amp; status</b>).</div>')
EMPTY_FLAGS = '<div class="ma-empty">Items that need a human check are listed here after a run.</div>'


def _sec(n, title, hint, cls):
    return (f'<div class="ma-sec {cls}"><span class="ma-num">{n}</span>'
            f'<div><h2>{html.escape(title)}</h2><p>{html.escape(hint)}</p></div></div>')


# ---------------------------------------------------------------------------------------
# result rendering (all text from the models is escaped)
# ---------------------------------------------------------------------------------------
def _chips(ids):
    ids = sorted({int(i) for i in ids or []})
    return "".join(f'<span class="ma-ref">{i}</span>' for i in ids) or '<span class="ma-dim">–</span>'


def _val(v):
    s = "" if v is None else str(v)
    if s.strip().lower() in ("", "unspecified", "none", "null"):
        return '<span class="ma-dim">unspecified</span>'
    return html.escape(s)


def _flags_table(flags):
    if not flags:
        return '<p class="ma-dim">No flags. Nothing needs a manual check.</p>'
    rows = "".join(
        f'<tr><td class="c"><span class="ma-badge b">{html.escape(str(f.get("stage", "")))}</span></td>'
        f'<td><span class="ma-badge">{html.escape(str(f.get("type", "")))}</span></td>'
        f'<td>{html.escape(str(f.get("item") or ""))}</td>'
        f'<td>{html.escape(str(f.get("reason", "")))}</td></tr>' for f in flags)
    return ('<div class="ma-tablewrap"><table class="ma-table"><thead><tr>'
            '<th class="c">Stage</th><th>Type</th><th>Item</th><th>Reason</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def render_flags_html(flags):
    return f'<div class="ma-rec">{_flags_table(flags)}</div>'


def render_record_html(res):
    r, e = res["record"], html.escape
    minutes, flags = res["minutes"] or [], res["flags"] or []
    sections = [("rec-summary", "Summary", None), ("rec-minutes", "Minutes", len(minutes)),
                ("rec-decisions", "Key decisions", len(r["decisions"])),
                ("rec-proposals", "Proposals, not agreed", len(r["proposals"])),
                ("rec-actions", "Action items", len(r["tasks"])), ("rec-flags", "Flags for review", len(flags))]
    toc = "".join(f'<a href="#{i}"><b>{n}</b>{e(t)}' + (f'<i>{c}</i>' if c is not None else "") + "</a>"
                  for n, (i, t, c) in enumerate(sections, 1))
    stats = [("s1", len(r["decisions"]), "Decisions"), ("s2", len(r["proposals"]), "Proposals"),
             ("s3", len(r["tasks"]), "Action items"), ("s4", len(flags), "Flags"),
             ("s5", len(res["segments"]), "Transcript lines"), ("s6", f'{res["seconds"]:g} s', "Run time")]
    cards = "".join(f'<div class="ma-stat {c}"><b>{e(str(v))}</b><span>{e(l)}</span></div>' for c, v, l in stats)
    m = res["models"]
    meta = (f'{e(res["source"])} · generated {e(res["generated_at"])}<br>'
            f'{e(m["stt"])} → {e(m["refinement"])} → {e(m["documentation"])}')

    def head(n, i, title):
        return f'<h3 id="{i}"><em>{n}</em>{e(title)}</h3>'

    out = [f'<div class="ma-rec"><div class="ma-meta">{meta}</div>',
           f'<div class="ma-stats">{cards}</div><nav class="ma-toc" aria-label="Contents">{toc}</nav>',
           head(1, "rec-summary", "Summary"),
           f'<p class="ma-prose">{e(res["summary"]) if res["summary"] else "<span class=ma-dim>No summary generated.</span>"}</p>',
           head(2, "rec-minutes", "Minutes")]
    for k, mi in enumerate(minutes, 1):
        out.append(f'<h4><em>2.{k}</em>{e(str(mi.get("topic", "")))}</h4>'
                   f'<p class="ma-prose">{e(str(mi.get("text", "")))}</p>')
    if not minutes:
        out.append('<p class="ma-dim">No minutes generated.</p>')

    out.append(head(3, "rec-decisions", "Key decisions"))
    if r["decisions"]:
        out.append('<ol class="ma-rows">' + "".join(
            f'<li><span class="n">{k}</span><span class="t">{e(d["text"])}</span>'
            f'<span class="r">{_chips(d["proposal_lines"] + d["confirmation_lines"])}</span></li>'
            for k, d in enumerate(r["decisions"], 1)) + "</ol>")
    else:
        out.append('<p class="ma-dim">No agreed decisions.</p>')

    out.append(head(4, "rec-proposals", "Proposals discussed, not agreed"))
    if r["proposals"]:
        out.append('<ul class="ma-rows prop">' + "".join(
            f'<li><span class="n">{k}</span><span class="t">{e(p["text"])}</span>'
            f'<span class="r">{_chips(p["proposal_lines"])}</span></li>'
            for k, p in enumerate(r["proposals"], 1)) + "</ul>")
    else:
        out.append('<p class="ma-dim">None.</p>')

    out.append(head(5, "rec-actions", "Action items"))
    if r["tasks"]:
        rows = "".join(
            f'<tr><td class="c num">{k}</td><td>{e(t["text"])}</td><td class="c">{_val(t["owner"])}</td>'
            f'<td class="c">{_val(t["deadline"])}</td><td class="c">{_chips(t["evidence_lines"])}</td></tr>'
            for k, t in enumerate(r["tasks"], 1))
        out.append('<div class="ma-tablewrap"><table class="ma-table"><thead><tr><th class="c">#</th><th>Task</th>'
                   '<th class="c">Owner</th><th class="c">Deadline</th><th class="c">Lines</th></tr></thead>'
                   f'<tbody>{rows}</tbody></table></div>')
    else:
        out.append('<p class="ma-dim">No action items.</p>')

    out += [head(6, "rec-flags", "Flags for review"), _flags_table(flags), "</div>"]
    return "".join(out)


# ------------------------------------------------------------------ helpers

def _default_glossary():
    try:
        with open(GLOSSARY, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _empty(status, running=False):
    """Return the 6-tuple Gradio outputs expect when results are not ready."""
    return status, "", "", RUNNING_RECORD if running else EMPTY_RECORD, EMPTY_FLAGS, None


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
    Yields (status, raw, refined, record_html, flags_html, files) tuples.
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
        yield _empty(f"❌ File not found: {audio_path}. Please verify the path.")
        return

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED:
        yield _empty(
            f"❌ Unsupported file type '{ext or 'none'}'.\n\n"
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
            box["error"] = f"{e.stage}: {e.message}"
        except Exception as e:
            box["error"] = f"Unexpected error ({type(e).__name__}): {e}"
        finally:
            msgs.put(None)  # sentinel — tells the loop to stop

    threading.Thread(target=worker, daemon=True).start()

    log = [f"▶ {os.path.basename(audio_path)}"]
    yield _empty("⏳ Starting pipeline…", running=True)

    # Stream progress messages to the status box
    while True:
        m = msgs.get()
        if m is None:
            break
        log.append(str(m))
        yield _empty("⏳ Processing…\n" + "\n".join(log[-14:]), running=True)

    # --- Error path ---
    if "error" in box:
        yield _empty(
            f"❌ {box['error']}\n\n"
            "Log:\n" + "\n".join(log[-10:])
        )
        return

    # --- Success path ---
    res = box["result"]
    r = res["record"]

    raw_text = "\n".join(res["raw_transcript"])
    refined_text = "\n".join(res["refined_transcript"])

    changed = sum(1 for s in res["segments"] if s.get("changed"))
    n_dec = len(r["decisions"])
    n_prop = len(r["proposals"])
    n_task = len(r["tasks"])
    n_flag = len(res["flags"])

    status_text = (
        f"✅ Done in {res['seconds']:.1f} s · {len(res['segments'])} transcript segments "
        f"({changed} refined) · {n_dec} decisions (agreed) · {n_prop} proposals (not agreed) · "
        f"{n_task} action items · {n_flag} flags for review\n"
        f"Models: {res['models']['stt']} → "
        f"{res['models']['refinement']} → "
        f"{res['models']['documentation']}"
    )

    # ZIP all outputs so user can download everything at once
    zip_path = shutil.make_archive(job + "_all_outputs", "zip", job)
    all_files = list(res["files"].values()) + [zip_path]

    yield (status_text, raw_text, refined_text, render_record_html(res),
           render_flags_html(res["flags"]), all_files)


def clear():
    """Reset inputs and outputs (order matches the `outputs` list of the Clear button)."""
    return ((None, "", None, _default_glossary(), False,
             os.getenv("WHISPER_MODEL", "openai/whisper-small")) + _empty(""))


# ------------------------------------------------------------------ Gradio UI

def _filter(fn, **kw):
    """Keep only the keyword arguments that `fn` accepts (Gradio's API differs between versions)."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in params}


def _opts(component, **kw):
    return _filter(component.__init__, **kw)


def _theme(gr):
    try:
        return gr.themes.Base(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate")
    except Exception:
        return None


def _style_kwargs(gr, fn):
    """theme/css go to Blocks() in Gradio 4/5 and to launch() in Gradio 6: whichever `fn` accepts."""
    given = _filter(fn, css=CSS, theme=_theme(gr))
    return {k: v for k, v in given.items() if v is not None}


def build_app():
    import gradio as gr
    copy = _opts(gr.Textbox, show_copy_button=True) or _opts(gr.Textbox, buttons=["copy"])

    with gr.Blocks(title="Meeting Assistant", **_style_kwargs(gr, gr.Blocks.__init__)) as demo:
        gr.HTML(intro())

        with gr.Row(**_opts(gr.Row, equal_height=False)):

            # ── 1 · Upload & settings ─────────────────────────────────────
            with gr.Column(scale=1, min_width=300, elem_id="sec-input"):
                gr.HTML(_sec(1, "Upload & settings", "Recording, domain terms and options", "c1"))
                audio_file = gr.File(
                    label="Meeting recording (drag & drop or browse)",
                    type="filepath",
                )
                gr.HTML(f'<p class="ma-dim" style="font-size:.8rem;margin:0">'
                        f'Supported: {html.escape(SUPPORTED_FMT)}</p>')
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
                glossary = gr.Textbox(
                    label="Domain glossary (optional, one term per line)",
                    placeholder="Kubernetes\nPostgreSQL\nCI/CD\n…",
                    value=_default_glossary(),
                    lines=6,
                )
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
                with gr.Row():
                    run_btn = gr.Button("▶  Run", variant="primary", elem_classes=["ma-run"])
                    clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["ma-clear"])

            # ── 2 · Run & status ──────────────────────────────────────────
            with gr.Column(scale=2, elem_id="sec-status"):
                gr.HTML(_sec(2, "Run & status", "Live progress of the six pipeline stages", "c2"))
                status = gr.Textbox(
                    label="Status",
                    lines=11,
                    interactive=False,
                    placeholder=("Upload a recording (or use the microphone) and press ▶ Run.\n\n"
                                 "⏱ First run downloads Whisper + Qwen model weights "
                                 "(~7 GB total) and may take a few minutes."),
                    elem_classes=["ma-mono"],
                    **copy,
                )

        # ── 3 · Results ───────────────────────────────────────────────────
        with gr.Column(elem_id="sec-results"):
            gr.HTML(_sec(3, "Results", "Everything is also available as downloads (3.5)", "c3"))
            with gr.Tabs():
                with gr.Tab("3.1  Meeting record"):
                    record = gr.HTML(EMPTY_RECORD)
                with gr.Tab("3.2  Raw transcript"):
                    raw = gr.Textbox(
                        label="Raw transcript (Whisper, top hypothesis)",
                        lines=22,
                        interactive=False,
                        elem_classes=["ma-mono"],
                        **copy,
                    )
                with gr.Tab("3.3  Refined transcript"):
                    refined = gr.Textbox(
                        label="Refined transcript (language model 1)",
                        lines=22,
                        interactive=False,
                        elem_classes=["ma-mono"],
                        **copy,
                    )
                with gr.Tab("3.4  Flags"):
                    flags = gr.HTML(EMPTY_FLAGS)
                with gr.Tab("3.5  Downloads"):
                    files = gr.File(
                        label="Outputs (Markdown, JSON, transcripts, ZIP of everything)",
                        file_count="multiple",
                        interactive=False,
                    )

        gr.HTML(f'<div class="ma-foot">{html.escape(FOOT)}</div>')

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
            **_filter(gr.Button.click, show_progress="minimal"),
        )
        clear_btn.click(
            fn=clear,
            inputs=None,
            outputs=[audio_file, audio_path_input, audio_mic, glossary, diarize, whisper_model,
                     status, raw, refined, record, flags, files],
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

    import gradio as gr
    demo = build_app()
    demo.queue().launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
        show_error=True,
        **_style_kwargs(gr, demo.launch),
    )
