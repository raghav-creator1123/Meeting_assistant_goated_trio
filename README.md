[README.md](https://github.com/user-attachments/files/33141394/README.md)
# Meeting Assistant

Turns a meeting recording into a transcript and a structured meeting record: a summary, minutes per topic, agreed decisions, proposals that were not agreed, and action items with owners and deadlines. Every item cites the transcript lines it came from, and anything the pipeline could not confirm is listed as a flag for review.

Everything runs locally on a GPU with open models (Whisper + Qwen2.5). No API keys are needed. A hosted-API backend (Groq + Gemini) is optional.

## Quick start

Run all commands from the repository root (the folder with `.env` and `requirements.txt`). The code lives in `app/`.

**Windows**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
copy .env.example .env
python app\setup.py
python app\app.py
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cp .env.example .env
python app/setup.py
python app/app.py
```

Then open http://localhost:7860 in a browser, upload or record a meeting, and press **Run**. On Windows, after the first-time setup you can start the app by double-clicking `run_app.bat`.

If your GPU has less than 15 GB of memory, edit `.env` before the first run (see [Requirements](#requirements)). If you have no GPU, see the API backend under [Optional features](#optional-features).

## How it works

```
Audio
 └─► 1. Whisper ─────────────── 5-best hypotheses per segment
      └─► 2. Refinement (LLM 1) ── corrects mis-heard words, names, terms; code guards protect
           │                        numbers and negation
           └─► 3. Stage A (LLM 2) ── extracts topics, decisions, tasks, each citing line numbers
                └─► 4. Stage B (code) ── drops uncited items; "agreed" needs a confirmation line;
                     │                   owners/deadlines must be stated on the cited lines
                     └─► 5. Stage D (LLM 2) ── re-checks each item against its cited lines
                          └─► 6. Stage C (LLM 2) ── writes summary + minutes from the verified record
                               └─► Markdown + JSON record, transcripts, flags
```

Only one language model is on the GPU at a time: Whisper runs and is unloaded, then the refinement model, then the documentation model.

| Role | Default model | Precision |
|---|---|---|
| Speech-to-text | `openai/whisper-small` | fp16 |
| LLM 1: refinement | `Qwen/Qwen2.5-3B-Instruct` | 16-bit |
| LLM 2: Stages A, D, C | `Qwen/Qwen2.5-7B-Instruct` | 4-bit (bitsandbytes) |

No model is fine-tuned. Behaviour comes from the prompts and the code checks. The exact prompts and model instructions are in **[docs/PROMPTS.md](docs/PROMPTS.md)**.

## Outputs

Each run writes a folder under `outputs/app_runs/<timestamp>/` (or `--out` on the command line):

| File | Contents |
|---|---|
| `raw_transcript.txt` | Whisper's top hypothesis per segment |
| `refined_transcript.txt` | After LLM correction, numbered `[id] SPEAKER: text` |
| `meeting_record.md` | Summary · minutes · decisions · proposals · action items · flags |
| `meeting_record.json` | The same record, machine-readable |
| `nbest_segments.json` | All Whisper hypotheses and timings per segment |
| `<timestamp>_all_outputs.zip` | Everything above, next to the run folder (app only) |

## Requirements

**Hardware.** An NVIDIA GPU with CUDA. Pick the model profile that fits your GPU memory (set in `.env`):

| GPU memory | Refinement | Documentation | Notes |
|---|---|---|---|
| 15 GB or more (Colab T4, RTX 3090, A100) | Qwen2.5-3B, 16-bit | Qwen2.5-7B, 4-bit | Default; best quality |
| 8-12 GB | Qwen2.5-3B, 4-bit | Qwen2.5-7B, 4-bit | |
| 6 GB (RTX 4050/4060 laptop) | Qwen2.5-3B, 4-bit | Qwen2.5-3B, 4-bit | 7B does not fit; tested setup |
| No GPU | any | any | `LOCAL_ALLOW_CPU=1`, very slow; or use the API backend |

Disk for model weights in the Hugging Face cache: about 8 GB for the 3B-only profile, about 25 GB with Qwen2.5-7B (its full weights are downloaded, then loaded in 4-bit). Set `HF_HOME` to move the cache.

**Software.**

- Python 3.10 or 3.11 (tested on 3.11.9)
- NVIDIA driver with CUDA 12.x and a CUDA build of PyTorch
- `ffmpeg` on the PATH, for mp3, m4a, mp4 and other non-WAV audio
- Python packages in [`requirements.txt`](requirements.txt); tested versions are listed at its top

## Setup

```bash
# 1. Get the code
git clone <this repository> meeting_assistant
cd meeting_assistant

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux / macOS

# 3. PyTorch with CUDA first (pick the index matching your driver: cu121, cu124, cu126 ...)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 4. Everything else
pip install -r requirements.txt

# 5. Configuration
copy .env.example .env              # Windows
# cp .env.example .env              # Linux / macOS
#    then edit .env: choose the model profile for your GPU (see the table above)

# 6. One-time setup check, then a quick smoke test
python app/setup.py
python app/check.py
```

Install ffmpeg if it is missing: `winget install Gyan.FFmpeg` (Windows), `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS).

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The first run downloads Whisper and Qwen weights from Hugging Face (several GB). Later runs start from the cache.

## Run

**Web app**

```bash
python app/app.py             # http://localhost:7860
python app/app.py --share     # also a public gradio.live link
python app/app.py --port 8080
python app/app.py --api       # use the Groq + Gemini backend instead of local models
```

On Windows you can also double-click `run_app.bat`, or run `run_app.bat --share` from a terminal; it uses `.venv` if present and accepts the same options as `app/app.py`. Stop the app with Ctrl+C in the terminal. In the app: upload or record audio, edit the glossary if needed, tick *Speaker labels* if configured, press **Run**, follow the live status, then view and download the outputs.

**Command line**

```bash
python app/pipeline.py meeting.mp3 --out outputs/my_meeting
python app/pipeline.py meeting.mp3 --out outputs/my_meeting --glossary prompts/glossary.txt --diarize
python app/pipeline.py --segments nbest.json --out outputs/test   # skip speech-to-text
```

`--segments` takes a JSON list of `{"id": 1, "hypotheses": ["...", ...], "speaker": "SPEAKER_00"}`.

## Configuration

Runtime settings live in `.env` (see [`.env.example`](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `local` in `.env.example` | `local` (GPU) or `api` (Groq + Gemini) |
| `REFINE_LOCAL_MODEL` / `REFINE_LOCAL_QUANT` | `Qwen/Qwen2.5-3B-Instruct` / `16bit` | Refinement model and precision (`16bit` or `4bit`) |
| `DOC_LOCAL_MODEL` / `DOC_LOCAL_QUANT` | `Qwen/Qwen2.5-7B-Instruct` / `4bit` | Documentation model and precision |
| `LOCAL_MAX_INPUT` | `24000` | Longest prompt in tokens; longer requests use smaller prompts |
| `LOCAL_ALLOW_CPU` | unset | `1` lets local models run on the CPU |
| `LOCAL_KEEP_ALL` | unset | `1` keeps both models on the GPU (large GPUs) |
| `STAGEA_CHUNK_LINES` | `100` (local) | Transcript lines per extraction call |
| `WHISPER_MODEL` | `openai/whisper-small` | Any Hugging Face Whisper model |
| `HF_HOME` | Hugging Face default | Where model weights are cached |
| `MEETING_TZ` | system time | Timezone for timestamps in the record |
| `HF_TOKEN` | unset | For speaker labels (pyannote) |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | unset | API backend only |

Prompt settings live in `prompts/`:

| File | What it sets |
|---|---|
| `prompts/glossary.txt` | Domain terms given to the refinement model, one per line (also editable in the app) |
| `prompts/refine_config.json` | Refinement batch size (20 lines) and whether the worked example is included |
| `prompts/stageA_config.json` | Which few-shot examples Stage A uses (`ami+synthetic`) |

## Prompts and model instructions

[docs/PROMPTS.md](docs/PROMPTS.md) lists every system prompt, few-shot example, user-turn format and output schema exactly as sent. It is generated from the code; after changing a prompt, regenerate it:

```bash
python tools/export_prompts.py
```

In short:

- **Refinement**: rules that allow fixing only mis-heard words (terms, acronyms, names) and forbid changing numbers, negations, names or commitments; one hand-written worked example; the glossary and the last 3 corrected lines as context; 20 lines per call; temperature 0.
- **Stage A**: rules for what counts as an agreed decision, a proposal and a task (never guess an owner); 3 few-shot excerpts from the AMI Meeting Corpus plus 1 hand-written example of a parked proposal and an owner-less task; the `Transcript: … + instruction` format; temperature 0.
- **Stage D**: checks each item against only the lines it cites; temperature 0.
- **Stage C**: writes summary and minutes from the verified record only; temperature 0.2.

## Data

`stageA/stageA_fewshot.json` holds the three AMI excerpts used in the Stage A prompt and is committed, so the AMI corpus is **not** needed to run the app.

To rebuild the examples you need the AMI annotations as JSON (`ami_json_for_stageA.zip`, made with the [ami-and-icsi-corpora](https://github.com/guokan-shang/ami-and-icsi-corpora) converter scripts: `dialogueActs`, `summlink`, `adjacencyPairs`, `topics`). Put the zip in this folder or its parent and run `python setup.py`, or call the builder directly:

```bash
python build_stageA_prompts_from_ami.py --ami-json path/to/ami_json --out-dir stageA
```

AMI's official test meetings (ES2004, ES2014, IS1009, TS3003, TS3007, EN2002) are never used as examples; they are written to `stageA/stageA_eval.jsonl`.

## Repository layout

```
meeting_assistant/
├── app.py                      Gradio web app (start here)
├── pipeline.py                 End-to-end orchestrator and CLI
├── asr.py                      Whisper 5-best transcription, audio checks, speaker labels
├── refine.py                   Refinement prompt, batching and meaning guards
├── stageA_prompt.py            Extraction prompt (Stage A) and evidence rules (Stage B)
├── document.py                 Verification (Stage D), writing (Stage C), Markdown/JSON output
├── llm.py                      Local GPU and API model clients, JSON handling
├── setup.py                    One-time setup (AMI examples, config files)
├── check.py                    Smoke test, no models loaded
├── build_stageA_prompts_from_ami.py   Builds the Stage A few-shot examples from AMI
├── eval_utils.py               Optional evaluation helpers (WER, extraction metrics)
├── make_hp_fewshot_prompts.py  Optional: builds a HyPoradise test set for the refinement prompt
├── tools/export_prompts.py     Regenerates docs/PROMPTS.md
├── docs/PROMPTS.md             All prompts and model instructions
├── prompts/                    Glossary and prompt settings
├── stageA/                     AMI few-shot examples, held-out meetings, assembled example prompt
├── requirements.txt
├── .env.example
└── run_app.bat                 Windows launcher
```

`outputs/`, `data/` and `.env` are not committed.

## Optional features

**Speaker labels.** Accept the terms of [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) on Hugging Face, `pip install pyannote.audio`, set `HF_TOKEN=hf_...` in `.env`, then tick *Speaker labels* in the app (or pass `--diarize`).

**API backend.** Set `LLM_BACKEND=api` (or run `python app/app.py --api`) and add `GROQ_API_KEY` and `GEMINI_API_KEY` to `.env`. Refinement then runs on Groq, documentation on Gemini; no GPU is needed for the language models.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No GPU found` | PyTorch is a CPU build: reinstall with the CUDA index URL (Setup step 3) |
| GPU out of memory, or Stage A very slow | Use a smaller profile in `.env` (Qwen2.5-3B, 4-bit), or lower `STAGEA_CHUNK_LINES` |
| `smaller_prompt_used` flag in the record | The prompt did not fit; the run continued with fewer examples or shorter excerpts |
| Audio file rejected | Install ffmpeg, or convert to WAV |
| Model download fills the C: drive | Set `HF_HOME` to a folder on another drive |

## Acknowledgements

- AMI Meeting Corpus (CC BY 4.0), for the Stage A few-shot examples
- HyPoradise (Chen et al., NeurIPS 2023), the N-best correction method the refinement step follows
- Rennard et al., *Abstractive Meeting Summarization: A Survey* (arXiv 2208.04163), the Stage A prompt format
- Whisper (OpenAI) and Qwen2.5 (Alibaba Cloud); check each model's Hugging Face card for its licence
