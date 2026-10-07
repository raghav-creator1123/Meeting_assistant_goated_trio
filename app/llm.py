"""
API clients for the two language-model roles.

  Refinement model  -> Groq        (REFINE_*)
  Documentation     -> Gemini      (DOC_*)   used for Stage A, C and D

Both are reached through the `openai` package, because Groq and Gemini expose
OpenAI-compatible endpoints. Settings come from .env (see .env.example).

LLM_BACKEND=local runs both roles on the local GPU instead (Hugging Face transformers):
  refinement     REFINE_LOCAL_MODEL (default Qwen/Qwen2.5-3B-Instruct, 16-bit)
  documentation  DOC_LOCAL_MODEL    (default Qwen/Qwen2.5-7B-Instruct, 4-bit)
Only one model is kept on the GPU at a time (set LOCAL_KEEP_ALL=1 on large GPUs).

Each role can use its own backend: REFINE_BACKEND / DOC_BACKEND ("local" or "api") override
LLM_BACKEND, e.g. DOC_BACKEND=api with LLM_BACKEND=local refines on the GPU and documents
with Gemini.

Rate limits (API backend): calls to one model are spaced at least API_MIN_INTERVAL seconds
apart (default 7, i.e. under 10 requests per minute), and a response format the API rejected
once is not tried again. CALLS counts the requests sent to each model. Short waits that the provider asks for are honoured automatically.
If DOC_FALLBACK_MODEL is set (a Groq model), documentation calls switch to it
when Gemini is rate-limited; the pipeline records when that happened.
"""

import json
import os
import re
import time
from collections import Counter
from types import SimpleNamespace

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULTS = {
    "REFINE_BASE_URL": "https://api.groq.com/openai/v1",
    "REFINE_MODEL": "openai/gpt-oss-20b",   # Groq; replaced automatically if Groq retires it
    "DOC_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "DOC_MODEL": "",          # empty -> pick the newest Gemini Flash model automatically
    "DOC_FALLBACK_MODEL": "auto",  # Groq model used if Gemini is out of quota; "auto" picks one
}
KEY_VARS = {"refine": ["GROQ_API_KEY", "REFINE_API_KEY"],
            "doc": ["GEMINI_API_KEY", "DOC_API_KEY"]}


CALLS = Counter()            # model -> API requests sent (including rejected ones)
_LAST_CALL = {}              # model -> time of the last API request
_BAD_FORMATS = {}            # model -> response_format types the API rejected


def backend(role):
    """'local' or 'api' for role 'refine' or 'doc'."""
    prefix = "REFINE" if role == "refine" else "DOC"
    return (os.getenv(f"{prefix}_BACKEND") or os.getenv("LLM_BACKEND") or "api").lower()


class LLMError(RuntimeError):
    """An error with a message that can be shown to the user as is."""


class RequestTooLarge(LLMError):
    """The prompt exceeds the model's per-request / per-minute token limit (free tiers)."""


class _LocalTooLarge(Exception):
    """Raised by the local backend when a prompt is too long or the GPU runs out of memory."""


# ------------------------------------------------------------------ local GPU backend
LOCAL_DEFAULTS = {"refine": ("Qwen/Qwen2.5-3B-Instruct", "16bit"),
                  "doc": ("Qwen/Qwen2.5-7B-Instruct", "4bit")}
_LOADED = {}                      # model_id -> (tokenizer, model)


def unload_local(keep=None):
    """Free GPU memory used by local models (all except `keep`)."""
    import gc
    for mid in [m for m in _LOADED if m != keep]:
        del _LOADED[mid]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _check_cpu_memory(model_id):
    """CPU models load in float32 (4 bytes per weight). If that cannot fit, the weight loader
    aborts the whole process natively, so fail early with a readable message instead."""
    try:
        import psutil
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
        with init_empty_weights():
            empty = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id))
        need = sum(p.numel() for p in empty.parameters()) * 4 * 1.1
        free = psutil.virtual_memory().available
    except Exception:
        return                                # cannot estimate: try loading anyway
    if need > free:
        raise LLMError(f"{model_id} needs about {need / 1e9:.0f} GB of RAM on the CPU, but only "
                       f"{free / 1e9:.1f} GB is free. Install CUDA PyTorch to use the GPU "
                       f"(see run_app.bat), choose a smaller model, or use --api.")


def _load_local(model_id, quant):
    if model_id in _LOADED:
        return _LOADED[model_id]
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.is_available():
        device = "cuda"
    elif os.getenv("LOCAL_ALLOW_CPU") == "1":
        device = "cpu"
    else:
        raise LLMError("No GPU found. In Colab: Runtime -> Change runtime type -> T4 GPU, "
                       "then restart and run the setup cells again.")
    if os.getenv("LOCAL_KEEP_ALL") != "1":
        unload_local()                        # one model on the GPU at a time
    kwargs = {"device_map": device}
    dtype = torch.float16 if device == "cuda" else torch.float32
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
    if quant == "4bit" and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    if device == "cpu":
        _check_cpu_memory(model_id)
    print(f"[llm] loading {model_id} ({quant}, {device}); the first run downloads the weights...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        try:      # memory-efficient attention: long prompts need far less GPU memory
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa",
                                                         **kwargs)
        except (ValueError, ImportError, TypeError):
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        unload_local()
        raise LLMError(f"Not enough GPU memory to load {model_id} ({quant}). "
                       f"Set the *_LOCAL_QUANT values in .env to 4bit or use a smaller model.") from e
    model.eval()
    print(f"[llm] {model_id} ready in {time.time() - t0:.0f}s")
    _LOADED[model_id] = (tok, model)
    return tok, model


class LocalChatClient:
    """Looks like an OpenAI client, but generates with a local transformers model."""

    def __init__(self, model_id, quant="4bit", max_input=24000):
        self.model_id, self.quant, self.max_input = model_id, quant, max_input
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=model_id)])

    def _create(self, model=None, messages=None, temperature=0.0, max_tokens=None,
                response_format=None, **_):
        import torch
        tok, mdl = _load_local(self.model_id, self.quant)
        msgs = [dict(m) for m in messages]
        if response_format:                   # no constrained decoding: ask clearly instead
            msgs[-1]["content"] += "\n\nRespond with the JSON object only, no other text."
        tmpl = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = tok.apply_chat_template(msgs, enable_thinking=False, **tmpl)  # Qwen3: no thinking
        except TypeError:
            text = tok.apply_chat_template(msgs, **tmpl)
        enc = tok(text, return_tensors="pt").to(mdl.device)
        n_in = enc["input_ids"].shape[1]
        if n_in > self.max_input:
            raise _LocalTooLarge(f"prompt has {n_in} tokens; the limit is {self.max_input}")
        gen = {"max_new_tokens": max_tokens or 2048, "do_sample": bool(temperature),
               "pad_token_id": tok.pad_token_id or tok.eos_token_id}
        if temperature:
            gen.update(temperature=temperature, top_p=0.9)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                out = mdl.generate(**enc, **gen)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise _LocalTooLarge(f"GPU ran out of memory on a {n_in}-token prompt") from e
        reply = tok.decode(out[0, n_in:], skip_special_tokens=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def _env(name):
    return os.getenv(name) or DEFAULTS.get(name, "")


def make_client(role, fallback=True):
    """role = 'refine' or 'doc'. Returns (client, model_name).

    With fallback=True the client is a ModelChain: refinement moves to other available
    Groq models when one runs out of daily tokens; documentation moves from Gemini to
    DOC_FALLBACK_MODEL (a Groq model) if that is set. Use fallback=False when evaluating
    a single model.
    """
    prefix = "REFINE" if role == "refine" else "DOC"
    if backend(role) == "local":
        mid = os.getenv(f"{prefix}_LOCAL_MODEL") or LOCAL_DEFAULTS[role][0]
        quant = os.getenv(f"{prefix}_LOCAL_QUANT") or LOCAL_DEFAULTS[role][1]
        return LocalChatClient(mid, quant, int(os.getenv("LOCAL_MAX_INPUT", "24000"))), mid
    from openai import OpenAI
    key = next((os.getenv(v) for v in KEY_VARS[role] if os.getenv(v)), None)
    if not key:
        names = " or ".join(KEY_VARS[role])
        raise LLMError(f"Missing API key for the {'refinement' if role == 'refine' else 'documentation'} "
                       f"model. Set {names} in your .env file.")
    # no hidden SDK retries: every request is counted, paced and retried in create_with_wait
    client = OpenAI(api_key=key, base_url=_env(f"{prefix}_BASE_URL"), max_retries=0, timeout=180)
    wanted = _env(f"{prefix}_MODEL")
    if role == "refine":
        model = pick_groq_model(client, wanted)
        if fallback:   # other Groq models have their own daily token budgets
            backups = [m for m in groq_candidates(client) if m != model][:3]
            client = ModelChain([(client, model)] + [(client, m) for m in backups])
    else:
        model = wanted or pick_flash_model(client)
        if not fallback:
            return client, model
        # two other Gemini Flash models, used when this one is overloaded or out of quota
        entries = [(client, model)] + [(client, m) for m in flash_models(client)
                                       if m != model][:2]
        fb_model = _env("DOC_FALLBACK_MODEL").strip()
        has_groq = any(os.getenv(v) for v in KEY_VARS["refine"])
        if fb_model and fb_model.lower() != "none" and has_groq:
            groq_client, refine_model = make_client("refine", fallback=False)
            options = groq_candidates(groq_client)
            # the named model first (if Groq has it), then other Groq models not used for
            # refinement; "auto", a typo or a retired name just falls through to these
            backups = ([fb_model] if fb_model in options else []) + \
                [m for m in options if m not in (fb_model, refine_model)]
            entries += [(groq_client, m) for m in backups[:3]]
        if len(entries) > 1:
            client = ModelChain(entries)
    return client, model


class ModelChain:
    """Looks like an OpenAI client. Calls go to the first (client, model) entry; when that
    model hits a daily quota (or must wait longer than `max_wait`), the chain moves on to the
    next entry and stays there. `used` lists every model that answered at least once."""

    def __init__(self, entries, max_wait=90):
        self.entries, self.pos, self.max_wait = entries, 0, max_wait
        self.used, self.switches = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = entries[0][0].models

    @property
    def model(self):
        return self.entries[self.pos][1]

    def _create(self, **kw):
        from openai import APIStatusError, RateLimitError
        while True:
            client, model = self.entries[self.pos]
            _pace(model)
            try:
                resp = client.chat.completions.create(**{**kw, "model": model})
                if model not in self.used:
                    self.used.append(model)
                return resp
            except RateLimitError as e:
                delay = _retry_delay(e)
                long_wait = _is_daily(e) or (delay is not None and delay > self.max_wait)
                if not long_wait or self.pos + 1 >= len(self.entries):
                    raise                     # short wait -> create_with_wait retries
                why, msg = "is out of quota", _short(e, 160)
            except APIStatusError as e:
                if not _overloaded(e) or self.pos + 1 >= len(self.entries):
                    raise                     # last model overloaded -> create_with_wait waits
                why, msg = "is overloaded", _short(e, 160)
            nxt = self.entries[self.pos + 1][1]
            self.switches.append((model, nxt, msg))
            print(f"[llm] {model} {why}; switching to {nxt}.")
            self.pos += 1


# preferred Groq chat models, best first (substring match); retired ones are simply skipped
GROQ_PREFERENCE = ["llama-3.3-70b", "gpt-oss-120b", "llama-4-maverick", "kimi-k2", "qwen3-32b",
                   "llama-4-scout", "qwen", "gpt-oss-20b", "llama-3.1-8b"]
GROQ_SKIP = ("whisper", "guard", "tts", "playai", "distil", "prompt-guard", "orpheus", "compound")


def groq_candidates(client):
    """Available Groq chat models, in preference order."""
    try:
        names = [m.id for m in client.models.list()]
    except Exception:
        return []
    chat = [n for n in names if not any(x in n.lower() for x in GROQ_SKIP)]
    ordered = [n for pref in GROQ_PREFERENCE for n in chat if pref in n]
    return list(dict.fromkeys(ordered + sorted(chat)))


def pick_groq_model(client, wanted=""):
    """Use `wanted` if Groq still serves it; otherwise the best available chat model."""
    try:
        names = [m.id for m in client.models.list()]
    except Exception:
        return wanted or GROQ_PREFERENCE[0]       # can't list: let the call report the problem
    if wanted and wanted in names:
        return wanted
    chat = [n for n in names if not any(s in n.lower() for s in GROQ_SKIP)]
    for pref in GROQ_PREFERENCE:
        for n in chat:
            if pref in n:
                if wanted:
                    print(f"[llm] Groq model '{wanted}' is not available; using '{n}' instead.")
                return n
    if chat:
        return sorted(chat)[0]
    raise LLMError("No Groq chat model is available for this key. Set REFINE_MODEL in .env.")


def flash_models(client):
    """General-purpose Gemini Flash models the key can use, newest first."""
    try:
        names = [m.id.replace("models/", "") for m in client.models.list()]
    except Exception as e:
        raise LLMError(f"Could not list Gemini models ({e}). Set DOC_MODEL in .env.") from e
    skip = ("lite", "image", "tts", "live", "audio", "thinking", "embedding", "preview-tts",
            "omni", "latest")
    flash = [n for n in names if "gemini" in n and "flash" in n and not any(s in n for s in skip)]

    def version(n):
        nums = re.findall(r"\d+(?:\.\d+)?", n)
        return (float(nums[0]) if nums else 0, "preview" not in n and "exp" not in n, n)
    return sorted(flash, key=version, reverse=True)


def pick_flash_model(client):
    """Choose the newest general-purpose Gemini Flash model the key can use."""
    flash = flash_models(client)
    if not flash:
        raise LLMError("No Gemini Flash model found for this key. Set DOC_MODEL in .env.")
    return flash[0]


def _retry_delay(err):
    """Seconds the provider asks us to wait (Gemini 'retryDelay': '37s', Groq 'try again in 3m31.6s')."""
    t = str(err)
    m = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?", t, re.I)
    if m and any(m.groups()):
        h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + sec
    m = re.search(r"retry(?:[_ ]?delay|[_ ]?after)?[\"':\s]*(?:in\s*)?(\d+(?:\.\d+)?)\s*s", t, re.I)
    return float(m.group(1)) if m else None


def _is_daily(err):
    t = str(err).lower()
    return "perday" in t.replace(" ", "") or "per day" in t or "daily" in t


def _short(err, n=300):
    return " ".join(str(err).split())[:n]


def _pace(model):
    """Space API requests to one model API_MIN_INTERVAL seconds apart, and count them."""
    gap = float(os.getenv("API_MIN_INTERVAL", "7"))
    wait = _LAST_CALL.get(model, 0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[model] = time.time()
    CALLS[model] += 1


def _overloaded(err):
    """Server-side trouble (e.g. Gemini 503 'high demand'); not the caller's fault or quota."""
    return getattr(err, "status_code", None) in (500, 502, 503, 504)


def create_with_wait(client, max_wait=90, tries=4, **kwargs):
    """chat.completions.create that waits out per-minute rate limits and server overload."""
    from openai import APIStatusError, RateLimitError
    for attempt in range(tries):
        if not isinstance(client, (LocalChatClient, ModelChain)):   # a ModelChain paces itself
            _pace(kwargs.get("model"))
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            delay = _retry_delay(e)
            if _is_daily(e) or attempt == tries - 1 or (delay and delay > max_wait):
                raise
            time.sleep(min(delay or 20 * (attempt + 1), max_wait))
        except APIStatusError as e:
            if not _overloaded(e) or attempt == tries - 1:
                raise
            wait = 10 * 2 ** attempt                    # 10, 20, 40 s
            print(f"[llm] the API is overloaded ({e.status_code}); retrying in {wait} s...")
            time.sleep(wait)


def chat_json(client, model, messages, schema=None, temperature=0.0, max_tokens=None):
    """One chat call that must return JSON. Tries json_schema, then json_object, then plain."""
    from openai import APIStatusError, APIConnectionError, RateLimitError, AuthenticationError
    kwargs = dict(model=model, temperature=temperature, messages=messages)
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    formats = ([{"type": "json_schema", "json_schema": schema}] if schema else []) + \
              [{"type": "json_object"}, None]
    if isinstance(client, LocalChatClient):   # local: one JSON request, then one repair try
        formats = [{"type": "json_object"}, {"type": "json_object"}]
    else:                                     # skip formats this model already rejected
        bad = _BAD_FORMATS.get(model, set())
        formats = [f for f in formats if (f or {}).get("type") not in bad] or [None]
    last, bad_reply = None, None
    for fmt in formats:
        try:
            if bad_reply is not None:         # previous answer was not valid JSON: say so
                kwargs["messages"] = list(messages) + [
                    {"role": "assistant", "content": bad_reply[:4000]},
                    {"role": "user", "content": "That was not valid JSON. Reply again with only "
                                                "the corrected JSON object."}]
            resp = create_with_wait(client, **kwargs,
                                    **({"response_format": fmt} if fmt else {}))
            content = resp.choices[0].message.content or ""
            try:
                return parse_json(content)
            except (json.JSONDecodeError, ValueError):
                bad_reply = content
                raise
        except _LocalTooLarge as e:
            raise RequestTooLarge(f"Request too large for {model} on this GPU ({e}).") from e
        except RateLimitError as e:
            kind = "daily" if _is_daily(e) else "per-minute"
            raise LLMError(f"Rate limit reached for {model} ({kind} quota). "
                           f"{'It resets tomorrow; ' if kind == 'daily' else 'Wait a minute and '}"
                           f"try again. Provider says: {_short(e)}") from e
        except AuthenticationError as e:
            raise LLMError(f"The API key for {model} was rejected. Check it in .env.") from e
        except APIConnectionError as e:
            raise LLMError(f"Could not reach the API for {model}. Check your internet "
                           f"connection.") from e
        except APIStatusError as e:
            if _overloaded(e):
                raise LLMError(f"{model} is overloaded (HTTP {e.status_code}), and so were the "
                               f"backup models after several retries. This is on the provider's "
                               f"side and does not use your quota; try again in a few minutes.") from e
            if e.status_code == 413 or "request too large" in str(e).lower() or \
                    "context_length" in str(e).lower():
                raise RequestTooLarge(f"Request too large for {getattr(client, 'model', model)} "
                                      f"(free-tier token limit). Provider says: {_short(e)}") from e
            if e.status_code == 404 or "decommission" in str(e).lower():
                raise LLMError(f"Model '{model}' is not available (retired or misspelled). Set a "
                               f"current model name in .env. Provider says: {_short(e)}") from e
            last = e            # e.g. response_format not supported -> try the next format
            if fmt and e.status_code == 400:
                _BAD_FORMATS.setdefault(model, set()).add(fmt["type"])
        except (json.JSONDecodeError, ValueError) as e:
            last = e
    raise LLMError(f"{model} did not return valid JSON ({last}).")


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)       # JSON wrapped in prose
        if m:
            return json.loads(m.group(0))
        raise
