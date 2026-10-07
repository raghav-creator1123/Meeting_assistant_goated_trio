"""
Speech-to-text: Whisper with 5-best hypotheses per segment.

1. Whisper (Hugging Face transformers) transcribes the whole file and returns
   time-stamped segments.
2. Each segment is decoded again with beam search (num_beams = n_best,
   num_return_sequences = n_best) to get its N-best list, as in HyPoradise.
3. Optional speaker labels with pyannote (gated on Hugging Face: accept the
   pyannote/speaker-diarization-3.1 terms and set HF_TOKEN).

Returns a list of segments:
  {"id": 1, "start": 0.0, "end": 4.2, "speaker": "SPEAKER_00" | None,
   "hypotheses": [...up to 5...], "score": -0.21, "flags": [...]}
"""

import os

SUPPORTED = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac", ".wma")
SR = 16000


class AudioError(ValueError):
    """Problem with the input file; the message is shown to the user."""


def load_audio(path):
    if not path or not os.path.exists(path):
        raise AudioError("No audio file was found. Please upload a meeting recording.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise AudioError(f"Unsupported file type '{ext or 'none'}'. "
                         f"Please upload one of: {', '.join(SUPPORTED)}.")
    if os.path.getsize(path) == 0:
        raise AudioError("The uploaded file is empty (0 bytes).")
    try:
        import librosa
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        raise AudioError(f"The file could not be read as audio ({type(e).__name__}). It may be "
                         f"corrupted, or ffmpeg may be missing for this format.") from e
    import numpy as np
    if audio.size < SR:
        raise AudioError("The recording is shorter than one second.")
    if float(np.sqrt(np.mean(audio ** 2))) < 1e-4:
        raise AudioError("The recording appears to be silent.")
    return audio


class Transcriber:
    def __init__(self, model_name=None, n_best=5, device=None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
        self.model_name = model_name or os.getenv("WHISPER_MODEL", "openai/whisper-small")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_best = min(n_best, 2) if self.device == "cpu" else n_best
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, dtype=self.dtype).to(self.device)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model,
                             tokenizer=self.processor.tokenizer,
                             feature_extractor=self.processor.feature_extractor,
                             chunk_length_s=30, device=self.device)

    def nbest(self, clip):
        import torch
        feats = self.processor(clip, sampling_rate=SR, return_tensors="pt").input_features
        with torch.no_grad():
            out = self.model.generate(feats.to(self.device, self.dtype), num_beams=self.n_best,
                                      num_return_sequences=self.n_best, max_new_tokens=220,
                                      output_scores=True, return_dict_in_generate=True,
                                      language="en", task="transcribe")
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.tolist() if out.sequences_scores is not None else [0] * len(texts)
        seen, hyps, best = set(), [], scores[0] if scores else 0.0
        for t in texts:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                hyps.append(t)
        return hyps, best

    def transcribe(self, path, progress=None, audio=None):
        import numpy as np
        audio = load_audio(path) if audio is None else audio
        result = self.pipe(audio.copy(), return_timestamps=True,
                           generate_kwargs={"language": "en", "task": "transcribe"})
        chunks = result.get("chunks") or [{"timestamp": (0.0, None), "text": result.get("text", "")}]
        segments = []
        for i, ch in enumerate(chunks):
            start, end = ch["timestamp"]
            start = float(start or 0.0)
            end = float(end) if end is not None else len(audio) / SR
            clip = audio[int(start * SR):int(end * SR)]
            if clip.size < SR // 4:                       # under 0.25 s: keep 1-best
                hyps, score = [ch["text"].strip()], 0.0
            else:
                hyps, score = self.nbest(clip[:30 * SR])
                if not hyps:
                    hyps = [ch["text"].strip()]
            flags = []
            if score < -1.0:
                flags.append("low_asr_confidence")
            if clip.size and float(np.sqrt(np.mean(clip ** 2))) < 5e-4:
                flags.append("near_silent_segment")      # Whisper can invent text here
            hyps = [h for h in hyps if h]
            if hyps:
                segments.append({"id": len(segments) + 1, "start": round(start, 2),
                                 "end": round(end, 2), "speaker": None,
                                 "hypotheses": hyps[:self.n_best],
                                 "score": round(score, 3), "flags": flags})
            if progress:
                progress(f"Transcribing segment {i + 1}/{len(chunks)}")
        if not segments:
            raise AudioError("No speech was detected in the recording.")
        return segments


def add_speakers(path, segments):
    """Optional diarization with pyannote. Leaves speakers as None if unavailable."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return segments, "Speaker labels skipped (HF_TOKEN not set)."
    try:
        from pyannote.audio import Pipeline
        diar = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        turns = [(t.start, t.end, spk) for t, _, spk in diar(path).itertracks(yield_label=True)]
    except Exception as e:
        return segments, f"Speaker labels skipped ({type(e).__name__}: {e})."
    names = {}
    for seg in segments:
        overlap = {}
        for s, e, spk in turns:
            o = min(e, seg["end"]) - max(s, seg["start"])
            if o > 0:
                overlap[spk] = overlap.get(spk, 0) + o
        if overlap:
            spk = max(overlap, key=overlap.get)
            seg["speaker"] = names.setdefault(spk, f"SPEAKER_{len(names):02d}")
    return segments, f"{len(names)} speakers detected."
