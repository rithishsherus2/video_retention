"""Audio-track signals: short-term loudness curve, silence/dead-air
detection, and a word-timestamped transcript via faster-whisper (local,
free, no API -- runs on CPU)."""
from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass

import numpy as np
import pandas as pd


def extract_audio_wav(video_path: str, out_path: str | None = None, sr: int = 16000) -> str:
    """Mono 16kHz WAV -- what faster-whisper expects, and plenty for the
    loudness analysis too."""
    if out_path is None:
        out_path = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")
    return out_path


def loudness_timeline(wav_path: str, window_s: float = 0.5, silence_db: float = -40.0) -> pd.DataFrame:
    """Short-term RMS loudness (dBFS) in fixed windows, plus a silence
    flag. This is an RMS proxy, not true LUFS -- good enough at this scale
    for spotting abrupt level jumps and dead air, which is what the
    'audio smooth / redundant dialogue' rule cares about."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    win = max(1, int(window_s * sr))
    rows = []
    for start in range(0, len(audio), win):
        chunk = audio[start:start + win]
        if len(chunk) == 0:
            continue
        rms = np.sqrt(np.mean(chunk ** 2)) + 1e-9
        db = 20 * np.log10(rms)
        rows.append({"t": start / sr, "rms_db": float(db), "is_silent": bool(db < silence_db)})
    return pd.DataFrame(rows)


@dataclass
class TranscriptWord:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    """A natural speech unit (Whisper's own segmentation, roughly a
    sentence/phrase) -- the right granularity for 'was this said again
    elsewhere', as opposed to individual words."""
    text: str
    start: float
    end: float


@dataclass
class TranscriptResult:
    words: list[TranscriptWord]
    segments: list[TranscriptSegment]
    language: str | None
    language_probability: float
    model_used: str


_whisper_models: dict[str, "object"] = {}  # keyed by model_size -- see bug note below


def _get_whisper(model_size: str):
    # NOTE: this used to be a single global cache regardless of model_size,
    # which meant escalating to a bigger model on low confidence would
    # silently keep returning the first (smaller) cached instance. Keyed
    # by size now so escalation actually loads the bigger model.
    if model_size not in _whisper_models:
        from faster_whisper import WhisperModel
        _whisper_models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_models[model_size]


# Explicit, visible confidence policy -- deliberately NOT relying on
# faster-whisper's internal defaults so this is auditable and tunable in
# our own code. Product decision: only trust a segment if the model is
# MORE THAN 80% confident it's actually speech (no_speech_prob < 0.20,
# i.e. speech-confidence > 80%), AND reasonably confident in the specific
# words (avg_logprob). Tested against a real case where forcing output
# past these bars produced no_speech_prob 0.83-0.91 and avg_logprob
# -1.3 to -1.4 text that read as incoherent garbage in valid script, not
# real words; a cleaner case (Whisper 'medium' on an isolated vocal
# track) landed at no_speech_prob 0.586-0.737 with avg_logprob -0.4 --
# still short of the 80% speech-confidence bar, so still correctly
# rejected under this policy even though it looked more plausible.
MAX_NO_SPEECH_PROB = 0.20  # i.e. require > 80% speech confidence
MIN_AVG_LOGPROB = -1.0


def _transcribe_once(audio_path: str, model_size: str, language: str | None) -> TranscriptResult:
    model = _get_whisper(model_size)
    segments_iter, info = model.transcribe(audio_path, word_timestamps=True, language=language)

    words: list[TranscriptWord] = []
    segments: list[TranscriptSegment] = []
    n_dropped = 0
    for seg in segments_iter:
        if seg.no_speech_prob > MAX_NO_SPEECH_PROB or seg.avg_logprob < MIN_AVG_LOGPROB:
            n_dropped += 1
            continue
        segments.append(TranscriptSegment(text=seg.text.strip(), start=seg.start, end=seg.end))
        for w in (seg.words or []):
            words.append(TranscriptWord(word=w.word.strip(), start=w.start, end=w.end))

    if n_dropped:
        print(f"[audio] dropped {n_dropped} low-confidence segment(s) (no_speech_prob > "
              f"{MAX_NO_SPEECH_PROB} or avg_logprob < {MIN_AVG_LOGPROB}) -- not treated as real transcript.")

    return TranscriptResult(
        words=words, segments=segments,
        language=info.language, language_probability=info.language_probability,
        model_used=model_size,
    )


_separator = None


def _get_separator():
    global _separator
    if _separator is None:
        from demucs.api import Separator
        _separator = Separator(model="htdemucs", device="cpu", progress=False)
    return _separator


def _has_meaningful_audio(wav_path: str, threshold_db: float = -50.0) -> bool:
    """Skip the expensive separation step entirely on audio that's just
    silent -- no point isolating vocals from nothing."""
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) == 0:
        return False
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    return 20 * np.log10(rms) > threshold_db


def separate_vocals(audio_path: str, out_path: str | None = None) -> str:
    """Isolate the vocal stem via Demucs (source separation) so ASR isn't
    fighting instrumentation for the same frequencies -- the standard
    approach for transcribing sung vocals, not something Whisper does on
    its own no matter the model size."""
    from demucs.audio import save_audio

    if out_path is None:
        out_path = tempfile.mktemp(suffix="_vocals.wav")
    separator = _get_separator()
    _origin, separated = separator.separate_audio_file(audio_path)
    save_audio(separated["vocals"], out_path, samplerate=separator.samplerate)
    return out_path


def transcribe(
    video_path: str, model_size: str = "small", wav_path: str | None = None, language: str | None = None,
    confidence_floor: float = 0.6, escalation_ladder: tuple[str, ...] = ("small", "medium"),
    try_vocal_separation: bool = True, vocal_separation_model: str = "medium",
) -> TranscriptResult:
    """Local transcription with word- and segment-level timestamps.

    Language defaults to auto-detect (language=None) -- for a real product
    reviewing arbitrary client content you can't assume the language ahead
    of time. Auto-detect CAN misfire on short/ambiguous audio (seen in
    testing: 'base' misidentified Telugu as Tamil and returned near-
    nothing), so rather than requiring the caller to already know and pass
    the right language, this escalates to a bigger model in the ladder
    when Whisper's own reported confidence (info.language_probability) is
    below confidence_floor. If a language IS passed explicitly, no
    escalation happens -- forcing a known language is faster and skips
    this entirely.

    If the full mix produces no usable segments (either genuine silence,
    or -- the case this was built for -- dialogue/lyrics buried under
    background music) and try_vocal_separation is set, this falls back to
    isolating the vocal track (see separate_vocals) and retrying there.
    Every segment, from either path, still has to clear the confidence
    gate in _transcribe_once (>80% speech confidence) before it's kept --
    separation makes real content easier to find, it doesn't lower the
    bar for trusting what comes out.
    """
    audio_path = wav_path or extract_audio_wav(video_path)

    result = _transcribe_once(audio_path, escalation_ladder[0], language)
    if language is None and result.language_probability < confidence_floor:
        for bigger_model in escalation_ladder[1:]:
            print(f"[audio] language detection confidence low ({result.language} @ "
                  f"{result.language_probability:.2f} with '{result.model_used}') -- "
                  f"escalating to '{bigger_model}'...")
            result = _transcribe_once(audio_path, bigger_model, language)
            if result.language_probability >= confidence_floor:
                break
        else:
            print(f"[audio] still low confidence ({result.language} @ {result.language_probability:.2f}) "
                  f"after escalating through {escalation_ladder} -- keeping best available result. "
                  f"Consider passing an explicit language= if you know it.")

    if result.segments or not try_vocal_separation or not _has_meaningful_audio(audio_path):
        return result

    print("[audio] no confident transcript from the full mix -- trying vocal separation "
          "(isolating vocals from music/background audio)...")
    try:
        vocals_path = separate_vocals(audio_path)
    except Exception as e:
        print(f"[audio] vocal separation failed ({e}), keeping original (empty) result.")
        return result

    vocals_result = _transcribe_once(vocals_path, vocal_separation_model, language)
    if vocals_result.segments:
        print(f"[audio] recovered {len(vocals_result.segments)} confident segment(s) from isolated vocals.")
        return vocals_result

    print("[audio] still nothing above the confidence bar after vocal separation -- "
          "treating this as confirmed no reliable dialogue/lyrics, not a failure to try harder.")
    return result


def transcript_to_text(words: list[TranscriptWord]) -> str:
    return " ".join(w.word for w in words)
