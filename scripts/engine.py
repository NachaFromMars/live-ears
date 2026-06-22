"""
WhisperEngine — faster-whisper wrapper for transcription.

Loads model once, transcribes audio chunks efficiently on GPU.
Supports language detection, beam search, and word timestamps.
"""

import sys
import time
from typing import Optional, List, Tuple

import numpy as np

from .config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
    MIN_CHUNK_S,
)


class TranscribeResult:
    """Single transcription result with metadata."""

    __slots__ = ("text", "start", "end", "language", "probability", "words")

    def __init__(
        self,
        text: str,
        start: float = 0.0,
        end: float = 0.0,
        language: str = "",
        probability: float = 0.0,
        words: Optional[List] = None,
    ):
        self.text = text.strip()
        self.start = start
        self.end = end
        self.language = language
        self.probability = probability
        self.words = words or []

    def __repr__(self) -> str:
        return f"TranscribeResult('{self.text[:50]}...', lang={self.language}, p={self.probability:.2f})"


class WhisperEngine:
    """
    Faster-whisper transcription engine.
    
    Usage:
        engine = WhisperEngine(model="large-v3")
        engine.load()
        result = engine.transcribe(audio_array)
        print(result.text)
    """

    def __init__(
        self,
        model: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE,
        beam_size: int = WHISPER_BEAM_SIZE,
        language: Optional[str] = WHISPER_LANGUAGE,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model = None
        self._loaded = False

    # ── Model Loading ────────────────────────────────────────────

    def load(self) -> None:
        """Load the Whisper model. Call once before transcribing."""
        if self._loaded:
            return

        from faster_whisper import WhisperModel

        print(
            f"[WhisperEngine] Loading {self.model_name} on {self.device} ({self.compute_type})...",
            file=sys.stderr,
        )
        t0 = time.time()

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        elapsed = time.time() - t0
        print(f"[WhisperEngine] Model loaded in {elapsed:.1f}s", file=sys.stderr)
        self._loaded = True

    def ensure_loaded(self) -> None:
        """Ensure model is loaded, load if not."""
        if not self._loaded:
            self.load()

    # ── Transcription ────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
    ) -> TranscribeResult:
        """
        Transcribe a numpy audio array (float32, 16kHz mono).
        
        Args:
            audio: float32 numpy array, mono, 16kHz
            language: Override language (None = auto-detect or use default)
            word_timestamps: Include word-level timestamps
            initial_prompt: Context prompt for better accuracy
            
        Returns:
            TranscribeResult with text, timestamps, language, probability
        """
        self.ensure_loaded()

        # Skip too-short audio
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_CHUNK_S:
            return TranscribeResult(text="", start=0.0, end=duration)

        # Ensure correct dtype
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        lang = language or self.language

        segments, info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam_size,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            vad_filter=False,  # We handle VAD separately
            without_timestamps=not word_timestamps,
        )

        # Collect all segments
        texts = []
        words_all = []
        seg_start = 0.0
        seg_end = 0.0

        for seg in segments:
            texts.append(seg.text.strip())
            seg_end = seg.end
            if seg_start == 0.0 and seg.start > 0:
                seg_start = seg.start
            if word_timestamps and seg.words:
                for w in seg.words:
                    words_all.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

        full_text = " ".join(texts)

        return TranscribeResult(
            text=full_text,
            start=seg_start,
            end=seg_end,
            language=info.language,
            probability=info.language_probability,
            words=words_all,
        )

    def transcribe_segments(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> List[TranscribeResult]:
        """
        Transcribe and return individual segments (useful for diarization alignment).
        """
        self.ensure_loaded()

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_CHUNK_S:
            return []

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        lang = language or self.language

        segments, info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=False,
        )

        results = []
        for seg in segments:
            words = []
            if seg.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in seg.words
                ]
            results.append(TranscribeResult(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                language=info.language,
                probability=info.language_probability,
                words=words,
            ))

        return results

    # ── Language Detection ───────────────────────────────────────

    def detect_language(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Detect language from audio sample.
        Returns (language_code, probability).
        """
        self.ensure_loaded()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Use first 30s max for detection
        max_samples = SAMPLE_RATE * 30
        sample = audio[:max_samples]

        _, info = self._model.transcribe(
            sample,
            beam_size=1,
            without_timestamps=True,
        )

        return info.language, info.language_probability

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_info(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "language": self.language or "auto",
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"WhisperEngine({self.model_name}, {self.device}, {status})"
