"""
VAD — Silero Voice Activity Detection wrapper.

Detects speech segments in audio, filters silence, saves GPU compute.
Uses silero-vad pip package (faster load, no torch.hub download).
"""

import sys
import time
from typing import List, Optional

import numpy as np
import torch

from .config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
)

# Silero VAD v6 requires fixed chunk sizes
VAD_CHUNK_SIZE_16K = 512   # 32ms at 16kHz
VAD_CHUNK_SIZE_8K = 256    # 32ms at 8kHz


class SpeechSegment:
    """A detected speech segment with start/end times."""

    __slots__ = ("start", "end", "audio")

    def __init__(self, start: float, end: float, audio: Optional[np.ndarray] = None):
        self.start = start   # seconds
        self.end = end       # seconds
        self.audio = audio   # float32 mono 16kHz (optional)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"SpeechSegment({self.start:.2f}s-{self.end:.2f}s, dur={self.duration:.2f}s)"


class VoiceActivityDetector:
    """
    Silero VAD wrapper for speech detection.
    
    Usage:
        vad = VoiceActivityDetector()
        vad.load()
        segments = vad.detect(audio_array)
        for seg in segments:
            print(f"Speech: {seg.start:.1f}s - {seg.end:.1f}s")
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate
        self._chunk_size = VAD_CHUNK_SIZE_16K if sample_rate == 16000 else VAD_CHUNK_SIZE_8K
        self._model = None
        self._loaded = False

    # ── Model Loading ────────────────────────────────────────────

    def load(self) -> None:
        """Load Silero VAD model."""
        if self._loaded:
            return

        from silero_vad import load_silero_vad

        print("[VAD] Loading Silero VAD...", file=sys.stderr)
        t0 = time.time()

        self._model = load_silero_vad()
        self._loaded = True

        elapsed = time.time() - t0
        print(f"[VAD] Loaded in {elapsed:.1f}s", file=sys.stderr)

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Detection ────────────────────────────────────────────────

    def detect(
        self,
        audio: np.ndarray,
        return_audio: bool = True,
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio array.
        
        Args:
            audio: float32 numpy array, mono, 16kHz
            return_audio: if True, attach audio slice to each segment
            
        Returns:
            List of SpeechSegment with start/end times
        """
        self.ensure_loaded()

        from silero_vad import get_speech_timestamps

        # Convert to torch tensor
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio_tensor = torch.from_numpy(audio)

        # Get speech timestamps
        timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            min_speech_duration_ms=self.min_speech_ms,
            return_seconds=False,  # Return in samples
        )

        segments = []
        for ts in timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            start_sec = start_sample / self.sample_rate
            end_sec = end_sample / self.sample_rate

            seg_audio = None
            if return_audio:
                seg_audio = audio[start_sample:end_sample]

            segments.append(SpeechSegment(
                start=start_sec,
                end=end_sec,
                audio=seg_audio,
            ))

        return segments

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Quick check: does this audio chunk contain speech?
        
        Processes audio in small chunks and returns True if any chunk has speech.
        """
        confidence = self.get_avg_confidence(audio)
        return confidence >= self.threshold

    def get_confidence(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability for a SINGLE chunk (must be exactly 512 samples for 16kHz).
        
        For variable-length audio, use get_avg_confidence() instead.
        """
        self.ensure_loaded()

        if len(audio_chunk) != self._chunk_size:
            raise ValueError(
                f"Chunk must be exactly {self._chunk_size} samples, got {len(audio_chunk)}. "
                f"Use get_avg_confidence() for variable-length audio."
            )

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        audio_tensor = torch.from_numpy(audio_chunk)
        return self._model(audio_tensor, self.sample_rate).item()

    def get_avg_confidence(self, audio: np.ndarray) -> float:
        """
        Get average speech probability across all chunks in audio.
        
        Handles any length audio by chunking internally.
        """
        self.ensure_loaded()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Process in chunks
        num_chunks = len(audio) // self._chunk_size
        if num_chunks == 0:
            # Pad short audio
            padded = np.zeros(self._chunk_size, dtype=np.float32)
            padded[:len(audio)] = audio
            audio_tensor = torch.from_numpy(padded)
            return self._model(audio_tensor, self.sample_rate).item()

        confidences = []
        for i in range(num_chunks):
            start = i * self._chunk_size
            end = start + self._chunk_size
            chunk = audio[start:end]
            audio_tensor = torch.from_numpy(chunk)
            conf = self._model(audio_tensor, self.sample_rate).item()
            confidences.append(conf)

        return sum(confidences) / len(confidences)

    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """
        Get ratio of chunks that contain speech (0.0 - 1.0).
        """
        self.ensure_loaded()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        num_chunks = len(audio) // self._chunk_size
        if num_chunks == 0:
            return 0.0

        speech_chunks = 0
        for i in range(num_chunks):
            start = i * self._chunk_size
            end = start + self._chunk_size
            chunk = audio[start:end]
            audio_tensor = torch.from_numpy(chunk)
            conf = self._model(audio_tensor, self.sample_rate).item()
            if conf >= self.threshold:
                speech_chunks += 1

        return speech_chunks / num_chunks

    # ── Streaming Helper ─────────────────────────────────────────

    def reset_states(self) -> None:
        """Reset VAD internal states (call between separate audio streams)."""
        if self._model is not None:
            self._model.reset_states()

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def chunk_size(self) -> int:
        """Required chunk size for single-chunk processing."""
        return self._chunk_size

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"VoiceActivityDetector(threshold={self.threshold}, {status})"
