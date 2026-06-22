"""
Diarization — pyannote-audio wrapper for speaker diarization.

Segments audio by speaker, labels as Speaker_0, Speaker_1, etc.
Requires HuggingFace token for model download (one-time).

NOTE: Step 3.1 provides basic diarization pipeline. For production,
set HF_TOKEN in environment or config.py to download models.
"""

import sys
import time
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch

from .config import (
    SAMPLE_RATE,
    DIARIZE_MIN_SPEAKERS,
    DIARIZE_MAX_SPEAKERS,
    DIARIZE_HF_TOKEN,
)


@dataclass
class SpeakerSegment:
    """A segment with speaker label."""
    start: float        # seconds
    end: float          # seconds
    speaker: str        # "Speaker_0", "Speaker_1", etc.

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"SpeakerSegment({self.start:.2f}s-{self.end:.2f}s, {self.speaker})"


class SpeakerDiarizer:
    """
    PyAnnote-audio wrapper for speaker diarization.
    
    Usage:
        diarizer = SpeakerDiarizer()
        diarizer.load()
        segments = diarizer.diarize(audio_array)
        for seg in segments:
            print(f"{seg.speaker}: {seg.start:.1f}s - {seg.end:.1f}s")
    """

    def __init__(
        self,
        min_speakers: int = DIARIZE_MIN_SPEAKERS,
        max_speakers: int = DIARIZE_MAX_SPEAKERS,
        hf_token: Optional[str] = None,
    ):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token = hf_token or DIARIZE_HF_TOKEN or os.getenv("HF_TOKEN", "")
        
        self._pipeline = None
        self._loaded = False
        self._mock_mode = False

    # ── Model Loading ────────────────────────────────────────────

    def load(self, use_mock: bool = False) -> None:
        """
        Load pyannote diarization pipeline.
        
        Args:
            use_mock: If True, use mock diarization (for testing without HF token)
        """
        if self._loaded:
            return

        if use_mock or not self.hf_token:
            print("[Diarization] Using MOCK mode (no HF token)", file=sys.stderr)
            self._mock_mode = True
            self._loaded = True
            return

        from pyannote.audio import Pipeline

        print("[Diarization] Loading pyannote pipeline (requires HF token)...", file=sys.stderr)
        t0 = time.time()

        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))

            elapsed = time.time() - t0
            print(f"[Diarization] Loaded in {elapsed:.1f}s", file=sys.stderr)
            self._loaded = True

        except Exception as e:
            print(f"[Diarization] Load failed: {e}", file=sys.stderr)
            print("[Diarization] Falling back to MOCK mode", file=sys.stderr)
            self._mock_mode = True
            self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Diarization ──────────────────────────────────────────────

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio: float32 mono 16kHz numpy array
            num_speakers: If known, specify exact speaker count (None = auto-detect)
            
        Returns:
            List of SpeakerSegment with start/end/speaker labels
        """
        self.ensure_loaded()

        if self._mock_mode:
            return self._mock_diarize(audio, num_speakers)

        # Convert to torch
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # pyannote expects dict with waveform and sample_rate
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

        input_dict = {
            "waveform": waveform,
            "sample_rate": SAMPLE_RATE,
        }

        # Run diarization
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["min_speakers"] = self.min_speakers
            kwargs["max_speakers"] = self.max_speakers

        diarization = self._pipeline(input_dict, **kwargs)

        # Convert to SpeakerSegment list
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker=f"Speaker_{speaker}",
            ))

        return segments

    def _mock_diarize(
        self,
        audio: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """
        Mock diarization: split audio into equal segments by speaker count.
        
        This is a placeholder for testing without pyannote models.
        """
        duration = len(audio) / SAMPLE_RATE
        n_speakers = num_speakers or 2  # Default 2 speakers in mock

        # Split audio into equal segments
        segment_duration = duration / n_speakers
        segments = []

        for i in range(n_speakers):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            segments.append(SpeakerSegment(
                start=start,
                end=min(end, duration),
                speaker=f"Speaker_{i}",
            ))

        return segments

    def get_speaker_timeline(
        self,
        segments: List[SpeakerSegment],
    ) -> List[Tuple[float, float, str]]:
        """
        Convert SpeakerSegment list to timeline format: [(start, end, speaker), ...]
        """
        return [(s.start, s.end, s.speaker) for s in segments]

    def get_speakers(self, segments: List[SpeakerSegment]) -> List[str]:
        """Get unique speaker labels from segments."""
        return sorted(set(s.speaker for s in segments))

    def count_speakers(self, segments: List[SpeakerSegment]) -> int:
        """Count unique speakers in segments."""
        return len(self.get_speakers(segments))

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        mode = " (MOCK)" if self._mock_mode else ""
        return f"SpeakerDiarizer({status}{mode})"
