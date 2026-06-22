"""
Merger — Align speaker diarization with transcription timestamps.

Takes speaker segments from diarization and text segments from Whisper,
merges them by timestamp overlap to produce speaker-labeled transcripts.
"""

import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from .diarize import SpeakerSegment
from .engine import TranscribeResult


@dataclass
class SpeakerTranscript:
    """A transcript segment with speaker label."""
    text: str
    start: float
    end: float
    speaker: str
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"[{self.speaker}] {self.text[:50]}..."


class DiarizationMerger:
    """
    Merge speaker diarization with transcription.
    
    Usage:
        merger = DiarizationMerger()
        
        # From diarization
        speaker_segments = diarizer.diarize(audio)
        
        # From transcription
        transcripts = engine.transcribe_segments(audio)
        
        # Merge
        labeled = merger.merge(speaker_segments, transcripts)
        
        for item in labeled:
            print(f"[{item.speaker}] {item.text}")
    """

    def __init__(self, overlap_threshold: float = 0.5):
        """
        Args:
            overlap_threshold: Minimum overlap ratio (0-1) to assign speaker to text segment.
                              0.5 = at least 50% overlap required.
        """
        self.overlap_threshold = overlap_threshold

    def merge(
        self,
        speaker_segments: List[SpeakerSegment],
        transcripts: List[TranscribeResult],
    ) -> List[SpeakerTranscript]:
        """
        Merge speaker segments with transcript segments.
        
        Args:
            speaker_segments: List of SpeakerSegment from diarization
            transcripts: List of TranscribeResult from Whisper
            
        Returns:
            List of SpeakerTranscript with text and speaker labels
        """
        if not transcripts:
            return []

        if not speaker_segments:
            # No diarization, label all as "Unknown"
            return [
                SpeakerTranscript(
                    text=t.text,
                    start=t.start,
                    end=t.end,
                    speaker="Unknown",
                    confidence=t.probability,
                )
                for t in transcripts
            ]

        merged = []

        for transcript in transcripts:
            # Find best matching speaker
            speaker = self._find_speaker_for_segment(
                transcript.start,
                transcript.end,
                speaker_segments,
            )

            merged.append(SpeakerTranscript(
                text=transcript.text,
                start=transcript.start,
                end=transcript.end,
                speaker=speaker,
                confidence=transcript.probability,
            ))

        return merged

    def _find_speaker_for_segment(
        self,
        start: float,
        end: float,
        speaker_segments: List[SpeakerSegment],
    ) -> str:
        """
        Find the speaker with most overlap with given time range.
        
        Returns speaker label or "Unknown" if no sufficient overlap.
        """
        segment_duration = end - start
        if segment_duration <= 0:
            return "Unknown"

        best_speaker = "Unknown"
        best_overlap = 0.0

        for spk_seg in speaker_segments:
            overlap = self._calculate_overlap(start, end, spk_seg.start, spk_seg.end)
            overlap_ratio = overlap / segment_duration

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = spk_seg.speaker

        # Only assign if overlap meets threshold
        if best_overlap < self.overlap_threshold:
            return "Unknown"

        return best_speaker

    @staticmethod
    def _calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
        """Calculate overlap duration between two time ranges."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0.0, overlap_end - overlap_start)

    def merge_simple(
        self,
        speaker_segments: List[SpeakerSegment],
        full_text: str,
    ) -> List[Tuple[str, str]]:
        """
        Simple merge: split full text by speaker segments (no word timestamps).
        
        Returns:
            List of (speaker, text) tuples
        """
        if not speaker_segments:
            return [("Unknown", full_text)]

        # This is a rough approximation without word timestamps
        # Split text by character count proportional to segment duration
        total_duration = speaker_segments[-1].end
        text_len = len(full_text)

        results = []
        char_pos = 0

        for seg in speaker_segments:
            # Estimate character count for this segment
            ratio = seg.duration / total_duration
            chars_in_seg = int(text_len * ratio)

            seg_text = full_text[char_pos:char_pos + chars_in_seg].strip()
            if seg_text:
                results.append((seg.speaker, seg_text))

            char_pos += chars_in_seg

        # Append remaining text to last speaker
        if char_pos < text_len:
            remaining = full_text[char_pos:].strip()
            if remaining:
                if results:
                    results[-1] = (results[-1][0], results[-1][1] + " " + remaining)
                else:
                    results.append(("Unknown", remaining))

        return results

    def format_transcript(
        self,
        items: List[SpeakerTranscript],
        include_timestamps: bool = False,
    ) -> str:
        """
        Format merged transcript as readable text.
        
        Args:
            items: List of SpeakerTranscript
            include_timestamps: If True, include [HH:MM:SS] timestamps
            
        Returns:
            Formatted transcript string
        """
        lines = []
        current_speaker = None

        for item in items:
            if item.speaker != current_speaker:
                # New speaker, new line
                if include_timestamps:
                    ts = self._format_timestamp(item.start)
                    lines.append(f"\n[{ts}] [{item.speaker}]")
                else:
                    lines.append(f"\n[{item.speaker}]")
                current_speaker = item.speaker

            lines.append(item.text)

        return " ".join(lines).strip()

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def group_by_speaker(
        self,
        items: List[SpeakerTranscript],
    ) -> dict[str, List[SpeakerTranscript]]:
        """Group transcripts by speaker."""
        groups: dict = {}
        for item in items:
            if item.speaker not in groups:
                groups[item.speaker] = []
            groups[item.speaker].append(item)
        return groups

    def get_speaker_text(
        self,
        items: List[SpeakerTranscript],
        speaker: str,
    ) -> str:
        """Get all text for a specific speaker."""
        texts = [item.text for item in items if item.speaker == speaker]
        return " ".join(texts)
