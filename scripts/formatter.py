"""
Formatter — Output format converters for transcription results.

Supports text, JSON, SRT subtitle, and agent-optimized formats.
"""

import json
from typing import List, Union
from datetime import datetime

from .engine import TranscribeResult
from .merger import SpeakerTranscript


class OutputFormatter:
    """Format transcription outputs."""

    @staticmethod
    def format_text(
        items: Union[List[TranscribeResult], List[SpeakerTranscript]],
        include_timestamps: bool = True,
        include_speakers: bool = False,
    ) -> str:
        """
        Plain text format.
        
        Example:
            [12:30:01] [Speaker_0] Hello world
            [12:30:03] [Speaker_1] Hi there
        """
        lines = []

        for item in items:
            parts = []

            if include_timestamps:
                if hasattr(item, 'start'):
                    ts = _format_timestamp(item.start)
                    parts.append(f"[{ts}]")
                else:
                    ts = datetime.now().strftime("%H:%M:%S")
                    parts.append(f"[{ts}]")

            if include_speakers and hasattr(item, 'speaker'):
                parts.append(f"[{item.speaker}]")

            parts.append(item.text)
            lines.append(" ".join(parts))

        return "\n".join(lines)

    @staticmethod
    def format_json(
        items: Union[List[TranscribeResult], List[SpeakerTranscript]],
    ) -> str:
        """
        JSON Lines format (one JSON object per line).
        
        Example:
            {"ts": "12:30:01", "speaker": "Speaker_0", "text": "Hello", "confidence": 0.95}
        """
        lines = []

        for item in items:
            obj = {"text": item.text}

            if hasattr(item, 'start'):
                obj["start"] = round(item.start, 3)
                obj["end"] = round(item.end, 3)
                obj["ts"] = _format_timestamp(item.start)

            if hasattr(item, 'speaker'):
                obj["speaker"] = item.speaker

            if hasattr(item, 'language'):
                obj["lang"] = item.language

            if hasattr(item, 'probability'):
                obj["confidence"] = round(item.probability, 3)
            elif hasattr(item, 'confidence'):
                obj["confidence"] = round(item.confidence, 3)

            lines.append(json.dumps(obj, ensure_ascii=False))

        return "\n".join(lines)

    @staticmethod
    def format_srt(
        items: Union[List[TranscribeResult], List[SpeakerTranscript]],
    ) -> str:
        """
        SRT subtitle format.
        
        Example:
            1
            00:00:01,000 --> 00:00:03,200
            [Speaker_0] Hello world
        """
        lines = []
        index = 1

        for item in items:
            if not hasattr(item, 'start') or not hasattr(item, 'end'):
                continue

            lines.append(str(index))
            start_srt = _format_srt_time(item.start)
            end_srt = _format_srt_time(item.end)
            lines.append(f"{start_srt} --> {end_srt}")

            text = item.text
            if hasattr(item, 'speaker'):
                text = f"[{item.speaker}] {text}"

            lines.append(text)
            lines.append("")  # Blank line
            index += 1

        return "\n".join(lines)

    @staticmethod
    def format_agent(
        items: Union[List[TranscribeResult], List[SpeakerTranscript]],
        max_tokens: int = 500,
    ) -> str:
        """
        Agent-optimized format: concise, ~500 tokens.
        
        Groups by speaker, summarizes timestamps.
        """
        if not items:
            return "(No speech detected)"

        # Group by speaker if available
        if all(hasattr(item, 'speaker') for item in items):
            groups = {}
            for item in items:
                speaker = item.speaker
                if speaker not in groups:
                    groups[speaker] = []
                groups[speaker].append(item.text)

            lines = []
            for speaker, texts in groups.items():
                combined = " ".join(texts)
                # Truncate if too long
                if len(combined) > 200:
                    combined = combined[:200] + "..."
                lines.append(f"**{speaker}:** {combined}")

            return "\n\n".join(lines)

        # Plain text grouping
        texts = [item.text for item in items]
        combined = " ".join(texts)

        # Truncate if needed
        if len(combined) > 800:
            combined = combined[:800] + "..."

        return combined


def _format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
