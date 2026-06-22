"""
StreamPipeline — Chunked buffer with VAD gate and overlap handling.

Manages real-time audio streaming: buffers audio, applies VAD filter,
handles chunk overlap, and yields speech segments for transcription.
"""

import sys
import threading
import queue
from typing import Optional, Generator, Callable
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import (
    SAMPLE_RATE,
    MIN_CHUNK_S,
    MAX_CHUNK_S,
    OVERLAP_S,
    VAD_THRESHOLD,
)
from .vad import VoiceActivityDetector


@dataclass
class AudioChunk:
    """An audio chunk ready for transcription."""
    audio: np.ndarray           # float32 mono 16kHz
    start_time: float           # seconds from stream start
    end_time: float             # seconds from stream start
    speech_ratio: float = 0.0   # ratio of speech in chunk

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"AudioChunk({self.start_time:.2f}s-{self.end_time:.2f}s, speech={self.speech_ratio:.1%})"


class StreamPipeline:
    """
    Real-time audio streaming pipeline with VAD gating.
    
    Features:
    - Ring buffer for continuous audio
    - VAD filtering (skip silence)
    - Overlap handling between chunks
    - Thread-safe queue for output
    
    Usage:
        pipeline = StreamPipeline()
        pipeline.start()
        
        # Feed audio
        pipeline.feed(audio_chunk)
        
        # Get speech chunks
        for chunk in pipeline.output():
            transcribe(chunk.audio)
        
        pipeline.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        min_chunk_s: float = MIN_CHUNK_S,
        max_chunk_s: float = MAX_CHUNK_S,
        overlap_s: float = OVERLAP_S,
        vad_threshold: float = VAD_THRESHOLD,
        use_vad: bool = True,
    ):
        self.sample_rate = sample_rate
        self.min_chunk_samples = int(min_chunk_s * sample_rate)
        self.max_chunk_samples = int(max_chunk_s * sample_rate)
        self.overlap_samples = int(overlap_s * sample_rate)
        self.vad_threshold = vad_threshold
        self.use_vad = use_vad

        # Buffer
        self._buffer: deque = deque()
        self._buffer_samples = 0
        self._stream_time = 0.0  # seconds since start

        # VAD
        self._vad: Optional[VoiceActivityDetector] = None
        if use_vad:
            self._vad = VoiceActivityDetector(threshold=vad_threshold)

        # Output queue
        self._output_queue: queue.Queue = queue.Queue()

        # State
        self._running = False
        self._lock = threading.Lock()

        # Speech state for VAD-based chunking
        self._in_speech = False
        self._speech_buffer: list = []
        self._speech_start_time = 0.0

    def start(self) -> None:
        """Start the pipeline."""
        self._running = True
        if self._vad is not None:
            self._vad.load()
        print("[StreamPipeline] Started", file=sys.stderr)

    def stop(self) -> None:
        """Stop the pipeline, flush remaining buffer."""
        self._running = False
        self._flush_speech_buffer(force=True)
        print("[StreamPipeline] Stopped", file=sys.stderr)

    def feed(self, audio: np.ndarray) -> None:
        """
        Feed audio chunk into the pipeline.
        
        Args:
            audio: float32 mono 16kHz numpy array
        """
        if not self._running:
            return

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        duration = len(audio) / self.sample_rate

        with self._lock:
            if self.use_vad and self._vad is not None:
                # VAD-based chunking
                self._process_with_vad(audio, duration)
            else:
                # Fixed-size chunking (no VAD)
                self._process_fixed_chunks(audio, duration)

            self._stream_time += duration

    def _process_with_vad(self, audio: np.ndarray, duration: float) -> None:
        """Process audio with VAD-based speech detection."""
        speech_ratio = self._vad.get_speech_ratio(audio)
        is_speech = speech_ratio >= 0.1  # At least 10% speech

        if is_speech:
            if not self._in_speech:
                # Speech started
                self._in_speech = True
                self._speech_start_time = self._stream_time
                self._speech_buffer = []

            self._speech_buffer.append(audio)

            # Check if max length reached
            total_samples = sum(len(a) for a in self._speech_buffer)
            if total_samples >= self.max_chunk_samples:
                self._flush_speech_buffer(force=True)
        else:
            if self._in_speech:
                # Speech ended
                self._flush_speech_buffer(force=False)
                self._in_speech = False

    def _process_fixed_chunks(self, audio: np.ndarray, duration: float) -> None:
        """Process audio with fixed-size chunks (no VAD)."""
        self._buffer.append(audio)
        self._buffer_samples += len(audio)

        while self._buffer_samples >= self.max_chunk_samples:
            self._emit_chunk_from_buffer()

    def _flush_speech_buffer(self, force: bool = False) -> None:
        """Flush accumulated speech to output."""
        if not self._speech_buffer:
            return

        total_samples = sum(len(a) for a in self._speech_buffer)

        # Only emit if min length met (or forced)
        if total_samples < self.min_chunk_samples and not force:
            return

        # Concatenate
        combined = np.concatenate(self._speech_buffer)
        end_time = self._stream_time

        # Compute speech ratio
        speech_ratio = 1.0  # Already filtered by VAD
        if self._vad is not None:
            speech_ratio = self._vad.get_speech_ratio(combined)

        chunk = AudioChunk(
            audio=combined,
            start_time=self._speech_start_time,
            end_time=end_time,
            speech_ratio=speech_ratio,
        )

        self._output_queue.put(chunk)
        self._speech_buffer = []

    def _emit_chunk_from_buffer(self) -> None:
        """Emit a chunk from the fixed buffer."""
        # Collect samples
        collected = []
        collected_samples = 0
        target = self.max_chunk_samples

        while collected_samples < target and self._buffer:
            chunk = self._buffer.popleft()
            needed = target - collected_samples

            if len(chunk) <= needed:
                collected.append(chunk)
                collected_samples += len(chunk)
                self._buffer_samples -= len(chunk)
            else:
                # Split chunk
                collected.append(chunk[:needed])
                self._buffer.appendleft(chunk[needed:])
                collected_samples += needed
                self._buffer_samples -= needed

        if not collected:
            return

        combined = np.concatenate(collected)
        start_time = self._stream_time - len(combined) / self.sample_rate
        end_time = self._stream_time

        speech_ratio = 1.0
        if self._vad is not None:
            speech_ratio = self._vad.get_speech_ratio(combined)

        chunk = AudioChunk(
            audio=combined,
            start_time=start_time,
            end_time=end_time,
            speech_ratio=speech_ratio,
        )

        self._output_queue.put(chunk)

    def get_chunk(self, timeout: float = None) -> Optional[AudioChunk]:
        """Get next output chunk, or None on timeout."""
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def output(self, timeout: float = 2.0) -> Generator[AudioChunk, None, None]:
        """Generator yielding output chunks."""
        while self._running or not self._output_queue.empty():
            chunk = self.get_chunk(timeout=timeout)
            if chunk is not None:
                yield chunk

    def clear(self) -> None:
        """Clear all buffers."""
        with self._lock:
            self._buffer.clear()
            self._buffer_samples = 0
            self._speech_buffer = []
            self._in_speech = False
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    break

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def buffer_duration(self) -> float:
        """Current buffer duration in seconds."""
        return self._buffer_samples / self.sample_rate

    @property
    def stream_time(self) -> float:
        """Total stream time in seconds."""
        return self._stream_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
