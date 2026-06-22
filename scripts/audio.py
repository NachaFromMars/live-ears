"""
AudioCapture — Mic and file audio input with ring buffer.

Supports:
- Real-time mic capture (sounddevice)
- File input (wav, mp3, ogg, flac via soundfile/ffmpeg)
- Ring buffer for streaming pipelines
- 16kHz mono output (Whisper native format)
"""

import queue
import threading
import wave
from pathlib import Path
from typing import Optional, Generator

import numpy as np

from .config import (
    SAMPLE_RATE,
    CHANNELS,
    DTYPE,
    BUFFER_DURATION_S,
)


class AudioCapture:
    """Captures audio from mic or file, outputs 16kHz mono float32 chunks."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        buffer_seconds: float = BUFFER_DURATION_S,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds
        self.device = device  # None = system default mic
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stream = None

    # ── Mic Capture ──────────────────────────────────────────────

    def start_mic(self) -> None:
        """Start capturing from microphone."""
        import sounddevice as sd

        self._running = True
        block_size = int(self.sample_rate * self.buffer_seconds)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=DTYPE,
            blocksize=block_size,
            device=self.device,
            callback=self._mic_callback,
        )
        self._stream.start()

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback — push audio chunks to queue."""
        if status:
            import sys
            print(f"[AudioCapture] Status: {status}", file=sys.stderr)
        if self._running:
            # Copy to avoid buffer reuse issues
            self._queue.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten())

    def stop_mic(self) -> None:
        """Stop microphone capture."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = None) -> Optional[np.ndarray]:
        """Get next audio chunk from queue. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def mic_stream(self, timeout: float = 5.0) -> Generator[np.ndarray, None, None]:
        """Generator yielding audio chunks from mic. Stops when self._running is False."""
        while self._running:
            chunk = self.get_chunk(timeout=timeout)
            if chunk is not None:
                yield chunk

    # ── File Input ───────────────────────────────────────────────

    def read_file(
        self,
        file_path: str,
        chunk_seconds: Optional[float] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Read audio file and yield chunks of float32 mono 16kHz.
        
        If chunk_seconds is None, yields entire file as one array.
        Otherwise yields fixed-size chunks for streaming simulation.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio = self._load_audio_file(str(path))

        if chunk_seconds is None:
            yield audio
        else:
            chunk_size = int(self.sample_rate * chunk_seconds)
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) > 0:
                    yield chunk

    def _load_audio_file(self, file_path: str) -> np.ndarray:
        """Load any audio file to float32 mono 16kHz numpy array."""
        ext = Path(file_path).suffix.lower()

        # Try soundfile first (wav, flac, ogg)
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Stereo to mono
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            return audio.astype(np.float32)
        except Exception:
            pass

        # Fallback: ffmpeg for mp3 and other formats
        return self._load_with_ffmpeg(file_path)

    def _load_with_ffmpeg(self, file_path: str) -> np.ndarray:
        """Load audio via ffmpeg subprocess (handles mp3, m4a, webm, etc.)."""
        import subprocess

        cmd = [
            "ffmpeg", "-i", file_path,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-v", "quiet",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, check=True, timeout=120
            )
            audio = np.frombuffer(result.stdout, dtype=np.float32)
            return audio
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg or use wav/flac/ogg files."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg error: {e.stderr.decode()[:200]}")

    # ── Resample ─────────────────────────────────────────────────

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using linear interpolation (simple, no deps)."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # ── Record to File ───────────────────────────────────────────

    def record_to_file(
        self,
        output_path: str,
        duration_seconds: float,
        show_progress: bool = True,
    ) -> str:
        """Record from mic for fixed duration and save as WAV."""
        import sounddevice as sd

        total_frames = int(self.sample_rate * duration_seconds)
        
        if show_progress:
            import sys
            print(f"Recording {duration_seconds}s...", file=sys.stderr)

        audio = sd.rec(
            total_frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=DTYPE,
            device=self.device,
        )
        sd.wait()

        if audio.ndim > 1:
            audio = audio[:, 0]

        # Save as WAV
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        if show_progress:
            print(f"Saved: {path}", file=sys.stderr)

        return str(path)

    # ── Utilities ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def list_devices() -> str:
        """List available audio devices."""
        import sounddevice as sd
        return str(sd.query_devices())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_mic()
