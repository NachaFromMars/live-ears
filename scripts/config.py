"""
Live-Ears Configuration — centralized constants and defaults.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"
PROFILES_DIR = SKILL_DIR / "profiles"
MODELS_DIR = SKILL_DIR / "models"
TESTS_DIR = SKILL_DIR / "tests"

# Ensure dirs exist
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Audio Defaults ───────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper native sample rate
CHANNELS = 1                 # Mono
DTYPE = "float32"            # Audio dtype for sounddevice
BLOCK_DURATION_MS = 30       # VAD frame size (ms)
BUFFER_DURATION_S = 3.0      # Transcription buffer (seconds)
MIN_CHUNK_S = 1.0            # Min chunk to transcribe
MAX_CHUNK_S = 10.0           # Max chunk before forced flush
OVERLAP_S = 0.5              # Overlap between chunks

# ── Whisper Defaults ─────────────────────────────────────────────
WHISPER_MODEL = "large-v3"   # Best accuracy
WHISPER_DEVICE = "cuda"      # GPU
WHISPER_COMPUTE = "float16"  # VRAM efficient
WHISPER_BEAM_SIZE = 5
WHISPER_LANGUAGE = None       # Auto-detect (or "vi", "en", etc.)

# ── VAD Defaults ─────────────────────────────────────────────────
VAD_THRESHOLD = 0.5          # Speech probability threshold
VAD_MIN_SILENCE_MS = 500     # Min silence to end segment
VAD_MIN_SPEECH_MS = 250      # Min speech to start segment
VAD_WINDOW_SIZE = 512        # Silero VAD window

# ── Diarization Defaults ────────────────────────────────────────
DIARIZE_MIN_SPEAKERS = 1
DIARIZE_MAX_SPEAKERS = 10
DIARIZE_HF_TOKEN = os.environ.get("HF_TOKEN", "")  # pyannote needs HF token

# ── Speaker ID Defaults ─────────────────────────────────────────
ID_THRESHOLD = 0.75          # Cosine similarity threshold for match
ID_EMBEDDING_DIM = 192       # ECAPA-TDNN embedding dimension
ENROLL_DURATION_S = 15       # Default enrollment duration
ENROLL_MIN_DURATION_S = 5    # Minimum enrollment duration

# ── WebSocket Defaults ───────────────────────────────────────────
WS_HOST = "127.0.0.1"
WS_PORT = 8765

# ── Output Formats ───────────────────────────────────────────────
FORMAT_TEXT = "text"
FORMAT_JSON = "json"
FORMAT_SRT = "srt"
FORMAT_AGENT = "agent"
VALID_FORMATS = [FORMAT_TEXT, FORMAT_JSON, FORMAT_SRT, FORMAT_AGENT]
DEFAULT_FORMAT = FORMAT_TEXT

# ── Manifest ─────────────────────────────────────────────────────
MANIFEST_FILE = PROFILES_DIR / "manifest.json"
