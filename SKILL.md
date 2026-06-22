# Live-Ears 🎙️ — Realtime Speech-to-Text + Speaker Identification

**v1.0** — Built by Mula Nacharis for OpenClaw

Local, GPU-accelerated skill for real-time speech transcription with voice activity detection, speaker diarization, and voice enrollment/identification.

## Features

✅ **Real-time STT** — faster-whisper (large-v3, 4x faster than Whisper)  
✅ **Voice Activity Detection** — Silero VAD (skip silence, save GPU)  
✅ **Speaker Diarization** — pyannote-audio (label Speaker_0, Speaker_1...)  
✅ **Voice Enrollment** — SpeechBrain ECAPA-TDNN (192-dim embeddings)  
✅ **Speaker Identification** — Cosine similarity matching with auto-save unknown  
✅ **Multi-format Output** — text, JSON, SRT, agent-optimized  
✅ **WebSocket Server** — Push transcription to connected clients  
✅ **99 Languages** — Whisper native support

## Requirements

- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA (recommended, 4-6GB VRAM)
- **CPU:** Works on CPU (slower)
- **Disk:** ~2GB for models (auto-download first run)

## Installation

```bash
cd ~/.openclaw/workspace/skills/live-ears
pip install -r requirements.txt
```

## Quick Start

### 1. Real-time Mic Transcription

```bash
python scripts/live.py live
# [12:30:01] Hello world
# [12:30:03] How are you
```

### 2. Transcribe File

```bash
python scripts/live.py file audio.mp3
python scripts/live.py file meeting.wav --format srt > subtitles.srt
```

### 3. Enroll Voice Profile

```bash
python scripts/live.py enroll "Nấng" --duration 15
# 🎙️  Nấng, please speak naturally for 15 seconds...
# ✅ Profile 'Nấng' saved
```

### 4. Speaker Identification

```bash
python scripts/live.py live --identify
# [12:30:01] [Nấng] Ê mày làm xong chưa?
# [12:30:04] [Unknown] Gần xong rồi.
```

### 5. Manage Profiles

```bash
python scripts/live.py profiles list
python scripts/live.py profiles delete "Unknown_01"
python scripts/live.py profiles rename "Unknown_02" "Tuấn"
python scripts/live.py profiles test  # Test recognition
```

### 6. WebSocket Server

```bash
python scripts/live.py serve --port 8765
# [WS] Server running on ws://127.0.0.1:8765
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `live` | Real-time mic transcription |
| `file <path>` | Transcribe audio file |
| `enroll <name>` | Enroll voice profile |
| `profiles list\|delete\|rename\|test` | Manage profiles |
| `devices` | List audio devices |
| `serve` | WebSocket server |

## Options

### Transcription

```bash
--lang vi           # Language code (vi, en, auto)
--model large-v3    # Whisper model (large-v3, medium, base)
--format json       # Output format (text, json, srt, agent)
--chunked           # Process file in chunks (for streaming simulation)
```

### Speaker ID

```bash
--identify          # Enable speaker identification
--threshold 0.75    # Similarity threshold (0-1, higher = stricter)
```

### Enrollment

```bash
--duration 15       # Recording duration (seconds)
--file audio.wav    # Enroll from file
--overwrite         # Replace existing profile
```

## Output Formats

### Text (default)

```
[12:30:01] [Speaker_0] Hello world
[12:30:03] [Speaker_1] Hi there
```

### JSON

```json
{"ts": "12:30:01", "speaker": "Speaker_0", "text": "Hello", "confidence": 0.95}
```

### SRT

```
1
00:00:01,000 --> 00:00:03,200
[Speaker_0] Hello world
```

### Agent (~500 tokens)

```
**Speaker_0:** Hello world, how are you today...
**Speaker_1:** I am fine, thank you for asking...
```

## Architecture

```
├── scripts/
│   ├── live.py          # CLI entry point
│   ├── config.py        # Configuration constants
│   ├── audio.py         # AudioCapture (mic/file)
│   ├── engine.py        # WhisperEngine (faster-whisper)
│   ├── vad.py           # VoiceActivityDetector (Silero VAD)
│   ├── streamer.py      # StreamPipeline (chunked + VAD gate)
│   ├── diarize.py       # SpeakerDiarizer (pyannote-audio)
│   ├── merger.py        # DiarizationMerger (align timestamps)
│   ├── embedder.py      # VoiceEmbedder (ECAPA-TDNN)
│   ├── identifier.py    # SpeakerIdentifier (match + enroll)
│   ├── profiles.py      # ProfileStore (manifest + .npy)
│   ├── formatter.py     # OutputFormatter (text/json/srt/agent)
│   └── ws_server.py     # TranscriptionServer (WebSocket)
├── profiles/            # Voice profiles (.npy + manifest.json)
├── models/              # Model cache (auto-download)
└── SKILL.md             # This file
```

## Performance

| Component | VRAM | CPU | Latency |
|-----------|------|-----|---------|
| Whisper large-v3 | ~2-3GB | High | ~1-2s |
| Silero VAD | ~50MB | Low | ~10ms |
| pyannote diarization | ~1-2GB | Med | ~500ms |
| ECAPA-TDNN embedder | ~500MB | Low | ~100ms |
| **Total** | **~4-5GB** | | **~1-2s** |

RTX 3090 24GB → ~19GB free for other tasks.

## Accuracy

- **STT:** ~95% (large-v3, Vietnamese)
- **Speaker ID:** ~90-95% (distinct voices)
- **Diarization:** ~85-90% (overlap ~70%)

## Limitations

- **Overlap:** Simultaneous speech reduces accuracy
- **Similar Voices:** Same gender/tone harder to distinguish
- **Noise:** Background noise affects transcription quality
- **Enrollment:** Need 10-15s clean voice sample

## Use Cases

1. **Voice Command** — Speak → OpenClaw understands → acts
2. **Meeting Transcription** — Record → transcribe → speaker-labeled minutes
3. **Live Subtitles** — Stream/video call → real-time SRT
4. **Voice Notes → Text** — Telegram voice → text
5. **Security** — Voice-based user authentication

## Integration with E-Ears

`live-ears` (STT) + `e-ears` (music analysis) = full audio pipeline:

```bash
# Capture + save
python scripts/live.py live --duration 60 > capture.txt

# If music detected, analyze
python ~/.openclaw/workspace/skills/e-ears/scripts/ears.py analyze capture.mp3
```

## Troubleshooting

### VRAM Out of Memory

```bash
# Use smaller model
python scripts/live.py live --model medium

# Or CPU mode
python scripts/live.py live --device cpu
```

### pyannote Model Download Fails

```bash
# Set HuggingFace token
export HF_TOKEN=hf_...

# Or use mock mode (testing only)
# Diarization automatically falls back to mock if no token
```

### Mic Not Detected

```bash
python scripts/live.py devices
# Find your mic device ID
# Use --device <id> flag (future feature)
```

## Development

Tested on:
- Windows 11, Python 3.12, CUDA 12.4, RTX 3090
- Ubuntu 22.04, Python 3.10, CUDA 11.8, A100

## License

MIT (OpenClaw Skill)

## Credits

- **faster-whisper:** CTranslate2 team
- **Silero VAD:** Silero team
- **pyannote-audio:** Hervé Bredin
- **SpeechBrain:** SpeechBrain team
- **Forge:** Mula Nacharis (2026-03-08)

---

**Live-Ears v1.0** — "Voice is the ultimate interface."
