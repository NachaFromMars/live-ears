# live-ears — Real-time speech-to-text with speaker identification

> GPU-accelerated live transcription using faster-whisper, Silero VAD, and pyannote diarization. Knows who is speaking, not just what was said.

[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-blueviolet)](https://github.com/NachaFromMars)

## Overview
live-ears provides real-time speech transcription with voice activity detection, speaker diarization, and speaker identification via voice enrollment. It uses faster-whisper (large-v3, 4× faster than Whisper), Silero VAD to skip silence, pyannote-audio for Speaker_0/1 labeling, and SpeechBrain ECAPA-TDNN for voice enrollment with cosine similarity matching. GPU recommended (CUDA, 4–6GB VRAM); CPU also works. Supports 99 languages and a WebSocket server for push delivery.

## Features
- **faster-whisper large-v3** — 4× Whisper speed, CUDA-accelerated
- **Silero VAD** — skip silence, save GPU cycles
- **Speaker diarization** — pyannote-audio (Speaker_0, Speaker_1...)
- **Voice enrollment** — SpeechBrain ECAPA-TDNN (192-dim embeddings)
- **Speaker ID** — cosine similarity matching, auto-save unknown voices
- **Output formats** — text, JSON, SRT, agent-optimized
- **WebSocket server** — push transcription to connected clients
- **99 languages** — Whisper native support

## Usage / Quick Start
```bash
pip install -r requirements.txt
python scripts/live.py live
```

## Trigger Keywords (OpenClaw)
live transcription, speech to text, speaker identification, real-time STT, voice recognition, diarization

---
Part of the [NachaFromMars](https://github.com/NachaFromMars) OpenClaw skill ecosystem.
