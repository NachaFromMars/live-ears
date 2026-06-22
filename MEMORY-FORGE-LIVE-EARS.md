# MEMORY-FORGE — LIVE-EARS 🎙️

## Project Info
- Skill: live-ears (Realtime STT + Speaker Identification)
- Location: ~/.openclaw/workspace/skills/live-ears/
- Forge method: SuperBuild-OpenBuild (đơn lập, 4000-6000 tok/step, thẩm định mỗi step)
- Model: claudible/claude-opus-4.6
- Started: 2026-07-14

## Architecture
- **Engine:** faster-whisper (CTranslate2) — large-v3 model
- **VAD:** Silero VAD 5.1 — filter silence, save GPU
- **Diarization:** pyannote-audio 3.3+ — tách speaker segments
- **Identification:** SpeechBrain ECAPA-TDNN — voice embedding 192-dim
- **Audio:** sounddevice — mic capture 16kHz mono
- **Output:** text/json/srt/agent formats + WebSocket server
- **VRAM:** ~4-5GB total (3090 24GB = dư sức)

## Forge Plan — 14 Steps (SuperBuild-OpenBuild)

### PHASE A: FOUNDATION (Steps 1-4)
| Step | Nội dung | Files | Tokens | Status |
|------|----------|-------|--------|--------|
| 1.1 | Project skeleton + requirements.txt + __init__.py | 4 files | 4000 | ✅ PASS |
| 1.2 | AudioCapture class (mic/file input, ring buffer, 16kHz mono) | 1 file | 5000 | ✅ PASS |
| 1.3 | WhisperEngine class (faster-whisper wrapper, model load, transcribe) | 1 file | 5000 | ✅ PASS |
| 1.4 | Basic CLI (live.py) — mic → whisper → stdout, test pipeline | 1 file | 4000 | ✅ PASS |

### PHASE B: VAD + STREAMING (Steps 2.1-2.3)
| Step | Nội dung | Files | Tokens | Status |
|------|----------|-------|--------|--------|
| 2.1 | Silero VAD wrapper (load model, detect speech, filter silence) | 1 file | 4500 | ✅ PASS |
| 2.2 | StreamPipeline (chunked buffer + VAD gate + overlap handling) | 1 file | 5500 | ✅ PASS |
| 2.3 | WebSocket server (ws://localhost:8765 push text) | 1 file | 4000 | ✅ PASS |

### PHASE C: SPEAKER DIARIZATION (Steps 3.1-3.2)
| Step | Nội dung | Files | Tokens | Status |
|------|----------|-------|--------|--------|
| 3.1 | PyAnnote diarization wrapper (segment → speaker labels) | 1 file | 5000 | ✅ PASS |
| 3.2 | Merger — align diarization + transcription by timestamp | 1 file | 5000 | ✅ PASS |

### PHASE D: VOICE ENROLLMENT + ID (Steps 4.1-4.3)
| Step | Nội dung | Files | Tokens | Status |
|------|----------|-------|--------|--------|
| 4.1 | SpeechBrain ECAPA-TDNN embedding extractor | 1 file | 4500 | ✅ PASS |
| 4.2 | ProfileStore (manifest.json, save/load/delete/rename .npy) | 1 file | 5000 | ✅ PASS |
| 4.3 | Identifier (cosine similarity match, threshold, unknown auto-save) | 1 file | 5500 | ✅ PASS |

### PHASE E: POLISH + SHIP (Steps 5.1-5.2)
| Step | Nội dung | Files | Tokens | Status |
|------|----------|-------|--------|--------|
| 5.1 | Formatter (text/json/srt/agent output) + CLI commands hoàn chỉnh | 2 files | 5500 | ⬜ PENDING |
| 5.2 | SKILL.md + tests + package | 4 files | 5000 | ⬜ PENDING |

## Quy Trình Mỗi Step
1. **Code** — viết đơn lập, 4000-6000 tokens, focus chất lượng
2. **Test** — chạy thử, verify hoạt động
3. **Thẩm Định** — review code quality, edge cases, error handling
4. **PASS** → cập nhật MEMORY-FORGE → tiếp step sau
5. **FAIL** → fix → test lại → PASS mới tiếp

## Thẩm Định Criteria (mỗi step)
- [ ] Code chạy không lỗi
- [ ] Error handling đầy đủ (try/except, graceful shutdown)
- [ ] Type hints + docstrings
- [ ] Edge cases covered
- [ ] Tương thích với các step trước
- [ ] VRAM/memory efficient
- [ ] Windows compatible (3090 machine)

## Dependencies
```
faster-whisper>=1.1.0
silero-vad>=5.1
pyannote.audio>=3.3.0
speechbrain>=1.0.0
sounddevice>=0.5.0
numpy>=1.26.0
websockets>=12.0
torch>=2.0.0
torchaudio>=2.0.0
```

## Progress Log
<!-- Cập nhật sau mỗi step hoàn thành -->

---
