#!/usr/bin/env python3
"""
Live-Ears CLI — Realtime STT + Speaker ID.

Commands:
    live      - Real-time mic transcription
    file      - Transcribe audio file
    enroll    - Enroll voice profile
    profiles  - Manage profiles (list/delete/rename)
    devices   - List audio devices
    serve     - WebSocket server
"""

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import *
from scripts.audio import AudioCapture
from scripts.engine import WhisperEngine
from scripts.identifier import SpeakerIdentifier
from scripts.formatter import OutputFormatter


def cmd_live(args):
    """Real-time mic transcription."""
    engine = WhisperEngine(model=args.model, language=args.lang)
    engine.load()

    identifier = None
    if args.identify:
        identifier = SpeakerIdentifier(threshold=args.threshold)
        identifier.load()
        print(f"[live-ears] Speaker ID: {identifier.count_profiles()} profiles loaded", file=sys.stderr)

    capture = AudioCapture(sample_rate=SAMPLE_RATE, buffer_seconds=args.buffer)

    running = [True]
    def on_signal(sig, frame):
        running[0] = False
        print("\n[live-ears] Stopping...", file=sys.stderr)
    signal.signal(signal.SIGINT, on_signal)

    print("[live-ears] Listening... (Ctrl+C to stop)", file=sys.stderr)
    capture.start_mic()

    try:
        while running[0]:
            chunk = capture.get_chunk(timeout=2.0)
            if chunk is None:
                continue

            result = engine.transcribe(chunk, language=args.lang)
            if not result.text:
                continue

            speaker = None
            if identifier:
                speaker, conf = identifier.identify(chunk)
                if conf < args.threshold:
                    speaker = "Unknown"

            ts = datetime.now().strftime("%H:%M:%S")

            if args.format == "json":
                import json
                obj = {"ts": ts, "text": result.text, "lang": result.language, "confidence": round(result.probability, 3)}
                if speaker:
                    obj["speaker"] = speaker
                print(json.dumps(obj, ensure_ascii=False), flush=True)
            else:
                prefix = f"[{ts}]"
                if speaker:
                    prefix += f" [{speaker}]"
                print(f"{prefix} {result.text}", flush=True)

    finally:
        capture.stop_mic()
        print("[live-ears] Stopped.", file=sys.stderr)


def cmd_file(args):
    """Transcribe audio file."""
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    engine = WhisperEngine(model=args.model, language=args.lang)
    engine.load()

    identifier = None
    if args.identify:
        identifier = SpeakerIdentifier(threshold=args.threshold)
        identifier.load()

    capture = AudioCapture(sample_rate=SAMPLE_RATE)

    print(f"[live-ears] Transcribing: {args.path}", file=sys.stderr)
    t0 = time.time()

    results = []
    chunk_seconds = args.buffer if args.chunked else None

    for chunk in capture.read_file(args.path, chunk_seconds=chunk_seconds):
        result = engine.transcribe(chunk, language=args.lang)
        if not result.text:
            continue

        if identifier:
            speaker, conf = identifier.identify(chunk)
            result.speaker = speaker if conf >= args.threshold else "Unknown"

        results.append(result)

    # Format output
    formatter = OutputFormatter()

    if args.format == "json":
        print(formatter.format_json(results))
    elif args.format == "srt":
        print(formatter.format_srt(results))
    elif args.format == "agent":
        print(formatter.format_agent(results))
    else:
        print(formatter.format_text(results, include_speakers=args.identify))

    elapsed = time.time() - t0
    print(f"\n[live-ears] Done in {elapsed:.1f}s", file=sys.stderr)


def cmd_enroll(args):
    """Enroll voice profile."""
    identifier = SpeakerIdentifier()
    identifier.load()

    capture = AudioCapture(sample_rate=SAMPLE_RATE)

    if args.file:
        # Enroll from file
        if not Path(args.file).exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        audio_chunks = list(capture.read_file(args.file))
        audio = np.concatenate(audio_chunks)
    else:
        # Record from mic
        print(f"🎙️  {args.name}, please speak naturally for {args.duration} seconds...", file=sys.stderr)
        import sounddevice as sd
        audio = sd.rec(
            int(args.duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

    # Enroll
    profile = identifier.enroll(args.name, audio, overwrite=args.overwrite)
    print(f"✅ Profile '{args.name}' saved (ID: {profile.id})", file=sys.stderr)


def cmd_profiles(args):
    """Manage profiles."""
    identifier = SpeakerIdentifier()

    if args.action == "list":
        profiles = identifier.list_profiles()
        if not profiles:
            print("No profiles enrolled.")
            return

        print(f"📋 {len(profiles)} profile(s):\n")
        for p in profiles:
            print(f"  • {p.name} (ID: {p.id})")
            print(f"    Created: {p.created}")
            if p.updated:
                print(f"    Updated: {p.updated}")
            print()

    elif args.action == "delete":
        identifier.delete_profile(args.name)
        print(f"✅ Deleted: {args.name}")

    elif args.action == "rename":
        identifier.rename_profile(args.old_name, args.new_name)
        print(f"✅ Renamed: {args.old_name} → {args.new_name}")

    elif args.action == "test":
        identifier.load()
        capture = AudioCapture(sample_rate=SAMPLE_RATE)

        print(f"🎙️  Say something for {args.duration} seconds to test recognition...", file=sys.stderr)
        import sounddevice as sd
        audio = sd.rec(
            int(args.duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        name, confidence = identifier.identify(audio)
        print(f"Recognized: {name} (confidence: {confidence:.2f})")


def cmd_devices(args):
    """List audio devices."""
    print(AudioCapture.list_devices())


def cmd_serve(args):
    """WebSocket server."""
    from scripts.ws_server import TranscriptionServer
    import asyncio

    server = TranscriptionServer(host=args.host, port=args.port)

    async def run():
        await server.start()
        print(f"[WS] Server running on {server.address}")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()

    asyncio.run(run())


def build_parser():
    parser = argparse.ArgumentParser(prog="live-ears", description="Realtime STT + Speaker ID")
    sub = parser.add_subparsers(dest="command", help="Command")

    # live
    p_live = sub.add_parser("live", help="Real-time mic transcription")
    p_live.add_argument("--lang", type=str, default=WHISPER_LANGUAGE)
    p_live.add_argument("--model", type=str, default=WHISPER_MODEL)
    p_live.add_argument("--buffer", type=float, default=BUFFER_DURATION_S)
    p_live.add_argument("--format", type=str, default=DEFAULT_FORMAT, choices=VALID_FORMATS)
    p_live.add_argument("--identify", action="store_true", help="Enable speaker ID")
    p_live.add_argument("--threshold", type=float, default=ID_THRESHOLD)
    p_live.set_defaults(func=cmd_live)

    # file
    p_file = sub.add_parser("file", help="Transcribe audio file")
    p_file.add_argument("path", type=str)
    p_file.add_argument("--lang", type=str, default=WHISPER_LANGUAGE)
    p_file.add_argument("--model", type=str, default=WHISPER_MODEL)
    p_file.add_argument("--buffer", type=float, default=BUFFER_DURATION_S)
    p_file.add_argument("--chunked", action="store_true")
    p_file.add_argument("--format", type=str, default=DEFAULT_FORMAT, choices=VALID_FORMATS)
    p_file.add_argument("--identify", action="store_true")
    p_file.add_argument("--threshold", type=float, default=ID_THRESHOLD)
    p_file.set_defaults(func=cmd_file)

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll voice profile")
    p_enroll.add_argument("name", type=str)
    p_enroll.add_argument("--duration", type=float, default=ENROLL_DURATION_S)
    p_enroll.add_argument("--file", type=str, help="Enroll from audio file")
    p_enroll.add_argument("--overwrite", action="store_true")
    p_enroll.set_defaults(func=cmd_enroll)

    # profiles
    p_prof = sub.add_parser("profiles", help="Manage profiles")
    p_prof.add_argument("action", type=str, choices=["list", "delete", "rename", "test"])
    p_prof.add_argument("name", type=str, nargs="?")
    p_prof.add_argument("new_name", type=str, nargs="?")
    p_prof.add_argument("old_name", type=str, nargs="?")
    p_prof.add_argument("--duration", type=float, default=5.0)
    p_prof.set_defaults(func=cmd_profiles)

    # devices
    p_dev = sub.add_parser("devices", help="List audio devices")
    p_dev.set_defaults(func=cmd_devices)

    # serve
    p_serve = sub.add_parser("serve", help="WebSocket server")
    p_serve.add_argument("--host", type=str, default=WS_HOST)
    p_serve.add_argument("--port", type=int, default=WS_PORT)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main():
    import numpy as np  # Needed for enroll
    import time
    from datetime import datetime

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
