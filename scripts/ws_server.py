"""
WebSocket Server — Push transcription text to connected clients.

Enables real-time subtitle display, live transcription UI, and integration
with external apps via WebSocket protocol.
"""

import asyncio
import sys
import json
from typing import Set, Optional, Callable
from datetime import datetime

import websockets
from websockets.server import WebSocketServerProtocol

from .config import WS_HOST, WS_PORT


class TranscriptionServer:
    """
    WebSocket server for pushing transcription results.
    
    Usage:
        server = TranscriptionServer()
        
        async def main():
            await server.start()
            
            # Push text to all clients
            await server.broadcast("Hello world")
            
            # Or push structured data
            await server.broadcast_json({
                "text": "Hello",
                "ts": "12:30:45",
                "lang": "en"
            })
            
            await server.stop()
        
        asyncio.run(main())
    """

    def __init__(
        self,
        host: str = WS_HOST,
        port: int = WS_PORT,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ):
        self.host = host
        self.port = port
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect

        self._clients: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._running = False

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            return

        print(f"[WS] Starting server on ws://{self.host}:{self.port}", file=sys.stderr)

        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )

        self._running = True
        print(f"[WS] Server started", file=sys.stderr)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        # Close all client connections
        if self._clients:
            await asyncio.gather(
                *[client.close() for client in self._clients],
                return_exceptions=True,
            )

        # Stop server
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        print("[WS] Server stopped", file=sys.stderr)

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a new client connection."""
        client_addr = websocket.remote_address
        print(f"[WS] Client connected: {client_addr}", file=sys.stderr)

        self._clients.add(websocket)

        if self.on_connect:
            try:
                self.on_connect(websocket)
            except Exception as e:
                print(f"[WS] on_connect error: {e}", file=sys.stderr)

        try:
            # Keep connection alive, client doesn't send messages (we only push)
            async for message in websocket:
                # Optional: handle client commands (e.g., "ping")
                if message == "ping":
                    await websocket.send("pong")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            print(f"[WS] Client disconnected: {client_addr}", file=sys.stderr)

            if self.on_disconnect:
                try:
                    self.on_disconnect(websocket)
                except Exception as e:
                    print(f"[WS] on_disconnect error: {e}", file=sys.stderr)

    async def broadcast(self, message: str) -> int:
        """
        Send text message to all connected clients.
        
        Returns:
            Number of clients that received the message.
        """
        if not self._clients:
            return 0

        tasks = [client.send(message) for client in self._clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        return success_count

    async def broadcast_json(self, data: dict) -> int:
        """
        Send JSON object to all connected clients.
        
        Returns:
            Number of clients that received the message.
        """
        message = json.dumps(data, ensure_ascii=False)
        return await self.broadcast(message)

    async def send_to(self, client: WebSocketServerProtocol, message: str) -> None:
        """Send message to a specific client."""
        try:
            await client.send(message)
        except Exception as e:
            print(f"[WS] Send error: {e}", file=sys.stderr)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def address(self) -> str:
        return f"ws://{self.host}:{self.port}"


# ── Helper for blocking usage in threads ────────────────────────────────

class TranscriptionServerSync:
    """
    Synchronous wrapper for TranscriptionServer for use in threaded environments.
    
    Usage:
        server = TranscriptionServerSync()
        server.start_in_thread()
        
        # Push from any thread
        server.push("Hello world")
        server.push_json({"text": "Hello", "ts": "12:30:45"})
        
        server.stop()
    """

    def __init__(self, host: str = WS_HOST, port: int = WS_PORT):
        self.host = host
        self.port = port
        self._server = TranscriptionServer(host, port)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = None

    def start_in_thread(self) -> None:
        """Start server in a background thread."""
        import threading

        def run_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._server.start())
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()

        # Wait for server to start
        import time
        time.sleep(0.5)

    def stop(self) -> None:
        """Stop the server."""
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)

    def push(self, message: str) -> None:
        """Push text message to all clients (thread-safe)."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._server.broadcast(message), self._loop)

    def push_json(self, data: dict) -> None:
        """Push JSON object to all clients (thread-safe)."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._server.broadcast_json(data), self._loop)

    @property
    def is_running(self) -> bool:
        return self._server.is_running

    @property
    def client_count(self) -> int:
        return self._server.client_count

    @property
    def address(self) -> str:
        return self._server.address
