import asyncio
import json
import secrets
import sys
import time

import httpx
import websockets
from fastapi import Header, HTTPException, status as fastapi_status
import credentials
from settings import (
    CENTRAL_SERVER_URL,
    HEARTBEAT_INTERVAL,
    MODEL_KEY,
    WS_MAX_ATTEMPTS,
    WS_BACKOFF_BASE,
    WS_BACKOFF_MAX,
    CREDENTIALS_WAIT_INTERVAL,
)


def sanitize_header(value: str) -> str:
    return value.encode("latin-1", errors="replace").decode("latin-1")


def _wait_for_credentials(context: str) -> str:
    warned = False
    while True:
        api_key = credentials.get_api_key()
        if api_key and CENTRAL_SERVER_URL:
            if warned:
                print(f"✅ {context}: credentials ready")
            return api_key

        if not warned:
            if not CENTRAL_SERVER_URL:
                print(f"⚠️  {context}: CENTRAL_SERVER_URL is not configured")
            if not api_key:
                print(f"⚠️  {context}: provider API key not ready yet")
            print(f"   Retrying every {CREDENTIALS_WAIT_INTERVAL}s...")
            warned = True

        time.sleep(CREDENTIALS_WAIT_INTERVAL)


async def verify_server_identity(x_api_key: str = Header(None)):
    expected = credentials.get_shared_secret()
    provided = (x_api_key or "").strip()

    if not expected:
        print("🚫 Auth rejected: no shared secret loaded on this provider")
        raise HTTPException(
            status_code=fastapi_status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    if not secrets.compare_digest(provided.encode(), expected.encode()):
        print(
            f"🚫 Auth rejected: shared secret mismatch "
            f"(received len={len(provided)}, expected len={len(expected)})"
        )
        raise HTTPException(
            status_code=fastapi_status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    return provided


async def activate_with_token(token: str, central_url: str) -> dict:
    print("🔑 Activating provider with token...")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{central_url.rstrip('/')}/api/v1/providers/activate",
                json={"token": token},
            )
        except Exception as e:
            print(f"❌ Cannot reach central server: {e}")
            sys.exit(1)

    if response.status_code != 200:
        print(f"❌ Activation failed (HTTP {response.status_code}): {response.text}")
        print("   A token can only be used once — check for an existing")
        print("   credentials file, or request a new token.")
        sys.exit(1)

    try:
        data = response.json()
    except json.JSONDecodeError:
        print("❌ Activation response is not valid JSON")
        sys.exit(1)

    missing = [k for k in ("api_key", "server_to_provider_key") if not data.get(k)]
    if missing:
        print(f"❌ Activation response is missing field(s): {', '.join(missing)}")
        sys.exit(1)

    print(f"✅ Activated as: {data.get('provider_name', 'unknown')}")
    return data


def connect_to_central_registry():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    attempts = 0

    while True:
        api_key = _wait_for_credentials("WebSocket")

        ws_url = (
            CENTRAL_SERVER_URL.replace("http://", "ws://")
            .replace("https://", "wss://")
            .rstrip("/")
        )
        uri = f"{ws_url}/api/v1/providers/connect"
        websocket = None

        try:
            print(f"🔌 Connecting to central registry: {uri}...")
            websocket = loop.run_until_complete(
                websockets.connect(
                    uri,
                    additional_headers={
                        "X-Provider-Key": api_key,
                        "X-Model": MODEL_KEY,
                    },
                    ping_interval=20,
                    ping_timeout=60,
                )
            )
            attempts = 0
            print("✅ Connected to central server (presence active)")

            while True:
                loop.run_until_complete(websocket.recv())

        except Exception as e:
            attempts += 1
            print(f"❌ Registry disconnected (error: {e})")

            if attempts >= WS_MAX_ATTEMPTS:
                print(f"\n💥 Critical: registry unreachable after {attempts} attempts.")
                print(f"   Details: {e}")
                print("   Exiting process...")
                sys.exit(1)

            delay = min(WS_BACKOFF_BASE * (2 ** (attempts - 1)), WS_BACKOFF_MAX)
            print(
                f"🔄 Reconnecting in {delay}s (attempt {attempts}/{WS_MAX_ATTEMPTS})..."
            )
            time.sleep(delay)

        finally:
            if websocket is not None:
                try:
                    loop.run_until_complete(websocket.close())
                except Exception:
                    pass


def send_heartbeat_sync():
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    with httpx.Client(timeout=10.0, limits=limits) as client:
        while True:
            api_key = _wait_for_credentials("Heartbeat")

            try:
                response = client.post(
                    f"{CENTRAL_SERVER_URL.rstrip('/')}/api/v1/providers/heartbeat",
                    headers={"X-API-Key": api_key},
                    json=True,
                )
                response.raise_for_status()
                print("💓 Heartbeat sent")

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                print(f"⚠️  Network/DNS issue: {e}. Retrying next cycle...")
            except httpx.HTTPStatusError as e:
                print(f"🚫 Heartbeat rejected: HTTP {e.response.status_code}")
                if e.response.status_code in (401, 403):
                    print("   The central server does not recognize this API key.")
            except Exception as e:
                print(f"❓ Unexpected heartbeat error: {e}")

            time.sleep(HEARTBEAT_INTERVAL)
