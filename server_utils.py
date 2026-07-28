import asyncio
import os
import time
import json
import httpx
from fastapi import HTTPException, Header, status as fastapi_status
import websockets
import sys
from settings import (
    SHARED_SECRET,
    CREDENTIALS_FILE,
    CENTRAL_SERVER_URL,
    PROVIDER_API_KEY,
    MODEL_KEY,
    HEARTBEAT_INTERVAL,
)


def sanitize_header(value: str) -> str:
    return value.encode("latin-1", errors="replace").decode("latin-1")


async def verify_server_identity(x_api_key: str = Header(None)):
    if not SHARED_SECRET or x_api_key != SHARED_SECRET:
        raise HTTPException(
            status_code=fastapi_status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return x_api_key


async def activate_with_token(token: str, central_url: str) -> dict:
    if os.path.exists(CREDENTIALS_FILE):
        print("🔑 Loading saved credentials...")
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)

    print("🔑 Activating provider with token...")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{central_url.rstrip('/')}/api/v1/providers/activate",
                json={"token": token},
            )
            if response.status_code != 200:
                print(f"❌ Activation failed: {response.text}")
                sys.exit(1)
            data = response.json()
            os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump(data, f)
            print(f"✅ Activated as: {data['provider_name']}")
            return data
        except Exception as e:
            print(f"❌ Cannot reach central server: {e}")
            sys.exit(1)


def connect_to_central_registry():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    attempts = 0

    while True:
        if not CENTRAL_SERVER_URL or not PROVIDER_API_KEY:
            print("⚠️ WebSocket: credentials not ready, retrying in 10s...")
            time.sleep(10)
            continue
        try:
            ws_url = (
                CENTRAL_SERVER_URL.replace("http://", "ws://")
                .replace("https://", "wss://")
                .rstrip("/")
            )
            uri = f"{ws_url}/api/v1/providers/connect"

            headers = {
                "X-Provider-Key": PROVIDER_API_KEY,
                "X-Model": MODEL_KEY,
            }
            print(f"🔌 Attempting to connect to the central registry: {uri}...")
            websocket = loop.run_until_complete(
                websockets.connect(
                    uri,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=60,
                )
            )
            attempts = 0
            print("✅ Connected to the central server (Active presence)")

            while True:
                loop.run_until_complete(websocket.recv())

        except Exception as e:
            attempts += 1
            if attempts > 3:
                print(
                    f"\nCritical Error: Failed to connect to registry after {attempts} attempts."
                )
                print(f"Details: {e}")
                print("Exiting process...")
                exit(1)
            print(f"❌ Register disconnection (Error: {e})")
            print("🔄 Attempting to reconnect in 10 seconds...")
            time.sleep(10)


def send_heartbeat_sync():
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    with httpx.Client(timeout=10.0, limits=limits) as client:
        while True:
            if not CENTRAL_SERVER_URL or not PROVIDER_API_KEY:
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            try:
                response = client.post(
                    f"{CENTRAL_SERVER_URL.rstrip('/')}/api/v1/providers/heartbeat",
                    headers={"X-API-Key": PROVIDER_API_KEY},
                    json=True,
                )
                response.raise_for_status()
                print(f"💓 Heartbeat sent")

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                print(f"⚠️  Network/DNS issue: {e}. Retrying next time...")
            except httpx.HTTPStatusError as e:
                print(f"🚫 Server returned an error: {e.response.status_code}")
            except Exception as e:
                print(f"❓ Unexpected error: {e}")

            time.sleep(HEARTBEAT_INTERVAL)
