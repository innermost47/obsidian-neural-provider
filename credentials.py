import json, os
from pathlib import Path
from settings import CREDENTIALS_FILE
from settings import PROVIDER_API_KEY as ENV_API_KEY
from settings import SHARED_SECRET as ENV_SHARED_SECRET

_state = {"api_key": "", "shared_secret": ""}


def get_api_key() -> str:
    return _state["api_key"]


def get_shared_secret() -> str:
    return _state["shared_secret"]


def set_credentials(api_key: str, shared_secret: str, persist: bool = True):
    _state["api_key"] = (api_key or "").strip()
    _state["shared_secret"] = (shared_secret or "").strip()
    if not persist:
        return
    p = Path(CREDENTIALS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "api_key": _state["api_key"],
                "server_to_provider_key": _state["shared_secret"],
            }
        )
    )
    os.replace(tmp, p)
    os.chmod(p, 0o600)


def load_from_file() -> bool:
    p = Path(CREDENTIALS_FILE)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        api_key = (data.get("api_key") or "").strip()
        secret = (data.get("server_to_provider_key") or "").strip()
        if not api_key or not secret:
            print(f"⚠️  {CREDENTIALS_FILE} incomplete, ignored")
            return False
        set_credentials(api_key, secret, persist=False)
        return True
    except Exception as e:
        print(f"⚠️ Unable to read {CREDENTIALS_FILE}: {e}")
        return False


def load_from_env() -> bool:
    if ENV_API_KEY and ENV_SHARED_SECRET:
        set_credentials(ENV_API_KEY, ENV_SHARED_SECRET, persist=False)
        return True
    return False
