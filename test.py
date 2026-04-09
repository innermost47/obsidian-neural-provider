import hashlib
import httpx
import re
import sys

GITHUB_URL = "https://raw.githubusercontent.com/innermost47/obsidian-neural-provider/main/provider.py"


def hash_github():
    r = httpx.get(GITHUB_URL, timeout=10)
    if r.status_code != 200:
        print(f"GitHub fetch failed: HTTP {r.status_code}")
        return
    content = r.content.replace(b"\r\n", b"\n")
    h = hashlib.sha256(content).hexdigest()
    print(f"[GITHUB] {h}")
    print(f"[GITHUB] size: {len(content)} bytes")


def hash_local(path: str):
    with open(path, "rb") as f:
        content = f.read().replace(b"\r\n", b"\n")
    h = hashlib.sha256(content).hexdigest()
    print(f"[LOCAL]  {h}")
    print(f"[LOCAL]  size: {len(content)} bytes")


def diff_stripped(path: str):
    def clean_code_for_hashing(text: str) -> bytes:
        cleaned = re.sub(r"\s+", "", text)
        return cleaned.encode("utf-8")

    r = httpx.get(GITHUB_URL, timeout=10)
    github_stripped = clean_code_for_hashing(r.text)

    with open(path, "r", encoding="utf-8") as f:
        local_text = f.read()
    local_stripped = clean_code_for_hashing(local_text)

    h_github = hashlib.sha256(github_stripped).hexdigest()
    h_local = hashlib.sha256(local_stripped).hexdigest()

    print(f"[GITHUB stripped] {h_github} ({len(github_stripped)} bytes)")
    print(f"[LOCAL  stripped] {h_local} ({len(local_stripped)} bytes)")
    print(f"[MATCH] {h_github == h_local}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test.py github")
        print("  python test.py local <path>")
        print("  python test.py diff <path>")
        sys.exit(1)
    if sys.argv[1] == "github":
        hash_github()
    elif sys.argv[1] == "local":
        if len(sys.argv) < 3:
            print("Provide file path")
            sys.exit(1)
        hash_local(sys.argv[2])
    elif sys.argv[1] == "diff":
        if len(sys.argv) < 3:
            print("Provide file path")
            sys.exit(1)
        diff_stripped(sys.argv[2])
