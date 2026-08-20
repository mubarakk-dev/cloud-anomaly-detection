"""Fail fast if the running container does not expose a healthy model."""

import json
import urllib.request


with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as response:
    payload = json.load(response)
assert payload["status"] == "ok"
print(json.dumps(payload, indent=2))
