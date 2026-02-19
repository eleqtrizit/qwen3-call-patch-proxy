# Qwen3 Call Patch Proxy

An HTTP proxy that fixes malformed tool calls from Qwen3-Coder LLM models before forwarding them to OpenCode or any OpenAI-compatible client.

> Primarily developed for [OpenCode](https://github.com/sst/opencode), but works with any client using the OpenAI streaming API.

## What It Fixes

- **Fragmented tool calls** — consolidates arguments split across multiple SSE events
- **Wrong parameter types** — converts string-encoded arrays, objects, and booleans to their proper types
- **Missing parameters** — injects sensible defaults for required fields the model omits
- **Invalid tool call IDs** — rewrites IDs to the `call_<hex>` format OpenCode expects
- **XML-format tool calls** — converts `<function=...>` syntax to JSON tool call format

## Tested Models

- `unsloth/Qwen3-Coder-30B-A3B-Instruct`
- `cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ`

## Demo

<img src="images/OpenCode-Qwen3-Coder-GameOfLife.png" alt="OpenCode creating Game of Life with Qwen3-Coder" width="600">

*OpenCode successfully completing a Conway's Game of Life implementation via the proxy*

---

## Installation

### Via uvx (no install)

```bash
uvx git+https://github.com/eleqtrizit/qwen3-call-patch-proxy
```

### Via uv (recommended)

```bash
uv tool install git+https://github.com/eleqtrizit/qwen3-call-patch-proxy
```

### Via pip

```bash
pip install git+https://github.com/eleqtrizit/qwen3-call-patch-proxy
```

---

## Usage

```
OpenCode ──→ Proxy :7999 ──→ Qwen3 Server :8080
              [fixes SSE stream]
```

Start the proxy (defaults: listen on `0.0.0.0:7999`, forward to `http://127.0.0.1:8080`):

```bash
qwen3-call-patch-proxy [--host HOST] [--port PORT] [--target-url URL] [--verbose]
```

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host address to listen on |
| `--port` | `7999` | Port to listen on |
| `--target-url` | `http://127.0.0.1:8080` | Target URL to proxy requests to |
| `--verbose` | `false` | Log each SSE stream response returned to the client |

Example with custom target:

```bash
qwen3-call-patch-proxy --port 7999 --target-url http://127.0.0.1:8080
```

Then configure OpenCode to use `http://127.0.0.1:7999/v1` as the base URL:

```json
{
  "provider": {
    "qwen3-coder": {
      "name": "Qwen3 Coder (via proxy)",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ": {}
      },
      "options": {
        "baseURL": "http://127.0.0.1:7999/v1"
      }
    }
  }
}
```

---

## Management Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/_health` | GET | Returns proxy status and buffer counts |
| `/_reload` | POST | Reloads `tool_fixes.yaml` without restart |

---

## Logging

Detailed logs are written to `{system_temp}/proxy_detailed.log` (e.g. `/tmp/proxy_detailed.log`).

---

## Documentation

See [DETAILED_GUIDE.md](DETAILED_GUIDE.md) for configuration reference, fix rules, architecture details, and troubleshooting.

