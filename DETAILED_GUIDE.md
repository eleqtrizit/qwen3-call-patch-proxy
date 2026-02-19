# Qwen3 Call Patch Proxy — Detailed Guide

## Table of Contents

1. [Architecture](#architecture)
2. [Configuration](#configuration)
3. [Fix Rules Reference](#fix-rules-reference)
4. [Tool Name Inference](#tool-name-inference)
5. [XML Tool Call Conversion](#xml-tool-call-conversion)
6. [Management API](#management-api)
7. [Logging](#logging)
8. [Troubleshooting](#troubleshooting)
9. [Extending with Custom Fix Rules](#extending-with-custom-fix-rules)

---

## Architecture

```
┌──────────┐        ┌───────────────────────────┐        ┌──────────────────┐
│ OpenCode │──HTTP─▶│  qwen3-call-patch-proxy   │──HTTP─▶│  Qwen3 / vLLM   │
│ (client) │◀──SSE──│  :7999                    │◀──SSE──│  :8080           │
└──────────┘        │                           │        └──────────────────┘
                    │  per-request state        │
                    │  ┌─────────────────────┐  │
                    │  │ ToolBuffer          │  │
                    │  │  accumulates frags  │  │
                    │  └────────┬────────────┘  │
                    │           │ complete?      │
                    │           ▼                │
                    │  ┌─────────────────────┐  │
                    │  │ ToolFixEngine       │  │
                    │  │  applies YAML rules │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

### Request lifecycle

1. Client sends a request to the proxy.
2. Proxy forwards the request verbatim to the upstream model server.
3. Proxy streams the response back to the client, intercepting each `data:` SSE event.
4. For each event containing `delta.tool_calls`, arguments are accumulated in a `ToolBuffer`.
5. When `is_json_complete()` returns true for the buffer, the assembled arguments are parsed and passed through `ToolFixEngine`.
6. The fixed event (with a properly-formatted `call_<hex>` ID) is sent to the client.
7. On stream end (`[DONE]`), any incomplete buffers are closed out and sent.

---

## Configuration

### CLI arguments

See `qwen3-call-patch-proxy --help` or the README for the full argument reference. The relevant runtime paths are:

| Constant | Default | Description |
|---|---|---|
| `CONFIG_FILE` | bundled `tool_fixes.yaml` | Fix rules file |
| `LOG_FILE` | `{tempdir}/proxy_detailed.log` | Detailed log path |

### tool_fixes.yaml

Fix rules are declared in `tool_fixes.yaml`, bundled inside the package. The file is loaded at startup. You can trigger a live reload without restarting via `POST /_reload`.

#### File structure

```yaml
tools:
  <tool_name>:
    fixes:
      - name: "<rule_name>"
        parameter: "<parameter_name>"
        condition: "<condition>"
        action: "<action>"
        # action-specific fields (see below)

settings:
  buffer_timeout: 120       # seconds before abandoning an incomplete buffer
  max_buffer_size: 1048576  # max bytes per tool call buffer (1 MB)
  detailed_logging: true    # log fixed argument values to file
  case_sensitive_tools: false
```

---

## Fix Rules Reference

### Conditions

| Condition | Triggers when… |
|---|---|
| `is_string` | The parameter exists and its value is a `str` |
| `missing` | The parameter key is absent from the arguments object |
| `missing_or_empty` | The parameter is absent, `None`, `""`, `[]`, or `{}` |
| `exists` | The parameter key is present (any value) |
| `invalid_enum` | The parameter value is not in `valid_values` list |

### Actions

| Action | Description | Required extra fields |
|---|---|---|
| `parse_json_array` | `json.loads()` the string value into a list | `fallback_value` (used if parsing fails) |
| `parse_json_object` | `json.loads()` the string value into a dict | — |
| `convert_string_to_boolean` | Maps `"true"/"1"/"yes"/"on"` → `True`, else `False` | — |
| `set_default` | Sets the parameter to a fixed value | `default_value` |
| `remove_parameter` | Deletes the parameter from the arguments object | — |
| `convert_tool_to_write` | Changes the tool name from `read` to `write` when `content` is present | — |

### Built-in rules (from bundled `tool_fixes.yaml`)

| Tool | Rule | What it fixes |
|---|---|---|
| `todowrite` | `todos_string_to_array` | `todos` passed as JSON string → array |
| `bash` | `missing_description` | Injects `"Execute the given shell command"` when `description` is absent |
| `edit` | `replace_all_string_to_boolean` | `replaceAll`/`replace_all` string → boolean |
| `multiedit` | `edits_validation` | `edits` passed as JSON string → array |
| `glob` | `default_path` | Injects `"."` when `path` is missing |
| `grep` | `output_mode_validation` | Resets invalid `output_mode` to `"files_with_matches"` |
| `task` | `default_subagent_type` | Injects `"general-purpose"` when `subagent_type` is absent |
| `read` | `convert_read_with_content_to_write` | Converts `read+content` call to a `write` call |

---

## Tool Name Inference

When Qwen3-Coder emits fragmented tool call arguments without naming the tool in the initial event, the proxy infers the tool name by examining the accumulated argument keys:

| Inferred tool | Key pattern |
|---|---|
| `todowrite` | `"todos"` present |
| `bash` | `"command"` present, no `"edits"` |
| `multiedit` | `"file_path"` + `"edits"` |
| `edit` | `"filePath"/"file_path"` + `"oldString"/"old_string"` + `"newString"/"new_string"` |
| `grep` | `"pattern"` + `"output_mode"` |
| `glob` | `"pattern"` alone |
| `webfetch` | `"url"` + `"prompt"` |
| `websearch` | `"query"` present |
| `write` | `"content"` + `"file_path"/"filePath"` |
| `read` | `"file_path"/"filePath"` alone |
| `task` | `"description"` + `"prompt"` + `"subagent_type"` |
| `notebookedit` | `"notebook_path"` + `"new_source"` |
| `list` | `"path"` alone |

---

## XML Tool Call Conversion

Some Qwen3 checkpoints emit tool calls in an XML format instead of JSON:

```
<function=glob><parameter=pattern>**/*.py</parameter></function>
```

The proxy detects this in the content stream and converts it to a standard JSON tool call event before forwarding:

```json
{
  "index": 0,
  "id": "call_<24 hex chars>",
  "function": {
    "name": "glob",
    "arguments": "{\"pattern\": \"**/*.py\"}"
  }
}
```

The raw XML content is suppressed (replaced with an empty string) so it is not displayed to the user.

---

## Management API

### `GET /_health`

Returns current proxy state. Example response:

```json
{
  "status": "healthy",
  "active_requests": 1,
  "total_buffers": 2,
  "config_loaded": true,
  "target_host": "http://127.0.0.1:8080",
  "uptime": "unknown"
}
```

### `POST /_reload`

Reloads `tool_fixes.yaml` from disk without restarting the proxy. Useful when editing custom rules.

```bash
curl -X POST http://localhost:7999/_reload
```

Response:

```json
{"status": "success", "message": "Configuration reloaded"}
```

---

## Logging

| Logger | Destination | Content |
|---|---|---|
| `console` | stdout | Tool call events, fixes applied, XML conversions |
| `qwen3_call_patch_proxy` | `{tempdir}/proxy_detailed.log` | Full SSE event payloads, buffer state, all debug output |

The log file path is printed at startup. On Linux/macOS it is typically `/tmp/proxy_detailed.log`; on Windows it is `%TEMP%\proxy_detailed.log`.

To tail the log in real time:

```bash
tail -f /tmp/proxy_detailed.log
```

---

## Troubleshooting

### Proxy starts but tool calls still fail

1. Check `/_health` to confirm the proxy is running and `config_loaded` is `true`.
2. Tail the log file — fixed events are logged at `DEBUG` level with the full JSON payload.
3. Confirm your client points to port `7999`, not `8080` directly.

### `Config file not found, using defaults`

The bundled `tool_fixes.yaml` was not found inside the installed package. Reinstall cleanly:

```bash
uv tool uninstall qwen3-call-patch-proxy
uv tool install git+https://github.com/yourusername/qwen3-call-patch-proxy
```

### Buffer never completes (tool call never sent)

The model may be generating a tool call whose JSON never closes. Signs: no tool call arrives at the client, the log shows the buffer growing. Reduce `buffer_timeout` so incomplete buffers are flushed sooner, or check whether the model is producing valid JSON at all.

### `502 Backend server disconnected`

The upstream model server closed the connection mid-stream. This is normal for interrupted requests. If it happens consistently, check the model server logs.

---

## Extending with Custom Fix Rules

Add entries to `src/qwen3_call_patch_proxy/tool_fixes.yaml`, then call `POST /_reload` or restart.

**Example: ensure `write` tool always has a non-empty `content`**

```yaml
tools:
  write:
    fixes:
      - name: "empty_content_default"
        parameter: "content"
        condition: "missing_or_empty"
        action: "set_default"
        default_value: ""
        description: "Prevent null content on write"
```

**Example: fix a custom tool's enum parameter**

```yaml
tools:
  mymodel_search:
    fixes:
      - name: "fix_search_type"
        parameter: "search_type"
        condition: "invalid_enum"
        valid_values: ["semantic", "keyword", "hybrid"]
        action: "set_default"
        default_value: "semantic"
        description: "Default invalid search_type to semantic"
```

All fix conditions and actions documented in [Fix Rules Reference](#fix-rules-reference) are available.
