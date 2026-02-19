#!/usr/bin/env python3
"""
Debug test script that simulates the glob repeated call scenario.

Runs a mock LLM backend that sends SSE tool call events (in various
malformed/fragmented states) and captures what the proxy sends to the
"client" side so we can see exactly what Cursor receives.
"""

import asyncio
import json
import os
import sys
from aiohttp import web


# ---------------------------------------------------------------------------
# Scenarios – each returns a list of SSE chunks the mock backend will stream
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _delta(tool_calls=None, content=None, finish=None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    event: dict = {"choices": [{"delta": delta}]}
    if finish:
        event["choices"][0]["finish_reason"] = finish
    return event


def scenario_complete_named():
    """Single SSE event with a fully-formed glob call (happy path)."""
    return [
        _sse(_delta(tool_calls=[
            {"index": 0, "id": "call_aaaa111100001111aaaabbbb",
             "function": {"name": "glob",
                          "arguments": '{"glob_pattern": "**/*.py"}'}}
        ])),
        _sse(_delta(finish="tool_calls")),
        "data: [DONE]\n\n",
    ]


def scenario_fragmented_named():
    """Glob call split over multiple named events with the same call_id."""
    return [
        # Header chunk – name + id + first fragment
        _sse(_delta(tool_calls=[
            {"index": 0, "id": "call_bbbb222200002222bbbbcccc",
             "function": {"name": "glob",
                          "arguments": '{"glob_pat'}}
        ])),
        # Continuation – same id, same name, rest of arguments
        _sse(_delta(tool_calls=[
            {"index": 0, "id": "call_bbbb222200002222bbbbcccc",
             "function": {"name": "glob",
                          "arguments": 'tern": "**/*.py"}'}}
        ])),
        _sse(_delta(finish="tool_calls")),
        "data: [DONE]\n\n",
    ]


def scenario_qwen3_real_split():
    """
    Reproduces the actual Qwen3 behaviour seen in the logs:
    - SSE1: named call with id + name + arguments = '{'  (just the opening brace)
    - SSE2: anonymous fragment with the rest of the arguments
    The proxy must merge them and produce {"pattern":"**/*.py","path":"."}.
    """
    return [
        # Named call header – only the opening brace as arguments
        _sse(_delta(tool_calls=[
            {"index": 0, "id": "call_qwen3real0000000000000001",
             "function": {"name": "glob",
                          "arguments": "{"}}
        ])),
        # Anonymous fragment – rest of the JSON (no id, no name)
        _sse(_delta(tool_calls=[
            {"index": 0,
             "function": {"arguments": '"pattern": "**/*.py"}'}}
        ])),
        _sse(_delta(finish="tool_calls")),
        "data: [DONE]\n\n",
    ]


def scenario_incomplete_json_recovery():
    """
    Single named event whose arguments are missing the closing brace –
    the proxy must recover it at stream end.
    """
    return [
        _sse(_delta(tool_calls=[
            {"index": 0, "id": "call_cccc333300003333ccccdddd",
             "function": {"name": "glob",
                          "arguments": '{"glob_pattern": "**/*.py"'}}
        ])),
        _sse(_delta(finish="tool_calls")),
        "data: [DONE]\n\n",
    ]


def scenario_xml_glob():
    """Glob expressed as an XML tool call in the content stream."""
    return [
        _sse(_delta(content="<function=glob><parameter=glob_pattern>**/*.py</parameter></function>")),
        _sse(_delta(finish="tool_calls")),
        "data: [DONE]\n\n",
    ]


def scenario_duplicate_events():
    """
    Simulate what would cause Cursor to see the same call twice.
    The proxy should NOT emit duplicates.
    """
    call = {"index": 0, "id": "call_dddd444400004444ddddeeee",
            "function": {"name": "glob",
                         "arguments": '{"glob_pattern": "**/*.py"}'}}
    return [
        _sse(_delta(tool_calls=[call])),
        _sse(_delta(tool_calls=[call])),   # intentional duplicate from backend
        _sse(_delta(finish="tool_calls")),
        "data: [DONE]\n\n",
    ]


SCENARIOS = {
    "complete": scenario_complete_named,
    "fragmented": scenario_fragmented_named,
    "qwen3_real": scenario_qwen3_real_split,
    "recovery": scenario_incomplete_json_recovery,
    "xml": scenario_xml_glob,
    "duplicate": scenario_duplicate_events,
}

# ---------------------------------------------------------------------------
# Mock backend server
# ---------------------------------------------------------------------------
MOCK_PORT = int(os.environ.get("MOCK_PORT", 8080))
PROXY_PORT = int(os.environ.get("PROXY_PORT", 7999))

current_scenario_chunks: list[str] = []


async def mock_llm_handler(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(status=200, reason="OK")
    resp.headers["Content-Type"] = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    await resp.prepare(request)

    print(f"  [mock-backend] Streaming {len(current_scenario_chunks)} chunks", flush=True)
    for chunk in current_scenario_chunks:
        await resp.write(chunk.encode())
        await asyncio.sleep(0.01)

    await resp.write_eof()
    return resp


# ---------------------------------------------------------------------------
# Client: collect what the proxy sends back
# ---------------------------------------------------------------------------

async def collect_proxy_output() -> list[dict]:
    import aiohttp

    captured: list[dict] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions",
                json={"model": "qwen3", "messages": [], "stream": True},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        captured.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        print(f"  [client] Error collecting proxy output: {exc}", flush=True)
    return captured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_scenario(name: str, chunks: list[str]):
    global current_scenario_chunks
    current_scenario_chunks = chunks

    print(f"\n{'='*60}", flush=True)
    print(f"SCENARIO: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    events = await collect_proxy_output()

    tool_calls_seen: list[dict] = []
    for ev in events:
        delta = ev.get("choices", [{}])[0].get("delta", {})
        tcs = delta.get("tool_calls", [])
        for tc in tcs:
            fn = tc.get("function", {})
            tool_calls_seen.append({
                "id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
            })

    if tool_calls_seen:
        print(f"  Tool calls sent to client ({len(tool_calls_seen)} total):", flush=True)
        for i, tc in enumerate(tool_calls_seen):
            print(f"    [{i}] name={tc['name']!r}  id={tc['id']!r}", flush=True)
            try:
                parsed = json.loads(tc["arguments"] or "{}")
                print(f"        args={json.dumps(parsed)}", flush=True)
            except Exception:
                print(f"        args(raw)={tc['arguments']!r}", flush=True)

        # Duplicate check
        sigs = [f"{tc['name']}|{tc['arguments']}" for tc in tool_calls_seen]
        dups = [s for s in sigs if sigs.count(s) > 1]
        if dups:
            print(f"\n  ⚠️  DUPLICATES DETECTED: {set(dups)}", flush=True)
        else:
            print(f"\n  ✅  No duplicates.", flush=True)
    else:
        print("  ❌  No tool calls captured from proxy output!", flush=True)


async def main():
    # Start mock backend
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", mock_llm_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", MOCK_PORT)
    await site.start()
    print(f"Mock backend listening on :{MOCK_PORT}", flush=True)

    # Give proxy time to start if it's not already running
    await asyncio.sleep(0.5)

    # Run selected or all scenarios
    scenarios_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(SCENARIOS.keys())
    for name in scenarios_to_run:
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name!r}. Available: {list(SCENARIOS.keys())}", flush=True)
            continue
        await run_scenario(name, SCENARIOS[name]())
        await asyncio.sleep(0.3)

    await runner.cleanup()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
