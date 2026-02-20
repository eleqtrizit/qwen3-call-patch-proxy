#!/usr/bin/env python3
"""
Test streaming XML tool call detection across multiple SSE events
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import process_sse_event, RequestState, request_states
import asyncio
import json
import uuid

async def test_streaming_xml_detection():
    """Test that XML tool calls are detected when streamed across multiple events"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("Testing streaming XML detection:")
    
    # Simulate the exact streaming pattern from the logs
    events = [
        # Regular content first
        {"choices": [{"delta": {"content": "I'll scan the source code to identify areas for improvement. Let me start by examining the project structure and key files.\n\n"}, "index": 0}]},
        
        # XML fragments exactly as they appear in the logs
        {"choices": [{"delta": {"content": "<"}, "index": 0}]},
        {"choices": [{"delta": {"content": "function"}, "index": 0}]},
        {"choices": [{"delta": {"content": "=g"}, "index": 0}]},
        {"choices": [{"delta": {"content": "lob"}, "index": 0}]},
        {"choices": [{"delta": {"content": ">\n"}, "index": 0}]},
        {"choices": [{"delta": {"content": "<"}, "index": 0}]},
        {"choices": [{"delta": {"content": "parameter"}, "index": 0}]},
        {"choices": [{"delta": {"content": "="}, "index": 0}]},
        {"choices": [{"delta": {"content": "pattern"}, "index": 0}]},
        {"choices": [{"delta": {"content": ">\n"}, "index": 0}]},
        {"choices": [{"delta": {"content": "src"}, "index": 0}]},
        {"choices": [{"delta": {"content": "/"}, "index": 0}]},
        {"choices": [{"delta": {"content": "semantic"}, "index": 0}]},
        {"choices": [{"delta": {"content": "_h"}, "index": 0}]},
        {"choices": [{"delta": {"content": "ar"}, "index": 0}]},
        {"choices": [{"delta": {"content": "vest"}, "index": 0}]},
        {"choices": [{"delta": {"content": "/src"}, "index": 0}]},
        {"choices": [{"delta": {"content": "/common"}, "index": 0}]},
        {"choices": [{"delta": {"content": "/cli"}, "index": 0}]},
        {"choices": [{"delta": {"content": "/"}, "index": 0}]},
        {"choices": [{"delta": {"content": "commands"}, "index": 0}]},
        {"choices": [{"delta": {"content": "/*."}, "index": 0}]},
        {"choices": [{"delta": {"content": "py"}, "index": 0}]},
        {"choices": [{"delta": {"content": "\n"}, "index": 0}]},
        {"choices": [{"delta": {"content": "</"}, "index": 0}]},
        {"choices": [{"delta": {"content": "parameter"}, "index": 0}]},
        {"choices": [{"delta": {"content": ">\n"}, "index": 0}]},
        {"choices": [{"delta": {"content": "</"}, "index": 0}]},
        {"choices": [{"delta": {"content": "function"}, "index": 0}]},
        {"choices": [{"delta": {"content": ">\n"}, "index": 0}]},
        {"choices": [{"delta": {"content": "</tool_call>"}, "index": 0}]},
    ]
    
    tool_call_generated = False
    tool_call_event = None
    
    # Process events one by one
    for i, event in enumerate(events):
        print(f"Processing event {i+1}: {event['choices'][0]['delta']['content'][:20]}...")
        result = await process_sse_event(event, request_id)
        
        # Check if XML was converted to tool call
        delta = result["choices"][0]["delta"]
        if "tool_calls" in delta and delta["tool_calls"]:
            tool_call_generated = True
            tool_call_event = result
            print(f"  → XML converted to tool call at event {i+1}!")
            break
        elif delta.get("content") == "":
            print(f"  → Content suppressed (XML detected but not complete yet)")
        else:
            print(f"  → Content passed through: {delta.get('content', '')[:30]}...")
    
    assert tool_call_generated, "No tool call was generated from streaming XML"
    
    # Verify the tool call
    tool_call = tool_call_event["choices"][0]["delta"]["tool_calls"][0]
    
    print(f"\nGenerated tool call:")
    print(f"  Function name: {tool_call['function']['name']}")
    print(f"  Call ID: {tool_call['id']}")
    print(f"  Arguments: {tool_call['function']['arguments']}")
    
    # Parse arguments
    args = json.loads(tool_call['function']['arguments'])
    expected_pattern = "src/semantic_harvest/src/common/cli/commands/*.py"
    
    assert tool_call['function']['name'] == 'glob', \
        f"Expected function name 'glob', got '{tool_call['function']['name']}'"
    assert args.get('pattern') == expected_pattern, \
        f"Expected pattern '{expected_pattern}', got '{args.get('pattern')}'"
    assert tool_call['id'].startswith('call_'), \
        f"Call ID doesn't start with 'call_': {tool_call['id']}"
    assert 'index' in tool_call, "Tool call missing 'index' field"
    
    print("\n✓ Streaming XML detection successful!")
    
    # Cleanup
    if request_id in request_states:
        del request_states[request_id]

if __name__ == "__main__":
    asyncio.run(test_streaming_xml_detection())
    print("\n🎉 Streaming XML detection test passed!")
    sys.exit(0)
