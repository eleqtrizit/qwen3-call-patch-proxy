#!/usr/bin/env python3
"""
Test the real-world scenario where OpenCode should receive a proper tool call
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import process_sse_event, RequestState, request_states
import asyncio
import json
import uuid

async def test_end_to_end_glob_call():
    """Test that a fragmented glob call results in a proper tool call for OpenCode"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("Testing end-to-end glob tool call processing:")
    
    # Simulate the fragmented events that cause the issue
    events = [
        # Empty tool call header (gets suppressed)
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_test123",
                        "function": {"name": "glob", "arguments": ""}
                    }]
                },
                "index": 0
            }]
        },
        
        # JSON fragments
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{"}
                    }]
                },
                "index": 0
            }]
        },
        
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '"pattern": "src/**/*.py"'}
                    }]
                },
                "index": 0
            }]
        },
        
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "}"}
                    }]
                },
                "index": 0
            }]
        }
    ]
    
    # Process fragments
    tool_call_event = None
    for i, event in enumerate(events):
        print(f"Processing event {i+1}...")
        result = await process_sse_event(event, request_id)
        
        # Check if we got a complete tool call
        delta = result["choices"][0]["delta"]
        if "tool_calls" in delta and delta["tool_calls"]:
            tool_call_event = result
            print(f"  → Got complete tool call!")
            break
        else:
            print(f"  → Fragments suppressed (expected)")
    
    if not tool_call_event:
        print("✗ No complete tool call generated")
        return False
    
    # Verify the tool call is properly formatted for OpenCode
    tool_call = tool_call_event["choices"][0]["delta"]["tool_calls"][0]
    
    print(f"\nGenerated tool call:")
    print(f"  Function name: {tool_call['function']['name']}")
    print(f"  Call ID: {tool_call['id']}")
    print(f"  Has index: {'index' in tool_call}")
    print(f"  Arguments: {tool_call['function']['arguments']}")
    
    # Parse and verify arguments
    try:
        args = json.loads(tool_call['function']['arguments'])
        print(f"  Parsed args: {args}")
        
        # Check that the fix engine added the default path
        has_pattern = "pattern" in args
        has_path = "path" in args  # Should be added by fix engine
        pattern_correct = args.get("pattern") == "src/**/*.py"
        
        print(f"  Has pattern: {has_pattern}")
        print(f"  Has default path: {has_path}")
        print(f"  Pattern correct: {pattern_correct}")
        
        success = (
            tool_call['function']['name'] == 'glob' and
            tool_call['id'].startswith('call_') and
            'index' in tool_call and
            has_pattern and
            has_path and
            pattern_correct
        )
        
        if success:
            print("\n✓ Tool call properly formatted for OpenCode!")
        else:
            print("\n✗ Tool call format issues detected")
        
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]
        
        return success
        
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse arguments: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_end_to_end_glob_call())
    
    if success:
        print("\n🎉 End-to-end glob call test passed!")
        sys.exit(0)
    else:
        print("\n❌ End-to-end glob call test failed!")
        sys.exit(1)