#!/usr/bin/env python3
"""
Test that tool call IDs are in the correct format
"""
import sys
import os
import json
import asyncio
import re
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import (
    RequestState, 
    process_sse_event
)

async def test_id_format():
    """Test that generated tool call IDs match OpenCode expectations"""
    print("Testing tool call ID format:")
    
    # Create request state
    request_id = "test-id-format"
    request_state = RequestState(request_id=request_id)
    
    # Mock request_states
    from qwen3_call_patch_proxy import request_states
    request_states[request_id] = request_state
    
    try:
        # Create SSE event with fragments that will be consolidated
        event = {
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": "{"}},
                        {"index": 2, "function": {"arguments": '"todos": "[{\\"content\\": \\"Test\\", \\"id\\": \\"1\\"}]"'}},
                        {"index": 3, "function": {"arguments": "}"}}
                    ]
                }
            }]
        }
        
        print(f"  Original tool_calls: {len(event['choices'][0]['delta']['tool_calls'])}")
        
        # Process the event
        fixed_event = await process_sse_event(event, request_id)
        
        # Check the result
        tool_calls = fixed_event["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        
        tool_call = tool_calls[0]
        call_id = tool_call.get("id", "")
        
        print(f"  Generated ID: {call_id}")
        print(f"  ID type: {type(call_id)}")
        print(f"  ID length: {len(call_id)}")
        
        # Check if ID matches expected format: call_<24_hex_chars>
        id_pattern = r"^call_[a-f0-9]{24}$"
        assert re.match(id_pattern, call_id), \
            f"ID format doesn't match expected pattern {id_pattern!r}: {call_id}"
        print("  ✓ ID format matches OpenCode pattern")
        
        assert "index" in tool_call and isinstance(tool_call["index"], int), \
            f"Index field missing or wrong type: {tool_call.get('index')}"
        print("  ✓ Index field is present and numeric")
        
        assert "function" in tool_call and "name" in tool_call["function"], \
            "Function name is missing"
        print("  ✓ Function name is present")
        
        assert "function" in tool_call and "arguments" in tool_call["function"], \
            "Function arguments missing"
        args_str = tool_call["function"]["arguments"]
        args = json.loads(args_str)
        assert isinstance(args.get("todos"), list), \
            f"Arguments don't have proper todos array: {args}"
        print("  ✓ Arguments are valid JSON with todos array")
            
    finally:
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]

async def main():
    print("Testing tool call ID format...\n")
    
    await test_id_format()
    
    print("\n🎉 ID format test passed!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
