#!/usr/bin/env python3
"""
Test that arguments are properly replaced, not appended
"""
import sys
import os
import json
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import (
    RequestState, 
    process_sse_event
)

async def test_argument_replacement():
    """Test that fragments are replaced with fixed arguments"""
    print("Testing argument replacement:")
    
    # Create request state
    request_id = "test-replacement"
    request_state = RequestState(request_id=request_id)
    
    # Mock request_states
    from qwen3_call_patch_proxy import request_states
    request_states[request_id] = request_state
    
    try:
        # Create SSE event with fragments (simulating the problem case)
        event = {
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": "{"}},
                        {"index": 2, "function": {"arguments": '"todos": "[{\\"content\\": \\"Test task\\", \\"status\\": \\"pending\\", \\"id\\": \\"1\\"}]"'}},
                        {"index": 3, "function": {"arguments": "}"}}
                    ]
                }
            }]
        }
        
        print(f"  Original tool_calls count: {len(event['choices'][0]['delta']['tool_calls'])}")
        
        # Process the event
        fixed_event = await process_sse_event(event, request_id)
        
        # Check the result
        tool_calls = fixed_event["choices"][0]["delta"]["tool_calls"]
        print(f"  Fixed tool_calls count: {len(tool_calls)}")
        
        if len(tool_calls) == 1:
            tool_call = tool_calls[0]
            print(f"  Fixed tool name: {tool_call['function']['name']}")
            print(f"  Tool call has index: {'index' in tool_call}")
            if 'index' in tool_call:
                print(f"  Index value: {tool_call['index']}")
            
            # Parse the arguments to verify they're correct
            args_str = tool_call['function']['arguments']
            print(f"  Arguments string: {args_str[:100]}...")
            
            try:
                args = json.loads(args_str)
                if "todos" in args and isinstance(args["todos"], list):
                    print("  ✓ Arguments are valid JSON with todos array")
                    # Also check that index field is present
                    if 'index' in tool_call:
                        print("  ✓ Index field is present")
                        return True
                    else:
                        print("  ✗ Index field is missing (required by OpenCode)")
                        return False
                else:
                    print("  ✗ Arguments don't have proper todos array")
                    return False
            except json.JSONDecodeError as e:
                print(f"  ✗ Arguments are not valid JSON: {e}")
                return False
        else:
            print(f"  ✗ Expected 1 tool call, got {len(tool_calls)}")
            return False
            
    finally:
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]

async def main():
    print("Testing argument replacement fix...\n")
    
    success = await test_argument_replacement()
    
    if success:
        print("\n🎉 Argument replacement test passed!")
        return 0
    else:
        print("\n❌ Argument replacement test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))