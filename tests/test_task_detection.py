#!/usr/bin/env python3
"""
Test the exact scenario that was failing in the logs
"""
import sys
import os
import json
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import (
    RequestState, 
    process_sse_event,
    infer_tool_name_from_content
)

async def test_task_detection():
    """Test the exact content that was failing"""
    print("Testing task tool detection for the exact failing case:")
    
    # This is the exact content from the logs that was failing
    failing_content = '{"description": "Research Game of Life rules", "prompt": "Research Conway\'s Game of Life rules and requirements for implementation", "subagent_type": "general"}'
    
    print(f"Content: {failing_content}")
    
    # Test tool name inference
    tool_name = infer_tool_name_from_content(failing_content)
    print(f"Inferred tool name: '{tool_name}'")
    
    assert tool_name == "task", \
        f"Tool name should be 'task' but got '{tool_name}'"
    print("✓ Tool name correctly inferred as 'task'")
    
    # Create request state
    request_id = "test-task-detection"
    request_state = RequestState(request_id=request_id)
    
    # Mock request_states
    from qwen3_call_patch_proxy import request_states
    request_states[request_id] = request_state
    
    try:
        # Create SSE event simulating the failing case
        event = {
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": "{"}},
                        {"index": 2, "function": {"arguments": '"description": "Research Game of Life rules", "prompt": "Research Conway\'s Game of Life rules and requirements for implementation", "subagent_type": "general"'}},
                        {"index": 3, "function": {"arguments": "}"}}
                    ]
                }
            }]
        }
        
        print(f"\nOriginal tool_calls: {len(event['choices'][0]['delta']['tool_calls'])}")
        
        # Process the event
        fixed_event = await process_sse_event(event, request_id)
        
        delta = fixed_event["choices"][0]["delta"]
        assert "tool_calls" in delta, "No tool_calls in fixed event"
        
        tool_calls = delta["tool_calls"]
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        
        tool_call = tool_calls[0]
        call_id = tool_call.get("id", "")
        function_name = tool_call.get("function", {}).get("name", "")
        
        print(f"Generated tool call:")
        print(f"  ID: {call_id}")
        print(f"  Function name: '{function_name}'")
        
        assert function_name == "task", \
            f"Function name should be 'task' but got '{function_name}'"
        print("✓ Function name correctly set to 'task'")
        
        # Verify arguments are valid JSON
        args_str = tool_call.get("function", {}).get("arguments", "")
        args = json.loads(args_str)
        print("✓ Arguments are valid JSON")
        if "subagent_type" in args:
            print(f"✓ subagent_type: {args['subagent_type']}")
            
    finally:
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]

async def main():
    print("Testing task tool detection for the failing scenario...\n")
    
    await test_task_detection()
    
    print("\n🎉 Task detection test passed!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
