#!/usr/bin/env python3
"""
Test the complete flow from fragments to fixed tool calls
"""
import sys
import os
import json
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import (
    RequestState, 
    process_sse_event, 
    fix_engine,
    ToolBuffer
)

async def test_complete_todowrite_flow():
    """Test complete TodoWrite processing flow"""
    print("Testing complete TodoWrite flow:")
    
    # Create request state
    request_id = "test-123"
    request_state = RequestState(request_id=request_id)
    
    # Simulate the SSE events from the log
    events = [
        # Event with empty call_id tool
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_0c46c4a795a740ee99c4679c",
                        "function": {"arguments": ""}
                    }]
                }
            }]
        },
        
        # Event with indexed fragments
        {
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": "{"}},
                        {"index": 2, "function": {"arguments": '"todos": "[{\\"content\\": \\"Test task\\", \\"status\\": \\"pending\\", \\"id\\": \\"1\\"}]"'}},
                        {"index": 3, "function": {"arguments": "}"}}
                    ]
                }
            }]
        },
        
        # Finish event
        {
            "choices": [{
                "finish_reason": "tool_calls"
            }]
        }
    ]
    
    # Mock request_states
    from qwen3_call_patch_proxy import request_states
    request_states[request_id] = request_state
    
    try:
        # Process each event
        for i, event in enumerate(events):
            print(f"  Processing event {i+1}: {list(event.keys())}")
            fixed_event = await process_sse_event(event.copy(), request_id)
            
            # Check buffer state
            buffer_count = len(request_state.tool_buffers)
            print(f"    Buffers after event {i+1}: {buffer_count}")
            for buf_id, buf in request_state.tool_buffers.items():
                print(f"      Buffer {buf_id}: {len(buf.content)} chars, tool: {buf.tool_name}")
        
        # Check final state
        assert len(request_state.tool_buffers) == 0, \
            f"{len(request_state.tool_buffers)} buffers remaining after processing"
        print("  ✓ All buffers processed successfully")
            
    finally:
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]

async def test_tool_fix_application():
    """Test that tool fixes are actually applied"""
    print("\nTesting tool fix application:")
    
    # Test with TodoWrite content
    test_content = '{"todos": "[{\\"content\\": \\"Test\\", \\"status\\": \\"pending\\", \\"id\\": \\"1\\"}]"}'
    
    parsed = json.loads(test_content)
    print(f"  Original todos type: {type(parsed['todos'])}")
    
    # Apply fixes
    _, fixed = fix_engine.apply_fixes("todowrite", parsed, "test")
    print(f"  Fixed todos type: {type(fixed['todos'])}")
    
    assert isinstance(fixed['todos'], list), \
        f"todos fix failed: expected list, got {type(fixed['todos'])}"
    print("  ✓ todos converted to array successfully")

async def main():
    print("Testing complete proxy flow...\n")
    
    await test_complete_todowrite_flow()
    await test_tool_fix_application()
    
    print("\n🎉 All complete flow tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
