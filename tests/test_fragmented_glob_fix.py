#!/usr/bin/env python3
"""
Test the exact scenario from the logs where glob fragments are processed
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import process_sse_event, RequestState, request_states, infer_tool_name_from_content
import asyncio
import json
import uuid

async def test_fragmented_glob_scenario():
    """Test the exact fragmented glob scenario from the logs"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("Testing fragmented glob scenario from logs:")
    
    # Step 1: Empty tool call header (suppressed)
    event1 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_b4c5fae3bb0847e5ab8ac609",
                    "function": {"name": "glob", "arguments": ""}
                }]
            },
            "index": 0
        }]
    }
    
    # Step 2: Fragment with opening brace
    event2 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{"}
                }]
            },
            "index": 0
        }]
    }
    
    # Step 3: Fragment with pattern parameter
    event3 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '"pattern": "src/semantic_harvest/src/common/cli/commands/*.py"'}
                }]
            },
            "index": 0
        }]
    }
    
    # Step 4: Fragment with closing brace
    event4 = {
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
    
    # Step 5: Finish reason
    event5 = {
        "choices": [{
            "delta": {"content": ""},
            "finish_reason": "tool_calls",
            "index": 0
        }]
    }
    
    # Process events
    print("Processing events...")
    await process_sse_event(event1, request_id)
    await process_sse_event(event2, request_id)
    await process_sse_event(event3, request_id)
    await process_sse_event(event4, request_id)
    result = await process_sse_event(event5, request_id)
    
    # Check if tool name inference works on the final content
    final_content = '{"pattern": "src/semantic_harvest/src/common/cli/commands/*.py"}'
    inferred_tool = infer_tool_name_from_content(final_content)
    
    print(f"Final buffer content: {final_content}")
    print(f"Inferred tool name: '{inferred_tool}'")
    print(f"Tool inference working: {inferred_tool == 'glob'}")
    
    # Check the request state
    state = request_states.get(request_id)
    if state:
        print(f"Remaining buffers: {len(state.tool_buffers)}")
        for buf_id, buffer in state.tool_buffers.items():
            print(f"  Buffer {buf_id}: tool_name='{buffer.tool_name}', content='{buffer.content}'")
    
    # Cleanup
    if request_id in request_states:
        del request_states[request_id]
    
    return inferred_tool == 'glob'

if __name__ == "__main__":
    success = asyncio.run(test_fragmented_glob_scenario())
    
    if success:
        print("\n✓ Fragmented glob scenario test passed!")
        sys.exit(0)
    else:
        print("\n✗ Fragmented glob scenario test failed!")
        sys.exit(1)