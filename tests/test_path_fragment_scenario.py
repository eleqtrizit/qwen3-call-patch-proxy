#!/usr/bin/env python3
"""
Test the exact path-fragment scenario from the latest logs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import process_sse_event, RequestState, request_states, infer_tool_name_from_content
import asyncio
import json
import uuid

async def test_path_fragment_scenario():
    """Test the exact path fragment scenario from the logs"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("Testing path fragment scenario from latest logs:")
    
    # Step 1: Empty tool call header (suppressed)
    event1 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_bb43d274632e40158f2d1571", 
                    "function": {"name": "list", "arguments": ""}
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
    
    # Step 3: Fragment with path parameter (exact from logs)
    event3 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '"path": "/home/florath/devel/TEST/semantic-harvest-v2"'}
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
    final_content = '{"path": "/home/florath/devel/TEST/semantic-harvest-v2"}'
    inferred_tool = infer_tool_name_from_content(final_content)
    
    print(f"Final buffer content: {final_content}")
    print(f"Inferred tool name: '{inferred_tool}'")
    
    # Check the request state
    state = request_states.get(request_id)
    if state:
        print(f"Remaining buffers: {len(state.tool_buffers)}")
        for buf_id, buffer in state.tool_buffers.items():
            print(f"  Buffer {buf_id}: tool_name='{buffer.tool_name}', content='{buffer.content}'")
    
    # Cleanup
    if request_id in request_states:
        del request_states[request_id]
    
    assert inferred_tool == 'list', f"Expected 'list', got '{inferred_tool}'"
    print(f"Tool inference working: True")

async def test_end_to_end_list_call():
    """Test that a fragmented list call results in a proper tool call for OpenCode"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8] 
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("\nTesting end-to-end list tool call processing:")
    
    # Simulate the fragmented events
    events = [
        # Empty tool call header
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_test456",
                        "function": {"name": "list", "arguments": ""}
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
                        "function": {"arguments": '"path": "/home/user/project"'}
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
    
    assert tool_call_event is not None, "No complete tool call generated"
    
    # Verify the tool call
    tool_call = tool_call_event["choices"][0]["delta"]["tool_calls"][0]
    
    print(f"\nGenerated tool call:")
    print(f"  Function name: {tool_call['function']['name']}")
    print(f"  Call ID: {tool_call['id']}")  
    print(f"  Has index: {'index' in tool_call}")
    print(f"  Arguments: {tool_call['function']['arguments']}")
    
    # Parse and verify arguments
    args = json.loads(tool_call['function']['arguments'])
    print(f"  Parsed args: {args}")
    
    assert tool_call['function']['name'] == 'list', \
        f"Expected function name 'list', got '{tool_call['function']['name']}'"
    assert tool_call['id'].startswith('call_'), \
        f"Call ID doesn't start with 'call_': {tool_call['id']}"
    assert 'index' in tool_call, "Tool call missing 'index' field"
    assert "path" in args, f"Arguments missing 'path': {args}"
    assert args.get("path") == "/home/user/project", \
        f"Expected '/home/user/project', got '{args.get('path')}'"
    
    print("\n✓ List tool call properly formatted for OpenCode!")
    
    # Cleanup
    if request_id in request_states:
        del request_states[request_id]

if __name__ == "__main__":
    async def run_tests():
        await test_path_fragment_scenario()
        await test_end_to_end_list_call()
    
    asyncio.run(run_tests())
    print("\n🎉 Path fragment scenario tests passed!")
    sys.exit(0)
