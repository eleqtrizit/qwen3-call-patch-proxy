#!/usr/bin/env python3
"""
Test the exact filePath scenario from the latest logs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import process_sse_event, RequestState, request_states, infer_tool_name_from_content
import asyncio
import json
import uuid

async def test_filepath_fragment_scenario():
    """Test the exact filePath fragment scenario from the logs"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("Testing filePath fragment scenario from latest logs:")
    
    # Step 1: Empty tool call header (suppressed)
    event1 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_dda4b7a3e461465aa6b369a0", 
                    "function": {"name": "read", "arguments": ""}
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
    
    # Step 3: Fragment with filePath parameter (exact from logs)
    event3 = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '"filePath": "/home/florath/devel/TEST/semantic-harvest-v2/src/semantic_harvest/src/common/cli/main.py"'}
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
    final_content = '{"filePath": "/home/florath/devel/TEST/semantic-harvest-v2/src/semantic_harvest/src/common/cli/main.py"}'
    inferred_tool = infer_tool_name_from_content(final_content)
    
    print(f"Final buffer content: {final_content[:80]}...")
    print(f"Inferred tool name: '{inferred_tool}'")
    print(f"Tool inference working: {inferred_tool == 'read'}")
    
    # Check the request state
    state = request_states.get(request_id)
    if state:
        print(f"Remaining buffers: {len(state.tool_buffers)}")
        for buf_id, buffer in state.tool_buffers.items():
            print(f"  Buffer {buf_id}: tool_name='{buffer.tool_name}', content='{buffer.content[:50]}...'")
    
    # Cleanup
    if request_id in request_states:
        del request_states[request_id]
    
    return inferred_tool == 'read'

async def test_end_to_end_read_call():
    """Test that a fragmented read call results in a proper tool call for OpenCode"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8] 
    request_states[request_id] = RequestState(request_id=request_id)
    
    print("\nTesting end-to-end read tool call processing:")
    
    # Simulate the fragmented events
    events = [
        # Empty tool call header
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_test789",
                        "function": {"name": "read", "arguments": ""}
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
                        "function": {"arguments": '"filePath": "/path/to/file.py"'}
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
    
    # Verify the tool call
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
        
        has_file_path = "filePath" in args
        path_correct = args.get("filePath") == "/path/to/file.py"
        
        success = (
            tool_call['function']['name'] == 'read' and
            tool_call['id'].startswith('call_') and
            'index' in tool_call and
            has_file_path and
            path_correct
        )
        
        if success:
            print("\n✓ Read tool call properly formatted for OpenCode!")
        else:
            print("\n✗ Read tool call format issues detected")
        
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]
        
        return success
        
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse arguments: {e}")
        return False

if __name__ == "__main__":
    async def run_tests():
        test1 = await test_filepath_fragment_scenario()
        test2 = await test_end_to_end_read_call()
        return test1 and test2
    
    success = asyncio.run(run_tests())
    
    if success:
        print("\n🎉 FilePath fragment scenario tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some filePath fragment scenario tests failed!")
        sys.exit(1)