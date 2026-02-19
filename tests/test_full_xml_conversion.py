#!/usr/bin/env python3
"""
Test complete XML-to-JSON tool call conversion flow
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import process_sse_event, RequestState, request_states
import asyncio
import json
import uuid

async def test_complete_xml_conversion():
    """Test the complete flow of XML tool call conversion in SSE events"""
    
    # Create a test request state
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    # Simulate the exact XML content from the logs
    xml_content = """I'll scan the source code structure to identify potential improvement areas. Let me examine the project layout and key files to understand the current implementation.

<function=glob>
<parameter=pattern>
src/semantic_harvest/src/common/cli/commands/*.py
</parameter>
</function>
</tool_call>"""
    
    # Create SSE event with XML content
    sse_event = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk", 
        "created": 1755026110,
        "model": "unsloth/Qwen3-Coder-30B-A3B-Instruct",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": xml_content
                },
                "logprobs": None,
                "finish_reason": None
            }
        ]
    }
    
    print("Testing complete XML-to-JSON conversion:")
    print(f"Original content contains XML: {'<function=' in xml_content}")
    
    # Process the event
    fixed_event = await process_sse_event(sse_event, request_id)
    
    # Check results
    delta = fixed_event["choices"][0]["delta"]
    
    # Verify XML was converted to tool_calls
    has_tool_calls = "tool_calls" in delta
    content_cleared = delta.get("content", "") == ""
    
    print(f"Has tool_calls after processing: {has_tool_calls}")
    print(f"Content cleared: {content_cleared}")
    
    if has_tool_calls:
        tool_call = delta["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]
        
        try:
            arguments = json.loads(arguments_str)
            pattern = arguments.get("pattern", "")
            
            print(f"Function name: {function_name}")
            print(f"Pattern argument: {pattern}")
            print(f"Call ID format: {tool_call['id']}")
            print(f"Has index: {'index' in tool_call}")
            
            # Verify the conversion
            expected_pattern = "src/semantic_harvest/src/common/cli/commands/*.py"
            success = (
                function_name == "glob" and
                pattern == expected_pattern and
                tool_call["id"].startswith("call_") and
                "index" in tool_call and
                content_cleared
            )
            
            if success:
                print("✓ XML-to-JSON conversion successful!")
                return True
            else:
                print("✗ XML-to-JSON conversion failed validation")
                return False
                
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse arguments JSON: {e}")
            return False
    else:
        print("✗ No tool_calls found after processing")
        return False
    
    # Cleanup
    if request_id in request_states:
        del request_states[request_id]

async def test_mixed_content_with_xml():
    """Test content that has both text and XML tool calls"""
    
    request_id = str(uuid.uuid4())[:8]
    request_states[request_id] = RequestState(request_id=request_id)
    
    # Content with both regular text and XML tool call
    mixed_content = """Let me help you with that.

<function=bash>
<parameter=command>ls -la</parameter>
<parameter=description>List directory contents</parameter>
</function>
</tool_call>"""
    
    sse_event = {
        "choices": [{
            "delta": {"content": mixed_content},
            "index": 0
        }]
    }
    
    print("\nTesting mixed content with XML tool call:")
    
    fixed_event = await process_sse_event(sse_event, request_id)
    delta = fixed_event["choices"][0]["delta"]
    
    if "tool_calls" in delta:
        tool_call = delta["tool_calls"][0]
        args = json.loads(tool_call["function"]["arguments"])
        
        print(f"Function: {tool_call['function']['name']}")
        print(f"Arguments: {args}")
        print(f"Multiple parameters detected: {len(args) > 1}")
        
        success = (
            tool_call["function"]["name"] == "bash" and
            "command" in args and
            "description" in args and
            args["command"] == "ls -la"
        )
        
        if success:
            print("✓ Mixed content conversion successful!")
        else:
            print("✗ Mixed content conversion failed")
        
        # Cleanup
        if request_id in request_states:
            del request_states[request_id]
            
        return success
    else:
        print("✗ No tool_calls found in mixed content")
        return False

if __name__ == "__main__":
    async def run_tests():
        test1 = await test_complete_xml_conversion()
        test2 = await test_mixed_content_with_xml()
        return test1 and test2
    
    success = asyncio.run(run_tests())
    
    if success:
        print("\n🎉 All XML conversion flow tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some XML conversion flow tests failed!")
        sys.exit(1)