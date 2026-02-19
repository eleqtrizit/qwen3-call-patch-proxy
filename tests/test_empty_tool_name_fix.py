#!/usr/bin/env python3
"""
Test to verify that empty tool names don't get sent to OpenCode
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content, ToolBuffer
import asyncio
import json
from datetime import datetime

async def test_empty_tool_name_prevention():
    """Test that empty tool names are prevented from being sent"""
    
    # Test case 1: Content that can't be identified as any tool
    unidentifiable_content = '{"some_random_parameter": "value", "another_param": 123}'
    tool_name = infer_tool_name_from_content(unidentifiable_content)
    
    print(f"Test 1: Unidentifiable content")
    print(f"  Content: {unidentifiable_content}")
    print(f"  Inferred tool name: '{tool_name}'")
    print(f"  Is empty: {tool_name == ''}")
    
    # Test case 2: Partial JSON that might be in a buffer
    partial_content = '{"some_field": "incomplete'
    tool_name2 = infer_tool_name_from_content(partial_content)
    
    print(f"\nTest 2: Partial JSON content")
    print(f"  Content: {partial_content}")
    print(f"  Inferred tool name: '{tool_name2}'")
    print(f"  Is empty: {tool_name2 == ''}")
    
    # Test case 3: Valid content that should be identified
    valid_content = '{"command": "ls -la", "description": "List files"}'
    tool_name3 = infer_tool_name_from_content(valid_content)
    
    print(f"\nTest 3: Valid bash tool content")
    print(f"  Content: {valid_content}")
    print(f"  Inferred tool name: '{tool_name3}'")
    print(f"  Is empty: {tool_name3 == ''}")
    
    # Simulate buffer scenario
    buffer = ToolBuffer(
        call_id="test_call",
        content=unidentifiable_content,
        request_id="test_request"
    )
    buffer.tool_name = infer_tool_name_from_content(buffer.content)
    
    print(f"\nBuffer simulation:")
    print(f"  Buffer tool_name: '{buffer.tool_name}'")
    print(f"  Would be blocked from sending: {not buffer.tool_name}")
    
    # Test that the fix condition would work
    fixed_args = json.dumps({"test": "args"})
    would_send = bool(fixed_args and buffer.tool_name)
    
    print(f"  Would send tool call: {would_send}")
    
    return tool_name == '' and tool_name2 == '' and tool_name3 == 'bash' and not would_send

if __name__ == "__main__":
    result = asyncio.run(test_empty_tool_name_prevention())
    
    if result:
        print("\n✓ Empty tool name prevention test passed!")
        sys.exit(0)
    else:
        print("\n✗ Empty tool name prevention test failed!")
        sys.exit(1)