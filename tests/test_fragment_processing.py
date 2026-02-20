#!/usr/bin/env python3
"""
Test the fragment consolidation logic
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content, is_json_complete

def test_tool_name_inference():
    """Test tool name inference from content"""
    test_cases = [
        ('{"todos": "[{\\"id\\":\\"1\\"}]"}', "todowrite"),
        ('{"command": "ls -la", "description": "list files"}', "bash"),
        ('{"file_path": "test.py", "edits": []}', "multiedit"),
        ('{"pattern": "*.py", "path": "/home"}', "glob"),
        ('{"pattern": "function", "output_mode": "content"}', "grep"),
        ('{"unknown": "param"}', ""),
    ]
    
    print("Testing tool name inference:")
    for content, expected in test_cases:
        result = infer_tool_name_from_content(content)
        print(f"  {content[:30]}... -> {result}")
        assert result == expected, f"Expected '{expected}', got '{result}' for: {content[:50]}"
    
    print(f"Tool inference tests: {len(test_cases)}/{len(test_cases)} passed\n")

def test_fragment_consolidation():
    """Test fragment consolidation"""
    # Simulate the TodoWrite fragments from the log
    fragments = [
        "",  # Empty from call_id
        "{",  # Opening brace
        '"todos": "[{\\"content\\": \\"Research Conway\'s Game of Life rules and requirements\\", \\"status\\": \\"pending\\", \\"priority\\": \\"high\\", \\"id\\": \\"task1\\"}]"',  # Main content
        "}"  # Closing brace
    ]
    
    consolidated = "".join(fragments)
    print("Testing fragment consolidation:")
    print(f"  Consolidated: {consolidated}")
    
    # Test if it's complete JSON
    complete = is_json_complete(consolidated)
    print(f"  Is complete JSON: {complete}")
    
    # Test tool inference
    tool_name = infer_tool_name_from_content(consolidated)
    print(f"  Inferred tool: {tool_name}")
    
    # Test parsing
    import json
    parsed = json.loads(consolidated)
    print(f"  Parsed successfully: {type(parsed)}")
    print(f"  Has todos param: {'todos' in parsed}")
    assert 'todos' in parsed, "Consolidated JSON missing 'todos' key"

if __name__ == "__main__":
    print("Testing fragment processing improvements...\n")
    
    test_tool_name_inference()
    test_fragment_consolidation()
    
    print("🎉 All fragment processing tests passed!")
    sys.exit(0)
