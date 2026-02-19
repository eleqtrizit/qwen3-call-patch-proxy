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
    passed = 0
    for content, expected in test_cases:
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {content[:30]}... -> {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"Tool inference tests: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)

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
    try:
        import json
        parsed = json.loads(consolidated)
        print(f"  Parsed successfully: {type(parsed)}")
        print(f"  Has todos param: {'todos' in parsed}")
        return True
    except Exception as e:
        print(f"  Parse failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing fragment processing improvements...\n")
    
    all_passed = True
    all_passed &= test_tool_name_inference()
    all_passed &= test_fragment_consolidation()
    
    if all_passed:
        print("🎉 All fragment processing tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)