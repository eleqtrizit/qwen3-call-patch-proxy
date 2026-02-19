#!/usr/bin/env python3
"""
Test that Edit tool name inference works
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content

def test_edit_tool_inference():
    """Test Edit tool inference"""
    test_cases = [
        # Edit tool with filePath, oldString, newString
        ('{"filePath": "/path/to/file.py", "oldString": "old code", "newString": "new code"}', "edit"),
        # Edit tool with file_path, old_string, new_string  
        ('{"file_path": "/path/to/file.py", "old_string": "old code", "new_string": "new code"}', "edit"),
        # Write tool
        ('{"file_path": "/path/to/file.py", "content": "file content"}', "write"),
        ('{"filePath": "/path/to/file.py", "content": "file content"}', "write"),
        # MultiEdit tool
        ('{"file_path": "/path/to/file.py", "edits": [{"old_string": "old", "new_string": "new"}]}', "multiedit"),
    ]
    
    print("Testing tool name inference for Edit and related tools:")
    passed = 0
    for content, expected in test_cases:
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {content[:50]}... -> {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"Edit tool inference tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

if __name__ == "__main__":
    success = test_edit_tool_inference()
    if success:
        print("🎉 All Edit tool tests passed!")
        sys.exit(0)
    else:
        print("❌ Some Edit tool tests failed!")
        sys.exit(1)