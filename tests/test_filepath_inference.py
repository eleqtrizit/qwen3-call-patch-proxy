#!/usr/bin/env python3
"""
Test tool name inference for filePath (camelCase) and file_path (snake_case) parameters
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content

def test_filepath_inference():
    """Test that both camelCase and snake_case file path parameters are handled correctly"""
    
    test_cases = [
        # Read tool cases (file path only)
        ('{"filePath": "/home/user/test.py"}', "read"),
        ('{"file_path": "/home/user/test.py"}', "read"),
        
        # Write tool cases (file path + content)
        ('{"filePath": "/home/user/test.py", "content": "print(\\"hello\\")"}', "write"),
        ('{"file_path": "/home/user/test.py", "content": "print(\\"hello\\")"}', "write"),
        ('{"content": "print(\\"hello\\")", "filePath": "/home/user/test.py"}', "write"),
        ('{"content": "print(\\"hello\\")", "file_path": "/home/user/test.py"}', "write"),
        
        # The exact case from the latest logs
        ('{"filePath": "/home/florath/devel/TEST/semantic-harvest-v2/src/semantic_harvest/src/common/cli/main.py"}', "read"),
        
        # Edge cases
        ('{"filePath": "/test.py", "someOtherParam": "value"}', "read"),
        ('{"file_path": "/test.py", "limit": 100}', "read"),
        
        # Make sure other tools still work
        ('{"pattern": "*.py"}', "glob"),
        ('{"path": "/tmp"}', "list"),
        ('{"command": "ls"}', "bash"),
    ]
    
    print("Testing file path inference:")
    total = len(test_cases)
    
    for i, (content, expected) in enumerate(test_cases, 1):
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        print(f"  {i:2d}. {status} {expected:5s} -> {result:5s} | {content[:50]}...")
        assert result == expected, f"Expected '{expected}', got '{result}' for: {content[:60]}"
    
    print(f"\nFile path inference tests: {total}/{total} passed")

if __name__ == "__main__":
    success = test_filepath_inference()
    
    if success:
        print("🎉 File path inference tests passed!")
        sys.exit(0)
    else:
        print("❌ Some file path inference tests failed!")
        sys.exit(1)