#!/usr/bin/env python3
"""
Test tool name inference for path-only tool calls
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content

def test_path_only_inference():
    """Test that path-only JSON fragments are handled properly"""
    
    test_cases = [
        # The exact case from the latest logs
        ('{"path": "/home/florath/devel/TEST/semantic-harvest-v2"}', "list"),
        
        # Other path-only cases
        ('{"path": "/tmp"}', "list"),
        ('{"path": "."}', "list"),
        
        # Path with pattern (should be glob)
        ('{"path": "/src", "pattern": "*.py"}', "glob"),
        
        # Pattern with path (should still be glob)
        ('{"pattern": "*.py", "path": "/src"}', "glob"),
        
        # Pattern only (should be glob)
        ('{"pattern": "*.py"}', "glob"),
        
        # Make sure we don't break other tools
        ('{"command": "ls -la"}', "bash"),
        ('{"todos": "[{...}]"}', "todowrite"),
        ('{"file_path": "/test", "content": "hello"}', "write"),
    ]
    
    print("Testing path-only tool inference:")
    total = len(test_cases)
    
    for i, (content, expected) in enumerate(test_cases, 1):
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        print(f"  {i}. {status} {expected:10s} -> {result:10s} | {content[:50]}...")
        assert result == expected, f"Expected '{expected}', got '{result}' for: {content[:60]}"
    
    print(f"\nPath inference tests: {total}/{total} passed")

if __name__ == "__main__":
    success = test_path_only_inference()
    
    if success:
        print("🎉 Path-only inference tests passed!")
        sys.exit(0)
    else:
        print("❌ Some path-only inference tests failed!")
        sys.exit(1)