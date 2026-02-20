#!/usr/bin/env python3
"""
Test tool name inference for pattern-only glob calls
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content

def test_pattern_only_inference():
    """Test that pattern-only JSON is identified as glob"""
    
    test_cases = [
        # The exact case from the logs
        ('{"pattern": "src/semantic_harvest/src/common/cli/commands/*.py"}', "glob"),
        
        # Other pattern-only cases
        ('{"pattern": "*.py"}', "glob"),
        ('{"pattern": "**/*.js"}', "glob"),
        
        # Pattern with path (should still be glob)
        ('{"pattern": "*.py", "path": "/src"}', "glob"),
        
        # Pattern with output_mode (should be grep)
        ('{"pattern": "function", "output_mode": "content"}', "grep"),
        
        # Make sure we don't break other tools
        ('{"command": "ls -la"}', "bash"),
        ('{"todos": "[{...}]"}', "todowrite"),
    ]
    
    print("Testing pattern-only tool inference:")
    total = len(test_cases)
    
    for i, (content, expected) in enumerate(test_cases, 1):
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        print(f"  {i}. {status} {expected:10s} -> {result:10s} | {content[:50]}...")
        assert result == expected, f"Expected '{expected}', got '{result}' for: {content[:60]}"
    
    print(f"\nPattern inference tests: {total}/{total} passed")

if __name__ == "__main__":
    success = test_pattern_only_inference()
    
    if success:
        print("🎉 Pattern-only inference tests passed!")
        sys.exit(0)
    else:
        print("❌ Some pattern-only inference tests failed!")
        sys.exit(1)