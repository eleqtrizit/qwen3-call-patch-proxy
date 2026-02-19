#!/usr/bin/env python3
"""
Comprehensive test for tool name inference from JSON content
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content

def test_all_tools():
    """Test inference for all supported tools"""
    test_cases = [
        # TodoWrite tool
        ('{"todos": "[{\\"content\\": \\"Test\\", \\"id\\": \\"1\\"}]"}', "todowrite"),
        
        # Bash tool
        ('{"command": "ls -la", "description": "List files"}', "bash"),
        
        # Edit tool (camelCase)
        ('{"filePath": "/path/to/file.py", "oldString": "old", "newString": "new"}', "edit"),
        
        # Edit tool (snake_case)
        ('{"file_path": "/path/to/file.py", "old_string": "old", "new_string": "new"}', "edit"),
        
        # MultiEdit tool
        ('{"file_path": "/path/to/file.py", "edits": [{"old_string": "old", "new_string": "new"}]}', "multiedit"),
        
        # Glob tool
        ('{"pattern": "*.py", "path": "/src"}', "glob"),
        
        # Grep tool
        ('{"pattern": "function", "output_mode": "content"}', "grep"),
        
        # WebFetch tool
        ('{"url": "https://example.com", "prompt": "Get page content"}', "webfetch"),
        
        # WebSearch tool
        ('{"query": "python tutorial"}', "websearch"),
        
        # Write tool (camelCase)
        ('{"filePath": "/path/to/file.py", "content": "print(\\"hello\\")"}', "write"),
        
        # Write tool (snake_case)
        ('{"file_path": "/path/to/file.py", "content": "print(\\"hello\\")"}', "write"),
        
        # Task tool - THIS IS THE KEY ONE
        ('{"description": "Research Game of Life rules", "prompt": "Research Conway\'s Game of Life rules and requirements for implementation", "subagent_type": "general"}', "task"),
        
        # NotebookEdit tool
        ('{"notebook_path": "/path/to/notebook.ipynb", "new_source": "print(\\"test\\")"}', "notebookedit"),
        
        # Unknown tool
        ('{"unknown_param": "value"}', ""),
    ]
    
    print("Testing tool name inference for all supported tools:")
    passed = 0
    total = len(test_cases)
    
    for i, (content, expected) in enumerate(test_cases, 1):
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        tool_desc = expected if expected else "unknown"
        print(f"  {i:2d}. {status} {tool_desc:12s} -> {result:12s} | {content[:60]}...")
        
        if result == expected:
            passed += 1
        else:
            print(f"      Expected '{expected}', got '{result}'")
    
    print(f"\nTool inference tests: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    success = test_all_tools()
    
    if success:
        print("🎉 All tool inference tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tool inference tests failed!")
        sys.exit(1)