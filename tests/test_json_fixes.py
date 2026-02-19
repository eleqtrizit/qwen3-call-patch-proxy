#!/usr/bin/env python3
"""
Test JSON fixing functionality for malformed LLM output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_call_patch_proxy import ToolFixEngine, CONFIG_FILE

def test_malformed_json_fixes():
    """Test that the proxy can fix common JSON issues from LLMs"""
    engine = ToolFixEngine(CONFIG_FILE)
    
    # Test cases for malformed JSON that LLMs commonly generate
    test_cases = [
        {
            "name": "Single quotes to double quotes",
            "input": "{'content': 'Test task', 'status': 'pending', 'id': '1'}",
            "expected_content": "Test task"
        },
        {
            "name": "Mixed quotes",
            "input": "[{'content': \"Mixed quotes task\", 'status': 'completed'}]",
            "expected_length": 1
        },
        {
            "name": "Valid JSON (should pass through)",
            "input": '[{"content": "Valid task", "status": "pending"}]',
            "expected_length": 1
        }
    ]
    
    print("Testing JSON fixing functionality...")
    
    for test_case in test_cases:
        print(f"\n🧪 Test: {test_case['name']}")
        print(f"   Input: {test_case['input']}")
        
        # Simulate a todowrite tool call with malformed JSON
        args_obj = {"todos": test_case["input"]}
        
        try:
            final_tool_name, fixed_args = engine.apply_fixes("todowrite", args_obj, "test_req")
            
            print(f"   ✅ Tool: {final_tool_name}")
            print(f"   ✅ Fixed: {fixed_args['todos']}")
            
            # Validate the result
            if 'expected_content' in test_case:
                assert test_case['expected_content'] in str(fixed_args['todos'])
            if 'expected_length' in test_case:
                assert len(fixed_args['todos']) == test_case['expected_length']
                
            print(f"   ✅ Validation passed")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            raise

def test_tool_conversion():
    """Test read->write tool conversion"""
    engine = ToolFixEngine(CONFIG_FILE)
    
    print("\n🧪 Testing tool conversion (read→write)...")
    
    # Test read tool with content (should convert to write)
    args_obj = {
        "filePath": "/test/file.txt",
        "content": "File content here"
    }
    
    final_tool_name, fixed_args = engine.apply_fixes("read", args_obj, "test_req")
    
    print(f"   ✅ Converted: read → {final_tool_name}")
    print(f"   ✅ Args: {fixed_args}")
    
    assert final_tool_name == "write"
    assert "filePath" in fixed_args
    assert "content" in fixed_args
    print(f"   ✅ Conversion test passed")

if __name__ == "__main__":
    try:
        test_malformed_json_fixes()
        test_tool_conversion()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        sys.exit(1)