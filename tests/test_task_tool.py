#!/usr/bin/env python3
"""
Test that Task tool name inference and fixes work
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import infer_tool_name_from_content, ToolFixEngine, CONFIG_FILE

def test_task_tool_inference():
    """Test Task tool inference"""
    test_cases = [
        # Task tool with all required parameters
        ('{"description": "Research Game of Life rules", "prompt": "Research Conway\'s Game of Life rules and requirements for implementation", "subagent_type": "general"}', "task"),
        # Task tool missing subagent_type (should be inferred but might not have fixes)
        ('{"description": "Search for files", "prompt": "Find all Python files"}', ""),  # Won't be inferred without subagent_type
    ]
    
    print("Testing Task tool name inference:")
    passed = 0
    for content, expected in test_cases:
        result = infer_tool_name_from_content(content)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {content[:50]}... -> {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"Task tool inference tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_task_tool_fixes():
    """Test Task tool fixes"""
    print("\nTesting Task tool fixes:")
    
    # Create fix engine
    engine = ToolFixEngine(CONFIG_FILE)
    
    # Test case: missing subagent_type
    args_input = {
        "description": "Research something",
        "prompt": "Research this topic"
    }
    
    print(f"  Input: {args_input}")
    
    # Apply fixes
    fixed_args = engine.apply_fixes("task", args_input.copy(), "test-task")
    
    print(f"  Output: {fixed_args}")
    
    if "subagent_type" in fixed_args and fixed_args["subagent_type"] == "general-purpose":
        print("  ✓ Default subagent_type added successfully")
        return True
    else:
        print("  ✗ Default subagent_type not added")
        return False

if __name__ == "__main__":
    inference_success = test_task_tool_inference()
    fixes_success = test_task_tool_fixes()
    
    if inference_success and fixes_success:
        print("\n🎉 All Task tool tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some Task tool tests failed!")
        sys.exit(1)