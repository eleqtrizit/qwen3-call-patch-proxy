#!/usr/bin/env python3
"""
Test Edit tool boolean conversion
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import ToolFixEngine, CONFIG_FILE

def test_edit_boolean_conversion():
    """Test Edit tool boolean conversion"""
    print("Testing Edit tool boolean conversion:")
    
    # Create fix engine
    engine = ToolFixEngine(CONFIG_FILE)
    
    # Test cases for replaceAll conversion
    test_cases = [
        # String "True" -> boolean true
        ({"filePath": "test.py", "oldString": "old", "newString": "new", "replaceAll": "True"}, True),
        # String "true" -> boolean true
        ({"filePath": "test.py", "oldString": "old", "newString": "new", "replaceAll": "true"}, True),
        # String "False" -> boolean false
        ({"filePath": "test.py", "oldString": "old", "newString": "new", "replaceAll": "False"}, False),
        # String "false" -> boolean false
        ({"filePath": "test.py", "oldString": "old", "newString": "new", "replaceAll": "false"}, False),
        # String "1" -> boolean true
        ({"filePath": "test.py", "oldString": "old", "newString": "new", "replaceAll": "1"}, True),
        # String "0" -> boolean false
        ({"filePath": "test.py", "oldString": "old", "newString": "new", "replaceAll": "0"}, False),
    ]
    
    for i, (args_input, expected_bool) in enumerate(test_cases):
        print(f"  Test {i+1}: replaceAll='{args_input['replaceAll']}'")
        
        # Apply fixes
        _, fixed_args = engine.apply_fixes("edit", args_input.copy(), f"test-{i}")
        
        result_bool = fixed_args.get("replaceAll")
        result_type = type(result_bool)
        
        print(f"    Input: {args_input['replaceAll']} ({type(args_input['replaceAll'])})")
        print(f"    Output: {result_bool} ({result_type})")
        
        assert isinstance(result_bool, bool), \
            f"Expected bool type, got {result_type} for input '{args_input['replaceAll']}'"
        assert result_bool == expected_bool, \
            f"Expected {expected_bool}, got {result_bool} for input '{args_input['replaceAll']}'"
        print("    ✓ Conversion successful")
    
    print(f"Boolean conversion tests: {len(test_cases)}/{len(test_cases)} passed")

if __name__ == "__main__":
    test_edit_boolean_conversion()
    print("🎉 All Edit boolean tests passed!")
    sys.exit(0)
