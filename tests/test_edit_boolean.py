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
    
    passed = 0
    for i, (args_input, expected_bool) in enumerate(test_cases):
        print(f"  Test {i+1}: replaceAll='{args_input['replaceAll']}'")
        
        # Apply fixes
        fixed_args = engine.apply_fixes("edit", args_input.copy(), f"test-{i}")
        
        result_bool = fixed_args.get("replaceAll")
        result_type = type(result_bool)
        
        print(f"    Input: {args_input['replaceAll']} ({type(args_input['replaceAll'])})")
        print(f"    Output: {result_bool} ({result_type})")
        
        if isinstance(result_bool, bool) and result_bool == expected_bool:
            print("    ✓ Conversion successful")
            passed += 1
        else:
            print(f"    ✗ Expected {expected_bool} (bool), got {result_bool} ({result_type})")
    
    print(f"Boolean conversion tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

if __name__ == "__main__":
    success = test_edit_boolean_conversion()
    if success:
        print("🎉 All Edit boolean tests passed!")
        sys.exit(0)
    else:
        print("❌ Some Edit boolean tests failed!")
        sys.exit(1)