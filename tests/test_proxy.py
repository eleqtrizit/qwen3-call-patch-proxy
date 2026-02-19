#!/usr/bin/env python3
"""
Test script for the enhanced proxy functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import ToolFixEngine, is_json_complete, validate_json_syntax, CONFIG_FILE

def test_json_completion():
    """Test JSON completion detection"""
    test_cases = [
        ('{"key": "value"}', True),
        ('{"key": "value"', False),
        ('{"key": "value", "nested": {"inner": "val"}}', True),
        ('{"key": "value", "nested": {"inner": "val"}', False),
        ('{"array": [1, 2, 3]}', True),
        ('{"array": [1, 2, 3]', False),
        ('{"string": "with \\"quotes\\" inside"}', True),
        ('', False),
        ('{', False),
        ('{}', True),
    ]
    
    print("Testing JSON completion detection:")
    passed = 0
    for json_str, expected in test_cases:
        result = is_json_complete(json_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{json_str[:30]}...' -> {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"JSON completion tests: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)

def test_fix_engine():
    """Test the ToolFixEngine"""
    print("Testing ToolFixEngine:")
    
    # Test with default config
    engine = ToolFixEngine("nonexistent.yaml")  # Should fall back to defaults
    
    # Test TodoWrite fix
    test_args = {"todos": '[{"id": "1", "content": "test", "status": "pending"}]'}
    fixed_args = engine.apply_fixes("todowrite", test_args, "test-req")
    
    if isinstance(fixed_args["todos"], list):
        print("  ✓ TodoWrite todos string converted to array")
    else:
        print("  ✗ TodoWrite fix failed")
        return False
    
    # Test Bash fix
    test_args = {"command": "ls -la"}
    fixed_args = engine.apply_fixes("bash", test_args, "test-req")
    
    if "description" in fixed_args and fixed_args["description"]:
        print("  ✓ Bash description added")
    else:
        print("  ✗ Bash description fix failed")
        return False
    
    # Test case insensitivity
    test_args = {"command": "echo test"}
    fixed_args = engine.apply_fixes("BASH", test_args, "test-req")
    
    if "description" in fixed_args:
        print("  ✓ Case-insensitive tool matching works")
    else:
        print("  ✗ Case-insensitive matching failed")
        return False
    
    print("ToolFixEngine tests: All passed\n")
    return True

def test_yaml_config():
    """Test YAML configuration loading"""
    print("Testing YAML configuration:")
    
    try:
        engine = ToolFixEngine(CONFIG_FILE)
        if engine.config and 'tools' in engine.config:
            print("  ✓ YAML configuration loaded successfully")
            print(f"  ✓ Found {len(engine.config['tools'])} configured tools")
            return True
        else:
            print("  ✗ YAML configuration missing required structure")
            return False
    except Exception as e:
        print(f"  ✗ YAML configuration loading failed: {e}")
        return False

if __name__ == "__main__":
    print("Running proxy functionality tests...\n")
    
    all_passed = True
    all_passed &= test_json_completion()
    all_passed &= test_fix_engine() 
    all_passed &= test_yaml_config()
    
    if all_passed:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)