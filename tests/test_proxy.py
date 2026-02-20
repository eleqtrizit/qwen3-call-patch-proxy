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
    for json_str, expected in test_cases:
        result = is_json_complete(json_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{json_str[:30]}...' -> {result} (expected {expected})")
        assert result == expected, \
            f"JSON completion mismatch for '{json_str[:50]}': expected {expected}, got {result}"
    
    print(f"JSON completion tests: {len(test_cases)}/{len(test_cases)} passed\n")

def test_fix_engine():
    """Test the ToolFixEngine"""
    print("Testing ToolFixEngine:")
    
    # Test with default config
    engine = ToolFixEngine("nonexistent.yaml")  # Should fall back to defaults
    
    # Test TodoWrite fix
    test_args = {"todos": '[{"id": "1", "content": "test", "status": "pending"}]'}
    _, fixed_args = engine.apply_fixes("todowrite", test_args, "test-req")
    
    assert isinstance(fixed_args["todos"], list), \
        f"TodoWrite fix failed: todos is not a list: {type(fixed_args['todos'])}"
    print("  ✓ TodoWrite todos string converted to array")
    
    # Test Bash fix
    test_args = {"command": "ls -la"}
    _, fixed_args = engine.apply_fixes("bash", test_args, "test-req")
    
    assert "description" in fixed_args and fixed_args["description"], \
        f"Bash description fix failed: {fixed_args}"
    print("  ✓ Bash description added")
    
    # Test case insensitivity
    test_args = {"command": "echo test"}
    _, fixed_args = engine.apply_fixes("BASH", test_args, "test-req")
    
    assert "description" in fixed_args, \
        f"Case-insensitive matching failed: {fixed_args}"
    print("  ✓ Case-insensitive tool matching works")
    
    print("ToolFixEngine tests: All passed\n")

def test_yaml_config():
    """Test YAML configuration loading"""
    print("Testing YAML configuration:")
    
    engine = ToolFixEngine(CONFIG_FILE)
    assert engine.config and 'tools' in engine.config, \
        "YAML configuration missing required structure"
    print("  ✓ YAML configuration loaded successfully")
    print(f"  ✓ Found {len(engine.config['tools'])} configured tools")

if __name__ == "__main__":
    print("Running proxy functionality tests...\n")
    
    test_json_completion()
    test_fix_engine()
    test_yaml_config()
    
    print("🎉 All tests passed!")
    sys.exit(0)
