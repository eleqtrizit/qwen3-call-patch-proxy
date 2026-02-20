#!/usr/bin/env python3
"""
Test XML tool call detection and conversion to JSON format
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import detect_and_convert_xml_tool_call
import asyncio
import json

def test_xml_tool_call_detection():
    """Test detection and conversion of XML format tool calls"""
    
    test_cases = [
        # Test case 1: Simple glob tool call
        {
            "xml": "<function=glob>\n<parameter=pattern>\nsrc/semantic_harvest/src/common/cli/commands/*.py\n</parameter>\n</function>\n</tool_call>",
            "expected_function": "glob",
            "expected_args": {"pattern": "src/semantic_harvest/src/common/cli/commands/*.py"}
        },
        
        # Test case 2: Bash tool call
        {
            "xml": "<function=bash><parameter=command>ls -la</parameter></function>",
            "expected_function": "bash", 
            "expected_args": {"command": "ls -la"}
        },
        
        # Test case 3: Write tool call with multiple parameters
        {
            "xml": "<function=write><parameter=file_path>/tmp/test.py</parameter><parameter=content>print('hello')</parameter></function></tool_call>",
            "expected_function": "write",
            "expected_args": {"file_path": "/tmp/test.py", "content": "print('hello')"}
        },
        
        # Test case 4: Invalid XML (should return None)
        {
            "xml": "Just some regular text without tool calls",
            "expected_function": None,
            "expected_args": None
        },
        
        # Test case 5: Incomplete XML (should return None)
        {
            "xml": "<function=incomplete><parameter=test>",
            "expected_function": None,
            "expected_args": None
        }
    ]
    
    print("Testing XML tool call detection and conversion:")
    
    for i, test_case in enumerate(test_cases, 1):
        xml_content = test_case["xml"]
        expected_function = test_case["expected_function"]
        expected_args = test_case["expected_args"]
        
        result = detect_and_convert_xml_tool_call(xml_content)
        
        if expected_function is None:
            assert result is None, f"Case {i}: Expected None but got: {result}"
            print(f"  {i}. ✓ Correctly detected invalid XML")
        else:
            assert result is not None, f"Case {i}: Expected result but got None"
            assert result["function_name"] == expected_function, \
                f"Case {i}: Expected function '{expected_function}', got '{result['function_name']}'"
            assert result["arguments"] == expected_args, \
                f"Case {i}: Expected args {expected_args}, got {result['arguments']}"
            print(f"  {i}. ✓ {expected_function} -> {result['arguments']}")
    
    print(f"\nXML tool call tests: {len(test_cases)}/{len(test_cases)} passed")

def test_xml_content_scenarios():
    """Test various XML content scenarios"""
    
    scenarios = [
        # Mixed content with XML tool call
        "I'll help you scan the code.\n\n<function=glob>\n<parameter=pattern>\n*.py\n</parameter>\n</function>\n</tool_call>",
        
        # Multiple lines with spacing
        """Let me check the files.
        
<function=bash>
<parameter=command>
find . -name "*.py" | head -5
</parameter>
</function>
</tool_call>""",
        
        # Compact format
        "<function=read><parameter=file_path>/home/user/test.py</parameter></function>",
    ]
    
    print("\nTesting various XML content scenarios:")
    
    for i, content in enumerate(scenarios, 1):
        result = detect_and_convert_xml_tool_call(content)
        assert result is not None, f"Scenario {i}: Failed to detect tool call in content"
        print(f"  {i}. ✓ Detected: {result['function_name']} with {len(result['arguments'])} params")

if __name__ == "__main__":
    test_xml_tool_call_detection()
    test_xml_content_scenarios()
    
    print("\n🎉 All XML tool call conversion tests passed!")
    sys.exit(0)
