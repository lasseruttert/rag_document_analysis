#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test runner for RAG system tests.
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test_module(module_name):
    """Run tests from a module."""
    print(f"\n=== Running tests from {module_name} ===")
    try:
        # Import the test module
        test_module = __import__(module_name)
        
        # Find all test classes and methods
        test_count = 0
        passed_count = 0
        
        for attr_name in dir(test_module):
            attr = getattr(test_module, attr_name)
            
            # Check if it's a test class
            if attr_name.startswith('Test') and hasattr(attr, '__dict__'):
                print(f"\n--- Testing {attr_name} ---")
                test_instance = attr()
                
                # Run test methods
                for method_name in dir(test_instance):
                    if method_name.startswith('test_'):
                        test_count += 1
                        try:
                            method = getattr(test_instance, method_name)
                            method()
                            print(f"✓ {method_name}")
                            passed_count += 1
                        except Exception as e:
                            print(f"✗ {method_name}: {str(e)}")
                            traceback.print_exc()
        
        print(f"\n{module_name}: {passed_count}/{test_count} tests passed")
        return passed_count, test_count
        
    except Exception as e:
        print(f"Failed to run {module_name}: {str(e)}")
        traceback.print_exc()
        return 0, 0

def main():
    """Run all tests."""
    print("RAG System Test Runner")
    print("=" * 50)
    
    # Change to tests directory
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    if os.path.exists(test_dir):
        sys.path.insert(0, test_dir)
        os.chdir(test_dir)
    
    # List of test modules to run
    test_modules = []
    
    # Find test files
    for file in os.listdir('.'):
        if file.startswith('test_') and file.endswith('.py'):
            module_name = file[:-3]  # Remove .py extension
            test_modules.append(module_name)
    
    total_passed = 0
    total_tests = 0
    
    for module in test_modules:
        passed, total = run_test_module(module)
        total_passed += passed
        total_tests += total
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_tests - total_passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())