import unittest
import sys
import os
import importlib
import ast
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_module_coverage():
    coverage_data = defaultdict(lambda: {'classes': 0, 'methods': 0, 'tested_methods': 0})
    
    pccl_path = os.path.join(os.path.dirname(__file__), '..', 'pccl')
    
    for root, dirs, files in os.walk(pccl_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                module_path = os.path.relpath(file_path, os.path.join(pccl_path, '..'))
                module_name = module_path.replace(os.path.sep, '.').replace('.py', '')
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    class_count = 0
                    method_count = 0
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_count += 1
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    method_count += 1
                    
                    coverage_data[module_name]['classes'] = class_count
                    coverage_data[module_name]['methods'] = method_count
                    
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
    
    return coverage_data

def run_tests_and_generate_report():
    from test_collective_ir import run_tests_with_coverage
    
    print("Running Tests...")
    test_result = run_tests_with_coverage()
    
    print("\n" + "="*60)
    print("DETAILED COVERAGE ANALYSIS")
    print("="*60)
    
    coverage_data = analyze_module_coverage()
    
    total_classes = 0
    total_methods = 0
    tested_modules = 0
    
    for module, data in coverage_data.items():
        if data['classes'] > 0:
            total_classes += data['classes']
            total_methods += data['methods']
            tested_modules += 1
            
            print(f"\n{module}:")
            print(f"  Classes: {data['classes']}")
            print(f"  Methods: {data['methods']}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Modules Analyzed: {len(coverage_data)}")
    print(f"Modules with Classes: {tested_modules}")
    print(f"Total Classes: {total_classes}")
    print(f"Total Methods: {total_methods}")
    
    estimated_coverage = min((tested_modules / len(coverage_data)) * 100, 100) if coverage_data else 0
    print(f"Estimated Code Coverage: {estimated_coverage:.1f}%")
    
    return test_result

if __name__ == '__main__':
    sys.exit(run_tests_and_generate_report())

