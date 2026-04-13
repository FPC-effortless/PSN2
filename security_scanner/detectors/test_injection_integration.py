"""Integration tests for InjectionDetector

This test demonstrates the InjectionDetector working with realistic code examples.
"""

import ast
import tempfile
from pathlib import Path

from security_scanner.detectors.injection import InjectionDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
)


def test_realistic_eval_vulnerability():
    """Test detection in realistic code with eval vulnerability"""
    
    # Realistic code that might appear in a config parser
    code = """
import json

class ConfigParser:
    def parse_expression(self, expr_string):
        '''Parse and evaluate a mathematical expression'''
        # VULNERABLE: Using eval with user input
        result = eval(expr_string)
        return result
    
    def safe_parse(self, expr_string):
        '''Safe parsing using ast.literal_eval'''
        import ast
        result = ast.literal_eval(expr_string)
        return result
"""
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_file.write(code)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    file_info = FileInfo(
        path=str(path),
        file_type=FileType.PYTHON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )
    
    # Parse and detect
    ast_tree = ast.parse(code)
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect the eval() but not ast.literal_eval()
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert 'eval' in vuln.title.lower()
    assert vuln.line_number == 8  # Line with eval()
    assert 'eval(expr_string)' in vuln.code_snippet
    
    # Verify recommendation mentions ast.literal_eval as alternative
    assert 'ast.literal_eval' in vuln.recommendation
    
    # Clean up
    path.unlink()
    
    print(f"✓ Detected vulnerability: {vuln.title}")
    print(f"  Location: Line {vuln.line_number}")
    print(f"  Severity: {vuln.severity.value}")
    print(f"  Confidence: {vuln.confidence}")


def test_realistic_exec_vulnerability():
    """Test detection in code with exec vulnerability"""
    
    code = """
def execute_user_script(script_path):
    '''Execute a user-provided Python script'''
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # VULNERABLE: Executing arbitrary code from file
    exec(script_content)
"""
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_file.write(code)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    file_info = FileInfo(
        path=str(path),
        file_type=FileType.PYTHON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )
    
    ast_tree = ast.parse(code)
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    assert 'exec' in vuln.title.lower()
    assert vuln.severity == SeverityLevel.HIGH
    assert vuln.cwe_id == "CWE-94"
    
    path.unlink()
    
    print(f"✓ Detected vulnerability: {vuln.title}")
    print(f"  CWE: {vuln.cwe_id}")


def test_multiple_vulnerabilities_in_file():
    """Test detection of multiple vulnerabilities in one file"""
    
    code = """
class DynamicLoader:
    def load_and_execute(self, module_name, code_string):
        # VULNERABLE: Dynamic import
        module = __import__(module_name)
        
        # VULNERABLE: Code execution
        exec(code_string)
        
        return module
    
    def evaluate_expression(self, expr):
        # VULNERABLE: Expression evaluation
        return eval(expr)
"""
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_file.write(code)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    file_info = FileInfo(
        path=str(path),
        file_type=FileType.PYTHON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )
    
    ast_tree = ast.parse(code)
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect all three vulnerabilities
    assert len(vulnerabilities) == 3
    
    # Verify different types detected
    functions = {v.context['function'] for v in vulnerabilities}
    assert '__import__' in functions
    assert 'exec' in functions
    assert 'eval' in functions
    
    print(f"✓ Detected {len(vulnerabilities)} vulnerabilities in one file:")
    for vuln in vulnerabilities:
        print(f"  - {vuln.title} (Line {vuln.line_number})")
    
    path.unlink()


if __name__ == '__main__':
    print("Running InjectionDetector integration tests...\n")
    test_realistic_eval_vulnerability()
    print()
    test_realistic_exec_vulnerability()
    print()
    test_multiple_vulnerabilities_in_file()
    print("\n✓ All integration tests passed!")
