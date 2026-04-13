"""Unit tests for ASTAnalyzer helper class

Tests cover:
- User-controlled input detection
- Function name extraction
- Code snippet extraction
- Input validation detection
"""

import ast
import tempfile
from pathlib import Path

import pytest

from security_scanner.ast_analyzer import ASTAnalyzer


class TestIsUserControlled:
    """Tests for is_user_controlled() method"""
    
    def test_function_parameter_is_user_controlled(self):
        """Function parameters should be detected as user-controlled"""
        code = """
def process_data(user_input):
    result = eval(user_input)
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        
        # Find the eval call
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call):
                # The argument to eval is user_input (a Name node)
                arg = node.args[0]
                assert ASTAnalyzer.is_user_controlled(arg, tree)
    
    def test_input_function_is_user_controlled(self):
        """input() calls should be detected as user-controlled"""
        code = """
data = input("Enter data: ")
"""
        tree = ast.parse(code)
        
        # Find the input() call
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert ASTAnalyzer.is_user_controlled(node, tree)
    
    def test_file_read_is_user_controlled(self):
        """File read operations should be detected as user-controlled"""
        code = """
with open("file.txt") as f:
    data = f.read()
"""
        tree = ast.parse(code)
        
        # Find the read() call
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name == "read":
                    assert ASTAnalyzer.is_user_controlled(node, tree)
    
    def test_request_args_is_user_controlled(self):
        """Web request data should be detected as user-controlled"""
        code = """
username = request.args.get('username')
"""
        tree = ast.parse(code)
        
        # Find the request.args attribute
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "args":
                assert ASTAnalyzer.is_user_controlled(node, tree)
    
    def test_literal_is_not_user_controlled(self):
        """Literal values should not be detected as user-controlled"""
        code = """
data = "hardcoded string"
"""
        tree = ast.parse(code)
        
        # Find the string constant
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                assert not ASTAnalyzer.is_user_controlled(node, tree)
    
    def test_subscript_of_user_controlled_is_user_controlled(self):
        """Subscripts of user-controlled sources should be user-controlled"""
        code = """
def process(data):
    item = data[0]
"""
        tree = ast.parse(code)
        
        # Find the subscript node
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                assert ASTAnalyzer.is_user_controlled(node, tree)


class TestGetFunctionName:
    """Tests for get_function_name() method"""
    
    def test_simple_function_name(self):
        """Simple function calls should return the function name"""
        code = "eval('1+1')"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        assert ASTAnalyzer.get_function_name(call_node) == "eval"
    
    def test_attribute_function_name(self):
        """Attribute function calls should return the full path"""
        code = "os.path.join('a', 'b')"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        assert ASTAnalyzer.get_function_name(call_node) == "os.path.join"
    
    def test_module_function_name(self):
        """Module function calls should return module.function"""
        code = "torch.load('model.pt')"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        assert ASTAnalyzer.get_function_name(call_node) == "torch.load"
    
    def test_method_call_name(self):
        """Method calls should return the method name"""
        code = "obj.method()"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        assert ASTAnalyzer.get_function_name(call_node) == "obj.method"
    
    def test_non_call_node_returns_none(self):
        """Non-Call nodes should return None"""
        code = "x = 5"
        tree = ast.parse(code)
        assign_node = tree.body[0]
        
        assert ASTAnalyzer.get_function_name(assign_node) is None
    
    def test_nested_attribute_function_name(self):
        """Deeply nested attribute calls should return full path"""
        code = "a.b.c.d.method()"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        assert ASTAnalyzer.get_function_name(call_node) == "a.b.c.d.method"


class TestGetCodeSnippet:
    """Tests for get_code_snippet() method"""
    
    def test_extract_snippet_with_context(self):
        """Should extract code snippet with surrounding context lines"""
        # Create a temporary file
        code = """line 1
line 2
line 3
line 4
line 5
line 6
line 7
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            snippet = ASTAnalyzer.get_code_snippet(temp_path, 4, context_lines=2)
            
            # Should include lines 2-6 (4 +/- 2)
            assert "line 2" in snippet
            assert "line 3" in snippet
            assert "line 4" in snippet
            assert "line 5" in snippet
            assert "line 6" in snippet
            
            # Should not include lines 1 and 7
            assert "line 1" not in snippet
            assert "line 7" not in snippet
            
            # Target line should be highlighted with >>>
            assert ">>>" in snippet
            lines = snippet.split('\n')
            target_line = [l for l in lines if ">>>" in l][0]
            assert "line 4" in target_line
        finally:
            Path(temp_path).unlink()
    
    def test_snippet_at_file_start(self):
        """Should handle snippets at the start of file"""
        code = """line 1
line 2
line 3
line 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            snippet = ASTAnalyzer.get_code_snippet(temp_path, 1, context_lines=3)
            
            # Should start from line 1
            assert "line 1" in snippet
            assert ">>>" in snippet
        finally:
            Path(temp_path).unlink()
    
    def test_snippet_at_file_end(self):
        """Should handle snippets at the end of file"""
        code = """line 1
line 2
line 3
line 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            snippet = ASTAnalyzer.get_code_snippet(temp_path, 4, context_lines=3)
            
            # Should include line 4
            assert "line 4" in snippet
            assert ">>>" in snippet
        finally:
            Path(temp_path).unlink()
    
    def test_nonexistent_file(self):
        """Should return error message for nonexistent file"""
        snippet = ASTAnalyzer.get_code_snippet("/nonexistent/file.py", 1)
        assert "File not found" in snippet
    
    def test_line_numbers_in_snippet(self):
        """Snippet should include line numbers"""
        code = """line 1
line 2
line 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            snippet = ASTAnalyzer.get_code_snippet(temp_path, 2, context_lines=1)
            
            # Should have line numbers
            assert "1 |" in snippet
            assert "2 |" in snippet
            assert "3 |" in snippet
        finally:
            Path(temp_path).unlink()


class TestHasValidation:
    """Tests for has_validation() method"""
    
    def test_isinstance_validation(self):
        """isinstance() checks should be detected as validation"""
        code = """
def process(data):
    if isinstance(data, str):
        return data.upper()
"""
        tree = ast.parse(code)
        
        # Find the data Name node in the return statement
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "data":
                # Check if validation exists in the function context
                func_def = tree.body[0]
                if ASTAnalyzer.has_validation(node, func_def):
                    return  # Test passes
        
        pytest.fail("Should have detected isinstance validation")
    
    def test_type_check_validation(self):
        """type() checks should be detected as validation"""
        code = """
def process(value):
    if type(value) == int:
        return value * 2
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        
        # Find the value Name node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "value":
                if ASTAnalyzer.has_validation(node, func_def):
                    return  # Test passes
        
        pytest.fail("Should have detected type validation")
    
    def test_length_validation(self):
        """len() checks should be detected as validation"""
        code = """
def process(items):
    if len(items) > 0:
        return items[0]
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        
        # Find the items Name node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "items":
                if ASTAnalyzer.has_validation(node, func_def):
                    return  # Test passes
        
        pytest.fail("Should have detected length validation")
    
    def test_assert_validation(self):
        """assert statements should be detected as validation"""
        code = """
def process(value):
    assert value > 0
    return value * 2
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        
        # Find the value Name node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "value":
                if ASTAnalyzer.has_validation(node, func_def):
                    return  # Test passes
        
        pytest.fail("Should have detected assert validation")
    
    def test_no_validation(self):
        """Should return False when no validation is present"""
        code = """
def process(data):
    return data.upper()
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        
        # Find the data Name node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "data":
                assert not ASTAnalyzer.has_validation(node, func_def)
                return
    
    def test_comparison_validation(self):
        """Comparison operations should be detected as validation"""
        code = """
def process(age):
    if age >= 18:
        return "adult"
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        
        # Find the age Name node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "age":
                if ASTAnalyzer.has_validation(node, func_def):
                    return  # Test passes
        
        pytest.fail("Should have detected comparison validation")


class TestPrivateHelpers:
    """Tests for private helper methods"""
    
    def test_get_attribute_path(self):
        """Should build full attribute path"""
        code = "os.path.join"
        tree = ast.parse(code)
        attr_node = tree.body[0].value
        
        path = ASTAnalyzer._get_attribute_path(attr_node)
        assert path == "os.path.join"
    
    def test_references_variable(self):
        """Should detect if node references a variable"""
        code = "x + y"
        tree = ast.parse(code)
        binop = tree.body[0].value
        
        assert ASTAnalyzer._references_variable(binop, "x")
        assert ASTAnalyzer._references_variable(binop, "y")
        assert not ASTAnalyzer._references_variable(binop, "z")
    
    def test_is_validation_test_with_comparison(self):
        """Should recognize comparison as validation test"""
        code = "x > 5"
        tree = ast.parse(code)
        compare = tree.body[0].value
        
        assert ASTAnalyzer._is_validation_test(compare)
    
    def test_is_validation_test_with_isinstance(self):
        """Should recognize isinstance as validation test"""
        code = "isinstance(x, int)"
        tree = ast.parse(code)
        call = tree.body[0].value
        
        assert ASTAnalyzer._is_validation_test(call)
