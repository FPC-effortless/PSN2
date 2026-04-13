"""Input Validation Detector

This module implements detection of missing input validation vulnerabilities including:
- Functions accepting external input without type checking
- File path parameters without directory validation
- Numeric inputs without range validation
- String inputs without length limits
- Dataset loaders without file existence checks

The detector uses AST visitor pattern to identify functions with parameters
and analyzes whether proper validation is performed on inputs.
"""

import ast
from typing import List, Optional, Set

from security_scanner.ast_analyzer import ASTAnalyzer
from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    SeverityLevel,
    Vulnerability,
    VulnerabilityType,
)


class ValidationDetector(BaseDetector):
    """Detector for missing input validation vulnerabilities
    
    This detector identifies functions that accept external input without
    proper validation:
    - Type checking for external inputs
    - Directory validation for file paths
    - Range validation for numeric inputs
    - Length limits for string inputs
    - File existence checks for dataset loaders
    
    The detector performs context analysis to determine if validation is
    present and assigns severity based on the type of input.
    """
    
    # File operation functions that should validate paths
    FILE_OPERATIONS = {
        'open', 'read', 'write', 'load', 'save',
        'torch.load', 'torch.save',
        'pickle.load', 'pickle.dump',
        'json.load', 'json.dump',
        'yaml.load', 'yaml.dump',
    }
    
    # Dataset loader class patterns
    DATASET_LOADERS = {
        'Dataset', 'DataLoader', 'Loader', 'Reader'
    }
    
    # Type checking functions
    TYPE_CHECKING_FUNCTIONS = {
        'isinstance', 'type', 'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple'
    }
    
    # Path validation functions
    PATH_VALIDATION_FUNCTIONS = {
        'os.path.exists', 'os.path.isfile', 'os.path.isdir',
        'Path.exists', 'Path.is_file', 'Path.is_dir',
        'os.path.abspath', 'os.path.realpath', 'Path.resolve'
    }
    
    # Range/length validation patterns
    VALIDATION_PATTERNS = {
        'len', 'range', 'min', 'max', 'abs'
    }
    
    def __init__(self, config: DetectorConfig):
        """Initialize the validation detector
        
        Args:
            config: Configuration for the detector
        """
        super().__init__(config)
        self.vulnerabilities: List[Vulnerability] = []
        self.current_file: Optional[FileInfo] = None
        self.ast_tree: Optional[ast.AST] = None
    
    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        """Detect missing input validation vulnerabilities in a file
        
        Args:
            file_info: Information about the file being scanned
            ast_tree: Parsed AST tree for the Python file
            
        Returns:
            List of detected vulnerabilities
        """
        if not self.config.enabled:
            return []
        
        if ast_tree is None:
            return []
        
        # Reset state for this file
        self.vulnerabilities = []
        self.current_file = file_info
        self.ast_tree = ast_tree
        
        # Visit all function definitions in the AST
        self._visit_node(ast_tree)
        
        # Filter out suppressed vulnerabilities
        return [v for v in self.vulnerabilities if not self.should_suppress(v)]
    
    def _visit_node(self, node: ast.AST) -> None:
        """Recursively visit AST nodes to find function definitions
        
        Args:
            node: The AST node to visit
        """
        # Check if this is a function definition
        if isinstance(node, ast.FunctionDef):
            self._check_function(node)
        
        # Check if this is a class definition (for dataset loaders)
        if isinstance(node, ast.ClassDef):
            self._check_class(node)
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self._visit_node(child)
    
    def _check_function(self, node: ast.FunctionDef) -> None:
        """Check if a function has proper input validation
        
        Args:
            node: The FunctionDef node to analyze
        """
        # Skip functions with no parameters (except self/cls)
        params = [arg for arg in node.args.args if arg.arg not in ('self', 'cls')]
        if not params:
            return
        
        # Analyze each parameter
        for param in params:
            self._check_parameter(node, param)
    
    def _check_parameter(self, func_node: ast.FunctionDef, param: ast.arg) -> None:
        """Check if a parameter has proper validation
        
        Args:
            func_node: The function definition node
            param: The parameter to check
        """
        param_name = param.arg
        
        # Determine parameter type from name and usage
        param_type = self._infer_parameter_type(func_node, param_name)
        
        if param_type == 'path':
            self._check_path_parameter(func_node, param_name)
        elif param_type == 'numeric':
            self._check_numeric_parameter(func_node, param_name)
        elif param_type == 'string':
            self._check_string_parameter(func_node, param_name)
        elif param_type == 'external':
            self._check_external_input(func_node, param_name)
    
    def _infer_parameter_type(self, func_node: ast.FunctionDef, param_name: str) -> str:
        """Infer the type of parameter from its name and usage
        
        Args:
            func_node: The function definition node
            param_name: Name of the parameter
            
        Returns:
            Type category: 'path', 'numeric', 'string', 'external', or 'unknown'
        """
        # Check parameter name patterns
        param_lower = param_name.lower()
        
        # Path-related parameters
        if any(keyword in param_lower for keyword in ['path', 'file', 'dir', 'folder', 'location']):
            return 'path'
        
        # Numeric parameters
        if any(keyword in param_lower for keyword in ['size', 'count', 'num', 'index', 'idx', 'length', 'width', 'height']):
            return 'numeric'
        
        # String parameters
        if any(keyword in param_lower for keyword in ['name', 'text', 'str', 'message', 'content']):
            return 'string'
        
        # Check usage in function body
        for node in ast.walk(func_node):
            # Check if parameter is used in file operations
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name and any(op in func_name for op in self.FILE_OPERATIONS):
                    # Check if parameter is used as argument
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == param_name:
                            return 'path'
        
        # Check if parameter comes from external source
        if self._is_external_parameter(func_node, param_name):
            return 'external'
        
        return 'unknown'
    
    def _is_external_parameter(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if parameter represents external input
        
        Args:
            func_node: The function definition node
            param_name: Name of the parameter
            
        Returns:
            True if parameter is from external source
        """
        # Check if function is called with user input sources
        # This is a simplified check - in practice, would need data flow analysis
        func_name_lower = func_node.name.lower()
        
        # Functions that typically handle external input
        external_indicators = ['load', 'read', 'parse', 'process', 'handle', 'input']
        return any(indicator in func_name_lower for indicator in external_indicators)
    
    def _check_path_parameter(self, func_node: ast.FunctionDef, param_name: str) -> None:
        """Check if a file path parameter has proper validation
        
        Args:
            func_node: The function definition node
            param_name: Name of the path parameter
        """
        # Check for path validation in function body
        has_directory_validation = self._has_directory_validation(func_node, param_name)
        has_existence_check = self._has_existence_check(func_node, param_name)
        
        # If no directory validation, report HIGH severity vulnerability
        if not has_directory_validation:
            vuln = self._create_vulnerability(
                node=func_node,
                param_name=param_name,
                issue_type='path_directory',
                severity=SeverityLevel.HIGH,
                has_validation=False,
            )
            self.vulnerabilities.append(vuln)
    
    def _check_function_in_dataset_loader(self, func_node: ast.FunctionDef) -> None:
        """Check a function that is part of a dataset loader class
        
        Args:
            func_node: The function definition node
        """
        # Skip functions with no parameters (except self/cls)
        params = [arg for arg in func_node.args.args if arg.arg not in ('self', 'cls')]
        if not params:
            return
        
        # Analyze each parameter
        for param in params:
            param_name = param.arg
            param_type = self._infer_parameter_type(func_node, param_name)
            
            if param_type == 'path':
                # Check for directory validation
                has_directory_validation = self._has_directory_validation(func_node, param_name)
                if not has_directory_validation:
                    vuln = self._create_vulnerability(
                        node=func_node,
                        param_name=param_name,
                        issue_type='path_directory',
                        severity=SeverityLevel.HIGH,
                        has_validation=False,
                    )
                    self.vulnerabilities.append(vuln)
                
                # Check for existence check (specific to dataset loaders)
                has_existence_check = self._has_existence_check(func_node, param_name)
                if not has_existence_check:
                    vuln = self._create_vulnerability(
                        node=func_node,
                        param_name=param_name,
                        issue_type='path_existence',
                        severity=SeverityLevel.LOW,
                        has_validation=False,
                    )
                    self.vulnerabilities.append(vuln)
    
    def _check_numeric_parameter(self, func_node: ast.FunctionDef, param_name: str) -> None:
        """Check if a numeric parameter has range validation
        
        Args:
            func_node: The function definition node
            param_name: Name of the numeric parameter
        """
        has_range_validation = self._has_range_validation(func_node, param_name)
        
        if not has_range_validation:
            vuln = self._create_vulnerability(
                node=func_node,
                param_name=param_name,
                issue_type='numeric_range',
                severity=SeverityLevel.LOW,
                has_validation=False,
            )
            self.vulnerabilities.append(vuln)
    
    def _check_string_parameter(self, func_node: ast.FunctionDef, param_name: str) -> None:
        """Check if a string parameter has length validation
        
        Args:
            func_node: The function definition node
            param_name: Name of the string parameter
        """
        has_length_validation = self._has_length_validation(func_node, param_name)
        
        if not has_length_validation:
            vuln = self._create_vulnerability(
                node=func_node,
                param_name=param_name,
                issue_type='string_length',
                severity=SeverityLevel.MEDIUM,
                has_validation=False,
            )
            self.vulnerabilities.append(vuln)
    
    def _check_external_input(self, func_node: ast.FunctionDef, param_name: str) -> None:
        """Check if external input has type checking
        
        Args:
            func_node: The function definition node
            param_name: Name of the parameter
        """
        has_type_checking = self._has_type_checking(func_node, param_name)
        
        if not has_type_checking:
            vuln = self._create_vulnerability(
                node=func_node,
                param_name=param_name,
                issue_type='type_checking',
                severity=SeverityLevel.MEDIUM,
                has_validation=False,
            )
            self.vulnerabilities.append(vuln)
    
    def _has_directory_validation(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if path parameter has directory validation
        
        Args:
            func_node: The function definition node
            param_name: Name of the path parameter
            
        Returns:
            True if directory validation is present
        """
        for node in ast.walk(func_node):
            # Check for path validation function calls
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name:
                    # Check for path validation functions
                    for validation_func in self.PATH_VALIDATION_FUNCTIONS:
                        if validation_func in func_name:
                            # Check if it's validating our parameter
                            for arg in node.args:
                                if isinstance(arg, ast.Name) and arg.id == param_name:
                                    return True
                    
                    # Check for startswith() method calls on the parameter
                    if func_name == 'startswith':
                        # Check if it's called on our parameter or a variable derived from it
                        if isinstance(node.func, ast.Attribute):
                            # Check the object being called
                            obj = node.func.value
                            if isinstance(obj, ast.Call):
                                # Check if it's str(param_name)
                                inner_func = ASTAnalyzer.get_function_name(obj)
                                if inner_func == 'str' and obj.args:
                                    if isinstance(obj.args[0], ast.Name) and obj.args[0].id == param_name:
                                        return True
            
            # Check for startswith/in checks for allowed directories
            if isinstance(node, ast.Compare):
                if self._references_variable(node, param_name):
                    # Check if comparing against allowed paths
                    for comparator in node.comparators:
                        if isinstance(comparator, (ast.Str, ast.Constant)):
                            return True
                    # Check for 'in' operator with allowed paths
                    for op in node.ops:
                        if isinstance(op, ast.In):
                            return True
        
        return False
    
    def _has_existence_check(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if path parameter has existence check
        
        Args:
            func_node: The function definition node
            param_name: Name of the path parameter
            
        Returns:
            True if existence check is present
        """
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name:
                    # Check for existence check functions
                    existence_checks = ['exists', 'isfile', 'isdir', 'is_file', 'is_dir']
                    if any(check in func_name for check in existence_checks):
                        # Check if it's checking our parameter
                        for arg in node.args:
                            if isinstance(arg, ast.Name) and arg.id == param_name:
                                return True
                        
                        # Check if it's checking a variable derived from our parameter
                        # e.g., file_path = Path(data_path); file_path.exists()
                        if isinstance(node.func, ast.Attribute):
                            obj = node.func.value
                            if isinstance(obj, ast.Name):
                                # Check if this variable was assigned from our parameter
                                if self._is_variable_derived_from_param(func_node, obj.id, param_name):
                                    return True
        
        return False
    
    def _is_variable_derived_from_param(self, func_node: ast.FunctionDef, var_name: str, param_name: str) -> bool:
        """Check if a variable is derived from a parameter
        
        Args:
            func_node: The function definition node
            var_name: Name of the variable to check
            param_name: Name of the parameter
            
        Returns:
            True if the variable is derived from the parameter
        """
        for node in ast.walk(func_node):
            # Check for assignments: var_name = ... param_name ...
            if isinstance(node, ast.Assign):
                # Check if var_name is being assigned
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        # Check if the value references param_name
                        if self._references_variable(node.value, param_name):
                            return True
        
        return False
    
    def _has_range_validation(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if numeric parameter has range validation
        
        Args:
            func_node: The function definition node
            param_name: Name of the numeric parameter
            
        Returns:
            True if range validation is present
        """
        for node in ast.walk(func_node):
            # Check for comparison operations
            if isinstance(node, ast.Compare):
                if self._references_variable(node, param_name):
                    return True
            
            # Check for range/min/max calls
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name and any(v in func_name for v in ['range', 'min', 'max', 'clamp']):
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == param_name:
                            return True
        
        return False
    
    def _has_length_validation(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if string parameter has length validation
        
        Args:
            func_node: The function definition node
            param_name: Name of the string parameter
            
        Returns:
            True if length validation is present
        """
        for node in ast.walk(func_node):
            # Check for len() calls with comparison
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name == 'len':
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == param_name:
                            # Check if len() result is compared
                            return True
            
            # Check for comparison operations on the parameter
            if isinstance(node, ast.Compare):
                # Check if comparing len(param)
                if isinstance(node.left, ast.Call):
                    func_name = ASTAnalyzer.get_function_name(node.left)
                    if func_name == 'len':
                        for arg in node.left.args:
                            if isinstance(arg, ast.Name) and arg.id == param_name:
                                return True
        
        return False
    
    def _has_type_checking(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        """Check if parameter has type checking
        
        Args:
            func_node: The function definition node
            param_name: Name of the parameter
            
        Returns:
            True if type checking is present
        """
        # Check for type annotations
        for arg in func_node.args.args:
            if arg.arg == param_name and arg.annotation is not None:
                return True
        
        # Check for isinstance/type calls in function body
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                func_name = ASTAnalyzer.get_function_name(node)
                if func_name:
                    # Check for type checking functions
                    for type_func in self.TYPE_CHECKING_FUNCTIONS:
                        if type_func in func_name:
                            # Check if it's checking our parameter
                            for arg in node.args:
                                if isinstance(arg, ast.Name) and arg.id == param_name:
                                    return True
        
        return False
    
    def _is_dataset_loader(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is part of a dataset loader class
        
        Args:
            func_node: The function definition node
            
        Returns:
            True if function is in a dataset loader class
        """
        # Walk up to find parent class
        # This is simplified - in practice would need parent tracking
        func_name_lower = func_node.name.lower()
        return any(loader in func_name_lower for loader in ['load', 'read', 'dataset'])
    
    def _check_class(self, node: ast.ClassDef) -> None:
        """Check if a class is a dataset loader and validate its methods
        
        Args:
            node: The ClassDef node to analyze
        """
        # Check if this is a dataset loader class
        is_dataset_loader = any(
            loader in node.name for loader in self.DATASET_LOADERS
        )
        
        if not is_dataset_loader:
            return
        
        # Check methods in the class
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Check __init__ and load methods for path validation
                if item.name in ('__init__', 'load', 'read', '__getitem__'):
                    # Mark this function as part of a dataset loader
                    self._check_function_in_dataset_loader(item)
    
    def _references_variable(self, node: ast.AST, var_name: str) -> bool:
        """Check if a node references a specific variable
        
        Args:
            node: The AST node to check
            var_name: Name of the variable
            
        Returns:
            True if the node references the variable
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == var_name:
                return True
        return False
    
    def _create_vulnerability(
        self,
        node: ast.FunctionDef,
        param_name: str,
        issue_type: str,
        severity: SeverityLevel,
        has_validation: bool,
    ) -> Vulnerability:
        """Create a Vulnerability object for missing validation
        
        Args:
            node: The AST FunctionDef node
            param_name: Name of the parameter missing validation
            issue_type: Type of validation issue
            severity: Severity level for this vulnerability
            has_validation: Whether any validation is present
            
        Returns:
            Vulnerability object with all details
        """
        # Generate unique ID based on issue type
        issue_prefixes = {
            'path_directory': 'VAL',
            'path_existence': 'VAL',
            'numeric_range': 'VAL',
            'string_length': 'VAL',
            'type_checking': 'VAL',
        }
        prefix = issue_prefixes.get(issue_type, 'VAL')
        vuln_id = f"{prefix}{len(self.vulnerabilities) + 1:03d}"
        
        # Get line and column information
        line_number = node.lineno
        column = node.col_offset
        
        # Extract code snippet
        code_snippet = ASTAnalyzer.get_code_snippet(
            self.current_file.path,
            line_number,
            context_lines=2
        )
        
        # Generate title and description based on issue type
        title, description = self._get_title_description(issue_type, param_name, node.name)
        
        # Generate recommendation
        recommendation = self._get_recommendation(issue_type, param_name)
        
        # Determine CWE ID
        cwe_id = self._get_cwe_id(issue_type)
        
        # Calculate confidence score
        confidence = 0.8  # Medium confidence for validation issues
        
        return Vulnerability(
            id=vuln_id,
            title=title,
            description=description,
            severity=severity,
            vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
            file_path=self.current_file.path,
            line_number=line_number,
            column=column,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id,
            confidence=confidence,
            context={
                'parameter': param_name,
                'function': node.name,
                'issue_type': issue_type,
                'has_validation': has_validation,
            }
        )
    
    def _get_title_description(self, issue_type: str, param_name: str, func_name: str) -> tuple:
        """Generate title and description for validation issue
        
        Args:
            issue_type: Type of validation issue
            param_name: Name of the parameter
            func_name: Name of the function
            
        Returns:
            Tuple of (title, description)
        """
        titles = {
            'path_directory': f"File Path Parameter Without Directory Validation",
            'path_existence': f"File Path Parameter Without Existence Check",
            'numeric_range': f"Numeric Parameter Without Range Validation",
            'string_length': f"String Parameter Without Length Limit",
            'type_checking': f"External Input Without Type Checking",
        }
        
        descriptions = {
            'path_directory': (
                f"The function '{func_name}' accepts a file path parameter '{param_name}' "
                f"without validating that it is within allowed directories. This could allow "
                f"path traversal attacks where an attacker provides paths like '../../../etc/passwd' "
                f"to access files outside the intended directory."
            ),
            'path_existence': (
                f"The function '{func_name}' accepts a file path parameter '{param_name}' "
                f"without checking if the file exists. This could lead to runtime errors or "
                f"unexpected behavior when the file is not found."
            ),
            'numeric_range': (
                f"The function '{func_name}' accepts a numeric parameter '{param_name}' "
                f"without validating that it is within an acceptable range. This could lead to "
                f"integer overflow, buffer overflow, or unexpected behavior with extreme values."
            ),
            'string_length': (
                f"The function '{func_name}' accepts a string parameter '{param_name}' "
                f"without validating its length. This could lead to denial of service attacks "
                f"through extremely long strings or buffer overflow vulnerabilities."
            ),
            'type_checking': (
                f"The function '{func_name}' accepts external input parameter '{param_name}' "
                f"without type checking. This could lead to type confusion vulnerabilities or "
                f"unexpected behavior when the wrong type is provided."
            ),
        }
        
        title = titles.get(issue_type, "Missing Input Validation")
        description = descriptions.get(issue_type, "Input validation is missing.")
        
        return title, description
    
    def _get_recommendation(self, issue_type: str, param_name: str) -> str:
        """Generate remediation recommendation for validation issue
        
        Args:
            issue_type: Type of validation issue
            param_name: Name of the parameter
            
        Returns:
            Remediation recommendation string
        """
        recommendations = {
            'path_directory': (
                f"Add directory validation for the '{param_name}' parameter:\n"
                f"1. Use pathlib.Path.resolve() to get the absolute path\n"
                f"2. Check that the resolved path starts with an allowed base directory\n"
                f"3. Reject paths containing '..' or other traversal patterns\n\n"
                f"Example:\n"
                f"from pathlib import Path\n"
                f"allowed_dir = Path('/safe/directory').resolve()\n"
                f"file_path = Path({param_name}).resolve()\n"
                f"if not file_path.is_relative_to(allowed_dir):\n"
                f"    raise ValueError('Path outside allowed directory')"
            ),
            'path_existence': (
                f"Add file existence check for the '{param_name}' parameter:\n"
                f"1. Use os.path.exists() or Path.exists() to verify the file exists\n"
                f"2. Handle the case when the file is not found gracefully\n\n"
                f"Example:\n"
                f"from pathlib import Path\n"
                f"file_path = Path({param_name})\n"
                f"if not file_path.exists():\n"
                f"    raise FileNotFoundError(f'File not found: {{file_path}}')"
            ),
            'numeric_range': (
                f"Add range validation for the '{param_name}' parameter:\n"
                f"1. Define acceptable minimum and maximum values\n"
                f"2. Check that the value is within the valid range\n"
                f"3. Raise an error or clamp the value if out of range\n\n"
                f"Example:\n"
                f"if not (0 <= {param_name} <= 100):\n"
                f"    raise ValueError(f'{{param_name}} must be between 0 and 100')"
            ),
            'string_length': (
                f"Add length validation for the '{param_name}' parameter:\n"
                f"1. Define a maximum acceptable length\n"
                f"2. Check the length before processing\n"
                f"3. Reject or truncate strings that exceed the limit\n\n"
                f"Example:\n"
                f"MAX_LENGTH = 1000\n"
                f"if len({param_name}) > MAX_LENGTH:\n"
                f"    raise ValueError(f'String too long: {{len({param_name})}} > {{MAX_LENGTH}}')"
            ),
            'type_checking': (
                f"Add type checking for the '{param_name}' parameter:\n"
                f"1. Use type annotations to document expected types\n"
                f"2. Use isinstance() to validate the type at runtime\n"
                f"3. Raise TypeError if the wrong type is provided\n\n"
                f"Example:\n"
                f"if not isinstance({param_name}, str):\n"
                f"    raise TypeError(f'Expected str, got {{type({param_name}).__name__}}')"
            ),
        }
        
        return recommendations.get(issue_type, "Add appropriate input validation.")
    
    def _get_cwe_id(self, issue_type: str) -> str:
        """Get CWE ID for validation issue type
        
        Args:
            issue_type: Type of validation issue
            
        Returns:
            CWE ID string
        """
        cwe_ids = {
            'path_directory': 'CWE-22',  # Path Traversal
            'path_existence': 'CWE-73',  # External Control of File Name or Path
            'numeric_range': 'CWE-190',  # Integer Overflow or Wraparound
            'string_length': 'CWE-120',  # Buffer Copy without Checking Size of Input
            'type_checking': 'CWE-20',   # Improper Input Validation
        }
        
        return cwe_ids.get(issue_type, 'CWE-20')
