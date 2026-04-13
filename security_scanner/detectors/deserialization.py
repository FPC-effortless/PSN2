"""Deserialization Detector

This module implements detection of unsafe deserialization vulnerabilities including:
- torch.load() without weights_only=True parameter
- pickle.load() on untrusted data sources
- yaml.load() without SafeLoader
- JSON deserialization without schema validation

The detector uses AST visitor pattern to identify deserialization function calls
and analyzes parameters to determine if safe practices are followed.
"""

import ast
from typing import List, Optional

from security_scanner.ast_analyzer import ASTAnalyzer
from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    SeverityLevel,
    Vulnerability,
    VulnerabilityType,
)


class DeserializationDetector(BaseDetector):
    """Detector for unsafe deserialization vulnerabilities
    
    This detector identifies unsafe deserialization patterns that can lead to
    arbitrary code execution or data corruption:
    - torch.load(): PyTorch checkpoint loading without weights_only=True
    - pickle.load(): Python object deserialization (inherently unsafe)
    - yaml.load(): YAML parsing without SafeLoader
    - json.load()/json.loads(): JSON parsing without schema validation
    
    The detector performs parameter analysis to determine if safe practices
    are being followed and assigns severity based on the risk level.
    """
    
    # Mapping of deserialization functions to their base severity levels
    DESERIALIZATION_FUNCTIONS = {
        'torch.load': SeverityLevel.HIGH,
        'pickle.load': SeverityLevel.CRITICAL,
        'pickle.loads': SeverityLevel.CRITICAL,
        'yaml.load': SeverityLevel.HIGH,
        'yaml.unsafe_load': SeverityLevel.HIGH,
        'json.load': SeverityLevel.MEDIUM,
        'json.loads': SeverityLevel.MEDIUM,
    }
    
    def __init__(self, config: DetectorConfig):
        """Initialize the deserialization detector
        
        Args:
            config: Configuration for the detector
        """
        super().__init__(config)
        self.vulnerabilities: List[Vulnerability] = []
        self.current_file: Optional[FileInfo] = None
        self.ast_tree: Optional[ast.AST] = None
    
    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        """Detect unsafe deserialization vulnerabilities in a file
        
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
        
        # Visit all nodes in the AST
        self._visit_node(ast_tree)
        
        # Filter out suppressed vulnerabilities
        return [v for v in self.vulnerabilities if not self.should_suppress(v)]
    
    def _visit_node(self, node: ast.AST) -> None:
        """Recursively visit AST nodes to find deserialization function calls
        
        Args:
            node: The AST node to visit
        """
        # Check if this is a function call
        if isinstance(node, ast.Call):
            self._check_call(node)
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self._visit_node(child)
    
    def _check_call(self, node: ast.Call) -> None:
        """Check if a function call is an unsafe deserialization point
        
        Args:
            node: The Call node to analyze
        """
        func_name = ASTAnalyzer.get_function_name(node)
        
        if not func_name:
            return
        
        # Check if this is a deserialization function
        if func_name not in self.DESERIALIZATION_FUNCTIONS:
            return
        
        # Get base severity for this function
        base_severity = self.DESERIALIZATION_FUNCTIONS[func_name]
        
        # Check for safe usage patterns
        is_safe = self._is_safe_usage(node, func_name)
        
        # If usage is safe, don't report vulnerability
        if is_safe:
            return
        
        # Analyze context to determine if input is user-controlled
        user_controlled = self._is_input_user_controlled(node)
        
        # Check if there's validation present
        has_validation = ASTAnalyzer.has_validation(node, self.ast_tree)
        
        # Determine final severity based on context
        context = {
            'pattern': func_name,
            'base_severity': base_severity,
            'user_controlled': user_controlled,
            'has_validation': has_validation,
            'file_path': self.current_file.path,
        }
        
        severity = self.get_severity(context)
        
        # Generate vulnerability
        vuln = self._create_vulnerability(
            node=node,
            func_name=func_name,
            severity=severity,
            user_controlled=user_controlled,
            has_validation=has_validation,
        )
        
        self.vulnerabilities.append(vuln)
    
    def _is_safe_usage(self, node: ast.Call, func_name: str) -> bool:
        """Check if deserialization function is used safely
        
        Args:
            node: The Call node to analyze
            func_name: Name of the deserialization function
            
        Returns:
            True if usage is safe, False otherwise
        """
        # Check torch.load() for weights_only=True
        if func_name == 'torch.load':
            return self._has_weights_only_true(node)
        
        # Check yaml.load() for SafeLoader
        if func_name == 'yaml.load':
            return self._has_safe_loader(node)
        
        # pickle.load() and pickle.loads() are always unsafe
        if func_name in ['pickle.load', 'pickle.loads']:
            return False
        
        # json.load() and json.loads() - check for schema validation
        # For now, we flag all JSON deserialization as potentially unsafe
        # unless there's explicit validation in the context
        if func_name in ['json.load', 'json.loads']:
            # Check if there's validation in the surrounding context
            return ASTAnalyzer.has_validation(node, self.ast_tree)
        
        return False
    
    def _has_weights_only_true(self, node: ast.Call) -> bool:
        """Check if torch.load() has weights_only=True parameter
        
        Args:
            node: The Call node for torch.load()
            
        Returns:
            True if weights_only=True is present, False otherwise
        """
        for keyword in node.keywords:
            if keyword.arg == 'weights_only':
                # Check if the value is True
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value is True
                # Handle ast.NameConstant for older Python versions
                if isinstance(keyword.value, ast.NameConstant):
                    return keyword.value.value is True
        return False
    
    def _has_safe_loader(self, node: ast.Call) -> bool:
        """Check if yaml.load() uses SafeLoader
        
        Args:
            node: The Call node for yaml.load()
            
        Returns:
            True if SafeLoader is used, False otherwise
        """
        for keyword in node.keywords:
            if keyword.arg == 'Loader':
                # Check if the value is SafeLoader or yaml.SafeLoader
                if isinstance(keyword.value, ast.Name):
                    return keyword.value.id == 'SafeLoader'
                if isinstance(keyword.value, ast.Attribute):
                    attr_path = ASTAnalyzer._get_attribute_path(keyword.value)
                    return 'SafeLoader' in attr_path
        return False
    
    def _is_input_user_controlled(self, node: ast.Call) -> bool:
        """Determine if the input to a deserialization function is user-controlled
        
        Args:
            node: The Call node to analyze
            
        Returns:
            True if input is potentially user-controlled, False otherwise
        """
        # Check the first argument (the data source)
        if not node.args:
            return False
        
        first_arg = node.args[0]
        
        # Use ASTAnalyzer to check if the argument is user-controlled
        return ASTAnalyzer.is_user_controlled(first_arg, self.ast_tree)
    
    def _create_vulnerability(
        self,
        node: ast.Call,
        func_name: str,
        severity: SeverityLevel,
        user_controlled: bool,
        has_validation: bool,
    ) -> Vulnerability:
        """Create a Vulnerability object for a detected unsafe deserialization
        
        Args:
            node: The AST Call node containing the deserialization function
            func_name: Name of the deserialization function
            severity: Severity level for this vulnerability
            user_controlled: Whether input is user-controlled
            has_validation: Whether validation is present
            
        Returns:
            Vulnerability object with all details
        """
        # Generate unique ID
        vuln_id = f"DES{len(self.vulnerabilities) + 1:03d}"
        
        # Get line and column information
        line_number = node.lineno
        column = node.col_offset
        
        # Extract code snippet
        code_snippet = ASTAnalyzer.get_code_snippet(
            self.current_file.path,
            line_number,
            context_lines=2
        )
        
        # Generate title and description
        title = f"Unsafe Deserialization via {func_name}()"
        
        description = self._get_description(func_name, user_controlled, has_validation)
        
        # Generate recommendation
        recommendation = self._get_recommendation(func_name, user_controlled)
        
        # Determine CWE ID
        cwe_id = "CWE-502"  # Deserialization of Untrusted Data
        
        # Calculate confidence score
        confidence = 1.0
        if not user_controlled:
            confidence = 0.8
        if has_validation:
            confidence *= 0.7
        
        return Vulnerability(
            id=vuln_id,
            title=title,
            description=description,
            severity=severity,
            vulnerability_type=VulnerabilityType.UNSAFE_DESERIALIZATION,
            file_path=self.current_file.path,
            line_number=line_number,
            column=column,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id,
            confidence=confidence,
            context={
                'function': func_name,
                'user_controlled': user_controlled,
                'has_validation': has_validation,
            }
        )
    
    def _get_description(self, func_name: str, user_controlled: bool, has_validation: bool) -> str:
        """Generate description for a specific deserialization vulnerability
        
        Args:
            func_name: Name of the deserialization function
            user_controlled: Whether input is user-controlled
            has_validation: Whether validation is present
            
        Returns:
            Description string
        """
        descriptions = {
            'torch.load': (
                "The function torch.load() can execute arbitrary code when loading "
                "malicious checkpoint files. Without the weights_only=True parameter, "
                "PyTorch will deserialize arbitrary Python objects, which can lead to "
                "remote code execution."
            ),
            'pickle.load': (
                "The function pickle.load() can execute arbitrary code during deserialization. "
                "Pickle is inherently unsafe when used with untrusted data, as it can "
                "instantiate arbitrary Python objects and execute code."
            ),
            'pickle.loads': (
                "The function pickle.loads() can execute arbitrary code during deserialization. "
                "Pickle is inherently unsafe when used with untrusted data, as it can "
                "instantiate arbitrary Python objects and execute code."
            ),
            'yaml.load': (
                "The function yaml.load() without SafeLoader can execute arbitrary Python code. "
                "YAML deserialization without SafeLoader allows instantiation of arbitrary "
                "Python objects, which can lead to remote code execution."
            ),
            'yaml.unsafe_load': (
                "The function yaml.unsafe_load() can execute arbitrary Python code. "
                "This function explicitly allows unsafe deserialization and should never "
                "be used with untrusted data."
            ),
            'json.load': (
                "The function json.load() deserializes JSON data without schema validation. "
                "While JSON deserialization is generally safer than pickle or YAML, "
                "lack of schema validation can lead to unexpected data structures and "
                "potential security issues."
            ),
            'json.loads': (
                "The function json.loads() deserializes JSON data without schema validation. "
                "While JSON deserialization is generally safer than pickle or YAML, "
                "lack of schema validation can lead to unexpected data structures and "
                "potential security issues."
            ),
        }
        
        base_description = descriptions.get(
            func_name,
            f"The function {func_name}() performs unsafe deserialization."
        )
        
        if user_controlled:
            base_description += (
                " This call uses user-controlled input, which makes it a critical security risk. "
                "An attacker could provide malicious serialized data to execute arbitrary code."
            )
        else:
            base_description += (
                " While the input does not appear to be directly user-controlled, "
                f"using {func_name}() with untrusted data is inherently dangerous."
            )
        
        if has_validation:
            base_description += " Some input validation was detected, which reduces the risk."
        
        return base_description
    
    def _get_recommendation(self, func_name: str, user_controlled: bool) -> str:
        """Generate remediation recommendation for a specific function
        
        Args:
            func_name: Name of the deserialization function
            user_controlled: Whether input is user-controlled
            
        Returns:
            Remediation recommendation string
        """
        recommendations = {
            'torch.load': (
                "Use torch.load() with weights_only=True parameter to safely load model weights. "
                "This prevents arbitrary code execution by restricting deserialization to tensor data only. "
                "Example: torch.load('checkpoint.pt', weights_only=True)\n\n"
                "For additional security, verify checkpoint integrity using cryptographic signatures "
                "before loading."
            ),
            'pickle.load': (
                "Avoid using pickle.load() with untrusted data. Pickle is inherently unsafe for "
                "deserialization of untrusted data. Consider using safer alternatives:\n"
                "- For model checkpoints: Use torch.load() with weights_only=True\n"
                "- For data serialization: Use JSON, Protocol Buffers, or MessagePack\n"
                "- For configuration: Use JSON or YAML with SafeLoader\n\n"
                "If pickle must be used, implement cryptographic signature verification and "
                "load data only from trusted sources."
            ),
            'pickle.loads': (
                "Avoid using pickle.loads() with untrusted data. Pickle is inherently unsafe for "
                "deserialization of untrusted data. Consider using safer alternatives:\n"
                "- For model checkpoints: Use torch.load() with weights_only=True\n"
                "- For data serialization: Use JSON, Protocol Buffers, or MessagePack\n"
                "- For configuration: Use JSON or YAML with SafeLoader\n\n"
                "If pickle must be used, implement cryptographic signature verification and "
                "load data only from trusted sources."
            ),
            'yaml.load': (
                "Use yaml.safe_load() or yaml.load() with Loader=yaml.SafeLoader to safely parse YAML. "
                "SafeLoader restricts deserialization to simple Python objects and prevents "
                "arbitrary code execution.\n\n"
                "Example: yaml.load(file, Loader=yaml.SafeLoader)\n"
                "Or better: yaml.safe_load(file)"
            ),
            'yaml.unsafe_load': (
                "Replace yaml.unsafe_load() with yaml.safe_load() to safely parse YAML. "
                "The unsafe_load function explicitly allows arbitrary code execution and "
                "should never be used.\n\n"
                "Example: yaml.safe_load(file)"
            ),
            'json.load': (
                "Implement schema validation for JSON data to ensure it matches expected structure. "
                "Use libraries like jsonschema or pydantic to validate deserialized data:\n\n"
                "Example with jsonschema:\n"
                "import jsonschema\n"
                "data = json.load(file)\n"
                "jsonschema.validate(data, schema)\n\n"
                "Example with pydantic:\n"
                "from pydantic import BaseModel\n"
                "data = json.load(file)\n"
                "validated_data = MyModel(**data)"
            ),
            'json.loads': (
                "Implement schema validation for JSON data to ensure it matches expected structure. "
                "Use libraries like jsonschema or pydantic to validate deserialized data:\n\n"
                "Example with jsonschema:\n"
                "import jsonschema\n"
                "data = json.loads(string)\n"
                "jsonschema.validate(data, schema)\n\n"
                "Example with pydantic:\n"
                "from pydantic import BaseModel\n"
                "data = json.loads(string)\n"
                "validated_data = MyModel(**data)"
            ),
        }
        
        base_recommendation = recommendations.get(
            func_name,
            "Avoid deserializing untrusted data without proper validation."
        )
        
        if user_controlled:
            base_recommendation += (
                "\n\nCRITICAL: This call uses user-controlled input. "
                "Implement strict input validation, cryptographic signature verification, "
                "and consider using safer serialization formats."
            )
        
        return base_recommendation
