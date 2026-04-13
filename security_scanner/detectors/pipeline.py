"""Data Pipeline Detector

This module implements detection of security issues in data processing pipelines including:
- Data loaders with unvalidated file paths
- Shell command execution with user input (subprocess, os.system)
- Missing error handling for malformed input
- Dataset classes loading files without size limits
- Random seeds from external sources

The detector uses AST visitor pattern to identify data pipeline security issues.
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


class PipelineDetector(BaseDetector):
    """Detector for data pipeline security issues
    
    This detector identifies security issues in data processing pipelines:
    - Unvalidated file paths in data loaders
    - Shell command execution with user input (subprocess, os.system)
    - Missing error handling for malformed input
    - Dataset classes loading files without size limits
    - Random seeds from external sources
    
    The detector performs AST analysis to identify these patterns and assigns
    appropriate severity levels based on the security impact.
    """
    
    # Shell command execution functions (CRITICAL severity)
    SHELL_FUNCTIONS = {
        'os.system': SeverityLevel.CRITICAL,
        'subprocess.call': SeverityLevel.CRITICAL,
        'subprocess.run': SeverityLevel.CRITICAL,
        'subprocess.Popen': SeverityLevel.CRITICAL,
        'os.popen': SeverityLevel.CRITICAL,
        'commands.getoutput': SeverityLevel.CRITICAL,
        'commands.getstatusoutput': SeverityLevel.CRITICAL,
    }
    
    # File loading functions to check for validation
    FILE_LOADING_FUNCTIONS = {
        'open', 'load', 'read', 'read_csv', 'read_json', 'read_parquet',
        'load_dataset', 'DataLoader', 'Dataset', 'ImageFolder',
    }
    
    # Random seed functions
    RANDOM_SEED_FUNCTIONS = {
        'random.seed', 'np.random.seed', 'torch.manual_seed',
        'torch.cuda.manual_seed', 'torch.cuda.manual_seed_all',
        'random.Random', 'numpy.random.RandomState',
    }
    
    def __init__(self, config: DetectorConfig):
        """Initialize the pipeline detector
        
        Args:
            config: Configuration for the detector
        """
        super().__init__(config)
        self.vulnerabilities: List[Vulnerability] = []
        self.current_file: Optional[FileInfo] = None
        self.ast_tree: Optional[ast.AST] = None
    
    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        """Detect data pipeline security issues in a file
        
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
        """Recursively visit AST nodes to find data pipeline issues
        
        Args:
            node: The AST node to visit
        """
        # Check function calls
        if isinstance(node, ast.Call):
            self._check_shell_command(node)
            self._check_file_loading(node)
            self._check_random_seed(node)
        
        # Check class definitions for Dataset classes
        if isinstance(node, ast.ClassDef):
            self._check_dataset_class(node)
        
        # Check function definitions for error handling
        if isinstance(node, ast.FunctionDef):
            self._check_error_handling(node)
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self._visit_node(child)
    
    def _check_shell_command(self, node: ast.Call) -> None:
        """Check for shell command execution with user input
        
        Args:
            node: The Call node to analyze
        """
        func_name = ASTAnalyzer.get_function_name(node)
        
        if not func_name:
            return
        
        # Check if this is a shell command function
        if func_name not in self.SHELL_FUNCTIONS:
            return
        
        # Check if input is user-controlled
        user_controlled = False
        if node.args:
            # Check first argument (the command)
            user_controlled = ASTAnalyzer.is_user_controlled(node.args[0], self.ast_tree)
        
        # Check for shell=True in subprocess calls
        has_shell_true = False
        for keyword in node.keywords:
            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    has_shell_true = True
        
        # Always flag shell commands with user input as CRITICAL
        if user_controlled or has_shell_true:
            severity = SeverityLevel.CRITICAL
        else:
            severity = SeverityLevel.HIGH
        
        vuln = self._create_vulnerability(
            node=node,
            issue_type='shell_command_injection',
            severity=severity,
            func_name=func_name,
            user_controlled=user_controlled,
            has_shell_true=has_shell_true,
        )
        
        self.vulnerabilities.append(vuln)
    
    def _check_file_loading(self, node: ast.Call) -> None:
        """Check for file loading without path validation
        
        Args:
            node: The Call node to analyze
        """
        func_name = ASTAnalyzer.get_function_name(node)
        
        if not func_name:
            return
        
        # Check if this is a file loading function
        is_file_loading = any(pattern in func_name for pattern in self.FILE_LOADING_FUNCTIONS)
        
        if not is_file_loading:
            return
        
        # Check if path argument is user-controlled
        user_controlled = False
        has_validation = False
        
        if node.args:
            # Check first argument (usually the file path)
            first_arg = node.args[0]
            user_controlled = ASTAnalyzer.is_user_controlled(first_arg, self.ast_tree)
            has_validation = ASTAnalyzer.has_validation(first_arg, self.ast_tree)
        
        # Only flag if user-controlled and no validation
        if user_controlled and not has_validation:
            severity = SeverityLevel.MEDIUM
            
            vuln = self._create_vulnerability(
                node=node,
                issue_type='unvalidated_file_path',
                severity=severity,
                func_name=func_name,
                user_controlled=user_controlled,
                has_validation=has_validation,
            )
            
            self.vulnerabilities.append(vuln)
    
    def _check_dataset_class(self, node: ast.ClassDef) -> None:
        """Check Dataset classes for missing size limits
        
        Args:
            node: The ClassDef node to analyze
        """
        # Check if this is a Dataset class
        is_dataset = False
        for base in node.bases:
            if isinstance(base, ast.Name) and 'Dataset' in base.id:
                is_dataset = True
            elif isinstance(base, ast.Attribute) and 'Dataset' in base.attr:
                is_dataset = True
        
        if not is_dataset:
            return
        
        # Look for __getitem__ or load methods
        has_size_check = False
        load_methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name in ['__getitem__', 'load', 'load_data', 'read']:
                    load_methods.append(item)
                    # Check if there's size validation
                    if self._has_size_check(item):
                        has_size_check = True
        
        # Flag if Dataset loads files without size checks
        if load_methods and not has_size_check:
            # Use the class definition line
            vuln = self._create_vulnerability(
                node=node,
                issue_type='missing_size_limit',
                severity=SeverityLevel.MEDIUM,
                func_name=node.name,
                user_controlled=False,
                has_validation=False,
                context_info={'methods': [m.name for m in load_methods]},
            )
            
            self.vulnerabilities.append(vuln)
    
    def _has_size_check(self, func_node: ast.FunctionDef) -> bool:
        """Check if a function has size validation
        
        Args:
            func_node: The function definition to check
            
        Returns:
            True if size checks are present, False otherwise
        """
        # Look for size-related comparisons or checks
        for node in ast.walk(func_node):
            # Check for comparisons with size/length
            if isinstance(node, ast.Compare):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = ASTAnalyzer.get_function_name(child)
                        if func_name and ('len' in func_name or 'size' in func_name):
                            return True
                    if isinstance(child, ast.Attribute):
                        if child.attr in ['size', 'length', 'nbytes']:
                            return True
            
            # Check for explicit size limits in constants
            if isinstance(node, ast.Constant):
                if isinstance(node.value, int) and node.value > 1000:
                    # Likely a size limit constant
                    return True
        
        return False
    
    def _check_error_handling(self, node: ast.FunctionDef) -> None:
        """Check for missing error handling in data processing functions
        
        Args:
            node: The FunctionDef node to analyze
        """
        # Check if this is a data processing function
        is_data_processing = any(keyword in node.name.lower() for keyword in [
            'load', 'parse', 'process', 'read', 'decode', 'transform'
        ])
        
        if not is_data_processing:
            return
        
        # Check if function has try-except blocks
        has_error_handling = False
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                has_error_handling = True
                break
        
        # Check if function has file/data operations that could fail
        has_risky_operations = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = ASTAnalyzer.get_function_name(child)
                if func_name:
                    risky_patterns = ['open', 'load', 'read', 'parse', 'decode', 'json.loads', 'yaml.load']
                    if any(pattern in func_name for pattern in risky_patterns):
                        has_risky_operations = True
                        break
        
        # Flag if risky operations without error handling
        if has_risky_operations and not has_error_handling:
            vuln = self._create_vulnerability(
                node=node,
                issue_type='missing_error_handling',
                severity=SeverityLevel.LOW,
                func_name=node.name,
                user_controlled=False,
                has_validation=False,
            )
            
            self.vulnerabilities.append(vuln)
    
    def _check_random_seed(self, node: ast.Call) -> None:
        """Check for random seeds from external sources
        
        Args:
            node: The Call node to analyze
        """
        func_name = ASTAnalyzer.get_function_name(node)
        
        if not func_name:
            return
        
        # Check if this is a random seed function
        is_seed_function = any(pattern in func_name for pattern in self.RANDOM_SEED_FUNCTIONS)
        
        if not is_seed_function:
            return
        
        # Check if seed value is from external source
        external_source = False
        if node.args:
            first_arg = node.args[0]
            external_source = ASTAnalyzer.is_user_controlled(first_arg, self.ast_tree)
        
        # Only flag if seed is from external source
        if external_source:
            vuln = self._create_vulnerability(
                node=node,
                issue_type='external_random_seed',
                severity=SeverityLevel.LOW,
                func_name=func_name,
                user_controlled=external_source,
                has_validation=False,
            )
            
            self.vulnerabilities.append(vuln)
    
    def _create_vulnerability(
        self,
        node: ast.AST,
        issue_type: str,
        severity: SeverityLevel,
        func_name: str,
        user_controlled: bool,
        has_validation: bool,
        has_shell_true: bool = False,
        context_info: dict = None,
    ) -> Vulnerability:
        """Create a Vulnerability object for a detected pipeline issue
        
        Args:
            node: The AST node containing the issue
            issue_type: Type of issue detected
            severity: Severity level for this vulnerability
            func_name: Name of the function involved
            user_controlled: Whether input is user-controlled
            has_validation: Whether validation is present
            has_shell_true: Whether shell=True is used (for subprocess)
            context_info: Additional context information
            
        Returns:
            Vulnerability object with all details
        """
        if context_info is None:
            context_info = {}
        
        # Generate unique ID
        vuln_id = f"PIPE{len(self.vulnerabilities) + 1:03d}"
        
        # Get line and column information
        line_number = getattr(node, 'lineno', 1)
        column = getattr(node, 'col_offset', 0)
        
        # Extract code snippet
        code_snippet = ASTAnalyzer.get_code_snippet(
            self.current_file.path,
            line_number,
            context_lines=2
        )
        
        # Generate title, description, and recommendation
        title = self._get_title(issue_type, func_name)
        description = self._get_description(issue_type, func_name, user_controlled, has_validation, has_shell_true)
        recommendation = self._get_recommendation(issue_type, func_name)
        
        # Determine CWE ID
        cwe_id = self._get_cwe_id(issue_type)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(issue_type, user_controlled, has_validation)
        
        return Vulnerability(
            id=vuln_id,
            title=title,
            description=description,
            severity=severity,
            vulnerability_type=VulnerabilityType.DATA_PIPELINE,
            file_path=self.current_file.path,
            line_number=line_number,
            column=column,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id,
            confidence=confidence,
            context={
                'issue_type': issue_type,
                'function': func_name,
                'user_controlled': user_controlled,
                'has_validation': has_validation,
                'has_shell_true': has_shell_true,
                **context_info,
            }
        )
    
    def _get_title(self, issue_type: str, func_name: str) -> str:
        """Generate title for vulnerability
        
        Args:
            issue_type: Type of issue detected
            func_name: Name of the function involved
            
        Returns:
            Title string
        """
        titles = {
            'shell_command_injection': f"Shell Command Injection via {func_name}",
            'unvalidated_file_path': f"Unvalidated File Path in {func_name}",
            'missing_size_limit': f"Missing Size Limit in Dataset Class {func_name}",
            'missing_error_handling': f"Missing Error Handling in {func_name}",
            'external_random_seed': f"Random Seed from External Source in {func_name}",
        }
        
        return titles.get(issue_type, "Data Pipeline Security Issue")
    
    def _get_description(
        self,
        issue_type: str,
        func_name: str,
        user_controlled: bool,
        has_validation: bool,
        has_shell_true: bool,
    ) -> str:
        """Generate description for vulnerability
        
        Args:
            issue_type: Type of issue detected
            func_name: Name of the function involved
            user_controlled: Whether input is user-controlled
            has_validation: Whether validation is present
            has_shell_true: Whether shell=True is used
            
        Returns:
            Description string
        """
        descriptions = {
            'shell_command_injection': (
                f"The function {func_name} executes shell commands which can lead to command injection attacks. "
                + ("This call uses user-controlled input, making it a critical security risk. " if user_controlled else "")
                + ("The shell=True parameter is used, which increases the attack surface. " if has_shell_true else "")
                + "An attacker could inject malicious commands to be executed by the system."
            ),
            'unvalidated_file_path': (
                f"The function {func_name} loads files using user-controlled paths without validation. "
                "This could allow an attacker to access files outside the intended directory, "
                "leading to unauthorized data access or path traversal attacks."
            ),
            'missing_size_limit': (
                f"The Dataset class {func_name} loads files without checking their size. "
                "This could lead to denial of service attacks by loading extremely large files, "
                "causing memory exhaustion or performance degradation."
            ),
            'missing_error_handling': (
                f"The function {func_name} performs data processing operations without proper error handling. "
                "Malformed input could cause unexpected crashes or expose sensitive error information. "
                "Proper error handling is essential for robust data pipeline security."
            ),
            'external_random_seed': (
                f"The function {func_name} uses a random seed from an external source. "
                "This could allow an attacker to control the random number generation, "
                "potentially compromising the security of cryptographic operations or model training."
            ),
        }
        
        return descriptions.get(issue_type, "Data pipeline security issue detected.")
    
    def _get_recommendation(self, issue_type: str, func_name: str) -> str:
        """Generate remediation recommendation
        
        Args:
            issue_type: Type of issue detected
            func_name: Name of the function involved
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'shell_command_injection': (
                "Avoid executing shell commands with user input:\n"
                "1. Use subprocess with shell=False and pass arguments as a list\n"
                "2. Validate and sanitize all input before use\n"
                "3. Use an allow-list of permitted commands\n"
                "4. Consider using safer alternatives (e.g., Python libraries instead of shell commands)\n\n"
                "Example:\n"
                "# BAD: subprocess.run(user_input, shell=True)\n"
                "# GOOD: subprocess.run(['ls', validated_path], shell=False)"
            ),
            'unvalidated_file_path': (
                "Validate file paths before loading:\n"
                "1. Use pathlib.Path.resolve() to normalize paths\n"
                "2. Check that resolved path is within allowed directory\n"
                "3. Validate file extensions against an allow-list\n"
                "4. Check file existence and permissions\n\n"
                "Example:\n"
                "from pathlib import Path\n"
                "allowed_dir = Path('/data').resolve()\n"
                "file_path = Path(user_input).resolve()\n"
                "if not file_path.is_relative_to(allowed_dir):\n"
                "    raise ValueError('Invalid path')"
            ),
            'missing_size_limit': (
                "Implement size limits for file loading:\n"
                "1. Check file size before loading\n"
                "2. Set maximum file size limits (e.g., 100MB)\n"
                "3. Use streaming for large files\n"
                "4. Implement memory limits for data processing\n\n"
                "Example:\n"
                "MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB\n"
                "if os.path.getsize(file_path) > MAX_FILE_SIZE:\n"
                "    raise ValueError('File too large')"
            ),
            'missing_error_handling': (
                "Add proper error handling for data processing:\n"
                "1. Wrap risky operations in try-except blocks\n"
                "2. Handle specific exceptions (IOError, ValueError, etc.)\n"
                "3. Log errors without exposing sensitive information\n"
                "4. Provide graceful degradation for malformed input\n\n"
                "Example:\n"
                "try:\n"
                "    data = json.loads(input_data)\n"
                "except json.JSONDecodeError as e:\n"
                "    logger.error('Invalid JSON input')\n"
                "    return None"
            ),
            'external_random_seed': (
                "Use secure random seed generation:\n"
                "1. Generate seeds internally using secure random sources\n"
                "2. Do not accept seeds from user input\n"
                "3. Use secrets module for cryptographic operations\n"
                "4. Document seed generation for reproducibility\n\n"
                "Example:\n"
                "import secrets\n"
                "seed = secrets.randbits(32)  # Secure random seed\n"
                "torch.manual_seed(seed)"
            ),
        }
        
        return recommendations.get(issue_type, "Review and secure the data pipeline.")
    
    def _get_cwe_id(self, issue_type: str) -> str:
        """Get CWE ID for issue type
        
        Args:
            issue_type: Type of issue detected
            
        Returns:
            CWE ID string
        """
        cwe_ids = {
            'shell_command_injection': 'CWE-78',  # OS Command Injection
            'unvalidated_file_path': 'CWE-22',  # Path Traversal
            'missing_size_limit': 'CWE-400',  # Uncontrolled Resource Consumption
            'missing_error_handling': 'CWE-755',  # Improper Handling of Exceptional Conditions
            'external_random_seed': 'CWE-330',  # Use of Insufficiently Random Values
        }
        
        return cwe_ids.get(issue_type, 'CWE-20')  # Improper Input Validation
    
    def _calculate_confidence(self, issue_type: str, user_controlled: bool, has_validation: bool) -> float:
        """Calculate confidence score for vulnerability
        
        Args:
            issue_type: Type of issue detected
            user_controlled: Whether input is user-controlled
            has_validation: Whether validation is present
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence levels by issue type
        confidence_map = {
            'shell_command_injection': 1.0,
            'unvalidated_file_path': 0.8,
            'missing_size_limit': 0.7,
            'missing_error_handling': 0.6,
            'external_random_seed': 0.7,
        }
        
        confidence = confidence_map.get(issue_type, 0.8)
        
        # Adjust based on context
        if not user_controlled and issue_type in ['shell_command_injection', 'unvalidated_file_path']:
            confidence *= 0.7
        
        if has_validation:
            confidence *= 0.5
        
        return min(1.0, confidence)
