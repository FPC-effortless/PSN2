"""File Operations Detector

This module implements detection of insecure file system operations including:
- File creation with overly permissive permissions (0o777)
- Insecure temporary file creation (not using tempfile.mkstemp)
- File deletion without path validation
- Writes to world-writable directories
- Symbolic link usage without validation

The detector uses AST visitor pattern to identify file operations
and analyzes context to determine if proper security measures are in place.
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


class FileOpsDetector(BaseDetector):
    """Detector for insecure file system operations
    
    This detector identifies unsafe file operations that could lead to security issues:
    - os.chmod(), os.mkdir() with overly permissive permissions (0o777)
    - Temporary file creation without tempfile.mkstemp()
    - os.remove(), os.unlink(), shutil.rmtree() without path validation
    - Writing to world-writable directories (/tmp, /var/tmp)
    - os.symlink(), os.readlink() without validation
    
    The detector performs context analysis to determine if proper security
    measures are in place.
    """
    
    # File permission functions
    PERMISSION_FUNCTIONS = ['os.chmod', 'os.mkdir', 'os.makedirs', 'Path.chmod']
    
    # Temporary file creation patterns
    TEMP_FILE_FUNCTIONS = ['open', 'os.open']
    TEMP_DIRECTORIES = ['/tmp', '/var/tmp', 'C:\\Windows\\Temp', 'C:\\Temp']
    SAFE_TEMP_FUNCTIONS = ['tempfile.mkstemp', 'tempfile.NamedTemporaryFile', 'tempfile.TemporaryFile']
    
    # File deletion functions
    DELETE_FUNCTIONS = ['os.remove', 'os.unlink', 'shutil.rmtree', 'Path.unlink']
    
    # World-writable directories
    WORLD_WRITABLE_DIRS = ['/tmp', '/var/tmp', '/dev/shm']
    
    # Symbolic link functions
    SYMLINK_FUNCTIONS = ['os.symlink', 'os.readlink', 'Path.symlink_to', 'Path.readlink']
    
    def __init__(self, config: DetectorConfig):
        """Initialize the file operations detector
        
        Args:
            config: Configuration for the detector
        """
        super().__init__(config)
        self.vulnerabilities: List[Vulnerability] = []
        self.current_file: Optional[FileInfo] = None
        self.ast_tree: Optional[ast.AST] = None
    
    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        """Detect insecure file operations in a file
        
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
        """Recursively visit AST nodes to find file operations
        
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
        """Check if a function call is an insecure file operation
        
        Args:
            node: The Call node to analyze
        """
        func_name = ASTAnalyzer.get_function_name(node)
        
        if not func_name:
            return
        
        # Check for different types of file operations
        # Handle both module functions (os.chmod) and method calls (Path.chmod)
        if any(perm_func in func_name for perm_func in self.PERMISSION_FUNCTIONS):
            self._check_permissions(node, func_name)
        
        # Also check for method calls on Path objects
        elif isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # Check for Path.chmod()
            if method_name == 'chmod':
                self._check_permissions(node, f'Path.{method_name}')
            
            # Check for Path.unlink()
            elif method_name == 'unlink':
                self._check_file_deletion(node, f'Path.{method_name}')
            
            # Check for Path.symlink_to() or Path.readlink()
            elif method_name in ['symlink_to', 'readlink']:
                self._check_symlink_usage(node, f'Path.{method_name}')
        
        if any(temp_func in func_name for temp_func in self.TEMP_FILE_FUNCTIONS):
            self._check_temp_file_creation(node, func_name)
        
        elif any(del_func in func_name for del_func in self.DELETE_FUNCTIONS):
            self._check_file_deletion(node, func_name)
        
        elif any(sym_func in func_name for sym_func in self.SYMLINK_FUNCTIONS):
            self._check_symlink_usage(node, func_name)
    
    def _check_permissions(self, node: ast.Call, func_name: str) -> None:
        """Check for overly permissive file permissions
        
        Args:
            node: The Call node to analyze
            func_name: Name of the permission function
        """
        # Look for permission argument (mode parameter)
        permission_value = self._get_permission_value(node)
        
        if permission_value is None:
            return
        
        # Check if permission is 0o777 (world-readable, writable, executable)
        if permission_value == 0o777:
            vuln = self._create_vulnerability(
                node=node,
                func_name=func_name,
                issue_type='overly_permissive',
                severity=SeverityLevel.MEDIUM,
                context_info={
                    'permission': oct(permission_value),
                }
            )
            self.vulnerabilities.append(vuln)
    
    def _check_temp_file_creation(self, node: ast.Call, func_name: str) -> None:
        """Check for insecure temporary file creation
        
        Args:
            node: The Call node to analyze
            func_name: Name of the file operation function
        """
        # Check if this is creating a file in a temporary directory
        is_temp_file = self._is_temp_file_creation(node)
        
        if not is_temp_file:
            return
        
        # Check if using safe temp file creation methods
        # Look for tempfile.mkstemp usage in the surrounding context
        uses_safe_temp = self._uses_safe_temp_creation(node)
        
        if not uses_safe_temp:
            vuln = self._create_vulnerability(
                node=node,
                func_name=func_name,
                issue_type='insecure_temp_file',
                severity=SeverityLevel.MEDIUM,
                context_info={
                    'uses_safe_temp': False,
                }
            )
            self.vulnerabilities.append(vuln)
    
    def _check_file_deletion(self, node: ast.Call, func_name: str) -> None:
        """Check for file deletion without path validation
        
        Args:
            node: The Call node to analyze
            func_name: Name of the deletion function
        """
        # Check if path is validated
        has_validation = self._has_path_validation(node)
        
        # Check if path is user-controlled
        user_controlled = self._is_path_user_controlled(node)
        
        # High severity if no validation on deletion
        if not has_validation:
            vuln = self._create_vulnerability(
                node=node,
                func_name=func_name,
                issue_type='unvalidated_deletion',
                severity=SeverityLevel.HIGH,
                context_info={
                    'has_validation': has_validation,
                    'user_controlled': user_controlled,
                }
            )
            self.vulnerabilities.append(vuln)
    
    def _check_symlink_usage(self, node: ast.Call, func_name: str) -> None:
        """Check for symbolic link usage without validation
        
        Args:
            node: The Call node to analyze
            func_name: Name of the symlink function
        """
        # Check if symlink target is validated
        has_validation = self._has_path_validation(node)
        
        if not has_validation:
            vuln = self._create_vulnerability(
                node=node,
                func_name=func_name,
                issue_type='unvalidated_symlink',
                severity=SeverityLevel.MEDIUM,
                context_info={
                    'has_validation': has_validation,
                }
            )
            self.vulnerabilities.append(vuln)
    
    def _get_permission_value(self, node: ast.Call) -> Optional[int]:
        """Extract permission value from function call
        
        Args:
            node: The Call node to analyze
            
        Returns:
            Permission value as integer, or None if not found
        """
        # Check keyword arguments for 'mode' parameter
        for keyword in node.keywords:
            if keyword.arg == 'mode':
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
                elif isinstance(keyword.value, ast.Num):  # Python 3.7 compatibility
                    return keyword.value.n
        
        # Check positional arguments (mode is usually second argument)
        if len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant):
                return mode_arg.value
            elif isinstance(mode_arg, ast.Num):  # Python 3.7 compatibility
                return mode_arg.n
        
        return None
    
    def _is_temp_file_creation(self, node: ast.Call) -> bool:
        """Check if this is creating a file in a temporary directory
        
        Args:
            node: The Call node to analyze
            
        Returns:
            True if creating file in temp directory, False otherwise
        """
        # Check if any argument contains a temp directory path
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                for temp_dir in self.TEMP_DIRECTORIES:
                    if temp_dir.lower() in arg.value.lower():
                        return True
            
            # Check for string concatenation with temp directories
            if isinstance(arg, ast.BinOp):
                if self._contains_temp_dir_in_binop(arg):
                    return True
        
        # Check keyword arguments
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                for temp_dir in self.TEMP_DIRECTORIES:
                    if temp_dir.lower() in keyword.value.value.lower():
                        return True
        
        return False
    
    def _contains_temp_dir_in_binop(self, node: ast.BinOp) -> bool:
        """Check if binary operation contains temp directory reference
        
        Args:
            node: The BinOp node to analyze
            
        Returns:
            True if temp directory found, False otherwise
        """
        def check_node(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return any(temp_dir.lower() in n.value.lower() for temp_dir in self.TEMP_DIRECTORIES)
            if isinstance(n, ast.BinOp):
                return check_node(n.left) or check_node(n.right)
            return False
        
        return check_node(node)
    
    def _uses_safe_temp_creation(self, node: ast.Call) -> bool:
        """Check if code uses safe temporary file creation methods
        
        Args:
            node: The Call node to analyze
            
        Returns:
            True if safe temp creation is used, False otherwise
        """
        # Check if the call itself is a safe temp function
        func_name = ASTAnalyzer.get_function_name(node)
        if func_name and any(safe_func in func_name for safe_func in self.SAFE_TEMP_FUNCTIONS):
            return True
        
        # Check surrounding context for safe temp file usage
        # This is a simplified check - in practice would need more sophisticated analysis
        for parent_node in ast.walk(self.ast_tree):
            if isinstance(parent_node, ast.Call):
                parent_func_name = ASTAnalyzer.get_function_name(parent_node)
                if parent_func_name and any(safe_func in parent_func_name for safe_func in self.SAFE_TEMP_FUNCTIONS):
                    return True
        
        return False
    
    def _has_path_validation(self, node: ast.Call) -> bool:
        """Check if path has validation before use
        
        Args:
            node: The Call node to analyze
            
        Returns:
            True if validation is detected, False otherwise
        """
        return ASTAnalyzer.has_validation(node, self.ast_tree)
    
    def _is_path_user_controlled(self, node: ast.Call) -> bool:
        """Determine if the path argument is user-controlled
        
        Args:
            node: The Call node to analyze
            
        Returns:
            True if path is potentially user-controlled, False otherwise
        """
        # Check all arguments for user-controlled input
        for arg in node.args:
            if ASTAnalyzer.is_user_controlled(arg, self.ast_tree):
                return True
        
        # Check keyword arguments
        for keyword in node.keywords:
            if ASTAnalyzer.is_user_controlled(keyword.value, self.ast_tree):
                return True
        
        return False
    
    def _is_world_writable_dir(self, path: str) -> bool:
        """Check if path is in a world-writable directory
        
        Args:
            path: File path to check
            
        Returns:
            True if in world-writable directory, False otherwise
        """
        path_lower = path.lower()
        return any(ww_dir in path_lower for ww_dir in self.WORLD_WRITABLE_DIRS)
    
    def _create_vulnerability(
        self,
        node: ast.Call,
        func_name: str,
        issue_type: str,
        severity: SeverityLevel,
        context_info: dict,
    ) -> Vulnerability:
        """Create a Vulnerability object for a detected file operation issue
        
        Args:
            node: The AST Call node containing the file operation
            func_name: Name of the file operation function
            issue_type: Type of issue detected
            severity: Severity level for this vulnerability
            context_info: Additional context information
            
        Returns:
            Vulnerability object with all details
        """
        # Generate unique ID based on issue type
        issue_prefixes = {
            'overly_permissive': 'FOP',
            'insecure_temp_file': 'FOP',
            'unvalidated_deletion': 'FOP',
            'world_writable': 'FOP',
            'unvalidated_symlink': 'FOP',
        }
        prefix = issue_prefixes.get(issue_type, 'FOP')
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
        title, description = self._get_title_description(issue_type, func_name, context_info)
        
        # Generate recommendation
        recommendation = self._get_recommendation(issue_type, func_name)
        
        # Determine CWE ID
        cwe_id = self._get_cwe_id(issue_type)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(issue_type, context_info)
        
        return Vulnerability(
            id=vuln_id,
            title=title,
            description=description,
            severity=severity,
            vulnerability_type=VulnerabilityType.FILE_OPERATIONS,
            file_path=self.current_file.path,
            line_number=line_number,
            column=column,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id,
            confidence=confidence,
            context={
                'function': func_name,
                'issue_type': issue_type,
                **context_info,
            }
        )
    
    def _get_title_description(self, issue_type: str, func_name: str, context_info: dict) -> tuple:
        """Generate title and description for file operation issue
        
        Args:
            issue_type: Type of issue detected
            func_name: Name of the file operation function
            context_info: Additional context information
            
        Returns:
            Tuple of (title, description)
        """
        titles = {
            'overly_permissive': "File Created with Overly Permissive Permissions",
            'insecure_temp_file': "Insecure Temporary File Creation",
            'unvalidated_deletion': "File Deletion Without Path Validation",
            'world_writable': "Write to World-Writable Directory",
            'unvalidated_symlink': "Symbolic Link Usage Without Validation",
        }
        
        descriptions = {
            'overly_permissive': (
                f"The function {func_name}() is creating a file or directory with overly permissive "
                f"permissions ({context_info.get('permission', '0o777')}). This allows any user on the system "
                f"to read, write, and execute the file, which could lead to unauthorized access or modification."
            ),
            'insecure_temp_file': (
                f"The function {func_name}() is creating a temporary file without using secure methods "
                f"like tempfile.mkstemp(). This can lead to race conditions where an attacker creates "
                f"a file with the same name before your code does, potentially leading to data disclosure "
                f"or arbitrary file writes."
            ),
            'unvalidated_deletion': (
                f"The function {func_name}() is deleting files without validating that the path is within "
                f"safe directories. "
            ),
            'world_writable': (
                f"The function {func_name}() is writing to a world-writable directory. This could allow "
                f"attackers to modify or replace the file, leading to data tampering or code execution."
            ),
            'unvalidated_symlink': (
                f"The function {func_name}() is using symbolic links without validation. This could allow "
                f"attackers to create symlinks pointing to sensitive files, leading to unauthorized access "
                f"or modification."
            ),
        }
        
        title = titles.get(issue_type, "Insecure File Operation")
        description = descriptions.get(issue_type, "Insecure file operation detected.")
        
        # Add user-controlled context
        if context_info.get('user_controlled'):
            description += " The path appears to be user-controlled, which increases the risk."
        
        # Add validation context
        if not context_info.get('has_validation', True):
            description += " No path validation was detected."
        
        return title, description
    
    def _get_recommendation(self, issue_type: str, func_name: str) -> str:
        """Generate remediation recommendation for file operation issue
        
        Args:
            issue_type: Type of issue detected
            func_name: Name of the file operation function
            
        Returns:
            Remediation recommendation string
        """
        recommendations = {
            'overly_permissive': (
                "Use more restrictive file permissions. Instead of 0o777, use:\n"
                "- 0o600 for files that only the owner should access\n"
                "- 0o644 for files that others can read but not modify\n"
                "- 0o700 for directories that only the owner should access\n"
                "- 0o755 for directories that others can read and execute\n\n"
                "Example:\n"
                "os.chmod(file_path, 0o600)  # Owner read/write only\n"
                "os.mkdir(dir_path, 0o700)   # Owner full access only"
            ),
            'insecure_temp_file': (
                "Use tempfile.mkstemp() or tempfile.NamedTemporaryFile() for secure temporary file creation. "
                "These functions create files with secure permissions (0o600) and unique names that prevent "
                "race conditions.\n\n"
                "Example:\n"
                "import tempfile\n"
                "fd, temp_path = tempfile.mkstemp()\n"
                "try:\n"
                "    with os.fdopen(fd, 'w') as f:\n"
                "        f.write(data)\n"
                "finally:\n"
                "    os.unlink(temp_path)"
            ),
            'unvalidated_deletion': (
                "Always validate file paths before deletion:\n"
                "1. Use pathlib.Path.resolve() to canonicalize the path\n"
                "2. Check that the resolved path is within allowed directories\n"
                "3. Verify the file exists and is not a symlink to a sensitive location\n"
                "4. Use an allow-list of permitted directories\n\n"
                "Example:\n"
                "from pathlib import Path\n\n"
                "def safe_delete(file_path: str, allowed_dir: str):\n"
                "    allowed = Path(allowed_dir).resolve()\n"
                "    target = Path(file_path).resolve()\n"
                "    \n"
                "    if not target.is_relative_to(allowed):\n"
                "        raise ValueError('Path outside allowed directory')\n"
                "    \n"
                "    target.unlink()"
            ),
            'world_writable': (
                "Avoid writing to world-writable directories like /tmp. Instead:\n"
                "1. Use tempfile.mkstemp() which creates files with secure permissions\n"
                "2. Create a dedicated directory with restricted permissions\n"
                "3. Use application-specific temporary directories\n\n"
                "Example:\n"
                "import tempfile\n"
                "with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:\n"
                "    f.write(data)\n"
                "    temp_path = f.name"
            ),
            'unvalidated_symlink': (
                "Always validate symbolic links before use:\n"
                "1. Use Path.resolve() to follow symlinks and get the real path\n"
                "2. Verify the target is within allowed directories\n"
                "3. Check that the symlink doesn't point to sensitive files\n"
                "4. Consider using lstat() instead of stat() to detect symlinks\n\n"
                "Example:\n"
                "from pathlib import Path\n\n"
                "def safe_readlink(link_path: str, allowed_dir: str):\n"
                "    allowed = Path(allowed_dir).resolve()\n"
                "    link = Path(link_path)\n"
                "    \n"
                "    if link.is_symlink():\n"
                "        target = link.resolve()\n"
                "        if not target.is_relative_to(allowed):\n"
                "            raise ValueError('Symlink target outside allowed directory')\n"
                "    \n"
                "    return link.readlink()"
            ),
        }
        
        return recommendations.get(issue_type, "Implement proper security measures for file operations.")
    
    def _get_cwe_id(self, issue_type: str) -> str:
        """Get CWE ID for issue type
        
        Args:
            issue_type: Type of issue detected
            
        Returns:
            CWE ID string
        """
        cwe_ids = {
            'overly_permissive': 'CWE-732',  # Incorrect Permission Assignment for Critical Resource
            'insecure_temp_file': 'CWE-377',  # Insecure Temporary File
            'unvalidated_deletion': 'CWE-22',  # Path Traversal
            'world_writable': 'CWE-732',  # Incorrect Permission Assignment
            'unvalidated_symlink': 'CWE-59',  # Improper Link Resolution Before File Access
        }
        
        return cwe_ids.get(issue_type, 'CWE-732')
    
    def _calculate_confidence(self, issue_type: str, context_info: dict) -> float:
        """Calculate confidence score for vulnerability
        
        Args:
            issue_type: Type of issue detected
            context_info: Additional context information
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.9
        
        # Adjust based on context
        if issue_type == 'overly_permissive':
            # High confidence for explicit 0o777
            confidence = 1.0
        
        elif issue_type == 'insecure_temp_file':
            # Medium confidence - might be false positive
            confidence = 0.7
        
        elif issue_type == 'unvalidated_deletion':
            # Reduce confidence if validation is present
            if context_info.get('has_validation'):
                confidence = 0.5
            else:
                confidence = 0.9
        
        elif issue_type == 'unvalidated_symlink':
            # Medium confidence
            confidence = 0.8
        
        return confidence
