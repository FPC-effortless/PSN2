"""Configuration Detector

This module implements detection of security issues in configuration files including:
- Debug mode enabled in production (debug=true, DEBUG=1)
- Insecure default values
- Absolute paths to sensitive directories
- Missing required security settings
- World-readable files with sensitive data

The detector parses JSON and YAML configuration files to identify security misconfigurations.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
    Vulnerability,
    VulnerabilityType,
)


class ConfigDetector(BaseDetector):
    """Detector for configuration security issues
    
    This detector identifies security issues in JSON and YAML configuration files:
    - Debug mode enabled (debug=true, DEBUG=1, debug_mode=true)
    - Insecure default values (default passwords, weak settings)
    - Absolute paths to sensitive directories (/etc, /root, /home)
    - Missing required security settings
    - World-readable files containing sensitive data
    
    The detector parses configuration files and analyzes their content for
    security misconfigurations.
    """
    
    # Debug-related keys that indicate debug mode
    DEBUG_KEYS = ['debug', 'DEBUG', 'debug_mode', 'DEBUG_MODE', 'development', 'DEVELOPMENT']
    
    # Insecure default values to detect
    INSECURE_DEFAULTS = {
        'password': ['password', 'admin', 'default', '123456', 'changeme'],
        'secret': ['secret', 'default_secret'],
        'key': ['default_key', 'changeme'],
        'token': ['default_token'],
    }
    
    # Sensitive directory paths
    SENSITIVE_PATHS = ['/etc', '/root', '/home', '/var', '/usr/local', 'C:\\Windows', 'C:\\Users']
    
    # Required security settings (keys that should be present)
    REQUIRED_SECURITY_KEYS = ['ssl', 'tls', 'https', 'encryption', 'auth', 'authentication']
    
    def __init__(self, config: DetectorConfig):
        """Initialize the configuration detector
        
        Args:
            config: Configuration for the detector
        """
        super().__init__(config)
        self.vulnerabilities: List[Vulnerability] = []
        self.current_file: Optional[FileInfo] = None
        self.config_data: Optional[Dict[str, Any]] = None
    
    def detect(self, file_info: FileInfo, ast_tree: Optional[Any] = None) -> List[Vulnerability]:
        """Detect configuration security issues in a file
        
        Args:
            file_info: Information about the file being scanned
            ast_tree: Not used for config files (only for Python AST)
            
        Returns:
            List of detected vulnerabilities
        """
        if not self.config.enabled:
            return []
        
        # Only process JSON and YAML config files
        if file_info.file_type not in [FileType.CONFIG_JSON, FileType.CONFIG_YAML]:
            return []
        
        # Reset state for this file
        self.vulnerabilities = []
        self.current_file = file_info
        self.config_data = None
        
        # Parse configuration file
        if not self._parse_config_file(file_info):
            return []
        
        # Run all detection checks
        self._check_debug_mode()
        self._check_insecure_defaults()
        self._check_sensitive_paths()
        self._check_missing_security_settings()
        self._check_world_readable_sensitive_file()
        
        # Filter out suppressed vulnerabilities
        return [v for v in self.vulnerabilities if not self.should_suppress(v)]
    
    def _parse_config_file(self, file_info: FileInfo) -> bool:
        """Parse JSON or YAML configuration file
        
        Args:
            file_info: Information about the file to parse
            
        Returns:
            True if parsing succeeded, False otherwise
        """
        try:
            with open(file_info.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_info.file_type == FileType.CONFIG_JSON:
                self.config_data = json.loads(content)
            elif file_info.file_type == FileType.CONFIG_YAML:
                self.config_data = yaml.safe_load(content)
            
            return self.config_data is not None
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            # Invalid JSON/YAML - skip this file
            return False
        except Exception as e:
            # Other errors (file access, etc.) - skip this file
            return False
    
    def _check_debug_mode(self) -> None:
        """Check for debug mode enabled in configuration
        
        Detects debug flags set to true/1 which should not be enabled in production.
        """
        if not self.config_data:
            return
        
        # Recursively search for debug keys
        debug_findings = self._find_debug_keys(self.config_data, [])
        
        for key_path, value in debug_findings:
            # Check if debug is enabled
            if self._is_debug_enabled(value):
                vuln = self._create_vulnerability(
                    issue_type='debug_mode_enabled',
                    severity=SeverityLevel.MEDIUM,
                    key_path=key_path,
                    value=value,
                    line_number=1,  # Config files don't have line numbers in parsed form
                )
                self.vulnerabilities.append(vuln)
    
    def _find_debug_keys(self, data: Any, path: List[str]) -> List[tuple]:
        """Recursively find debug-related keys in configuration
        
        Args:
            data: Configuration data (dict, list, or primitive)
            path: Current path in the configuration tree
            
        Returns:
            List of (key_path, value) tuples for debug keys found
        """
        findings = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                
                # Check if this key is a debug key
                if key in self.DEBUG_KEYS:
                    findings.append(('.'.join(current_path), value))
                
                # Recursively search nested structures
                findings.extend(self._find_debug_keys(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = path + [f'[{i}]']
                findings.extend(self._find_debug_keys(item, current_path))
        
        return findings
    
    def _is_debug_enabled(self, value: Any) -> bool:
        """Check if a debug value indicates debug mode is enabled
        
        Args:
            value: The value to check
            
        Returns:
            True if debug is enabled, False otherwise
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value == 1
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'on', 'enabled']
        return False
    
    def _check_insecure_defaults(self) -> None:
        """Check for insecure default values in configuration
        
        Detects default passwords, secrets, keys, and tokens that should be changed.
        """
        if not self.config_data:
            return
        
        # Recursively search for insecure defaults
        insecure_findings = self._find_insecure_defaults(self.config_data, [])
        
        for key_path, key_type, value in insecure_findings:
            vuln = self._create_vulnerability(
                issue_type='insecure_default_value',
                severity=SeverityLevel.LOW,
                key_path=key_path,
                value=value,
                line_number=1,
                context_info={'key_type': key_type},
            )
            self.vulnerabilities.append(vuln)
    
    def _find_insecure_defaults(self, data: Any, path: List[str]) -> List[tuple]:
        """Recursively find insecure default values in configuration
        
        Args:
            data: Configuration data
            path: Current path in the configuration tree
            
        Returns:
            List of (key_path, key_type, value) tuples for insecure defaults found
        """
        findings = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                key_lower = key.lower()
                
                # Check if this key matches insecure default patterns
                for key_type, insecure_values in self.INSECURE_DEFAULTS.items():
                    if key_type in key_lower and isinstance(value, str):
                        if value.lower() in insecure_values:
                            findings.append(('.'.join(current_path), key_type, value))
                
                # Recursively search nested structures
                findings.extend(self._find_insecure_defaults(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = path + [f'[{i}]']
                findings.extend(self._find_insecure_defaults(item, current_path))
        
        return findings
    
    def _check_sensitive_paths(self) -> None:
        """Check for absolute paths to sensitive directories
        
        Detects hardcoded absolute paths to system directories that may be
        environment-specific or security-sensitive.
        """
        if not self.config_data:
            return
        
        # Recursively search for sensitive paths
        path_findings = self._find_sensitive_paths(self.config_data, [])
        
        for key_path, value in path_findings:
            vuln = self._create_vulnerability(
                issue_type='sensitive_path',
                severity=SeverityLevel.LOW,
                key_path=key_path,
                value=value,
                line_number=1,
            )
            self.vulnerabilities.append(vuln)
    
    def _find_sensitive_paths(self, data: Any, path: List[str]) -> List[tuple]:
        """Recursively find absolute paths to sensitive directories
        
        Args:
            data: Configuration data
            path: Current path in the configuration tree
            
        Returns:
            List of (key_path, value) tuples for sensitive paths found
        """
        findings = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                
                # Check if value is a sensitive path
                if isinstance(value, str) and self._is_sensitive_path(value):
                    findings.append(('.'.join(current_path), value))
                
                # Recursively search nested structures
                findings.extend(self._find_sensitive_paths(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = path + [f'[{i}]']
                findings.extend(self._find_sensitive_paths(item, current_path))
        
        return findings
    
    def _is_sensitive_path(self, value: str) -> bool:
        """Check if a string value is an absolute path to a sensitive directory
        
        Args:
            value: String value to check
            
        Returns:
            True if value is a sensitive path, False otherwise
        """
        # Check if it's an absolute path
        if not (value.startswith('/') or (len(value) > 2 and value[1] == ':')):
            return False
        
        # Check if it starts with any sensitive path
        for sensitive_path in self.SENSITIVE_PATHS:
            if value.startswith(sensitive_path):
                return True
        
        return False
    
    def _check_missing_security_settings(self) -> None:
        """Check for missing required security settings
        
        Detects when configuration files lack important security-related settings
        like SSL, TLS, encryption, or authentication configuration.
        """
        if not self.config_data:
            return
        
        # Flatten all keys in the configuration
        all_keys = self._get_all_keys(self.config_data)
        all_keys_lower = [k.lower() for k in all_keys]
        
        # Check if any required security keys are present
        has_security_setting = any(
            any(req_key in key for req_key in self.REQUIRED_SECURITY_KEYS)
            for key in all_keys_lower
        )
        
        # If no security settings found, flag as missing
        if not has_security_setting:
            vuln = self._create_vulnerability(
                issue_type='missing_security_settings',
                severity=SeverityLevel.MEDIUM,
                key_path='<root>',
                value=None,
                line_number=1,
            )
            self.vulnerabilities.append(vuln)
    
    def _get_all_keys(self, data: Any, path: List[str] = None) -> List[str]:
        """Get all keys in the configuration (flattened)
        
        Args:
            data: Configuration data
            path: Current path (for recursion)
            
        Returns:
            List of all key paths in the configuration
        """
        if path is None:
            path = []
        
        keys = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                keys.append('.'.join(current_path))
                keys.extend(self._get_all_keys(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = path + [f'[{i}]']
                keys.extend(self._get_all_keys(item, current_path))
        
        return keys
    
    def _check_world_readable_sensitive_file(self) -> None:
        """Check if configuration file is world-readable and contains sensitive data
        
        Detects configuration files with overly permissive file permissions that
        contain sensitive information like passwords, keys, or tokens.
        """
        if not self.current_file:
            return
        
        try:
            # Get file permissions
            file_stat = os.stat(self.current_file.path)
            file_mode = stat.filemode(file_stat.st_mode)
            
            # Check if file is world-readable (others have read permission)
            is_world_readable = bool(file_stat.st_mode & stat.S_IROTH)
            
            if not is_world_readable:
                return
            
            # Check if file contains sensitive data
            has_sensitive_data = self._contains_sensitive_data()
            
            if has_sensitive_data:
                vuln = self._create_vulnerability(
                    issue_type='world_readable_sensitive_file',
                    severity=SeverityLevel.HIGH,
                    key_path='<file>',
                    value=file_mode,
                    line_number=1,
                    context_info={'file_mode': file_mode},
                )
                self.vulnerabilities.append(vuln)
        
        except OSError:
            # Can't get file permissions - skip this check
            pass
    
    def _contains_sensitive_data(self) -> bool:
        """Check if configuration contains sensitive data
        
        Returns:
            True if configuration contains sensitive keys, False otherwise
        """
        if not self.config_data:
            return False
        
        # Sensitive key patterns
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential', 'api_key']
        
        # Get all keys
        all_keys = self._get_all_keys(self.config_data)
        all_keys_lower = [k.lower() for k in all_keys]
        
        # Check if any key contains sensitive patterns
        return any(
            any(pattern in key for pattern in sensitive_patterns)
            for key in all_keys_lower
        )
    
    def _create_vulnerability(
        self,
        issue_type: str,
        severity: SeverityLevel,
        key_path: str,
        value: Any,
        line_number: int,
        context_info: Dict[str, Any] = None,
    ) -> Vulnerability:
        """Create a Vulnerability object for a detected configuration issue
        
        Args:
            issue_type: Type of issue detected
            severity: Severity level for this vulnerability
            key_path: Path to the configuration key
            value: Value of the configuration key
            line_number: Line number (always 1 for parsed configs)
            context_info: Additional context information
            
        Returns:
            Vulnerability object with all details
        """
        if context_info is None:
            context_info = {}
        
        # Generate unique ID based on issue type
        issue_prefixes = {
            'debug_mode_enabled': 'CFG',
            'insecure_default_value': 'CFG',
            'sensitive_path': 'CFG',
            'missing_security_settings': 'CFG',
            'world_readable_sensitive_file': 'CFG',
        }
        prefix = issue_prefixes.get(issue_type, 'CFG')
        vuln_id = f"{prefix}{len(self.vulnerabilities) + 1:03d}"
        
        # Generate title and description
        title, description = self._get_title_description(issue_type, key_path, value, context_info)
        
        # Generate recommendation
        recommendation = self._get_recommendation(issue_type)
        
        # Determine CWE ID
        cwe_id = self._get_cwe_id(issue_type)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(issue_type, context_info)
        
        # Create code snippet (show key path and value)
        code_snippet = self._create_code_snippet(key_path, value)
        
        return Vulnerability(
            id=vuln_id,
            title=title,
            description=description,
            severity=severity,
            vulnerability_type=VulnerabilityType.CONFIGURATION,
            file_path=self.current_file.path,
            line_number=line_number,
            column=0,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id,
            confidence=confidence,
            context={
                'issue_type': issue_type,
                'key_path': key_path,
                'value': str(value) if value is not None else None,
                **context_info,
            }
        )
    
    def _get_title_description(
        self,
        issue_type: str,
        key_path: str,
        value: Any,
        context_info: Dict[str, Any],
    ) -> tuple:
        """Generate title and description for configuration issue
        
        Args:
            issue_type: Type of issue detected
            key_path: Path to the configuration key
            value: Value of the configuration key
            context_info: Additional context information
            
        Returns:
            Tuple of (title, description)
        """
        titles = {
            'debug_mode_enabled': "Debug Mode Enabled in Configuration",
            'insecure_default_value': "Insecure Default Value in Configuration",
            'sensitive_path': "Absolute Path to Sensitive Directory",
            'missing_security_settings': "Missing Required Security Settings",
            'world_readable_sensitive_file': "World-Readable File with Sensitive Data",
        }
        
        descriptions = {
            'debug_mode_enabled': (
                f"The configuration key '{key_path}' has debug mode enabled (value: {value}). "
                f"Debug mode should not be enabled in production environments as it may expose "
                f"sensitive information, enable verbose logging, or disable security features."
            ),
            'insecure_default_value': (
                f"The configuration key '{key_path}' contains an insecure default value ('{value}'). "
                f"Default values for {context_info.get('key_type', 'sensitive')} fields should be "
                f"changed to prevent unauthorized access. Using default credentials is a common "
                f"security vulnerability that attackers actively exploit."
            ),
            'sensitive_path': (
                f"The configuration key '{key_path}' contains an absolute path to a sensitive "
                f"directory ('{value}'). Hardcoded absolute paths can cause portability issues "
                f"and may expose sensitive system directories. Consider using relative paths or "
                f"environment variables."
            ),
            'missing_security_settings': (
                f"The configuration file does not contain any security-related settings such as "
                f"SSL/TLS, encryption, or authentication configuration. This may indicate that "
                f"security features are not properly configured."
            ),
            'world_readable_sensitive_file': (
                f"The configuration file has world-readable permissions ({value}) and contains "
                f"sensitive data such as passwords, keys, or tokens. This allows any user on the "
                f"system to read sensitive information, which could lead to unauthorized access."
            ),
        }
        
        title = titles.get(issue_type, "Configuration Security Issue")
        description = descriptions.get(issue_type, "Configuration security issue detected.")
        
        return title, description
    
    def _get_recommendation(self, issue_type: str) -> str:
        """Generate remediation recommendation for configuration issue
        
        Args:
            issue_type: Type of issue detected
            
        Returns:
            Remediation recommendation string
        """
        recommendations = {
            'debug_mode_enabled': (
                "Disable debug mode in production environments:\n"
                "1. Set debug flags to false or 0\n"
                "2. Use environment variables to control debug mode\n"
                "3. Ensure debug mode is only enabled in development environments\n"
                "4. Review logs to ensure no sensitive information is exposed\n\n"
                "Example:\n"
                "{\n"
                '  "debug": false,\n'
                '  "DEBUG_MODE": 0\n'
                "}"
            ),
            'insecure_default_value': (
                "Change default values to secure, unique values:\n"
                "1. Generate strong, random passwords and keys\n"
                "2. Use environment variables for sensitive values\n"
                "3. Never commit actual credentials to version control\n"
                "4. Use secrets management tools (e.g., HashiCorp Vault, AWS Secrets Manager)\n\n"
                "Example:\n"
                "{\n"
                '  "password": "${DB_PASSWORD}",  # Use environment variable\n'
                '  "api_key": "${API_KEY}"        # Use environment variable\n'
                "}"
            ),
            'sensitive_path': (
                "Use relative paths or environment variables instead of absolute paths:\n"
                "1. Use relative paths from the application root\n"
                "2. Use environment variables for system-specific paths\n"
                "3. Use path resolution libraries (e.g., pathlib in Python)\n"
                "4. Document required directory structure\n\n"
                "Example:\n"
                "{\n"
                '  "data_dir": "./data",           # Relative path\n'
                '  "config_dir": "${CONFIG_DIR}"   # Environment variable\n'
                "}"
            ),
            'missing_security_settings': (
                "Add required security settings to the configuration:\n"
                "1. Enable SSL/TLS for encrypted connections\n"
                "2. Configure authentication and authorization\n"
                "3. Enable encryption for sensitive data\n"
                "4. Set secure defaults for all security-related options\n\n"
                "Example:\n"
                "{\n"
                '  "ssl": true,\n'
                '  "tls_version": "1.3",\n'
                '  "authentication": "required",\n'
                '  "encryption": "AES-256"\n'
                "}"
            ),
            'world_readable_sensitive_file': (
                "Restrict file permissions to prevent unauthorized access:\n"
                "1. Set file permissions to 0o600 (owner read/write only)\n"
                "2. Ensure the file is owned by the application user\n"
                "3. Store sensitive configuration outside the web root\n"
                "4. Consider encrypting sensitive values in the configuration\n\n"
                "Example:\n"
                "chmod 600 config.json  # Owner read/write only\n"
                "chown appuser:appuser config.json"
            ),
        }
        
        return recommendations.get(issue_type, "Review and secure the configuration.")
    
    def _get_cwe_id(self, issue_type: str) -> str:
        """Get CWE ID for issue type
        
        Args:
            issue_type: Type of issue detected
            
        Returns:
            CWE ID string
        """
        cwe_ids = {
            'debug_mode_enabled': 'CWE-489',  # Active Debug Code
            'insecure_default_value': 'CWE-1188',  # Insecure Default Initialization
            'sensitive_path': 'CWE-426',  # Untrusted Search Path
            'missing_security_settings': 'CWE-1188',  # Insecure Default Initialization
            'world_readable_sensitive_file': 'CWE-732',  # Incorrect Permission Assignment
        }
        
        return cwe_ids.get(issue_type, 'CWE-16')  # Configuration
    
    def _calculate_confidence(self, issue_type: str, context_info: Dict[str, Any]) -> float:
        """Calculate confidence score for vulnerability
        
        Args:
            issue_type: Type of issue detected
            context_info: Additional context information
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence levels by issue type
        confidence_map = {
            'debug_mode_enabled': 0.9,
            'insecure_default_value': 0.8,
            'sensitive_path': 0.7,
            'missing_security_settings': 0.6,
            'world_readable_sensitive_file': 1.0,
        }
        
        return confidence_map.get(issue_type, 0.8)
    
    def _create_code_snippet(self, key_path: str, value: Any) -> str:
        """Create a code snippet showing the configuration issue
        
        Args:
            key_path: Path to the configuration key
            value: Value of the configuration key
            
        Returns:
            Code snippet string
        """
        if value is None:
            return f"{key_path}: <missing>"
        
        # Format value for display
        if isinstance(value, str):
            value_str = f'"{value}"'
        else:
            value_str = str(value)
        
        return f"{key_path}: {value_str}"
