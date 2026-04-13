"""Security Scanner Core Module

This module implements the main SecurityScanner class that orchestrates the security
scanning workflow. It handles:
- File discovery (Python, notebooks, config files)
- AST parsing for Python files
- Detector initialization and execution
- Result aggregation
- Cache integration for incremental scanning
"""

import ast
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from security_scanner.detectors.base import BaseDetector
from security_scanner.detectors.checkpoint import CheckpointDetector
from security_scanner.detectors.config import ConfigDetector
from security_scanner.detectors.dependency import DependencyDetector
from security_scanner.detectors.deserialization import DeserializationDetector
from security_scanner.detectors.file_ops import FileOpsDetector
from security_scanner.detectors.injection import InjectionDetector
from security_scanner.detectors.path_traversal import PathTraversalDetector
from security_scanner.detectors.pipeline import PipelineDetector
from security_scanner.detectors.secrets import SecretsDetector
from security_scanner.detectors.validation import ValidationDetector
from security_scanner.filters.false_positive import FalsePositiveFilter
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    ScanConfig,
    ScanResult,
    Vulnerability,
)
from security_scanner.parsers.notebook import NotebookParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultCache:
    """Stub implementation of result cache for incremental scanning
    
    This is a placeholder implementation that always returns False for cache hits.
    A full implementation would store file hashes and scan results to enable
    incremental scanning of only changed files.
    """
    
    def __init__(self, cache_dir: Optional[str]):
        """Initialize the result cache
        
        Args:
            cache_dir: Directory to store cache files (None to disable caching)
        """
        self.cache_dir = cache_dir
        self.enabled = cache_dir is not None
        
        if self.enabled:
            logger.info(f"Cache enabled at: {cache_dir}")
        else:
            logger.info("Cache disabled")
    
    def is_cached(self, file_path: str, last_modified: float) -> bool:
        """Check if file results are cached and up-to-date
        
        Args:
            file_path: Path to the file
            last_modified: Last modification timestamp
            
        Returns:
            True if cached results are available and current, False otherwise
        """
        # Stub implementation - always return False (no cache hit)
        return False
    
    def get_cached_results(self, file_path: str) -> List[Vulnerability]:
        """Retrieve cached scan results for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of cached vulnerabilities (empty for stub implementation)
        """
        return []
    
    def store_results(self, file_path: str, last_modified: float, vulnerabilities: List[Vulnerability]) -> None:
        """Store scan results in cache
        
        Args:
            file_path: Path to the file
            last_modified: Last modification timestamp
            vulnerabilities: List of vulnerabilities found
        """
        # Stub implementation - no-op
        pass


class SecurityScanner:
    """Main security scanner class that orchestrates vulnerability detection
    
    The SecurityScanner is responsible for:
    - Discovering files to scan (Python, notebooks, config files)
    - Parsing Python files into AST trees
    - Initializing and managing vulnerability detectors
    - Executing detectors on discovered files
    - Aggregating results from all detectors
    - Integrating with cache for incremental scanning
    
    Attributes:
        config: Configuration for the scan
        detectors: List of initialized vulnerability detectors
        cache: Result cache for incremental scanning
    """
    
    def __init__(self, config: ScanConfig):
        """Initialize the security scanner
        
        Args:
            config: ScanConfig instance with scanner settings
        """
        self.config = config
        self.detectors: List[BaseDetector] = []
        self.cache = ResultCache(config.cache_dir)
        self.notebook_parser = NotebookParser()
        self.fp_filter = FalsePositiveFilter(config.suppression_rules)
        
        # Initialize detectors
        self._initialize_detectors()
        
        logger.info(f"SecurityScanner initialized with {len(self.detectors)} detectors")
    
    def _initialize_detectors(self) -> None:
        """Initialize all vulnerability detectors
        
        Currently initializes the 10 implemented detectors:
        - InjectionDetector: Code injection vulnerabilities
        - SecretsDetector: Hardcoded credentials and secrets
        - PathTraversalDetector: Path traversal vulnerabilities
        - DeserializationDetector: Unsafe deserialization
        - ValidationDetector: Input validation issues
        - FileOpsDetector: Insecure file operations
        - ConfigDetector: Configuration security issues
        - CheckpointDetector: Model checkpoint security
        - PipelineDetector: Data pipeline security
        - DependencyDetector: Insecure dependencies
        """
        # Create default detector configuration
        detector_config = DetectorConfig(
            enabled=True,
            severity_overrides={},
            custom_patterns=[],
            exclusions=[]
        )
        
        # Initialize the 10 detectors
        self.detectors.append(InjectionDetector(detector_config))
        self.detectors.append(SecretsDetector(detector_config))
        self.detectors.append(PathTraversalDetector(detector_config))
        self.detectors.append(DeserializationDetector(detector_config))
        self.detectors.append(ValidationDetector(detector_config))
        self.detectors.append(FileOpsDetector(detector_config))
        self.detectors.append(ConfigDetector(detector_config))
        self.detectors.append(CheckpointDetector(detector_config))
        self.detectors.append(PipelineDetector(detector_config))
        self.detectors.append(DependencyDetector(detector_config))
        
        logger.info(f"Initialized detectors: {[d.name for d in self.detectors]}")
    
    def scan(self, target_path: str) -> ScanResult:
        """Execute security scan on target path
        
        This is the main entry point for scanning. It orchestrates the entire
        scanning workflow:
        1. Discover files to scan
        2. Parse Python files into AST trees
        3. Execute detectors on each file
        4. Aggregate results
        5. Return scan results
        
        Args:
            target_path: Path to directory or file to scan
            
        Returns:
            ScanResult containing all detected vulnerabilities and scan metadata
        """
        start_time = time.time()
        logger.info(f"Starting security scan of: {target_path}")
        
        # Discover files to scan
        files_to_scan = self.discover_files(target_path)
        logger.info(f"Discovered {len(files_to_scan)} files to scan")
        
        # Aggregate vulnerabilities from all files
        all_vulnerabilities: List[Vulnerability] = []
        files_scanned = 0
        
        # Scan each file
        for file_info in files_to_scan:
            # Check if file should be scanned (cache and exclusions)
            if not self.should_scan_file(file_info.path):
                logger.debug(f"Skipping file (cached or excluded): {file_info.path}")
                continue
            
            logger.debug(f"Scanning file: {file_info.path}")
            
            # Scan the file
            vulnerabilities = self._scan_file(file_info)
            all_vulnerabilities.extend(vulnerabilities)
            files_scanned += 1
            
            # Store results in cache
            self.cache.store_results(file_info.path, file_info.last_modified, vulnerabilities)
        
        # Calculate scan duration
        scan_duration = time.time() - start_time
        
        # Create scan result
        result = ScanResult(
            vulnerabilities=all_vulnerabilities,
            files_scanned=files_scanned,
            scan_duration=scan_duration,
            timestamp=datetime.now().isoformat(),
            target_path=target_path
        )
        
        logger.info(f"Scan complete: {len(all_vulnerabilities)} vulnerabilities found in {files_scanned} files")
        logger.info(f"Scan duration: {scan_duration:.2f} seconds")
        
        return result
    
    def discover_files(self, target_path: str) -> List[FileInfo]:
        """Discover Python, notebook, and config files to scan
        
        This method recursively searches the target path for files that should
        be scanned:
        - Python files (.py)
        - Jupyter notebooks (.ipynb)
        - JSON configuration files (.json)
        - YAML configuration files (.yaml, .yml)
        - Requirements files (requirements.txt)
        
        Files matching exclusion patterns are filtered out.
        
        Args:
            target_path: Path to directory or file to scan
            
        Returns:
            List of FileInfo objects for discovered files
        """
        discovered_files: List[FileInfo] = []
        target = Path(target_path)
        
        # Handle single file
        if target.is_file():
            file_info = self._create_file_info(target)
            if file_info:
                discovered_files.append(file_info)
            return discovered_files
        
        # Handle directory - recursively find files
        if not target.is_dir():
            logger.warning(f"Target path does not exist: {target_path}")
            return discovered_files
        
        # Walk directory tree
        for root, dirs, files in os.walk(target):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._is_excluded(os.path.join(root, d))]
            
            # Process files
            for filename in files:
                file_path = Path(root) / filename
                
                # Skip excluded files
                if self._is_excluded(str(file_path)):
                    continue
                
                # Create FileInfo if file type is supported
                file_info = self._create_file_info(file_path)
                if file_info:
                    discovered_files.append(file_info)
        
        return discovered_files
    
    def should_scan_file(self, file_path: str) -> bool:
        """Check if file should be scanned based on cache and config
        
        This method determines whether a file should be scanned by checking:
        1. If the file is in the cache and hasn't been modified
        2. If the file matches any exclusion patterns
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file should be scanned, False if it should be skipped
        """
        # Check exclusion patterns
        if self._is_excluded(file_path):
            return False
        
        # Check cache if incremental scanning is enabled
        if self.config.incremental and self.cache.enabled:
            try:
                # Get file modification time
                last_modified = os.path.getmtime(file_path)
                
                # Check if cached results are available and current
                if self.cache.is_cached(file_path, last_modified):
                    logger.debug(f"Using cached results for: {file_path}")
                    return False
            except OSError:
                # If we can't get file stats, scan it anyway
                pass
        
        return True
    
    def _scan_file(self, file_info: FileInfo) -> List[Vulnerability]:
        """Scan a single file with all detectors."""
        vulnerabilities: List[Vulnerability] = []

        # Parse Python files into AST; parse notebooks into code cells
        ast_tree = None
        if file_info.file_type == FileType.PYTHON:
            ast_tree = self._parse_python_file(file_info.path)
        elif file_info.file_type == FileType.NOTEBOOK:
            vulnerabilities.extend(self._scan_notebook(file_info))
            return self.fp_filter.filter(vulnerabilities)

        # Run all detectors on the file
        for detector in self.detectors:
            try:
                detector_vulns = detector.detect(file_info, ast_tree)
                vulnerabilities.extend(detector_vulns)
                logger.debug(f"{detector.name} found {len(detector_vulns)} issues in {file_info.path}")
            except Exception as e:
                logger.error(f"Error running {detector.name} on {file_info.path}: {e}")

        return self.fp_filter.filter(vulnerabilities)

    def _scan_notebook(self, file_info: FileInfo) -> List[Vulnerability]:
        """Scan a Jupyter notebook by extracting code cells and running detectors."""
        vulns: List[Vulnerability] = []
        content = self.notebook_parser.parse(file_info.path)
        if content is None:
            return vulns

        for cell in content.code_cells:
            try:
                cell_tree = ast.parse(cell.source_code)
            except SyntaxError:
                cell_tree = None

            # Create a synthetic FileInfo with the cell's line offset context
            cell_file_info = FileInfo(
                path=file_info.path,
                file_type=FileType.PYTHON,
                last_modified=file_info.last_modified,
                size=file_info.size,
            )
            for detector in self.detectors:
                try:
                    detector_vulns = detector.detect(cell_file_info, cell_tree)
                    vulns.extend(detector_vulns)
                except Exception as e:
                    logger.error(f"Error running {detector.name} on notebook cell: {e}")

        # Also scan output cells for secrets
        secrets_detector = next(
            (d for d in self.detectors if isinstance(d, SecretsDetector)), None
        )
        if secrets_detector:
            for output_cell in content.output_cells:
                output_vulns = secrets_detector.detect_in_code(
                    output_cell.output_text, file_info.path
                )
                vulns.extend(output_vulns)

        return vulns
    
    def _parse_python_file(self, file_path: str) -> Optional[ast.AST]:
        """Parse Python file into AST tree
        
        This method reads a Python file and parses it into an Abstract Syntax Tree.
        Parse errors are handled gracefully by logging the error and returning None.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Parsed AST tree, or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            # Parse into AST
            tree = ast.parse(source_code, filename=file_path)
            return tree
            
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _create_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """Create FileInfo object for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo object if file type is supported, None otherwise
        """
        # Determine file type
        file_type = self._get_file_type(file_path)
        if file_type is None:
            return None
        
        try:
            # Get file metadata
            stat = file_path.stat()
            
            return FileInfo(
                path=str(file_path),
                file_type=file_type,
                last_modified=stat.st_mtime,
                size=stat.st_size
            )
        except OSError as e:
            logger.warning(f"Error accessing file {file_path}: {e}")
            return None
    
    def _get_file_type(self, file_path: Path) -> Optional[FileType]:
        """Determine file type from extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileType enum value, or None if file type is not supported
        """
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        # Python files
        if suffix == '.py':
            return FileType.PYTHON
        
        # Jupyter notebooks
        if suffix == '.ipynb':
            return FileType.NOTEBOOK
        
        # JSON configuration files
        if suffix == '.json':
            return FileType.CONFIG_JSON
        
        # YAML configuration files
        if suffix in ['.yaml', '.yml']:
            return FileType.CONFIG_YAML
        
        # Requirements files
        if name == 'requirements.txt' or name.startswith('requirements') and name.endswith('.txt'):
            return FileType.REQUIREMENTS
        
        return None
    
    def _is_excluded(self, file_path: str) -> bool:
        """Check if file matches any exclusion patterns
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be excluded, False otherwise
        """
        # Convert to Path for easier manipulation
        path = Path(file_path)
        
        # Check against exclusion patterns
        for pattern in self.config.exclude_patterns:
            # Simple pattern matching (can be enhanced with glob/regex)
            if pattern in str(path):
                return True
            
            # Check if any part of the path matches the pattern
            if any(pattern in part for part in path.parts):
                return True
        
        # Common directories to exclude by default
        default_exclusions = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            '.hypothesis',
            'node_modules',
            'venv',
            'env',
            '.venv',
            '.env',
        ]
        
        for exclusion in default_exclusions:
            if exclusion in path.parts:
                return True
        
        return False
