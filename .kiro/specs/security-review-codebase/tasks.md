# Implementation Plan: Security Review Codebase

## Overview

This implementation plan breaks down the security-review-codebase feature into discrete coding tasks. The scanner will analyze Python code, Jupyter notebooks, and configuration files to detect security vulnerabilities using AST-based analysis, pattern matching, and context-aware detection.

The implementation follows a bottom-up approach: core data models → base detector framework → individual detectors → parsers → reporting → CLI → caching → false positive filtering. Each major component includes property-based tests to validate universal correctness properties.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create directory structure: `security_scanner/`, `security_scanner/detectors/`, `security_scanner/parsers/`, `security_scanner/reporting/`, `security_scanner/cache/`, `security_scanner/filters/`, `tests/`
  - Define core data models: `Vulnerability`, `SeverityLevel`, `VulnerabilityType`, `FileType`, `FileInfo`, `ScanResult`, `ScanConfig`, `DetectorConfig`
  - Create enums for severity levels and vulnerability types
  - Set up testing framework with pytest and Hypothesis for property-based testing
  - Create `requirements.txt` with dependencies: `pytest`, `hypothesis`, `nbformat` (for notebooks), `pyyaml`
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 13.5, 14.1_

- [ ] 2. Implement AST analysis helpers
  - [x] 2.1 Create `ASTAnalyzer` helper class
    - Implement `is_user_controlled()` to detect user input sources
    - Implement `get_function_name()` to extract fully qualified function names
    - Implement `get_code_snippet()` to extract code with context lines
    - Implement `has_validation()` to check for input validation patterns
    - _Requirements: 1.1-1.5, 3.1-3.4, 4.1-4.3, 5.1-5.5_

  - [ ]* 2.2 Write unit tests for AST helpers
    - Test function name extraction with various call patterns
    - Test user input detection with function parameters, file reads, network requests
    - Test code snippet extraction with edge cases (file start/end)
    - _Requirements: 1.1-1.5_

- [ ] 3. Implement base detector framework
  - [x] 3.1 Create `BaseDetector` abstract class
    - Define `detect()` abstract method accepting `FileInfo` and optional AST tree
    - Implement `get_severity()` for context-based severity determination
    - Implement `should_suppress()` for suppression rule checking
    - Add detector name and configuration properties
    - _Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 4.1-4.5_

  - [ ]* 3.2 Write unit tests for base detector
    - Test severity determination logic
    - Test suppression rule matching
    - Test detector configuration loading
    - _Requirements: 13.4_

- [ ] 4. Implement Code Injection Detector
  - [x] 4.1 Create `InjectionDetector` class extending `BaseDetector`
    - Implement AST visitor to detect `eval()`, `exec()`, `compile()`, `__import__()` calls
    - Implement context analysis to determine if input is user-controlled
    - Assign severity: HIGH for eval/exec/compile, MEDIUM for __import__
    - Generate vulnerability with code snippet and remediation recommendation
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 4.2 Write property test for dangerous function detection
    - **Property 1: Dangerous Function Detection**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    - Generate random Python code containing dangerous functions
    - Verify detection regardless of nesting level or context
    - Verify correct severity assignment for each function type

  - [ ]* 4.3 Write unit tests for injection detector
    - Test eval() detection with various code patterns
    - Test exec() detection with string and compiled code
    - Test compile() detection with user-controlled input
    - Test __import__() detection with dynamic module names
    - Test pickle.loads() detection (Critical severity)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 5. Implement Secrets Detector
  - [x] 5.1 Create `SecretsDetector` class extending `BaseDetector`
    - Define regex patterns for AWS keys, API keys, private keys, password assignments
    - Implement Shannon entropy calculation for high-entropy string detection
    - Implement variable name analysis (password, api_key, secret, token)
    - Assign severity: CRITICAL for AWS keys and private keys, HIGH for passwords
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 5.2 Write property test for secrets pattern matching
    - **Property 2: Secrets Pattern Matching**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    - Generate random strings matching credential patterns
    - Generate random strings NOT matching patterns (negative cases)
    - Verify 100% accuracy on pattern matching

  - [ ]* 5.3 Write unit tests for secrets detector
    - Test AWS access key pattern matching
    - Test generic API key detection with entropy analysis
    - Test private key detection (PEM format)
    - Test password assignment detection
    - Test false positives (comments, test data)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 6. Implement Path Traversal Detector
  - [x] 6.1 Create `PathTraversalDetector` class extending `BaseDetector`
    - Detect file operations: `open()`, `os.path.join()`, `Path()`, `shutil.copy()`, `shutil.move()`
    - Detect unsafe path construction with string concatenation
    - Detect paths containing ".." without validation
    - Check for `pathlib.Path.resolve()` usage
    - Assign severity: HIGH for unvalidated user paths, MEDIUM for os.path.join without validation
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 6.2 Write property test for path traversal detection
    - **Property 3: Path Traversal Detection**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    - Generate random file operation code with various path construction methods
    - Verify detection of string concatenation and ".." patterns
    - Verify no false positives on safe pathlib usage

  - [ ]* 6.3 Write unit tests for path traversal detector
    - Test string concatenation detection
    - Test os.path.join without validation
    - Test ".." in paths
    - Test open() with user-controlled paths
    - Test safe pathlib.Path.resolve() usage (no detection)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement Deserialization Detector
  - [x] 8.1 Create `DeserializationDetector` class extending `BaseDetector`
    - Detect `torch.load()` without `weights_only=True` parameter
    - Detect `pickle.load()` on untrusted data sources
    - Detect `yaml.load()` without `SafeLoader`
    - Detect JSON deserialization without schema validation
    - Assign severity: CRITICAL for pickle.load, HIGH for torch.load and yaml.load, MEDIUM for JSON
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 8.1, 8.2_

  - [ ]* 8.2 Write property test for unsafe deserialization detection
    - **Property 4: Unsafe Deserialization Detection**
    - **Validates: Requirements 4.1, 4.2, 4.3, 8.1, 8.2**
    - Generate random deserialization calls with various parameter combinations
    - Verify detection when safe parameters are missing
    - Verify no detection when safe parameters are present

  - [ ]* 8.3 Write unit tests for deserialization detector
    - Test torch.load() without weights_only=True
    - Test torch.load() with weights_only=True (no detection)
    - Test pickle.load() detection
    - Test yaml.load() without SafeLoader
    - Test yaml.safe_load() (no detection)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 8.1, 8.2, 8.5_

- [ ] 9. Implement Input Validation Detector
  - [x] 9.1 Create `ValidationDetector` class extending `BaseDetector`
    - Detect functions accepting external input without type checking
    - Detect file path parameters without directory validation
    - Detect numeric inputs without range validation
    - Detect string inputs without length limits
    - Detect dataset loaders without file existence checks
    - Assign severity: HIGH for file paths, MEDIUM for type checking and strings, LOW for numeric ranges
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 9.2 Write unit tests for validation detector
    - Test detection of missing type checks
    - Test detection of unvalidated file paths
    - Test detection of missing range validation
    - Test detection of missing length limits
    - Test detection of missing existence checks
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 10. Implement File Operations Detector
  - [x] 10.1 Create `FileOpsDetector` class extending `BaseDetector`
    - Detect file creation with overly permissive permissions (0o777)
    - Detect insecure temporary file creation (not using tempfile.mkstemp)
    - Detect file deletion without path validation
    - Detect writes to world-writable directories
    - Detect symbolic link usage without validation
    - Assign severity: HIGH for unvalidated deletions, MEDIUM for permissions and temp files
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 10.2 Write property test for file permission detection
    - **Property 5: File Permission Detection**
    - **Validates: Requirements 6.1**
    - Generate random file creation code with various permission values
    - Verify detection of overly permissive permissions (0o777)

  - [ ]* 10.3 Write property test for insecure temporary file detection
    - **Property 6: Insecure Temporary File Detection**
    - **Validates: Requirements 6.2, 6.3**
    - Generate random temporary file creation code
    - Verify detection of insecure patterns
    - Verify no detection when using tempfile.mkstemp

  - [ ]* 10.4 Write unit tests for file operations detector
    - Test 0o777 permission detection
    - Test insecure temp file creation
    - Test file deletion without validation
    - Test world-writable directory writes
    - Test symbolic link usage
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 11. Implement Configuration Detector
  - [x] 11.1 Create `ConfigDetector` class extending `BaseDetector`
    - Parse JSON and YAML configuration files
    - Detect debug mode enabled (debug=true, DEBUG=1)
    - Detect insecure default values
    - Detect absolute paths to sensitive directories
    - Detect missing required security settings
    - Detect world-readable files with sensitive data
    - Assign severity: HIGH for world-readable sensitive files, MEDIUM for debug mode, LOW for defaults
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 11.2 Write property test for configuration security detection
    - **Property 7: Configuration Security Detection**
    - **Validates: Requirements 7.1, 7.3**
    - Generate random configuration files with debug flags and sensitive paths
    - Verify detection of security issues with appropriate severity

  - [ ]* 11.3 Write unit tests for configuration detector
    - Test debug mode detection in JSON
    - Test insecure default values
    - Test sensitive path detection
    - Test missing security settings
    - Test world-readable file detection
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 12. Implement Checkpoint Security Detector
  - [x] 12.1 Create `CheckpointDetector` class extending `BaseDetector`
    - Detect torch.load() without weights_only=True (reuse deserialization logic)
    - Detect checkpoint loading from user-specified paths without validation
    - Detect missing checkpoint integrity verification
    - Detect missing metadata validation
    - Assign severity: HIGH for unsafe loading and unvalidated paths, MEDIUM for missing integrity checks
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 12.2 Write unit tests for checkpoint detector
    - Test torch.load() detection (covered by deserialization tests)
    - Test user-specified path detection
    - Test missing integrity verification
    - Test missing metadata validation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 13. Implement Data Pipeline Detector
  - [x] 13.1 Create `PipelineDetector` class extending `BaseDetector`
    - Detect data loaders with unvalidated file paths
    - Detect shell command execution with user input (subprocess, os.system)
    - Detect missing error handling for malformed input
    - Detect dataset classes loading files without size limits
    - Detect random seeds from external sources
    - Assign severity: CRITICAL for shell injection, MEDIUM for unvalidated paths and size limits, LOW for error handling
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 13.2 Write property test for shell command injection detection
    - **Property 8: Shell Command Injection Detection**
    - **Validates: Requirements 9.2**
    - Generate random shell command execution code with user-controlled input
    - Verify detection and CRITICAL severity assignment

  - [ ]* 13.3 Write unit tests for pipeline detector
    - Test data loader path validation
    - Test shell command injection (subprocess, os.system)
    - Test missing error handling
    - Test missing size limits
    - Test external random seeds
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 14. Implement Dependency Detector
  - [x] 14.1 Create `DependencyDetector` class extending `BaseDetector`
    - Detect imports of deprecated or insecure libraries
    - Detect usage of functions with known security issues
    - Parse requirements.txt and detect missing version pinning
    - Detect imports from untrusted sources
    - Assign severity: HIGH for insecure functions and untrusted sources, MEDIUM for deprecated libraries, LOW for version pinning
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 14.2 Write property test for requirements version pinning detection
    - **Property 9: Requirements Version Pinning Detection**
    - **Validates: Requirements 10.3**
    - Generate random requirements.txt files with and without version pinning
    - Verify detection of unpinned dependencies

  - [ ]* 14.3 Write unit tests for dependency detector
    - Test deprecated library detection
    - Test insecure function detection
    - Test missing version pinning in requirements.txt
    - Test untrusted source imports
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Implement Jupyter Notebook Parser
  - [x] 16.1 Create `NotebookParser` class
    - Implement `parse()` to load notebook JSON using nbformat
    - Implement `extract_code_cells()` to get executable code cells with line offsets
    - Implement `extract_output_cells()` to get output cells for secrets scanning
    - Handle malformed notebooks gracefully with error logging
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 16.2 Write property test for notebook secrets detection
    - **Property 10: Notebook Secrets Detection**
    - **Validates: Requirements 11.1, 11.5**
    - Generate random notebooks with credential patterns in code and output cells
    - Verify detection with appropriate severity

  - [ ]* 16.3 Write property test for notebook magic command detection
    - **Property 11: Notebook Magic Command Detection**
    - **Validates: Requirements 11.2, 11.4**
    - Generate random notebooks with shell magic commands (!, %run)
    - Verify detection of security risks

  - [ ]* 16.4 Write unit tests for notebook parser
    - Test code cell extraction with line offsets
    - Test output cell extraction
    - Test malformed notebook handling
    - Test secrets in code cells
    - Test secrets in output cells
    - Test shell magic commands (!, %run)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 17. Implement Vulnerability Aggregator and Report Generator
  - [x] 17.1 Create `ReportGenerator` class
    - Implement `group_by_severity()` to organize vulnerabilities by severity level
    - Implement `generate_json()` for JSON output format
    - Implement `generate_markdown()` for Markdown output format
    - Implement `generate_html()` for HTML output format
    - Include summary with vulnerability counts by severity
    - Include file path, line number, code snippet, and recommendation for each vulnerability
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 17.2 Write property test for vulnerability report grouping
    - **Property 12: Vulnerability Report Grouping**
    - **Validates: Requirements 12.1, 12.2, 12.3**
    - Generate random vulnerability lists with mixed severities
    - Verify correct grouping by severity
    - Verify all required fields present in each vulnerability

  - [ ]* 17.3 Write property test for vulnerability count calculation
    - **Property 13: Vulnerability Count Calculation**
    - **Validates: Requirements 12.4**
    - Generate random vulnerability lists
    - Verify sum of severity counts equals total count
    - Verify no vulnerabilities lost or duplicated

  - [ ]* 17.4 Write property test for report format validity
    - **Property 14: Report Format Validity**
    - **Validates: Requirements 12.5**
    - Generate random scan results
    - Verify JSON output is valid JSON
    - Verify Markdown output renders without errors
    - Verify HTML output is valid HTML

  - [ ]* 17.5 Write unit tests for report generator
    - Test grouping by severity
    - Test JSON format generation
    - Test Markdown format generation
    - Test HTML format generation
    - Test summary generation with counts
    - Test required fields in each vulnerability
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 18. Implement False Positive Filter
  - [x] 18.1 Create `FalsePositiveFilter` class
    - Implement `filter()` to process vulnerability lists
    - Implement `has_safe_wrapper()` to detect safe wrappers around dangerous functions
    - Implement `has_security_comment()` to detect security-related comments
    - Implement `is_suppressed()` to check suppression rules
    - Load suppression configuration from JSON file
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 18.2 Write property test for false positive suppression
    - **Property 15: False Positive Suppression**
    - **Validates: Requirements 13.4**
    - Generate random vulnerabilities and suppression rules
    - Verify vulnerabilities matching suppression rules are excluded

  - [ ]* 18.3 Write property test for confidence score validity
    - **Property 16: Confidence Score Validity**
    - **Validates: Requirements 13.5**
    - For any detected vulnerability, verify 0.0 <= confidence <= 1.0
    - Verify confidence is a float type

  - [ ]* 18.4 Write unit tests for false positive filter
    - Test safe wrapper detection
    - Test security comment detection
    - Test suppression rule matching
    - Test global suppression patterns
    - Test confidence score assignment
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 19. Implement Result Cache
  - [x] 19.1 Create `ResultCache` class
    - Implement `get_cached_result()` to retrieve cached results by file hash
    - Implement `store_result()` to save scan results with file hash
    - Implement `invalidate()` to remove cache entries
    - Use JSON for cache storage
    - Calculate file hashes using SHA256
    - _Requirements: 14.3, 14.4_

  - [ ]* 19.2 Write unit tests for result cache
    - Test cache storage and retrieval
    - Test cache invalidation
    - Test file hash calculation
    - Test cache hit/miss scenarios
    - Test corrupted cache handling
    - _Requirements: 14.3, 14.4_

- [ ] 20. Implement Security Scanner Core
  - [x] 20.1 Create `SecurityScanner` class
    - Implement `scan()` to orchestrate the scanning workflow
    - Implement `discover_files()` to find Python, notebook, and config files
    - Implement `should_scan_file()` to check cache and exclusion patterns
    - Initialize all detectors (10 detectors)
    - Integrate with cache for incremental scanning
    - Parse Python files into AST trees
    - Handle parse errors gracefully (log and continue)
    - _Requirements: 1.1-14.5_

  - [ ]* 20.2 Write integration tests for scanner core
    - Test file discovery with various patterns
    - Test cache integration
    - Test detector orchestration
    - Test AST parsing and error handling
    - Test exclusion pattern matching
    - _Requirements: 14.3, 14.4_

- [x] 21. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 22. Implement CLI Interface
  - [x] 22.1 Create `SecurityCLI` class
    - Implement argument parser with all CLI options (--path, --config, --output, --format, --severity, --incremental, --fail-on, --exclude, --cache-dir)
    - Implement `run()` to execute scan and generate report
    - Implement exit code logic based on severity thresholds
    - Load configuration from file if provided
    - Handle errors and display user-friendly messages
    - _Requirements: 14.1, 14.2_

  - [ ]* 22.2 Write property test for exit code determination
    - **Property 17: Exit Code Determination**
    - **Validates: Requirements 14.2**
    - Generate scan results with various severity distributions
    - Verify exit code is non-zero when threshold exceeded
    - Verify exit code is zero when no vulnerabilities exceed threshold

  - [ ]* 22.3 Write unit tests for CLI interface
    - Test argument parsing
    - Test configuration file loading
    - Test exit code logic
    - Test error handling
    - Test output format selection
    - _Requirements: 14.1, 14.2_

- [ ] 23. Create CLI entry point and package setup
  - [x] 23.1 Create `__main__.py` for CLI entry point
    - Import `SecurityCLI` and call `run()`
    - Handle keyboard interrupts gracefully
    - _Requirements: 14.1_

  - [x] 23.2 Create `setup.py` or `pyproject.toml` for package installation
    - Define package metadata
    - Define console script entry point: `security-scan`
    - List all dependencies
    - _Requirements: 14.1_

  - [ ]* 23.3 Write end-to-end tests
    - Test CLI execution on sample codebase
    - Test report generation in all formats
    - Test incremental scanning
    - Test suppression configuration
    - Test CI/CD integration scenario
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 24. Integration and wiring
  - [x] 24.1 Wire all components together
    - Ensure scanner initializes all 10 detectors
    - Ensure notebook parser integrates with detectors
    - Ensure false positive filter processes all vulnerabilities
    - Ensure report generator receives filtered results
    - Ensure cache integrates with incremental scanning
    - Test complete workflow: file discovery → AST parsing → detection → filtering → reporting
    - _Requirements: 1.1-14.5_

  - [ ]* 24.2 Write integration tests for complete workflow
    - Test scanning Python files with multiple vulnerability types
    - Test scanning Jupyter notebooks
    - Test scanning configuration files
    - Test incremental scanning with cache
    - Test suppression rules application
    - Test report generation in all formats
    - _Requirements: 1.1-14.5_

- [x] 25. Final checkpoint - Ensure all tests pass
  - Run complete test suite including all property-based tests (minimum 100 iterations each)
  - Verify all 17 properties pass
  - Verify all unit tests pass
  - Verify all integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (17 properties total)
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- The scanner uses Python AST analysis for accurate code inspection
- All detectors extend `BaseDetector` for consistent interface
- False positive filtering reduces noise through context-aware analysis
- Multiple output formats (JSON, Markdown, HTML) support various integration needs
- Caching enables fast incremental scanning for CI/CD pipelines
