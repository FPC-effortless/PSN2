# Design Document: Security Review Codebase

## Overview

The security-review-codebase feature provides automated security vulnerability detection for Python machine learning codebases. The system analyzes Python source files, Jupyter notebooks, and configuration files to identify common security issues including code injection, credential exposure, unsafe deserialization, path traversal, and insecure file operations.

The scanner uses Abstract Syntax Tree (AST) analysis for Python code inspection, combined with pattern matching for configuration files and specialized notebook parsing for Jupyter files. The design emphasizes low false-positive rates through context-aware analysis and configurable suppression mechanisms.

**Key Design Principles:**
- AST-based analysis for accurate Python code inspection
- Modular detector architecture for extensibility
- Severity-based vulnerability classification (Critical, High, Medium, Low, Info)
- Context-aware analysis to reduce false positives
- Multiple output formats (JSON, Markdown, HTML) for integration flexibility
- CLI-first design for CI/CD pipeline integration

## Architecture

### High-Level Architecture

```mermaid
graph TB
    CLI[CLI Interface] --> Scanner[Security Scanner]
    Scanner --> FileDiscovery[File Discovery]
    Scanner --> Cache[Result Cache]
    
    FileDiscovery --> PythonFiles[Python Files]
    FileDiscovery --> NotebookFiles[Jupyter Notebooks]
    FileDiscovery --> ConfigFiles[Config Files]
    
    Scanner --> DetectorEngine[Detector Engine]
    
    DetectorEngine --> InjectionDetector[Code Injection Detector]
    DetectorEngine --> SecretsDetector[Secrets Detector]
    DetectorEngine --> PathDetector[Path Traversal Detector]
    DetectorEngine --> DeserializationDetector[Deserialization Detector]
    DetectorEngine --> ValidationDetector[Input Validation Detector]
    DetectorEngine --> FileOpsDetector[File Operations Detector]
    DetectorEngine --> ConfigDetector[Configuration Detector]
    DetectorEngine --> CheckpointDetector[Checkpoint Security Detector]
    DetectorEngine --> PipelineDetector[Data Pipeline Detector]
    DetectorEngine --> DependencyDetector[Dependency Detector]
    
    DetectorEngine --> VulnerabilityAggregator[Vulnerability Aggregator]
    VulnerabilityAggregator --> FalsePositiveFilter[False Positive Filter]
    FalsePositiveFilter --> ReportGenerator[Report Generator]
    
    ReportGenerator --> JSONReport[JSON Output]
    ReportGenerator --> MarkdownReport[Markdown Output]
    ReportGenerator --> HTMLReport[HTML Output]
```

### Component Layers

1. **CLI Layer**: Command-line interface for user interaction and CI/CD integration
2. **Scanner Core**: Orchestrates file discovery, caching, and detector execution
3. **Detector Layer**: Modular vulnerability detectors implementing specific security checks
4. **Analysis Layer**: AST parsing, pattern matching, and context analysis
5. **Reporting Layer**: Vulnerability aggregation, filtering, and multi-format output generation

## Components and Interfaces

### 1. CLI Interface (`cli.py`)

**Responsibility**: Parse command-line arguments and orchestrate scanning workflow

**Interface**:
```python
class SecurityCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments"""
        
    def run(self, args: argparse.Namespace) -> int:
        """Execute security scan and return exit code"""
```

**Command-line Arguments**:
- `--path`: Target directory or file to scan (default: current directory)
- `--config`: Path to configuration file for suppression rules
- `--output`: Output file path (default: stdout)
- `--format`: Output format (json, markdown, html)
- `--severity`: Minimum severity level to report (critical, high, medium, low, info)
- `--incremental`: Scan only changed files (requires git)
- `--fail-on`: Exit with non-zero code on specified severity levels
- `--exclude`: Glob patterns for files to exclude
- `--cache-dir`: Directory for caching scan results

### 2. Security Scanner (`scanner.py`)

**Responsibility**: Orchestrate file discovery, detector execution, and result aggregation

**Interface**:
```python
class SecurityScanner:
    def __init__(self, config: ScanConfig):
        self.config = config
        self.detectors: List[BaseDetector] = []
        self.cache = ResultCache(config.cache_dir)
        
    def scan(self, target_path: str) -> ScanResult:
        """Execute security scan on target path"""
        
    def discover_files(self, target_path: str) -> List[FileInfo]:
        """Discover Python, notebook, and config files"""
        
    def should_scan_file(self, file_path: str) -> bool:
        """Check if file should be scanned based on cache and config"""
```

**Data Models**:
```python
@dataclass
class ScanConfig:
    target_path: str
    exclude_patterns: List[str]
    min_severity: SeverityLevel
    incremental: bool
    cache_dir: Optional[str]
    suppression_rules: Dict[str, Any]
    
@dataclass
class FileInfo:
    path: str
    file_type: FileType  # PYTHON, NOTEBOOK, CONFIG
    last_modified: float
    size: int
    
@dataclass
class ScanResult:
    vulnerabilities: List[Vulnerability]
    files_scanned: int
    scan_duration: float
    timestamp: str
```

### 3. Base Detector (`detectors/base.py`)

**Responsibility**: Define common interface for all vulnerability detectors

**Interface**:
```python
class BaseDetector(ABC):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST]) -> List[Vulnerability]:
        """Detect vulnerabilities in file"""
        
    def get_severity(self, context: Dict[str, Any]) -> SeverityLevel:
        """Determine vulnerability severity based on context"""
        
    def should_suppress(self, vuln: Vulnerability) -> bool:
        """Check if vulnerability should be suppressed"""
```

**Vulnerability Data Model**:
```python
@dataclass
class Vulnerability:
    id: str  # Unique identifier (e.g., "INJ001")
    title: str
    description: str
    severity: SeverityLevel
    file_path: str
    line_number: int
    column: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Any]  # Additional context for analysis
    
enum SeverityLevel:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
```

### 4. Code Injection Detector (`detectors/injection.py`)

**Responsibility**: Detect code injection vulnerabilities (eval, exec, compile, __import__)

**Detection Strategy**:
- AST visitor pattern to identify dangerous function calls
- Context analysis to determine if input is user-controlled
- Check for sanitization wrappers

**Key Patterns**:
```python
class InjectionDetector(BaseDetector):
    DANGEROUS_FUNCTIONS = {
        'eval': SeverityLevel.HIGH,
        'exec': SeverityLevel.HIGH,
        'compile': SeverityLevel.HIGH,
        '__import__': SeverityLevel.MEDIUM,
    }
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes"""
        if self._is_dangerous_call(node):
            if self._is_user_controlled(node):
                self._report_vulnerability(node)
```

### 5. Secrets Detector (`detectors/secrets.py`)

**Responsibility**: Detect hardcoded credentials, API keys, and sensitive data

**Detection Strategy**:
- Regex patterns for common credential formats
- Entropy analysis for high-entropy strings
- Variable name analysis (password, api_key, secret, token)
- Special handling for Jupyter notebook output cells

**Key Patterns**:
```python
class SecretsDetector(BaseDetector):
    PATTERNS = {
        'aws_access_key': r'AKIA[0-9A-Z]{16}',
        'generic_api_key': r'["\']([A-Za-z0-9_-]{20,})["\']',
        'private_key': r'-----BEGIN (RSA |EC )?PRIVATE KEY-----',
        'password_assignment': r'password\s*=\s*["\'][^"\']+["\']',
    }
    
    def detect_high_entropy_strings(self, node: ast.Str) -> bool:
        """Detect strings with high entropy (potential secrets)"""
        entropy = self._calculate_shannon_entropy(node.s)
        return entropy > 4.5 and len(node.s) > 16
```

### 6. Path Traversal Detector (`detectors/path_traversal.py`)

**Responsibility**: Detect path traversal vulnerabilities in file operations

**Detection Strategy**:
- Identify file operations with user-controlled paths
- Check for path validation and sanitization
- Detect string concatenation for path construction
- Verify use of pathlib.Path.resolve()

**Key Patterns**:
```python
class PathTraversalDetector(BaseDetector):
    FILE_OPERATIONS = ['open', 'os.path.join', 'Path', 'shutil.copy', 'shutil.move']
    
    def detect_unsafe_path_construction(self, node: ast.Call) -> bool:
        """Detect unsafe path construction patterns"""
        # Check for string concatenation with '..'
        # Check for os.path.join without validation
        # Check for open() with user-controlled paths
```

### 7. Deserialization Detector (`detectors/deserialization.py`)

**Responsibility**: Detect unsafe deserialization operations

**Detection Strategy**:
- Identify torch.load() without weights_only=True
- Detect pickle.load() on untrusted data
- Check yaml.load() without SafeLoader
- Verify JSON schema validation

**Key Patterns**:
```python
class DeserializationDetector(BaseDetector):
    def detect_unsafe_torch_load(self, node: ast.Call) -> bool:
        """Detect torch.load without weights_only=True"""
        if self._is_torch_load(node):
            has_weights_only = any(
                kw.arg == 'weights_only' and 
                isinstance(kw.value, ast.Constant) and 
                kw.value.value is True
                for kw in node.keywords
            )
            return not has_weights_only
```

### 8. Jupyter Notebook Parser (`parsers/notebook.py`)

**Responsibility**: Parse Jupyter notebooks and extract code cells for analysis

**Interface**:
```python
class NotebookParser:
    def parse(self, notebook_path: str) -> NotebookContent:
        """Parse notebook and extract code cells"""
        
    def extract_code_cells(self, notebook: Dict) -> List[CodeCell]:
        """Extract executable code cells"""
        
    def extract_output_cells(self, notebook: Dict) -> List[OutputCell]:
        """Extract output cells for secrets scanning"""
        
@dataclass
class CodeCell:
    cell_number: int
    source_code: str
    line_offset: int  # Line number in original notebook
    
@dataclass
class OutputCell:
    cell_number: int
    output_text: str
    output_type: str  # stream, display_data, execute_result
```

### 9. Configuration Detector (`detectors/config.py`)

**Responsibility**: Analyze configuration files for security issues

**Detection Strategy**:
- Parse JSON configuration files
- Check for debug mode in production
- Identify insecure default values
- Detect world-readable files with sensitive data

### 10. False Positive Filter (`filters/false_positive.py`)

**Responsibility**: Reduce false positives through context analysis

**Interface**:
```python
class FalsePositiveFilter:
    def __init__(self, suppression_config: Dict[str, Any]):
        self.suppression_rules = suppression_config
        
    def filter(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Filter out false positives"""
        
    def has_safe_wrapper(self, vuln: Vulnerability) -> bool:
        """Check if dangerous function is wrapped in safe context"""
        
    def has_security_comment(self, vuln: Vulnerability) -> bool:
        """Check for security-related comments explaining safe usage"""
        
    def is_suppressed(self, vuln: Vulnerability) -> bool:
        """Check if vulnerability is in suppression list"""
```

**Suppression Configuration Format**:
```json
{
  "suppressions": [
    {
      "id": "INJ001",
      "file": "psn2/config.py",
      "line": 42,
      "reason": "eval() used for safe config parsing with validated input"
    }
  ],
  "global_suppressions": {
    "test_*.py": ["INJ001", "SEC002"]
  }
}
```

### 11. Report Generator (`reporting/generator.py`)

**Responsibility**: Generate vulnerability reports in multiple formats

**Interface**:
```python
class ReportGenerator:
    def generate(self, scan_result: ScanResult, format: ReportFormat) -> str:
        """Generate report in specified format"""
        
    def generate_json(self, scan_result: ScanResult) -> str:
        """Generate JSON report"""
        
    def generate_markdown(self, scan_result: ScanResult) -> str:
        """Generate Markdown report"""
        
    def generate_html(self, scan_result: ScanResult) -> str:
        """Generate HTML report"""
        
    def group_by_severity(self, vulnerabilities: List[Vulnerability]) -> Dict[SeverityLevel, List[Vulnerability]]:
        """Group vulnerabilities by severity"""
```

**Report Structure**:
- Executive summary with vulnerability counts by severity
- Detailed vulnerability listings grouped by severity
- File-by-file breakdown
- Remediation recommendations
- Scan metadata (timestamp, duration, files scanned)

### 12. Result Cache (`cache/result_cache.py`)

**Responsibility**: Cache scan results for incremental scanning

**Interface**:
```python
class ResultCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "scan_cache.json"
        
    def get_cached_result(self, file_path: str, file_hash: str) -> Optional[List[Vulnerability]]:
        """Retrieve cached scan result if file unchanged"""
        
    def store_result(self, file_path: str, file_hash: str, vulnerabilities: List[Vulnerability]) -> None:
        """Store scan result in cache"""
        
    def invalidate(self, file_path: str) -> None:
        """Invalidate cache entry for file"""
```

## Data Models

### Core Data Structures

```python
# Vulnerability Classification
class VulnerabilityType(Enum):
    CODE_INJECTION = "code_injection"
    SECRETS_EXPOSURE = "secrets_exposure"
    PATH_TRAVERSAL = "path_traversal"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    INPUT_VALIDATION = "input_validation"
    FILE_OPERATIONS = "file_operations"
    CONFIGURATION = "configuration"
    CHECKPOINT_SECURITY = "checkpoint_security"
    DATA_PIPELINE = "data_pipeline"
    DEPENDENCY = "dependency"

# File Type Classification
class FileType(Enum):
    PYTHON = "python"
    NOTEBOOK = "notebook"
    CONFIG_JSON = "config_json"
    CONFIG_YAML = "config_yaml"
    REQUIREMENTS = "requirements"

# Detector Configuration
@dataclass
class DetectorConfig:
    enabled: bool = True
    severity_overrides: Dict[str, SeverityLevel] = field(default_factory=dict)
    custom_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
```

### AST Analysis Helpers

```python
class ASTAnalyzer:
    """Helper class for AST analysis"""
    
    @staticmethod
    def is_user_controlled(node: ast.AST, context: ast.AST) -> bool:
        """Determine if node value comes from user input"""
        # Check for function parameters, file reads, network requests
        
    @staticmethod
    def get_function_name(node: ast.Call) -> Optional[str]:
        """Extract fully qualified function name"""
        
    @staticmethod
    def get_code_snippet(file_path: str, line_number: int, context_lines: int = 3) -> str:
        """Extract code snippet with context"""
        
    @staticmethod
    def has_validation(node: ast.AST, context: ast.AST) -> bool:
        """Check if input has validation before use"""
```

## Error Handling

### Error Categories

1. **File Access Errors**: Handle permission denied, file not found
2. **Parse Errors**: Handle malformed Python files, invalid notebooks
3. **Configuration Errors**: Handle invalid suppression config
4. **Cache Errors**: Handle corrupted cache, disk full

### Error Handling Strategy

```python
class ScanError(Exception):
    """Base exception for scan errors"""
    pass

class FileAccessError(ScanError):
    """File cannot be accessed"""
    pass

class ParseError(ScanError):
    """File cannot be parsed"""
    pass

# Error handling in scanner
try:
    ast_tree = ast.parse(source_code)
except SyntaxError as e:
    logger.warning(f"Syntax error in {file_path}: {e}")
    # Continue scanning other files
    continue
```

### Graceful Degradation

- Skip files that cannot be parsed (log warning)
- Continue scanning if individual detector fails
- Provide partial results if scan is interrupted
- Cache partial results for resume capability

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, several properties can be consolidated:
- Properties 1.1-1.5 (dangerous function detection) can be combined into a single property about detecting any dangerous function call
- Properties 2.1-2.4 (secrets pattern matching) can be combined into a property about pattern matching accuracy
- Properties 4.1 and 8.1 are identical (torch.load detection) and should be merged
- Properties related to recommendation text (3.5, 4.5, 8.5, 10.5) are examples, not properties
- Input validation properties (5.1-5.5) require complex context analysis and are better suited for integration tests

### Property 1: Dangerous Function Detection

*For any* Python code containing calls to dangerous functions (eval, exec, compile, __import__, pickle.loads), the Security_Scanner SHALL detect the function call and assign the correct severity level based on the function type.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

### Property 2: Secrets Pattern Matching

*For any* string matching known credential patterns (API keys, AWS keys, private keys, password assignments), the Security_Scanner SHALL detect the pattern and flag it with appropriate severity.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

### Property 3: Path Traversal Detection

*For any* code performing file operations with string concatenation or containing ".." in paths, the Security_Scanner SHALL detect the unsafe path construction and flag it as a vulnerability.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 4: Unsafe Deserialization Detection

*For any* code using deserialization functions (torch.load, pickle.load, yaml.load) without safe parameters, the Security_Scanner SHALL detect the unsafe usage and flag it with appropriate severity.

**Validates: Requirements 4.1, 4.2, 4.3, 8.1, 8.2**

### Property 5: File Permission Detection

*For any* code creating files or directories with permission values, the Security_Scanner SHALL detect overly permissive permissions (0o777) and flag them as vulnerabilities.

**Validates: Requirements 6.1**

### Property 6: Insecure Temporary File Detection

*For any* code creating temporary files without using secure methods (tempfile.mkstemp), the Security_Scanner SHALL detect the insecure pattern and flag it.

**Validates: Requirements 6.2, 6.3**

### Property 7: Configuration Security Detection

*For any* configuration file (JSON/YAML) containing debug flags, insecure defaults, or sensitive paths, the Security_Scanner SHALL detect the security issue and flag it with appropriate severity.

**Validates: Requirements 7.1, 7.3**

### Property 8: Shell Command Injection Detection

*For any* code executing shell commands (subprocess, os.system) with user-controlled input, the Security_Scanner SHALL detect the injection risk and flag it as Critical severity.

**Validates: Requirements 9.2**

### Property 9: Requirements Version Pinning Detection

*For any* requirements.txt file, if a dependency lacks version pinning (no ==, >=, ~= specifier), the Security_Scanner SHALL flag it as a Low severity vulnerability.

**Validates: Requirements 10.3**

### Property 10: Notebook Secrets Detection

*For any* Jupyter notebook containing credential patterns in code cells or output cells, the Security_Scanner SHALL detect the exposed secrets and flag them with appropriate severity.

**Validates: Requirements 11.1, 11.5**

### Property 11: Notebook Magic Command Detection

*For any* Jupyter notebook containing shell magic commands (!, %run) with external input, the Security_Scanner SHALL detect the security risk and flag it.

**Validates: Requirements 11.2, 11.4**

### Property 12: Vulnerability Report Grouping

*For any* list of detected vulnerabilities, the report generator SHALL correctly group them by severity level and include all required fields (file path, line number, code snippet, recommendation).

**Validates: Requirements 12.1, 12.2, 12.3**

### Property 13: Vulnerability Count Calculation

*For any* list of vulnerabilities, the report summary SHALL accurately count vulnerabilities by severity level, with the sum of all severity counts equaling the total vulnerability count.

**Validates: Requirements 12.4**

### Property 14: Report Format Validity

*For any* scan result, the report generator SHALL produce valid output in all supported formats (JSON, Markdown, HTML) that can be parsed/rendered without errors.

**Validates: Requirements 12.5**

### Property 15: False Positive Suppression

*For any* vulnerability matching a suppression rule (by file, line, and ID), the Security_Scanner SHALL exclude it from the final report.

**Validates: Requirements 13.4**

### Property 16: Confidence Score Validity

*For any* detected vulnerability, the confidence score SHALL be a float value between 0.0 and 1.0 inclusive.

**Validates: Requirements 13.5**

### Property 17: Exit Code Determination

*For any* scan result, if the result contains vulnerabilities at or above the specified severity threshold, the CLI SHALL return a non-zero exit code.

**Validates: Requirements 14.2**

## Testing Strategy

### Dual Testing Approach

The security scanner requires both unit tests for specific examples and property-based tests for universal correctness guarantees:

- **Unit tests**: Verify specific vulnerability patterns, edge cases, and integration points
- **Property tests**: Verify detection logic holds across all valid code structures and patterns
- Together: Comprehensive coverage ensuring both concrete bugs and general correctness

### Property-Based Testing

**Property Test Library**: Use Hypothesis for Python property-based testing

**Configuration**: Each property test MUST run minimum 100 iterations to explore the input space

**Test Tagging**: Each property test MUST include a comment tag referencing the design property:
```python
# Feature: security-review-codebase, Property 1: Dangerous Function Detection
@given(dangerous_function=st.sampled_from(['eval', 'exec', 'compile', '__import__', 'pickle.loads']))
def test_dangerous_function_detection(dangerous_function):
    """Property: Any code with dangerous functions is detected"""
```

**Key Property Tests**:

1. **Property 1 - Dangerous Function Detection**:
   - Generate random Python AST trees containing dangerous function calls
   - Verify detection regardless of nesting level, context, or surrounding code
   - Verify correct severity assignment for each function type

2. **Property 2 - Secrets Pattern Matching**:
   - Generate random strings matching credential patterns
   - Generate random strings NOT matching patterns (negative cases)
   - Verify 100% accuracy on pattern matching

3. **Property 3 - Path Traversal Detection**:
   - Generate random file operation code with various path construction methods
   - Verify detection of string concatenation and ".." patterns
   - Verify no false positives on safe pathlib usage

4. **Property 4 - Unsafe Deserialization**:
   - Generate random deserialization calls with various parameter combinations
   - Verify detection when safe parameters are missing
   - Verify no detection when safe parameters are present

5. **Property 12 - Report Grouping**:
   - Generate random vulnerability lists with mixed severities
   - Verify correct grouping by severity
   - Verify all required fields are present in each vulnerability

6. **Property 13 - Count Calculation**:
   - Generate random vulnerability lists
   - Verify sum of severity counts equals total count
   - Verify no vulnerabilities are lost or duplicated

7. **Property 14 - Format Validity**:
   - Generate random scan results
   - Verify JSON output is valid JSON
   - Verify Markdown output renders without errors
   - Verify HTML output is valid HTML

8. **Property 16 - Confidence Score**:
   - For any detected vulnerability, verify 0.0 <= confidence <= 1.0
   - Verify confidence is a float type

9. **Property 17 - Exit Code Logic**:
   - Generate scan results with various severity distributions
   - Verify exit code is non-zero when threshold is exceeded
   - Verify exit code is zero when no vulnerabilities exceed threshold

### Unit Testing

**Test Coverage Areas**:
1. Individual detector logic with synthetic code samples
2. AST analysis helper functions
3. Pattern matching for secrets detection
4. Path traversal detection logic
5. Notebook parsing and cell extraction
6. False positive filtering rules
7. Report generation in all formats
8. Cache operations (store, retrieve, invalidate)

**Example Unit Tests**:
```python
def test_injection_detector_eval():
    """Test detection of eval() calls"""
    code = "result = eval(user_input)"
    detector = InjectionDetector(config)
    vulns = detector.detect_in_code(code)
    assert len(vulns) == 1
    assert vulns[0].severity == SeverityLevel.HIGH

def test_secrets_detector_aws_key():
    """Test detection of AWS access keys"""
    code = 'aws_key = "AKIAIOSFODNN7EXAMPLE"'
    detector = SecretsDetector(config)
    vulns = detector.detect_in_code(code)
    assert len(vulns) == 1
    assert "AWS" in vulns[0].title

def test_torch_load_without_weights_only():
    """Test detection of unsafe torch.load()"""
    code = "model = torch.load('checkpoint.pt')"
    detector = DeserializationDetector(config)
    vulns = detector.detect_in_code(code)
    assert len(vulns) == 1
    assert "weights_only" in vulns[0].recommendation

def test_recommendation_text_for_path_traversal():
    """Example test: Verify path traversal recommendations"""
    code = "path = base + user_input; open(path)"
    detector = PathTraversalDetector(config)
    vulns = detector.detect_in_code(code)
    assert len(vulns) > 0
    assert "pathlib.Path.resolve()" in vulns[0].recommendation

def test_torch_load_recommendation():
    """Example test: Verify torch.load recommendations"""
    code = "model = torch.load('checkpoint.pt')"
    detector = DeserializationDetector(config)
    vulns = detector.detect_in_code(code)
    assert "weights_only=True" in vulns[0].recommendation
```

### Integration Testing

**Test Scenarios**:
1. Scan entire PSN2 codebase and verify expected vulnerabilities
2. Test incremental scanning with git integration
3. Verify cache invalidation on file changes
4. Test CLI with various argument combinations
5. Verify exit codes for CI/CD integration
6. Test suppression configuration loading and application
7. Verify all output formats (JSON, Markdown, HTML)

**Example Integration Test**:
```python
def test_scan_psn2_codebase():
    """Integration test on real codebase"""
    scanner = SecurityScanner(config)
    result = scanner.scan("psn2/")
    
    # Verify expected vulnerabilities
    assert result.files_scanned > 0
    
    # Check for known issues
    checkpoint_vulns = [v for v in result.vulnerabilities 
                       if "checkpoint.py" in v.file_path]
    assert any("torch.load" in v.description for v in checkpoint_vulns)
```

### End-to-End Testing

**Test Scenarios**:
1. Run CLI scan on test repository with known vulnerabilities
2. Verify report generation and content accuracy
3. Test CI/CD integration with non-zero exit codes
4. Verify incremental scan performance improvement
5. Test suppression configuration workflow

## Performance Considerations

### Optimization Strategies

1. **Parallel File Processing**: Use multiprocessing for scanning multiple files
2. **Incremental Scanning**: Cache results and scan only changed files
3. **AST Caching**: Cache parsed AST trees for large files
4. **Lazy Loading**: Load detectors only when needed
5. **Early Exit**: Skip files based on extension and size limits

### Performance Targets

- Scan 1000 Python files in < 30 seconds
- Incremental scan of 10 changed files in < 5 seconds
- Memory usage < 500MB for large codebases
- Cache hit rate > 80% for incremental scans

### Scalability

```python
class ParallelScanner:
    def scan_parallel(self, files: List[FileInfo], num_workers: int = 4) -> List[Vulnerability]:
        """Scan files in parallel using multiprocessing"""
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(self._scan_file, files)
        return [v for result in results for v in result]
```

## Security Considerations

### Scanner Security

1. **Path Traversal Prevention**: Validate all file paths before access
2. **Resource Limits**: Limit file size and scan duration
3. **Sandboxing**: Do not execute scanned code
4. **Input Validation**: Validate configuration files and CLI arguments

### Privacy

1. **No Data Transmission**: All scanning happens locally
2. **Sensitive Data Handling**: Do not log or cache detected secrets
3. **Report Sanitization**: Option to redact sensitive information from reports

## Deployment and Integration

### Installation

```bash
pip install security-review-codebase
```

### CLI Usage

```bash
# Basic scan
security-scan --path ./psn2

# Scan with custom config
security-scan --path ./psn2 --config .security-scan.json

# Generate HTML report
security-scan --path ./psn2 --format html --output report.html

# CI/CD integration (fail on high/critical)
security-scan --path ./psn2 --fail-on high,critical

# Incremental scan
security-scan --path ./psn2 --incremental
```

### CI/CD Integration

**GitHub Actions Example**:
```yaml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Security Scan
        run: |
          pip install security-review-codebase
          security-scan --path . --fail-on high,critical --format json --output security-report.json
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: security-report
          path: security-report.json
```

### Configuration File

**.security-scan.json**:
```json
{
  "exclude_patterns": [
    "*/tests/*",
    "*/venv/*",
    "*/.git/*"
  ],
  "detectors": {
    "injection": {
      "enabled": true,
      "severity_overrides": {
        "eval": "critical"
      }
    },
    "secrets": {
      "enabled": true,
      "custom_patterns": [
        "custom_api_key_[A-Za-z0-9]{32}"
      ]
    }
  },
  "suppressions": [
    {
      "id": "INJ001",
      "file": "psn2/config.py",
      "line": 42,
      "reason": "Safe eval with validated input"
    }
  ],
  "min_severity": "medium",
  "max_file_size_mb": 10
}
```

## Future Enhancements

### Phase 2 Features

1. **Taint Analysis**: Track data flow from sources to sinks
2. **Machine Learning**: Use ML to reduce false positives
3. **Custom Detectors**: Plugin system for user-defined detectors
4. **IDE Integration**: VS Code extension for real-time scanning
5. **Remediation Suggestions**: Auto-fix capabilities for common issues
6. **Dependency Scanning**: Integration with vulnerability databases
7. **SARIF Output**: Support for SARIF format for tool interoperability

### Advanced Analysis

1. **Control Flow Analysis**: Understand execution paths
2. **Data Flow Analysis**: Track variable assignments and usage
3. **Type Inference**: Use type hints for better context
4. **Cross-File Analysis**: Detect vulnerabilities spanning multiple files

## References

Research findings that informed this design:

1. **AST-Based Security Analysis**: Python's Abstract Syntax Tree provides syntactically complete code representation, enabling accurate pattern detection without executing code ([source](https://appsecsanta.com/bandit)).

2. **PyTorch Security**: Recent vulnerabilities (CVE-2025-32434) demonstrate that even `torch.load()` with `weights_only=True` can be exploited, emphasizing the need for checkpoint integrity verification ([source](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)).

3. **Jupyter Notebook Risks**: Analysis of 5000 random notebooks revealed widespread credential exposure in both code cells and output cells, highlighting the need for comprehensive notebook scanning ([source](https://hackernoon.com/i-scanned-5000-random-jupyter-notebooks-from-github-heres-the-graveyard-of-secrets-i-found)).

4. **False Positive Reduction**: Context-aware analysis and security comment detection significantly reduce false positives while maintaining high detection rates ([source](https://www.codequiry.com/blog/your-static-analysis-tool-is-lying-to-you-about-security)).

5. **SAST Best Practices**: Static analysis tools should focus on actionable vulnerabilities with clear remediation paths, avoiding overwhelming developers with low-confidence findings ([source](https://rafter.so/blog/sast-static-analysis-guide)).
