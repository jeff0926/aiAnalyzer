"""
Code Analysis Agent Module

This module implements specialized analysis for source code files. It handles
various programming languages and provides insights about code quality,
patterns, and potential issues.

Features:
- Multi-language support
- Code quality analysis
- Pattern detection
- Dependency analysis
- Security scanning
- Complexity metrics
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from agents.base_agent import (
    AgentCapability,
    AgentConfig,
    AnalysisScope,
    BaseAgent,
    AnalysisError
)
from core.llm.llm_router import LLMRouter, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeMetrics:
    """Metrics for code analysis."""
    lines_total: int = 0
    lines_code: int = 0
    lines_comment: int = 0
    lines_blank: int = 0
    complexity: float = 0.0
    functions_count: int = 0
    classes_count: int = 0
    average_function_length: float = 0.0
    max_function_length: int = 0
    imports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class CodeIssue:
    """Code issue or suggestion."""
    type: str
    severity: str
    message: str
    line_number: int
    snippet: str
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None

class LanguageSupport:
    """Language-specific analysis support."""
    
    PATTERNS = {
        # File patterns for supported languages
        'python': {'*.py'},
        'javascript': {'*.js', '*.jsx', '*.ts', '*.tsx'},
        'java': {'*.java'},
        'cpp': {'*.cpp', '*.hpp', '*.cc', '*.h'},
        'go': {'*.go'},
        'rust': {'*.rs'},
    }
    
    COMMENT_MARKERS = {
        # Single-line comment markers
        'python': '#',
        'javascript': '//',
        'java': '//',
        'cpp': '//',
        'go': '//',
        'rust': '//',
    }
    
    COMPLEXITY_KEYWORDS = {
        # Keywords that indicate complexity
        'python': {'if', 'for', 'while', 'except', 'with', 'lambda'},
        'javascript': {'if', 'for', 'while', 'try', 'catch', 'switch'},
        'java': {'if', 'for', 'while', 'try', 'catch', 'switch'},
        'cpp': {'if', 'for', 'while', 'try', 'catch', 'switch'},
        'go': {'if', 'for', 'switch', 'select'},
        'rust': {'if', 'for', 'while', 'match', 'loop'},
    }
    
    @classmethod
    def get_language(cls, file_path: Path) -> Optional[str]:
        """Determine language from file path."""
        ext = file_path.suffix.lower()
        for lang, patterns in cls.PATTERNS.items():
            if any(Path(file_path).match(pattern) for pattern in patterns):
                return lang
        return None
    
    @classmethod
    def get_comment_marker(cls, language: str) -> str:
        """Get comment marker for language."""
        return cls.COMMENT_MARKERS.get(language, '#')
    
    @classmethod
    def get_complexity_keywords(cls, language: str) -> Set[str]:
        """Get complexity keywords for language."""
        return cls.COMPLEXITY_KEYWORDS.get(language, set())

class CodeAnalysisAgent(BaseAgent):
    """
    Agent specialized for source code analysis.
    """
    
    def _default_config(self) -> AgentConfig:
        """Provide default configuration."""
        all_patterns = set()
        for patterns in LanguageSupport.PATTERNS.values():
            all_patterns.update(patterns)
        
        return AgentConfig(
            capabilities={
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.SECURITY_SCAN
            },
            file_patterns=all_patterns,
            max_file_size=5 * 1024 * 1024,  # 5MB
            timeout=600,  # 10 minutes
            scope=AnalysisScope.STANDARD
        )
    
    async def _analyze_content(
        self,
        content: str,
        file_path: Path,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze source code content.
        
        Args:
            content: Source code content
            file_path: Path to source file
            context: Optional analysis context
            
        Returns:
            Analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Determine language
            language = LanguageSupport.get_language(file_path)
            if not language:
                raise AnalysisError(f"Unsupported file type: {file_path}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(content, language)
            
            # Collect code issues
            issues = await self._collect_issues(content, language, file_path)
            
            # Extract patterns and anti-patterns
            patterns = await self._extract_patterns(content, language, file_path)
            
            # Security scan
            security_issues = await self._security_scan(content, language, file_path)
            
            # Analyze dependencies
            dependencies = self._analyze_dependencies(content, language)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                content,
                metrics,
                issues,
                patterns,
                security_issues
            )
            
            return {
                'language': language,
                'metrics': {
                    'lines_total': metrics.lines_total,
                    'lines_code': metrics.lines_code,
                    'lines_comment': metrics.lines_comment,
                    'lines_blank': metrics.lines_blank,
                    'complexity': metrics.complexity,
                    'functions_count': metrics.functions_count,
                    'classes_count': metrics.classes_count,
                    'average_function_length': metrics.average_function_length,
                    'max_function_length': metrics.max_function_length
                },
                'issues': [
                    {
                        'type': issue.type,
                        'severity': issue.severity,
                        'message': issue.message,
                        'line_number': issue.line_number,
                        'snippet': issue.snippet,
                        'suggestion': issue.suggestion,
                        'rule_id': issue.rule_id
                    }
                    for issue in issues
                ],
                'patterns': patterns,
                'security_issues': security_issues,
                'dependencies': list(dependencies),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Code analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Code analysis failed: {str(e)}") from e
    
    def _calculate_metrics(self, content: str, language: str) -> CodeMetrics:
        """Calculate code metrics."""
        metrics = CodeMetrics()
        
        lines = content.splitlines()
        comment_marker = LanguageSupport.get_comment_marker(language)
        complexity_keywords = LanguageSupport.get_complexity_keywords(language)
        
        in_multiline_comment = False
        current_function_lines = 0
        max_function_lines = 0
        total_function_lines = 0
        functions_found = 0
        
        for line in lines:
            line = line.strip()
            
            # Count total lines
            metrics.lines_total += 1
            
            # Handle multi-line comments
            if language in {'python', 'javascript', 'java'}:
                if '"""' in line or "'''" in line:
                    in_multiline_comment = not in_multiline_comment
                    metrics.lines_comment += 1
                    continue
            
            # Skip empty lines
            if not line:
                metrics.lines_blank += 1
                continue
            
            # Count comments
            if in_multiline_comment:
                metrics.lines_comment += 1
            elif line.startswith(comment_marker):
                metrics.lines_comment += 1
            else:
                metrics.lines_code += 1
                
                # Calculate complexity
                for keyword in complexity_keywords:
                    if re.search(rf'\b{keyword}\b', line):
                        metrics.complexity += 1
                
                # Function analysis
                if 'def ' in line or 'function ' in line or '{' in line:
                    if current_function_lines > 0:
                        # End of previous function
                        max_function_lines = max(
                            max_function_lines,
                            current_function_lines
                        )
                        total_function_lines += current_function_lines
                        functions_found += 1
                    current_function_lines = 0
                else:
                    current_function_lines += 1
                
                # Class counting
                if 'class ' in line:
                    metrics.classes_count += 1
        
        # Handle last function
        if current_function_lines > 0:
            max_function_lines = max(max_function_lines, current_function_lines)
            total_function_lines += current_function_lines
            functions_found += 1
        
        metrics.functions_count = functions_found
        metrics.max_function_length = max_function_lines
        metrics.average_function_length = (
            total_function_lines / max(functions_found, 1)
        )
        
        return metrics
    
    async def _collect_issues(
        self,
        content: str,
        language: str,
        file_path: Path
    ) -> List[CodeIssue]:
        """Collect code issues and suggestions."""
        issues = []
        
        # Use LLM for advanced pattern recognition
        issue_prompts = [
            f"Analyze this {language} code for potential issues:\n{content}",
            "Identify code smells and anti-patterns.",
            "Suggest improvements for readability and maintainability."
        ]
        
        for prompt in issue_prompts:
            response = await self.llm_router.route_request(
                TaskType.CODE_ANALYSIS,
                prompt
            )
            
            # Process LLM response and extract issues
            # This is a placeholder - actual implementation would parse
            # the LLM response format
            if 'issues' in response:
                for issue_data in response['issues']:
                    issues.append(CodeIssue(**issue_data))
        
        return issues
    
    async def _extract_patterns(
        self,
        content: str,
        language: str,
        file_path: Path
    ) -> Dict[str, List[Dict]]:
        """Extract code patterns and anti-patterns."""
        patterns_prompt = (
            f"Identify design patterns and anti-patterns in this {language} "
            f"code:\n{content}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            patterns_prompt
        )
        
        return response.get('patterns', {
            'design_patterns': [],
            'anti_patterns': []
        })
    
    async def _security_scan(
        self,
        content: str,
        language: str,
        file_path: Path
    ) -> List[Dict]:
        """Perform security analysis."""
        security_prompt = (
            f"Analyze this {language} code for security vulnerabilities:\n{content}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.SECURITY_AUDIT,
            security_prompt
        )
        
        return response.get('security_issues', [])
    
    def _analyze_dependencies(
        self,
        content: str,
        language: str
    ) -> Set[str]:
        """Analyze code dependencies."""
        dependencies = set()
        
        # Language-specific dependency extraction
        if language == 'python':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            dependencies.add(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        dependencies.add(node.module)
            except:
                pass
        
        elif language == 'javascript':
            # Simple regex for import statements
            import_patterns = [
                r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]',
                r'require\([\'"](.+?)[\'"]\)'
            ]
            for pattern in import_patterns:
                dependencies.update(re.findall(pattern, content))
        
        # Add more language-specific extraction as needed
        
        return dependencies
    
    async def _generate_recommendations(
        self,
        content: str,
        metrics: CodeMetrics,
        issues: List[CodeIssue],
        patterns: Dict[str, List[Dict]],
        security_issues: List[Dict]
    ) -> List[Dict]:
        """Generate improvement recommendations."""
        context = {
            'metrics': metrics,
            'issues_count': len(issues),
            'patterns_found': patterns,
            'security_issues_count': len(security_issues)
        }
        
        recommendations_prompt = (
            f"Based on the analysis results, suggest improvements:\n"
            f"{json.dumps(context, indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            recommendations_prompt
        )
        
        return response.get('recommendations', [])