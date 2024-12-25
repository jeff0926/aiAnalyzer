import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFramework(str, Enum):
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    GTEST = "gtest"
    GO_TEST = "go"
    RSPEC = "rspec"

@dataclass
class TestCase:
    name: str
    file_path: str
    line_number: int
    description: Optional[str] = None
    assertions: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    mocks: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class TestMetrics:
    total_tests: int = 0
    assertions_count: int = 0
    fixtures_count: int = 0
    mocks_count: int = 0
    avg_complexity: float = 0.0
    coverage_percentage: float = 0.0

class TestPatterns:
    FILE_MAPPING = {
        'test_*.py': TestFramework.PYTEST,
        '*_test.py': TestFramework.PYTEST,
        '*Test.java': TestFramework.JUNIT,
        '*.spec.js': TestFramework.JEST,
        '*.test.js': TestFramework.JEST,
        '*_test.go': TestFramework.GO_TEST,
        '*_spec.rb': TestFramework.RSPEC
    }
    
    ASSERTION_PATTERNS = {
        TestFramework.PYTEST: [
            r'assert\s+',
            r'pytest\.raises',
            r'pytest\.warns',
            r'pytest\.approx'
        ],
        TestFramework.UNITTEST: [
            r'assert[A-Z][a-zA-Z]*\(',
            r'fail\(',
            r'failIf\(',
            r'failUnless\('
        ],
        TestFramework.JEST: [
            r'expect\(.+\)\.to',
            r'assert\.',
            r'should\.'
        ]
    }
    
    FIXTURE_PATTERNS = {
        TestFramework.PYTEST: [
            r'@pytest\.fixture',
            r'@fixture',
            r'request\.getfixturevalue'
        ],
        TestFramework.UNITTEST: [
            r'setUp\(',
            r'tearDown\(',
            r'setUpClass\(',
            r'tearDownClass\('
        ],
        TestFramework.JEST: [
            r'beforeEach\(',
            r'afterEach\(',
            r'beforeAll\(',
            r'afterAll\('
        ]
    }
    
    MOCK_PATTERNS = {
        TestFramework.PYTEST: [
            r'@mock\.',
            r'@patch\(',
            r'Mock\(',
            r'MagicMock\('
        ],
        TestFramework.UNITTEST: [
            r'mock\.',
            r'patch\(',
            r'Mock\(',
            r'MagicMock\('
        ],
        TestFramework.JEST: [
            r'jest\.mock\(',
            r'jest\.spyOn\(',
            r'createMock\(',
            r'mockImplementation\('
        ]
    }
    
    @classmethod
    def get_framework(cls, file_path: Path) -> Optional[TestFramework]:
        for pattern, framework in cls.FILE_MAPPING.items():
            if file_path.match(pattern):
                return framework
        return None
    
    @classmethod
    def find_assertions(cls, content: str, framework: TestFramework) -> List[str]:
        assertions = []
        patterns = cls.ASSERTION_PATTERNS.get(framework, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            assertions.extend(
                content[match.start():content.find('\n', match.start())].strip()
                for match in matches
            )
            
        return assertions
    
    @classmethod
    def find_fixtures(cls, content: str, framework: TestFramework) -> List[str]:
        fixtures = []
        patterns = cls.FIXTURE_PATTERNS.get(framework, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            fixtures.extend(
                content[match.start():content.find('\n', match.start())].strip()
                for match in matches
            )
            
        return fixtures
    
    @classmethod
    def find_mocks(cls, content: str, framework: TestFramework) -> List[str]:
        mocks = []
        patterns = cls.MOCK_PATTERNS.get(framework, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            mocks.extend(
                content[match.start():content.find('\n', match.start())].strip()
                for match in matches
            )
            
        return mocks
#####
class TestAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={AgentCapability.TEST},
            file_patterns={
                'test_*.py', '*_test.py', '*Test.java',
                '*.spec.js', '*.test.js', '*_test.go',
                '*_spec.rb'
            },
            max_file_size=2 * 1024 * 1024,
            timeout=300,
            scope=AnalysisScope.STANDARD
        )
    
    async def _analyze_content(
        self,
        content: str,
        file_path: Path,
        context: Optional[Dict] = None
    ) -> Dict:
        try:
            framework = TestPatterns.get_framework(file_path)
            if not framework:
                raise AnalysisError(f"Unsupported test framework: {file_path}")
            
            test_cases = self._extract_test_cases(content, file_path, framework)
            metrics = self._calculate_metrics(test_cases)
            quality_analysis = await self._analyze_quality(test_cases, framework)
            coverage_analysis = self._analyze_coverage(test_cases, content)
            best_practices = await self._check_best_practices(test_cases, framework)
            recommendations = await self._generate_recommendations(
                test_cases,
                metrics,
                quality_analysis,
                coverage_analysis,
                best_practices,
                framework
            )
            
            return {
                'framework': framework.value,
                'test_cases': [vars(tc) for tc in test_cases],
                'metrics': vars(metrics),
                'quality_analysis': quality_analysis,
                'coverage_analysis': coverage_analysis,
                'best_practices': best_practices,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Test analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Test analysis failed: {str(e)}") from e
    
    def _extract_test_cases(
        self,
        content: str,
        file_path: Path,
        framework: TestFramework
    ) -> List[TestCase]:
        if framework == TestFramework.PYTEST:
            return self._extract_pytest_cases(content, file_path)
        elif framework == TestFramework.UNITTEST:
            return self._extract_unittest_cases(content, file_path)
        elif framework in {TestFramework.JEST, TestFramework.MOCHA}:
            return self._extract_js_test_cases(content, file_path)
        return []
    
    def _extract_pytest_cases(
        self,
        content: str,
        file_path: Path
    ) -> List[TestCase]:
        test_cases = []
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_') or any(
                    decorator.id == 'pytest' 
                    for decorator in node.decorator_list 
                    if isinstance(decorator, ast.Name)
                ):
                    docstring = ast.get_docstring(node)
                    test_content = content[node.lineno:node.end_lineno or -1]
                    test_cases.append(TestCase(
                        name=node.name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=docstring,
                        assertions=TestPatterns.find_assertions(
                            test_content,
                            TestFramework.PYTEST
                        ),
                        fixtures=TestPatterns.find_fixtures(
                            test_content,
                            TestFramework.PYTEST
                        ),
                        mocks=TestPatterns.find_mocks(
                            test_content,
                            TestFramework.PYTEST
                        ),
                        tags=[d.id for d in node.decorator_list 
                              if isinstance(d, ast.Name)]
                    ))
        return test_cases
    
    def _extract_unittest_cases(
        self,
        content: str,
        file_path: Path
    ) -> List[TestCase]:
        test_cases = []
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(base.id == 'TestCase' 
                      for base in node.bases 
                      if isinstance(base, ast.Name)):
                    for method in node.body:
                        if (isinstance(method, ast.FunctionDef) and 
                            method.name.startswith('test_')):
                            docstring = ast.get_docstring(method)
                            test_content = content[method.lineno:method.end_lineno or -1]
                            test_cases.append(TestCase(
                                name=f"{node.name}.{method.name}",
                                file_path=str(file_path),
                                line_number=method.lineno,
                                description=docstring,
                                assertions=TestPatterns.find_assertions(
                                    test_content,
                                    TestFramework.UNITTEST
                                ),
                                fixtures=TestPatterns.find_fixtures(
                                    test_content,
                                    TestFramework.UNITTEST
                                ),
                                mocks=TestPatterns.find_mocks(
                                    test_content,
                                    TestFramework.UNITTEST
                                )
                            ))
        return test_cases
    
    def _extract_js_test_cases(
        self,
        content: str,
        file_path: Path
    ) -> List[TestCase]:
        test_cases = []
        describe_pattern = r'describe\([\'"](.+?)[\'"]\s*,\s*(?:async\s*)?function\s*\(\)\s*{'
        test_pattern = r'(?:it|test)\([\'"](.+?)[\'"]\s*,\s*(?:async\s*)?function\s*\(\)\s*{'
        
        for desc_match in re.finditer(describe_pattern, content):
            suite_name = desc_match.group(1)
            suite_start = desc_match.end()
            suite_content = content[suite_start:]
            suite_end = self._find_block_end(suite_content)
            suite_content = suite_content[:suite_end]
            
            for test_match in re.finditer(test_pattern, suite_content):
                test_name = test_match.group(1)
                test_start = test_match.end()
                test_content = suite_content[test_start:]
                test_end = self._find_block_end(test_content)
                test_content = test_content[:test_end]
                
                test_cases.append(TestCase(
                    name=f"{suite_name} - {test_name}",
                    file_path=str(file_path),
                    line_number=content[:suite_start + test_match.start()].count('\n') + 1,
                    assertions=TestPatterns.find_assertions(
                        test_content,
                        TestFramework.JEST
                    ),
                    fixtures=TestPatterns.find_fixtures(
                        test_content,
                        TestFramework.JEST
                    ),
                    mocks=TestPatterns.find_mocks(
                        test_content,
                        TestFramework.JEST
                    )
                ))
        return test_cases
    
    def _find_block_end(self, content: str) -> int:
        brace_count = 1
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
        return len(content)
    
    def _calculate_metrics(self, test_cases: List[TestCase]) -> TestMetrics:
        total_assertions = sum(len(tc.assertions) for tc in test_cases)
        total_fixtures = sum(len(tc.fixtures) for tc in test_cases)
        total_mocks = sum(len(tc.mocks) for tc in test_cases)
        
        return TestMetrics(
            total_tests=len(test_cases),
            assertions_count=total_assertions,
            fixtures_count=total_fixtures,
            mocks_count=total_mocks,
            avg_complexity=total_assertions / max(len(test_cases), 1)
        )
    
    async def _analyze_quality(
        self,
        test_cases: List[TestCase],
        framework: TestFramework
    ) -> Dict:
        quality_prompt = (
            f"Analyze these {framework.value} test cases for quality:\n"
            f"{json.dumps([vars(tc) for tc in test_cases], indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            quality_prompt
        )
        
        return response.get('quality_analysis', {
            'completeness': 0.0,
            'reliability': 0.0,
            'maintainability': 0.0,
            'issues': []
        })
    
    def _analyze_coverage(
        self,
        test_cases: List[TestCase],
        content: str
    ) -> Dict:
        assertions_per_line = {}
        for test_case in test_cases:
            for assertion in test_case.assertions:
                assertions_per_line[test_case.line_number] = \
                    assertions_per_line.get(test_case.line_number, 0) + 1
        
        total_lines = len(content.splitlines())
        covered_lines = len(assertions_per_line)
        
        return {
            'line_coverage': covered_lines / total_lines if total_lines > 0 else 0,
            'assertion_density': len(assertions_per_line) / max(len(test_cases), 1),
            'uncovered_lines': total_lines - covered_lines
        }
    
    async def _check_best_practices(
        self,
        test_cases: List[TestCase],
        framework: TestFramework
    ) -> List[Dict]:
        practices_prompt = (
            f"Check these {framework.value} test cases for best practices:\n"
            f"{json.dumps([vars(tc) for tc in test_cases], indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            practices_prompt
        )
        
        return response.get('best_practices', [])
    
    async def _generate_recommendations(
        self,
        test_cases: List[TestCase],
        metrics: TestMetrics,
        quality_analysis: Dict,
        coverage_analysis: Dict,
        best_practices: List[Dict],
        framework: TestFramework
    ) -> List[Dict]:
        context = {
            'framework': framework.value,
            'test_count': metrics.total_tests,
            'assertions_count': metrics.assertions_count,
            'avg_complexity': metrics.avg_complexity,
            'line_coverage': coverage_analysis['line_coverage'],
            'quality_score': quality_analysis.get('completeness', 0),
            'best_practices_issues': len(best_practices)
        }
        
        recommendations_prompt = (
            f"Based on the test analysis, suggest improvements:\n"
            f"{json.dumps(context, indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            recommendations_prompt
        )
        
        return response.get('recommendations', [])