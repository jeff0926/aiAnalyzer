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

class VulnerabilityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class VulnerabilityType(str, Enum):
    INJECTION = "injection"
    AUTH = "authentication"
    EXPOSURE = "sensitive_data_exposure"
    XXE = "xxe"
    ACCESS_CONTROL = "broken_access_control"
    MISCONFIG = "security_misconfiguration"
    XSS = "xss"
    DESERIALIZE = "insecure_deserialization"
    COMPONENTS = "vulnerable_components"
    LOGGING = "insufficient_logging"
    CRYPTO = "cryptographic_failures"

@dataclass
class SecurityVulnerability:
    level: VulnerabilityLevel
    type: VulnerabilityType
    description: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    references: List[str] = field(default_factory=list)
    false_positive_likelihood: float = 0.0

class SecurityPatterns:
    PATTERNS = {
        VulnerabilityType.INJECTION: {
            'sql': [
                r'execute\s*\(\s*[\'"].*?\%.*?[\'"]\s*%',
                r'execute\s*\(\s*[\'"].*?\{.*?\}.*?[\'"]\s*\.format',
                r'cursor\.execute\s*\([^,)]*\+',
            ],
            'command': [
                r'os\.system\s*\(',
                r'subprocess\.(?:call|run|Popen)\s*\(',
                r'eval\s*\(',
                r'exec\s*\('
            ],
            'nosql': [
                r'find\s*\(\s*\{.*?\$where\:',
                r'(?:update|remove)\s*\(\s*\{[^}]*\}\s*\)'
            ]
        },
        VulnerabilityType.AUTH: {
            'weak_crypto': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'random\s*\('
            ],
            'hardcoded': [
                r'password\s*=\s*[\'"][^\'"]+[\'"]',
                r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
                r'secret\s*=\s*[\'"][^\'"]+[\'"]'
            ]
        },
        VulnerabilityType.EXPOSURE: {
            'logging': [
                r'logging\.(?:debug|info|warning|error|critical)\s*\([^)]*password',
                r'print\s*\([^)]*token',
                r'console\.log\s*\([^)]*secret'
            ],
            'comments': [
                r'#.*password.*=',
                r'//.*api[-_]?key.*=',
                r'/\*.*secret.*\*/'
            ]
        },
        VulnerabilityType.XSS: {
            'reflected': [
                r'render\s*\([^)]*request\.',
                r'innerHTML\s*=',
                r'document\.write\s*\('
            ],
            'stored': [
                r'render_template\s*\([^)]*\)',
                r'dangerouslySetInnerHTML'
            ]
        },
        VulnerabilityType.CRYPTO: {
            'weak_cipher': [
                r'DES\.',
                r'RC4\.',
                r'MD5\.',
                r'SHA1\.'
            ],
            'insecure_random': [
                r'Math\.random\s*\(',
                r'random\.',
                r'rand\s*\('
            ],
            'weak_key': [
                r'keysize\s*=\s*1024',
                r'RSA\([^)]*1024\)'
            ]
        }
    }
    
    SEVERITY_RULES = {
        'sql_injection': VulnerabilityLevel.CRITICAL,
        'command_injection': VulnerabilityLevel.CRITICAL,
        'weak_crypto': VulnerabilityLevel.HIGH,
        'hardcoded_secrets': VulnerabilityLevel.HIGH,
        'xss': VulnerabilityLevel.HIGH,
        'information_exposure': VulnerabilityLevel.MEDIUM,
        'insecure_random': VulnerabilityLevel.MEDIUM
    }
    
    CWE_MAPPINGS = {
        VulnerabilityType.INJECTION: 'CWE-78',
        VulnerabilityType.AUTH: 'CWE-287',
        VulnerabilityType.EXPOSURE: 'CWE-200',
        VulnerabilityType.XXE: 'CWE-611',
        VulnerabilityType.ACCESS_CONTROL: 'CWE-284',
        VulnerabilityType.MISCONFIG: 'CWE-16',
        VulnerabilityType.XSS: 'CWE-79',
        VulnerabilityType.DESERIALIZE: 'CWE-502',
        VulnerabilityType.COMPONENTS: 'CWE-1104',
        VulnerabilityType.LOGGING: 'CWE-778',
        VulnerabilityType.CRYPTO: 'CWE-310'
    }
    
    @classmethod
    def find_vulnerabilities(
        cls,
        content: str,
        file_path: Path
    ) -> List[SecurityVulnerability]:
        vulnerabilities = []
        lines = content.splitlines()
        
        for vuln_type, categories in cls.PATTERNS.items():
            for category, patterns in categories.items():
                for pattern in patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            severity = cls.SEVERITY_RULES.get(
                                f"{category}_{vuln_type}",
                                VulnerabilityLevel.MEDIUM
                            )
                            
                            vulnerabilities.append(SecurityVulnerability(
                                level=severity,
                                type=vuln_type,
                                description=cls._generate_description(vuln_type, category),
                                line_number=i,
                                code_snippet=line.strip(),
                                recommendation=cls._generate_recommendation(
                                    vuln_type,
                                    category
                                ),
                                cwe_id=cls.CWE_MAPPINGS.get(vuln_type),
                                cvss_score=cls._calculate_cvss_score(severity),
                                false_positive_likelihood=cls._estimate_false_positive(
                                    line,
                                    category
                                )
                            ))
        
        return vulnerabilities
    
    @classmethod
    def _generate_description(
        cls,
        vuln_type: VulnerabilityType,
        category: str
    ) -> str:
        descriptions = {
            VulnerabilityType.INJECTION: {
                'sql': 'Potential SQL injection vulnerability due to string concatenation or formatting',
                'command': 'Possible command injection through unvalidated system command execution',
                'nosql': 'NoSQL injection risk in database query'
            },
            VulnerabilityType.AUTH: {
                'weak_crypto': 'Use of weak cryptographic functions',
                'hardcoded': 'Hardcoded credentials or secrets detected'
            },
            VulnerabilityType.EXPOSURE: {
                'logging': 'Sensitive data exposure in logs',
                'comments': 'Sensitive information in comments'
            },
            VulnerabilityType.XSS: {
                'reflected': 'Potential reflected XSS vulnerability',
                'stored': 'Possible stored XSS vulnerability'
            },
            VulnerabilityType.CRYPTO: {
                'weak_cipher': 'Use of weak or deprecated cipher',
                'insecure_random': 'Insecure random number generation',
                'weak_key': 'Insufficient key size for cryptographic operation'
            }
        }
        
        return descriptions.get(vuln_type, {}).get(
            category,
            f'Security vulnerability of type {vuln_type.value}'
        )
    
    @classmethod
    def _generate_recommendation(
        cls,
        vuln_type: VulnerabilityType,
        category: str
    ) -> str:
        recommendations = {
            VulnerabilityType.INJECTION: {
                'sql': 'Use parameterized queries or ORM',
                'command': 'Use subprocess.run with shell=False and input validation',
                'nosql': 'Use sanitized queries and input validation'
            },
            VulnerabilityType.AUTH: {
                'weak_crypto': 'Use strong hashing algorithms (e.g., bcrypt, Argon2)',
                'hardcoded': 'Use environment variables or secure secret management'
            },
            VulnerabilityType.EXPOSURE: {
                'logging': 'Implement proper log sanitization',
                'comments': 'Remove sensitive information from comments'
            },
            VulnerabilityType.XSS: {
                'reflected': 'Use proper output encoding and CSP headers',
                'stored': 'Implement input sanitization and validation'
            },
            VulnerabilityType.CRYPTO: {
                'weak_cipher': 'Use strong modern encryption algorithms',
                'insecure_random': 'Use cryptographically secure random number generation',
                'weak_key': 'Use appropriate key sizes (RSA ≥ 2048 bits)'
            }
        }
        
        return recommendations.get(vuln_type, {}).get(
            category,
            'Review and update security controls'
        )
    
    @classmethod
    def _calculate_cvss_score(cls, severity: VulnerabilityLevel) -> float:
        base_scores = {
            VulnerabilityLevel.CRITICAL: 9.0,
            VulnerabilityLevel.HIGH: 7.0,
            VulnerabilityLevel.MEDIUM: 5.0,
            VulnerabilityLevel.LOW: 3.0,
            VulnerabilityLevel.INFO: 1.0
        }
        return base_scores.get(severity, 5.0)
    
    @classmethod
    def _estimate_false_positive(cls, line: str, category: str) -> float:
        indicators = {
            'test': 0.8,
            'example': 0.7,
            'mock': 0.6,
            'sample': 0.6,
            'comment': 0.5
        }
        
        line_lower = line.lower()
        for indicator, probability in indicators.items():
            if indicator in line_lower:
                return probability
        
        return 0.2  # Base false positive likelihood
    
    class SecurityAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={
                AgentCapability.SECURITY_SCAN,
                AgentCapability.CODE_ANALYSIS
            },
            file_patterns={
                '*.py', '*.js', '*.java', '*.go',
                '*.rb', '*.php', '*.cs', '*.cpp',
                '*.h', '*.hpp', '*.c'
            },
            max_file_size=5 * 1024 * 1024,
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
            # Pattern-based vulnerability detection
            vulnerabilities = SecurityPatterns.find_vulnerabilities(content, file_path)
            
            # Calculate statistics
            severity_stats = {
                level: len([v for v in vulnerabilities if v.level == level])
                for level in VulnerabilityLevel
            }
            
            type_stats = {
                vtype: len([v for v in vulnerabilities if v.type == vtype])
                for vtype in VulnerabilityType
            }
            
            # Perform additional analyses
            llm_analysis = await self._perform_llm_analysis(content, file_path)
            dependency_scan = await self._scan_dependencies(file_path)
            compliance_check = await self._check_compliance(vulnerabilities)
            risk_score = self._calculate_risk_score(vulnerabilities, dependency_scan)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                vulnerabilities,
                llm_analysis,
                dependency_scan,
                compliance_check
            )
            
            return {
                'vulnerabilities': [vars(v) for v in vulnerabilities],
                'severity_stats': severity_stats,
                'type_stats': type_stats,
                'llm_analysis': llm_analysis,
                'dependency_scan': dependency_scan,
                'compliance': compliance_check,
                'recommendations': recommendations,
                'risk_score': risk_score
            }
            
        except Exception as e:
            logger.error(f"Security analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Security analysis failed: {str(e)}") from e
    
    async def _perform_llm_analysis(
        self,
        content: str,
        file_path: Path
    ) -> Dict:
        prompt = (
            f"Analyze this code for security vulnerabilities:\n"
            f"File: {file_path.name}\n"
            f"Content:\n{content}\n\n"
            f"Consider:\n"
            f"1. Potential security flaws not detectable by pattern matching\n"
            f"2. Business logic vulnerabilities\n"
            f"3. Security best practices violations\n"
            f"4. Data flow security concerns"
        )
        
        response = await self.llm_router.route_request(
            TaskType.SECURITY_AUDIT,
            prompt
        )
        
        return response.get('security_analysis', {
            'additional_vulnerabilities': [],
            'security_insights': [],
            'risk_factors': []
        })
    
    async def _scan_dependencies(self, file_path: Path) -> Dict:
        dependencies = {
            '.py': 'requirements.txt',
            '.js': 'package.json',
            '.java': 'pom.xml',
            '.go': 'go.mod',
            '.rb': 'Gemfile'
        }
        
        dep_file = dependencies.get(file_path.suffix)
        if dep_file:
            dep_path = file_path.parent / dep_file
            if dep_path.exists():
                content = dep_path.read_text()
                
                # Use LLM to analyze dependencies
                response = await self.llm_router.route_request(
                    TaskType.SECURITY_AUDIT,
                    f"Analyze these dependencies for security issues:\n{content}"
                )
                
                return response.get('dependency_analysis', {
                    'vulnerable_dependencies': [],
                    'outdated_dependencies': [],
                    'risk_score': 0.0
                })
        
        return {
            'error': 'No dependency file found',
            'vulnerable_dependencies': [],
            'risk_score': 0.0
        }
    
    async def _check_compliance(
        self,
        vulnerabilities: List[SecurityVulnerability]
    ) -> Dict:
        # Group vulnerabilities by type
        vuln_types = {v.type for v in vulnerabilities}
        
        # Check against compliance frameworks
        frameworks = {
            'OWASP_TOP_10': self._check_owasp_compliance(vuln_types),
            'PCI_DSS': self._check_pci_compliance(vuln_types),
            'GDPR': self._check_gdpr_compliance(vuln_types)
        }
        
        # Calculate overall compliance score
        total_score = sum(f['score'] for f in frameworks.values())
        max_score = len(frameworks) * 100
        
        return {
            'frameworks': frameworks,
            'overall_score': (total_score / max_score) * 100 if max_score > 0 else 0,
            'compliant': total_score / max_score >= 0.7 if max_score > 0 else False
        }
    
    def _check_owasp_compliance(self, vuln_types: Set[VulnerabilityType]) -> Dict:
        violations = []
        
        # OWASP Top 10 mapping
        owasp_mapping = {
            VulnerabilityType.INJECTION: 'A1:2021-Injection',
            VulnerabilityType.AUTH: 'A2:2021-Broken Authentication',
            VulnerabilityType.EXPOSURE: 'A3:2021-Sensitive Data Exposure',
            VulnerabilityType.XXE: 'A5:2021-Security Misconfiguration',
            VulnerabilityType.ACCESS_CONTROL: 'A1:2021-Broken Access Control',
            VulnerabilityType.XSS: 'A7:2021-Cross-Site Scripting',
            VulnerabilityType.DESERIALIZE: 'A8:2021-Insecure Deserialization',
            VulnerabilityType.COMPONENTS: 'A6:2021-Vulnerable Components',
            VulnerabilityType.LOGGING: 'A9:2021-Insufficient Logging',
            VulnerabilityType.CRYPTO: 'A2:2021-Cryptographic Failures'
        }
        
        for vuln_type in vuln_types:
            if vuln_type in owasp_mapping:
                violations.append(owasp_mapping[vuln_type])
        
        score = 100 - (len(violations) * 10)  # Deduct 10 points per violation
        
        return {
            'score': max(0, score),
            'violations': violations,
            'compliant': score >= 70
        }
    
    def _check_pci_compliance(self, vuln_types: Set[VulnerabilityType]) -> Dict:
        violations = []
        
        # PCI DSS requirements mapping
        pci_mapping = {
            VulnerabilityType.CRYPTO: 'Requirement 3: Protect stored data',
            VulnerabilityType.ACCESS_CONTROL: 'Requirement 7: Restrict access',
            VulnerabilityType.AUTH: 'Requirement 8: Authenticate access',
            VulnerabilityType.LOGGING: 'Requirement 10: Monitor networks'
        }
        
        for vuln_type in vuln_types:
            if vuln_type in pci_mapping:
                violations.append(pci_mapping[vuln_type])
        
        score = 100 - (len(violations) * 15)  # Deduct 15 points per violation
        
        return {
            'score': max(0, score),
            'violations': violations,
            'compliant': score >= 70
        }
    
    def _check_gdpr_compliance(self, vuln_types: Set[VulnerabilityType]) -> Dict:
        violations = []
        
        # GDPR requirements mapping
        gdpr_mapping = {
            VulnerabilityType.EXPOSURE: 'Article 32: Security of processing',
            VulnerabilityType.CRYPTO: 'Article 32: Encryption requirement',
            VulnerabilityType.LOGGING: 'Article 33: Breach notification'
        }
        
        for vuln_type in vuln_types:
            if vuln_type in gdpr_mapping:
                violations.append(gdpr_mapping[vuln_type])
        
        score = 100 - (len(violations) * 20)  # Deduct 20 points per violation
        
        return {
            'score': max(0, score),
            'violations': violations,
            'compliant': score >= 70
        }
    
    def _calculate_risk_score(
        self,
        vulnerabilities: List[SecurityVulnerability],
        dependency_scan: Dict
    ) -> float:
        # Base weights for different severity levels
        severity_weights = {
            VulnerabilityLevel.CRITICAL: 10.0,
            VulnerabilityLevel.HIGH: 7.5,
            VulnerabilityLevel.MEDIUM: 5.0,
            VulnerabilityLevel.LOW: 2.5,
            VulnerabilityLevel.INFO: 1.0
        }
        
        # Calculate weighted score from vulnerabilities
        vuln_score = sum(
            severity_weights[v.level] * (1 - v.false_positive_likelihood)
            for v in vulnerabilities
        )
        
        # Factor in dependency risks
        dep_score = dependency_scan.get('risk_score', 0.0)
        
        # Combine scores (70% vulnerabilities, 30% dependencies)
        total_score = (vuln_score * 0.7) + (dep_score * 0.3)
        
        # Normalize to 0-100 range
        return min(100, max(0, total_score))
    
    async def _generate_recommendations(
        self,
        vulnerabilities: List[SecurityVulnerability],
        llm_analysis: Dict,
        dependency_scan: Dict,
        compliance_check: Dict
    ) -> List[Dict]:
        context = {
            'vulnerability_count': len(vulnerabilities),
            'severity_distribution': {
                level.value: len([v for v in vulnerabilities if v.level == level])
                for level in VulnerabilityLevel
            },
            'compliance_issues': [
                issue for framework in compliance_check['frameworks'].values()
                for issue in framework.get('violations', [])
            ],
            'dependency_issues': dependency_scan.get('vulnerable_dependencies', [])
        }
        
        prompt = (
            f"Generate security improvement recommendations based on:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            f"Focus on:\n"
            f"1. Critical and high-severity issues\n"
            f"2. Compliance violations\n"
            f"3. Dependency vulnerabilities\n"
            f"4. Security best practices"
        )
        
        response = await self.llm_router.route_request(
            TaskType.SECURITY_AUDIT,
            prompt
        )
        
        return response.get('recommendations', [])