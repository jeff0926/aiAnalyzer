import json
import logging
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

class ARBSection(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    ARCHITECTURE_OVERVIEW = "architecture_overview"
    SYSTEM_CONTEXT = "system_context"
    TECH_STACK = "tech_stack"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    RISKS = "risks"
    RECOMMENDATIONS = "recommendations"
    APPENDICES = "appendices"

class ARBComponentType(str, Enum):
    SERVICE = "service"
    DATABASE = "database"
    QUEUE = "queue"
    CACHE = "cache"
    UI = "ui"
    API = "api"
    BATCH = "batch"
    STORAGE = "storage"
    INTEGRATION = "integration"

@dataclass
class ARBComponent:
    name: str
    type: ARBComponentType
    description: str
    tech_stack: List[str]
    dependencies: List[str] = field(default_factory=list)
    apis: List[Dict] = field(default_factory=list)
    data_flows: List[Dict] = field(default_factory=list)
    security_concerns: List[str] = field(default_factory=list)
    scalability_notes: List[str] = field(default_factory=list)
    monitoring: List[str] = field(default_factory=list)

@dataclass
class ARBRisk:
    category: str
    description: str
    impact: str
    likelihood: str
    mitigation: str
    status: str = "Open"
    owner: Optional[str] = None
    due_date: Optional[str] = None

class ARBPatterns:
    ARCHITECTURE_PATTERNS = {
        'microservices': {
            'indicators': [
                'service discovery',
                'api gateway',
                'circuit breaker',
                'service mesh'
            ],
            'tech_stack': [
                'kubernetes',
                'docker',
                'consul',
                'istio'
            ]
        },
        'event_driven': {
            'indicators': [
                'event bus',
                'message queue',
                'pub/sub',
                'event sourcing'
            ],
            'tech_stack': [
                'kafka',
                'rabbitmq',
                'sns',
                'eventbridge'
            ]
        },
        'serverless': {
            'indicators': [
                'function as a service',
                'api gateway',
                'cloud functions',
                'lambda'
            ],
            'tech_stack': [
                'aws lambda',
                'azure functions',
                'cloud run',
                'fargate'
            ]
        },
        'monolithic': {
            'indicators': [
                'single deployment',
                'shared database',
                'modular monolith',
                'layered architecture'
            ],
            'tech_stack': [
                'spring',
                'django',
                'rails',
                'laravel'
            ]
        }
    }
    
    TECH_CATEGORIES = {
        'frontend': {
            'frameworks': ['react', 'angular', 'vue', 'svelte'],
            'build_tools': ['webpack', 'vite', 'parcel', 'rollup'],
            'state_management': ['redux', 'mobx', 'vuex', 'recoil']
        },
        'backend': {
            'frameworks': ['spring', 'django', 'express', 'flask'],
            'orm': ['hibernate', 'sqlalchemy', 'sequelize', 'prisma'],
            'api': ['rest', 'graphql', 'grpc', 'soap']
        },
        'database': {
            'relational': ['postgresql', 'mysql', 'oracle', 'sqlserver'],
            'nosql': ['mongodb', 'cassandra', 'dynamodb', 'couchbase'],
            'cache': ['redis', 'memcached', 'elasticache']
        },
        'infrastructure': {
            'container': ['docker', 'kubernetes', 'ecs', 'openshift'],
            'cloud': ['aws', 'azure', 'gcp', 'alicloud'],
            'monitoring': ['prometheus', 'grafana', 'datadog', 'newrelic']
        }
    }
    
    RISK_CATEGORIES = {
        'technical': [
            'scalability',
            'reliability',
            'performance',
            'maintainability'
        ],
        'security': [
            'data_protection',
            'access_control',
            'compliance',
            'vulnerabilities'
        ],
        'operational': [
            'availability',
            'monitoring',
            'disaster_recovery',
            'incident_response'
        ],
        'organizational': [
            'resources',
            'skills',
            'process',
            'governance'
        ]
    }
    
    @classmethod
    def detect_architecture_pattern(cls, tech_stack: List[str]) -> List[str]:
        detected_patterns = []
        tech_stack_lower = [tech.lower() for tech in tech_stack]
        
        for pattern, attributes in cls.ARCHITECTURE_PATTERNS.items():
            # Check tech stack indicators
            tech_matches = sum(
                1 for tech in attributes['tech_stack']
                if any(tech.lower() in stack_item for stack_item in tech_stack_lower)
            )
            
            # Calculate match score
            if tech_matches >= 2:  # At least 2 matching technologies
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    @classmethod
    def categorize_tech_stack(cls, tech_stack: List[str]) -> Dict[str, List[str]]:
        categorized = {category: [] for category in cls.TECH_CATEGORIES}
        tech_stack_lower = [tech.lower() for tech in tech_stack]
        
        for category, subcategories in cls.TECH_CATEGORIES.items():
            for tech_type, technologies in subcategories.items():
                matches = [
                    tech for tech in tech_stack_lower
                    if any(t.lower() in tech for t in technologies)
                ]
                if matches:
                    categorized[category].extend(matches)
        
        return {k: v for k, v in categorized.items() if v}
    
    class ARBAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={
                AgentCapability.INFRASTRUCTURE,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.SECURITY_SCAN
            },
            file_patterns={'*'},  # Accept all files for analysis
            max_file_size=50 * 1024 * 1024,  # 50MB
            timeout=900,  # 15 minutes
            scope=AnalysisScope.DEEP
        )
    
    async def _analyze_content(
        self,
        content: str,
        file_path: Path,
        context: Optional[Dict] = None
    ) -> Dict:
        try:
            # Process analysis results from context
            analysis_results = context.get('analysis_results', {})
            repo_info = context.get('repo_info', {})
            
            # Generate ARB document sections
            executive_summary = await self._generate_executive_summary(
                analysis_results,
                repo_info
            )
            
            architecture_overview = await self._generate_architecture_overview(
                analysis_results
            )
            
            system_context = await self._generate_system_context(
                analysis_results
            )
            
            tech_stack = self._analyze_tech_stack(analysis_results)
            
            security_analysis = await self._generate_security_analysis(
                analysis_results
            )
            
            performance_analysis = await self._generate_performance_analysis(
                analysis_results
            )
            
            scalability_analysis = await self._generate_scalability_analysis(
                analysis_results,
                tech_stack
            )
            
            reliability_analysis = await self._generate_reliability_analysis(
                analysis_results,
                tech_stack
            )
            
            maintainability_analysis = await self._analyze_maintainability(
                analysis_results
            )
            
            risks = await self._identify_risks(
                analysis_results,
                tech_stack
            )
            
            recommendations = await self._generate_recommendations(
                analysis_results,
                risks,
                tech_stack
            )
            
            return {
                'arb_package': {
                    ARBSection.EXECUTIVE_SUMMARY: executive_summary,
                    ARBSection.ARCHITECTURE_OVERVIEW: architecture_overview,
                    ARBSection.SYSTEM_CONTEXT: system_context,
                    ARBSection.TECH_STACK: tech_stack,
                    ARBSection.SECURITY_ANALYSIS: security_analysis,
                    ARBSection.PERFORMANCE_ANALYSIS: performance_analysis,
                    ARBSection.SCALABILITY: scalability_analysis,
                    ARBSection.RELIABILITY: reliability_analysis,
                    ARBSection.MAINTAINABILITY: maintainability_analysis,
                    ARBSection.RISKS: risks,
                    ARBSection.RECOMMENDATIONS: recommendations
                },
                'metadata': {
                    'generated_at': context.get('timestamp'),
                    'repository': repo_info.get('name'),
                    'branch': repo_info.get('branch'),
                    'commit': repo_info.get('commit')
                }
            }
            
        except Exception as e:
            logger.error(f"ARB package generation failed: {str(e)}")
            raise AnalysisError(f"ARB package generation failed: {str(e)}") from e
    
    async def _generate_executive_summary(
        self,
        analysis_results: Dict,
        repo_info: Dict
    ) -> Dict:
        prompt = (
            f"Generate an executive summary for architecture review:\n"
            f"Repository: {repo_info.get('name')}\n"
            f"Key Metrics:\n"
            f"- Files: {analysis_results.get('files_count')}\n"
            f"- Components: {len(analysis_results.get('components', []))}\n"
            f"- Critical Issues: {analysis_results.get('critical_issues_count')}\n"
            f"Consider:\n"
            f"1. Overall architecture approach\n"
            f"2. Key technical decisions\n"
            f"3. Major risks and challenges\n"
            f"4. Strategic recommendations"
        )
        
        response = await self.llm_router.route_request(
            TaskType.ARCHITECTURE_REVIEW,
            prompt
        )
        
        return response.get('executive_summary', {
            'overview': '',
            'approach': '',
            'key_findings': [],
            'recommendations': []
        })
    
    async def _generate_architecture_overview(self, analysis_results: Dict) -> Dict:
        components = []
        for comp_data in analysis_results.get('components', []):
            components.append(ARBComponent(
                name=comp_data['name'],
                type=ARBComponentType(comp_data['type']),
                description=comp_data['description'],
                tech_stack=comp_data['technologies'],
                dependencies=comp_data.get('dependencies', []),
                apis=comp_data.get('apis', []),
                data_flows=comp_data.get('data_flows', []),
                security_concerns=comp_data.get('security_issues', []),
                scalability_notes=comp_data.get('scalability_notes', []),
                monitoring=comp_data.get('monitoring', [])
            ))
        
        # Detect architecture patterns
        all_tech = [tech for comp in components for tech in comp.tech_stack]
        patterns = ARBPatterns.detect_architecture_pattern(all_tech)
        
        return {
            'patterns': patterns,
            'components': [vars(comp) for comp in components],
            'interfaces': analysis_results.get('interfaces', []),
            'data_flows': analysis_results.get('data_flows', []),
            'deployment': analysis_results.get('deployment', {})
        }
    
    async def _generate_system_context(self, analysis_results: Dict) -> Dict:
        return {
            'external_systems': analysis_results.get('external_dependencies', []),
            'integration_points': analysis_results.get('integration_points', []),
            'data_flows': analysis_results.get('external_data_flows', []),
            'constraints': analysis_results.get('system_constraints', []),
            'assumptions': analysis_results.get('assumptions', [])
        }
    
    def _analyze_tech_stack(self, analysis_results: Dict) -> Dict:
        all_tech = []
        for comp in analysis_results.get('components', []):
            all_tech.extend(comp.get('technologies', []))
        
        categorized = ARBPatterns.categorize_tech_stack(all_tech)
        return {
            'categories': categorized,
            'major_frameworks': analysis_results.get('frameworks', []),
            'databases': analysis_results.get('databases', []),
            'infrastructure': analysis_results.get('infrastructure', []),
            'third_party_services': analysis_results.get('external_services', [])
        }
    
    async def _generate_security_analysis(self, analysis_results: Dict) -> Dict:
        security_data = analysis_results.get('security_analysis', {})
        
        return {
            'vulnerabilities': security_data.get('vulnerabilities', []),
            'compliance': security_data.get('compliance_status', {}),
            'data_protection': security_data.get('data_protection', {}),
            'access_control': security_data.get('access_control', {}),
            'security_testing': security_data.get('security_testing', {}),
            'recommendations': security_data.get('recommendations', [])
        }
    
    async def _generate_performance_analysis(self, analysis_results: Dict) -> Dict:
        performance_data = analysis_results.get('performance_analysis', {})
        
        return {
            'bottlenecks': performance_data.get('bottlenecks', []),
            'resource_usage': performance_data.get('resource_usage', {}),
            'response_times': performance_data.get('response_times', {}),
            'optimization_opportunities': performance_data.get('optimizations', []),
            'benchmarks': performance_data.get('benchmarks', {}),
            'recommendations': performance_data.get('recommendations', [])
        }
    
    async def _generate_scalability_analysis(
        self,
        analysis_results: Dict,
        tech_stack: Dict
    ) -> Dict:
        # Use LLM for scalability insights
        prompt = (
            f"Analyze system scalability based on:\n"
            f"Tech Stack: {json.dumps(tech_stack, indent=2)}\n"
            f"Architecture: {json.dumps(analysis_results.get('architecture_overview', {}), indent=2)}\n"
            f"Consider:\n"
            f"1. Horizontal and vertical scaling\n"
            f"2. Data scalability\n"
            f"3. Load balancing\n"
            f"4. Caching strategies"
        )
        
        response = await self.llm_router.route_request(
            TaskType.ARCHITECTURE_REVIEW,
            prompt
        )
        
        return response.get('scalability_analysis', {
            'scaling_approach': '',
            'bottlenecks': [],
            'recommendations': []
        })
    
    async def _generate_reliability_analysis(
        self,
        analysis_results: Dict,
        tech_stack: Dict
    ) -> Dict:
        # Use LLM for reliability insights
        prompt = (
            f"Analyze system reliability based on:\n"
            f"Tech Stack: {json.dumps(tech_stack, indent=2)}\n"
            f"Architecture: {json.dumps(analysis_results.get('architecture_overview', {}), indent=2)}\n"
            f"Consider:\n"
            f"1. Fault tolerance\n"
            f"2. High availability\n"
            f"3. Disaster recovery\n"
            f"4. Monitoring and observability"
        )
        
        response = await self.llm_router.route_request(
            TaskType.ARCHITECTURE_REVIEW,
            prompt
        )
        
        return response.get('reliability_analysis', {
            'fault_tolerance': '',
            'availability': '',
            'disaster_recovery': '',
            'monitoring': '',
            'recommendations': []
        })
    
    async def _analyze_maintainability(self, analysis_results: Dict) -> Dict:
        code_metrics = analysis_results.get('code_metrics', {})
        test_coverage = analysis_results.get('test_coverage', {})
        documentation = analysis_results.get('documentation_analysis', {})
        
        return {
            'code_quality': {
                'metrics': code_metrics,
                'issues': analysis_results.get('code_issues', []),
                'recommendations': analysis_results.get('code_recommendations', [])
            },
            'testing': {
                'coverage': test_coverage,
                'testing_practices': analysis_results.get('testing_practices', []),
                'recommendations': analysis_results.get('testing_recommendations', [])
            },
            'documentation': {
                'coverage': documentation.get('coverage', {}),
                'quality': documentation.get('quality', {}),
                'recommendations': documentation.get('recommendations', [])
            },
            'ci_cd': analysis_results.get('ci_cd_analysis', {})
        }
    
    async def _identify_risks(self, analysis_results: Dict, tech_stack: Dict) -> List[Dict]:
        risks = []
        
        # Technical risks
        tech_risks = await self._analyze_technical_risks(
            analysis_results,
            tech_stack
        )
        risks.extend(tech_risks)
        
        # Security risks
        security_risks = await self._analyze_security_risks(
            analysis_results
        )
        risks.extend(security_risks)
        
        # Operational risks
        operational_risks = await self._analyze_operational_risks(
            analysis_results,
            tech_stack
        )
        risks.extend(operational_risks)
        
        return [vars(risk) for risk in risks]
    
    async def _analyze_technical_risks(
        self,
        analysis_results: Dict,
        tech_stack: Dict
    ) -> List[ARBRisk]:
        prompt = (
            f"Identify technical risks based on:\n"
            f"Tech Stack: {json.dumps(tech_stack, indent=2)}\n"
            f"Architecture: {json.dumps(analysis_results.get('architecture_overview', {}), indent=2)}\n"
            f"Consider scalability, reliability, and maintainability risks."
        )
        
        response = await self.llm_router.route_request(
            TaskType.ARCHITECTURE_REVIEW,
            prompt
        )
        
        risks = []
        for risk_data in response.get('risks', []):
            risks.append(ARBRisk(
                category='technical',
                description=risk_data['description'],
                impact=risk_data['impact'],
                likelihood=risk_data['likelihood'],
                mitigation=risk_data['mitigation']
            ))
        
        return risks
    
    async def _analyze_security_risks(self, analysis_results: Dict) -> List[ARBRisk]:
        security_data = analysis_results.get('security_analysis', {})
        risks = []
        
        for vuln in security_data.get('vulnerabilities', []):
            risks.append(ARBRisk(
                category='security',
                description=vuln['description'],
                impact=vuln['severity'],
                likelihood='high' if vuln.get('exploitable', False) else 'medium',
                mitigation=vuln.get('remediation', '')
            ))
        
        return risks
    
    async def _analyze_operational_risks(
        self,
        analysis_results: Dict,
        tech_stack: Dict
    ) -> List[ARBRisk]:
        prompt = (
            f"Identify operational risks based on:\n"
            f"Tech Stack: {json.dumps(tech_stack, indent=2)}\n"
            f"Deployment: {json.dumps(analysis_results.get('deployment', {}), indent=2)}\n"
            f"Consider monitoring, disaster recovery, and operational processes."
        )
        
        response = await self.llm_router.route_request(
            TaskType.ARCHITECTURE_REVIEW,
            prompt
        )
        
        risks = []
        for risk_data in response.get('risks', []):
            risks.append(ARBRisk(
                category='operational',
                description=risk_data['description'],
                impact=risk_data['impact'],
                likelihood=risk_data['likelihood'],
                mitigation=risk_data['mitigation']
            ))
        
        return risks
    
    async def _generate_recommendations(
        self,
        analysis_results: Dict,
        risks: List[Dict],
        tech_stack: Dict
    ) -> List[Dict]:
        prompt = (
            f"Generate architecture recommendations based on:\n"
            f"Tech Stack: {json.dumps(tech_stack, indent=2)}\n"
            f"Risks: {json.dumps(risks, indent=2)}\n"
            f"Analysis: {json.dumps(analysis_results, indent=2)}\n"
            f"Consider:\n"
            f"1. Architectural improvements\n"
            f"2. Technical debt reduction\n"
            f"3. Security enhancements\n"
            f"4. Operational improvements\n"
            f"5. Future scalability"
        )
        
        response = await self.llm_router.route_request
        
        async def _generate_recommendations(
        self,
        analysis_results: Dict,
        risks: List[Dict],
        tech_stack: Dict
    ) -> List[Dict]:
        prompt = (
            f"Generate architecture recommendations based on:\n"
            f"Tech Stack: {json.dumps(tech_stack, indent=2)}\n"
            f"Risks: {json.dumps(risks, indent=2)}\n"
            f"Analysis: {json.dumps(analysis_results, indent=2)}\n"
            f"Consider:\n"
            f"1. Architectural improvements\n"
            f"2. Technical debt reduction\n"
            f"3. Security enhancements\n"
            f"4. Operational improvements\n"
            f"5. Future scalability"
        )
        
        response = await self.llm_router.route_request(
            TaskType.ARCHITECTURE_REVIEW,
            prompt
        )
        
        recommendations = response.get('recommendations', [])
        
        # Categorize recommendations
        categorized_recommendations = {
            'architectural': [],
            'technical_debt': [],
            'security': [],
            'operational': [],
            'scalability': []
        }
        
        for rec in recommendations:
            category = rec.get('category', 'architectural')
            if category in categorized_recommendations:
                categorized_recommendations[category].append({
                    'title': rec.get('title', ''),
                    'description': rec.get('description', ''),
                    'priority': rec.get('priority', 'medium'),
                    'effort': rec.get('effort', 'medium'),
                    'impact': rec.get('impact', 'medium'),
                    'timeline': rec.get('timeline', 'medium-term'),
                    'dependencies': rec.get('dependencies', []),
                    'risks': rec.get('risks', [])
                })
        
        return categorized_recommendations
    
    def _check_compliance(self, analysis_results: Dict) -> Dict:
        """Check architecture against common compliance frameworks."""
        compliance_results = {
            'frameworks': {},
            'violations': [],
            'recommendations': []
        }
        
        # Add compliance checks for common frameworks
        if 'security_analysis' in analysis_results:
            security = analysis_results['security_analysis']
            
            # Check SOC2 compliance
            compliance_results['frameworks']['SOC2'] = {
                'status': self._check_soc2_compliance(security),
                'controls': self._get_soc2_controls(security)
            }
            
            # Check GDPR compliance if applicable
            if self._is_gdpr_applicable(analysis_results):
                compliance_results['frameworks']['GDPR'] = {
                    'status': self._check_gdpr_compliance(security),
                    'controls': self._get_gdpr_controls(security)
                }
            
            # Check HIPAA compliance if applicable
            if self._is_hipaa_applicable(analysis_results):
                compliance_results['frameworks']['HIPAA'] = {
                    'status': self._check_hipaa_compliance(security),
                    'controls': self._get_hipaa_controls(security)
                }
        
        return compliance_results
    
    def _check_soc2_compliance(self, security_analysis: Dict) -> Dict:
        """Check SOC2 compliance controls."""
        return {
            'security': self._check_security_controls(security_analysis),
            'availability': self._check_availability_controls(security_analysis),
            'confidentiality': self._check_confidentiality_controls(security_analysis),
            'processing_integrity': self._check_processing_controls(security_analysis),
            'privacy': self._check_privacy_controls(security_analysis)
        }
    
    def _check_security_controls(self, security_analysis: Dict) -> Dict:
        controls = {
            'access_control': False,
            'encryption': False,
            'monitoring': False,
            'incident_response': False
        }
        
        if security_analysis.get('access_control'):
            controls['access_control'] = True
        if security_analysis.get('encryption'):
            controls['encryption'] = True
        if security_analysis.get('monitoring'):
            controls['monitoring'] = True
        if security_analysis.get('incident_response'):
            controls['incident_response'] = True
        
        return controls
    
    def _is_gdpr_applicable(self, analysis_results: Dict) -> bool:
        """Check if GDPR compliance is applicable."""
        # Check for EU data or users
        return any(
            'eu' in str(item).lower()
            for item in analysis_results.get('data_flows', [])
        )
    
    def _is_hipaa_applicable(self, analysis_results: Dict) -> bool:
        """Check if HIPAA compliance is applicable."""
        # Check for healthcare data
        return any(
            'health' in str(item).lower() or 'medical' in str(item).lower()
            for item in analysis_results.get('data_flows', [])
        )
    
    def _generate_documentation(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive architecture documentation."""
        return {
            'architecture_decision_records': self._generate_adrs(analysis_results),
            'component_documentation': self._generate_component_docs(analysis_results),
            'api_documentation': self._generate_api_docs(analysis_results),
            'deployment_documentation': self._generate_deployment_docs(analysis_results),
            'security_documentation': self._generate_security_docs(analysis_results)
        }
    
    def _generate_adrs(self, analysis_results: Dict) -> List[Dict]:
        """Generate Architecture Decision Records (ADRs)."""
        adrs = []
        
        # Generate ADRs for key architectural decisions
        patterns = analysis_results.get('architecture_overview', {}).get('patterns', [])
        for pattern in patterns:
            adrs.append({
                'title': f"Use of {pattern} Architecture",
                'status': 'Accepted',
                'context': f"Decision to implement {pattern} architecture pattern",
                'decision': f"Adopted {pattern} architecture for the system",
                'consequences': self._get_pattern_consequences(pattern)
            })
        
        # Generate ADRs for technology choices
        tech_stack = analysis_results.get('tech_stack', {})
        for category, technologies in tech_stack.items():
            if technologies:
                adrs.append({
                    'title': f"Technology Choice for {category}",
                    'status': 'Accepted',
                    'context': f"Selection of technologies for {category}",
                    'decision': f"Adopted {', '.join(technologies)}",
                    'consequences': self._get_tech_consequences(category, technologies)
                })
        
        return adrs