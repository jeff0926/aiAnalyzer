import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
import hcl2

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

class InfraFormat(str, Enum):
    TERRAFORM = "terraform"
    CLOUDFORMATION = "cloudformation"
    ARM = "arm"
    KUBERNETES = "kubernetes"
    HELM = "helm"
    ANSIBLE = "ansible"
    PULUMI = "pulumi"

@dataclass
class Resource:
    type: str
    name: str
    provider: str
    properties: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SecurityIssue:
    severity: str
    resource_type: str
    resource_name: str
    issue_type: str
    description: str
    recommendation: str
    references: List[str] = field(default_factory=list)

class InfraPatterns:
    FILE_MAPPING = {
        '.tf': InfraFormat.TERRAFORM,
        '.tfvars': InfraFormat.TERRAFORM,
        '.yaml': InfraFormat.KUBERNETES,
        '.yml': InfraFormat.KUBERNETES,
        '.template': InfraFormat.CLOUDFORMATION,
        '.bicep': InfraFormat.ARM,
        'Chart.yaml': InfraFormat.HELM,
        'playbook.yml': InfraFormat.ANSIBLE,
        'Pulumi.yaml': InfraFormat.PULUMI
    }
    
    SENSITIVE_PATTERNS = {
        'password': r'(?i)password|passwd|pwd',
        'key': r'(?i)key|secret|token|credential',
        'certificate': r'(?i)certificate|cert|pem|crt',
        'connection': r'(?i)connection_string|connstr|jdbc'
    }
    
    SECURITY_RULES = {
        'open_ports': {
            'pattern': r'port\s*:\s*"*\d+"*',
            'severity': 'high',
            'description': 'Potentially exposed port'
        },
        'public_access': {
            'pattern': r'public\s*=\s*true|public_access\s*=\s*true',
            'severity': 'high',
            'description': 'Public access enabled'
        },
        'weak_encryption': {
            'pattern': r'encryption\s*=\s*false|encrypted\s*=\s*false',
            'severity': 'critical',
            'description': 'Encryption disabled'
        }
    }
    
    @classmethod
    def get_format(cls, file_path: Path) -> Optional[InfraFormat]:
        if file_path.suffix == '.tf' or file_path.name.endswith('.tfvars'):
            return InfraFormat.TERRAFORM
        elif file_path.suffix in {'.yaml', '.yml'}:
            content = file_path.read_text()
            if 'apiVersion' in content and 'kind' in content:
                return InfraFormat.KUBERNETES
            elif 'AWSTemplateFormatVersion' in content:
                return InfraFormat.CLOUDFORMATION
            elif file_path.name == 'Chart.yaml':
                return InfraFormat.HELM
        return cls.FILE_MAPPING.get(file_path.suffix)
    
    @classmethod
    def find_security_issues(cls, resource: Resource) -> List[SecurityIssue]:
        issues = []
        
        # Check for sensitive data exposure
        for data_type, pattern in cls.SENSITIVE_PATTERNS.items():
            matches = []
            for key, value in cls._traverse_dict(resource.properties):
                if re.search(pattern, key, re.IGNORECASE):
                    matches.append(key)
            
            if matches:
                issues.append(SecurityIssue(
                    severity='high',
                    resource_type=resource.type,
                    resource_name=resource.name,
                    issue_type='sensitive_data',
                    description=f'Sensitive {data_type} found in properties: {", ".join(matches)}',
                    recommendation=f'Use secure parameter storage for {data_type}',
                    references=['https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html']
                ))
        
        # Check security rules
        for rule_name, rule in cls.SECURITY_RULES.items():
            for key, value in cls._traverse_dict(resource.properties):
                if re.search(rule['pattern'], f"{key}={value}", re.IGNORECASE):
                    issues.append(SecurityIssue(
                        severity=rule['severity'],
                        resource_type=resource.type,
                        resource_name=resource.name,
                        issue_type=rule_name,
                        description=rule['description'],
                        recommendation=f'Review and secure {key} configuration',
                        references=[]
                    ))
        
        return issues
    
    @classmethod
    def _traverse_dict(cls, d: Dict, path: str = '') -> List[tuple]:
        items = []
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                items.extend(cls._traverse_dict(value, current_path))
            else:
                items.append((current_path, str(value)))
        return items
####
class InfraAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={
                AgentCapability.INFRASTRUCTURE,
                AgentCapability.SECURITY_SCAN
            },
            file_patterns={
                '*.tf', '*.tfvars', '*.yaml', '*.yml',
                '*.template', '*.bicep', 'Chart.yaml',
                'playbook.yml', 'Pulumi.yaml'
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
            infra_format = InfraPatterns.get_format(file_path)
            if not infra_format:
                raise AnalysisError(f"Unsupported infrastructure format: {file_path}")
            
            parsed_content = self._parse_content(content, infra_format)
            
            if infra_format == InfraFormat.TERRAFORM:
                analysis = self._analyze_terraform(parsed_content)
            elif infra_format == InfraFormat.KUBERNETES:
                analysis = self._analyze_kubernetes(parsed_content)
            elif infra_format == InfraFormat.CLOUDFORMATION:
                analysis = self._analyze_cloudformation(parsed_content)
            else:
                analysis = self._analyze_generic(parsed_content, infra_format)
            
            resources = self._extract_resources(parsed_content, infra_format)
            security_analysis = self._analyze_security(resources)
            dependencies = self._analyze_dependencies(resources)
            cost_analysis = await self._analyze_cost(resources)
            recommendations = await self._generate_recommendations(
                analysis,
                security_analysis,
                dependencies,
                cost_analysis,
                infra_format
            )
            
            return {
                'format': infra_format.value,
                'analysis': analysis,
                'resources': [vars(r) for r in resources],
                'security': security_analysis,
                'dependencies': dependencies,
                'cost_analysis': cost_analysis,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Infrastructure analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Infrastructure analysis failed: {str(e)}") from e
    
    def _parse_content(self, content: str, format: InfraFormat) -> Dict:
        try:
            if format == InfraFormat.TERRAFORM:
                return hcl2.loads(content)
            elif format in {InfraFormat.KUBERNETES, InfraFormat.HELM}:
                return yaml.safe_load(content)
            elif format in {InfraFormat.CLOUDFORMATION, InfraFormat.ARM}:
                return json.loads(content)
            else:
                return {'content': content}
        except Exception as e:
            raise AnalysisError(f"Failed to parse {format.value} content: {str(e)}")
    
    def _analyze_terraform(self, content: Dict) -> Dict:
        resource_blocks = content.get('resource', {})
        data_blocks = content.get('data', {})
        variables = content.get('variable', {})
        outputs = content.get('output', {})
        
        resources = {}
        for resource_type, instances in resource_blocks.items():
            for name, config in instances.items():
                resources[f"{resource_type}.{name}"] = config
        
        return {
            'resource_count': len(resources),
            'data_block_count': len(data_blocks),
            'variable_count': len(variables),
            'output_count': len(outputs),
            'providers': list(content.get('provider', {}).keys()),
            'resources': resources
        }
    
    def _analyze_kubernetes(self, content: Dict) -> Dict:
        return {
            'api_version': content.get('apiVersion'),
            'kind': content.get('kind'),
            'namespace': content.get('metadata', {}).get('namespace', 'default'),
            'name': content.get('metadata', {}).get('name'),
            'labels': content.get('metadata', {}).get('labels', {}),
            'annotations': content.get('metadata', {}).get('annotations', {}),
            'spec': content.get('spec', {})
        }
    
    def _analyze_cloudformation(self, content: Dict) -> Dict:
        return {
            'template_version': content.get('AWSTemplateFormatVersion'),
            'description': content.get('Description'),
            'parameter_count': len(content.get('Parameters', {})),
            'resource_count': len(content.get('Resources', {})),
            'output_count': len(content.get('Outputs', {})),
            'resource_types': list(set(
                r.get('Type') for r in content.get('Resources', {}).values()
            ))
        }
    
    def _analyze_generic(self, content: Dict, format: InfraFormat) -> Dict:
        return {
            'format': format.value,
            'content_size': len(str(content)),
            'top_level_keys': list(content.keys())
        }
    
    def _extract_resources(self, content: Dict, format: InfraFormat) -> List[Resource]:
        resources = []
        
        if format == InfraFormat.TERRAFORM:
            for resource_type, instances in content.get('resource', {}).items():
                provider = resource_type.split('_')[0]
                for name, config in instances.items():
                    resources.append(Resource(
                        type=resource_type,
                        name=name,
                        provider=provider,
                        properties=config,
                        dependencies=self._find_terraform_dependencies(config),
                        tags=config.get('tags', {})
                    ))
                    
        elif format == InfraFormat.KUBERNETES:
            resources.append(Resource(
                type=content.get('kind'),
                name=content.get('metadata', {}).get('name'),
                provider='kubernetes',
                properties=content.get('spec', {}),
                tags=content.get('metadata', {}).get('labels', {})
            ))
            
        elif format == InfraFormat.CLOUDFORMATION:
            for name, resource in content.get('Resources', {}).items():
                resources.append(Resource(
                    type=resource.get('Type'),
                    name=name,
                    provider='aws',
                    properties=resource.get('Properties', {}),
                    dependencies=resource.get('DependsOn', []),
                    tags=resource.get('Properties', {}).get('Tags', {})
                ))
        
        return resources
    
    def _find_terraform_dependencies(self, config: Dict) -> List[str]:
        deps = []
        for value in str(config).split('${')[1:]:
            dep = value.split('}')[0]
            if dep:
                deps.append(dep)
        return deps
    
    def _analyze_security(self, resources: List[Resource]) -> Dict:
        issues = []
        for resource in resources:
            resource_issues = InfraPatterns.find_security_issues(resource)
            issues.extend([vars(issue) for issue in resource_issues])
        
        return {
            'total_issues': len(issues),
            'issues_by_severity': {
                severity: len([i for i in issues if i['severity'] == severity])
                for severity in {'critical', 'high', 'medium', 'low'}
            },
            'issues': issues
        }
    
    def _analyze_dependencies(self, resources: List[Resource]) -> Dict:
        dependency_graph = {}
        
        for resource in resources:
            dependency_graph[f"{resource.type}.{resource.name}"] = {
                'dependencies': resource.dependencies,
                'dependents': []
            }
        
        for resource_id, data in dependency_graph.items():
            for dep in data['dependencies']:
                if dep in dependency_graph:
                    dependency_graph[dep]['dependents'].append(resource_id)
        
        return dependency_graph
    
    async def _analyze_cost(self, resources: List[Resource]) -> Dict:
        cost_prompt = (
            "Estimate infrastructure costs for these resources:\n"
            f"{json.dumps([vars(r) for r in resources], indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.INFRASTRUCTURE,
            cost_prompt
        )
        
        return response.get('cost_analysis', {
            'estimated_monthly_cost': 0,
            'cost_breakdown': {},
            'notes': ['Cost analysis requires provider-specific pricing information']
        })
    
    async def _generate_recommendations(
        self,
        analysis: Dict,
        security_analysis: Dict,
        dependencies: Dict,
        cost_analysis: Dict,
        format: InfraFormat
    ) -> List[Dict]:
        context = {
            'format': format.value,
            'resource_count': len(analysis.get('resources', {})),
            'security_issues': security_analysis['total_issues'],
            'estimated_cost': cost_analysis.get('estimated_monthly_cost', 0)
        }
        
        recommendations_prompt = (
            f"Based on the infrastructure analysis, suggest improvements:\n"
            f"{json.dumps(context, indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.INFRASTRUCTURE,
            recommendations_prompt
        )
        
        return response.get('recommendations', [])
