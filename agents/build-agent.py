import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

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

class BuildFileType(str, Enum):
    DOCKERFILE = "dockerfile"
    COMPOSE = "docker-compose"
    MAKEFILE = "makefile"
    PACKAGE_JSON = "package.json"
    REQUIREMENTS = "requirements.txt"
    GEMFILE = "gemfile"
    PODFILE = "podfile"
    GRADLE = "gradle"
    POM = "pom"

@dataclass
class BuildDependency:
    name: str
    version: str
    type: str
    source: Optional[str] = None
    constraints: List[str] = field(default_factory=list)

@dataclass
class BuildStage:
    name: str
    commands: List[str]
    dependencies: List[BuildDependency]
    environment: Dict[str, str]
    artifacts: List[str]

class BuildPatterns:
    FILE_MAPPING = {
        'Dockerfile': BuildFileType.DOCKERFILE,
        'docker-compose.yml': BuildFileType.COMPOSE,
        'docker-compose.yaml': BuildFileType.COMPOSE,
        'Makefile': BuildFileType.MAKEFILE,
        'package.json': BuildFileType.PACKAGE_JSON,
        'requirements.txt': BuildFileType.REQUIREMENTS,
        'Gemfile': BuildFileType.GEMFILE,
        'Podfile': BuildFileType.PODFILE,
        'build.gradle': BuildFileType.GRADLE,
        'pom.xml': BuildFileType.POM
    }
    
    SECURITY_PATTERNS = {
        'exposed_port': r'EXPOSE\s+\d+',
        'root_user': r'USER\s+root',
        'sensitive_arg': r'ARG\s+(?:.*PASSWORD|.*SECRET|.*KEY)',
        'sudo_usage': r'sudo\s+',
        'wget_curl': r'(?:wget|curl)\s+http://',
        'latest_tag': r'FROM\s+[^:]+:latest'
    }
    
    VERSION_PATTERNS = {
        'exact': r'==\s*[\d.]+',
        'minimum': r'>=\s*[\d.]+',
        'maximum': r'<=\s*[\d.]+',
        'range': r'>=\s*[\d.]+\s*,\s*<=\s*[\d.]+',
        'caret': r'\^\s*[\d.]+',
        'tilde': r'~\s*[\d.]+',
    }
    
    @classmethod
    def get_file_type(cls, file_path: Path) -> Optional[BuildFileType]:
        return cls.FILE_MAPPING.get(file_path.name)
    
    @classmethod
    def find_security_issues(cls, content: str) -> List[Dict]:
        issues = []
        for issue_type, pattern in cls.SECURITY_PATTERNS.items():
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                issues.append({
                    'type': issue_type,
                    'line': content.count('\n', 0, match.start()) + 1,
                    'match': match.group()
                })
        return issues

class BuildAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={AgentCapability.BUILD},
            file_patterns=set(BuildPatterns.FILE_MAPPING.keys()),
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
            file_type = BuildPatterns.get_file_type(file_path)
            if not file_type:
                raise AnalysisError(f"Unsupported build file: {file_path}")
            
            if file_type == BuildFileType.DOCKERFILE:
                analysis = self._analyze_dockerfile(content)
            elif file_type == BuildFileType.COMPOSE:
                analysis = self._analyze_compose(content)
            elif file_type == BuildFileType.MAKEFILE:
                analysis = self._analyze_makefile(content)
            elif file_type == BuildFileType.PACKAGE_JSON:
                analysis = self._analyze_package_json(content)
            elif file_type == BuildFileType.REQUIREMENTS:
                analysis = self._analyze_requirements(content)
            else:
                analysis = self._analyze_generic(content, file_type)
            
            security_issues = BuildPatterns.find_security_issues(content)
            best_practices = await self._check_best_practices(content, file_type)
            dependencies = self._extract_dependencies(content, file_type)
            recommendations = await self._generate_recommendations(
                analysis,
                security_issues,
                best_practices,
                dependencies,
                file_type
            )
            
            return {
                'file_type': file_type.value,
                'analysis': analysis,
                'security_issues': security_issues,
                'best_practices': best_practices,
                'dependencies': [dep.__dict__ for dep in dependencies],
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Build analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Build analysis failed: {str(e)}") from e
    
    def _analyze_dockerfile(self, content: str) -> Dict:
        stages = []
        current_stage = {
            'base_image': None,
            'commands': [],
            'env_vars': {},
            'exposed_ports': [],
            'volumes': [],
            'working_dir': None,
            'user': None
        }
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            instruction = parts[0].upper()
            
            if instruction == 'FROM':
                if current_stage['commands']:
                    stages.append(current_stage)
                    current_stage = {
                        'base_image': None,
                        'commands': [],
                        'env_vars': {},
                        'exposed_ports': [],
                        'volumes': [],
                        'working_dir': None,
                        'user': None
                    }
                current_stage['base_image'] = parts[1]
                
            elif instruction == 'RUN':
                current_stage['commands'].append(' '.join(parts[1:]))
                
            elif instruction == 'ENV':
                key = parts[1]
                value = ' '.join(parts[2:])
                current_stage['env_vars'][key] = value
                
            elif instruction == 'EXPOSE':
                current_stage['exposed_ports'].extend(parts[1:])
                
            elif instruction == 'VOLUME':
                current_stage['volumes'].extend(parts[1:])
                
            elif instruction == 'WORKDIR':
                current_stage['working_dir'] = parts[1]
                
            elif instruction == 'USER':
                current_stage['user'] = parts[1]
        
        if current_stage['commands']:
            stages.append(current_stage)
        
        return {
            'stages': stages,
            'stage_count': len(stages),
            'total_commands': sum(len(s['commands']) for s in stages),
            'total_env_vars': sum(len(s['env_vars']) for s in stages),
            'unique_base_images': list(set(s['base_image'] for s in stages))
        }
    
    def _analyze_compose(self, content: str) -> Dict:
        try:
            compose = yaml.safe_load(content)
            services = {}
            
            for service_name, service_config in compose.get('services', {}).items():
                services[service_name] = {
                    'image': service_config.get('image'),
                    'build': service_config.get('build'),
                    'ports': service_config.get('ports', []),
                    'volumes': service_config.get('volumes', []),
                    'environment': service_config.get('environment', {}),
                    'depends_on': service_config.get('depends_on', []),
                    'networks': service_config.get('networks', [])
                }
            
            return {
                'version': compose.get('version'),
                'services': services,
                'networks': compose.get('networks', {}),
                'volumes': compose.get('volumes', {}),
                'service_count': len(services),
                'total_ports': sum(
                    len(s['ports']) for s in services.values()
                ),
                'total_volumes': sum(
                    len(s['volumes']) for s in services.values()
                )
            }
            
        except yaml.YAMLError as e:
            raise AnalysisError(f"Invalid docker-compose file: {str(e)}")
    
    def _analyze_makefile(self, content: str) -> Dict:
        targets = {}
        current_target = None
        
        for line in content.splitlines():
            line = line.rstrip()
            
            if not line or line.startswith('#'):
                continue
                
            if line[0] != '\t' and ':' in line:
                target_name = line.split(':')[0].strip()
                dependencies = line.split(':')[1].strip().split()
                current_target = target_name
                targets[current_target] = {
                    'dependencies': dependencies,
                    'commands': [],
                    'variables': {}
                }
                
            elif line.startswith('\t') and current_target:
                targets[current_target]['commands'].append(line.strip())
                
            elif '=' in line:
                var_name = line.split('=')[0].strip()
                var_value = line.split('=')[1].strip()
                if current_target:
                    targets[current_target]['variables'][var_name] = var_value
        
        return {
            'targets': targets,
            'target_count': len(targets),
            'total_commands': sum(
                len(t['commands']) for t in targets.values()
            ),
            'total_variables': sum(
                len(t['variables']) for t in targets.values()
            )
        }
    
    def _analyze_package_json(self, content: str) -> Dict:
        try:
            package = json.loads(content)
            scripts = package.get('scripts', {})
            dependencies = package.get('dependencies', {})
            dev_dependencies = package.get('devDependencies', {})
            
            return {
                'name': package.get('name'),
                'version': package.get('version'),
                'scripts': scripts,
                'dependencies': dependencies,
                'dev_dependencies': dev_dependencies,
                'total_scripts': len(scripts),
                'total_dependencies': len(dependencies),
                'total_dev_dependencies': len(dev_dependencies)
            }
            
        except json.JSONDecodeError as e:
            raise AnalysisError(f"Invalid package.json: {str(e)}")
    
    def _analyze_requirements(self, content: str) -> Dict:
        dependencies = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if '==' in line:
                name, version = line.split('==')
                dependencies.append({
                    'name': name.strip(),
                    'version': version.strip(),
                    'type': 'exact'
                })
            elif '>=' in line:
                name, version = line.split('>=')
                dependencies.append({
                    'name': name.strip(),
                    'version': version.strip(),
                    'type': 'minimum'
                })
        
        return {
            'dependencies': dependencies,
            'total_dependencies': len(dependencies)
        }
    
    def _analyze_generic(self, content: str, file_type: BuildFileType) -> Dict:
        return {
            'file_type': file_type.value,
            'line_count': len(content.splitlines()),
            'size_bytes': len(content.encode('utf-8'))
        }
    
    def _extract_dependencies(
        self,
        content: str,
        file_type: BuildFileType
    ) -> List[BuildDependency]:
        dependencies = []
        
        if file_type == BuildFileType.DOCKERFILE:
            base_images = re.findall(r'FROM\s+([^\s]+)', content)
            for image in base_images:
                name, tag = image.split(':') if ':' in image else (image, 'latest')
                dependencies.append(BuildDependency(
                    name=name,
                    version=tag,
                    type='image'
                ))
                
        elif file_type == BuildFileType.REQUIREMENTS:
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for pattern_type, pattern in BuildPatterns.VERSION_PATTERNS.items():
                        match = re.search(pattern, line)
                        if match:
                            name = line.split(match.group())[0].strip()
                            dependencies.append(BuildDependency(
                                name=name,
                                version=match.group(),
                                type=pattern_type
                            ))
                            break
        
        return dependencies
    
    async def _check_best_practices(
        self,
        content: str,
        file_type: BuildFileType
    ) -> List[Dict]:
        best_practices_prompt = (
            f"Analyze this {file_type.value} file for best practices:\n{content}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            best_practices_prompt
        )
        
        return response.get('best_practices', [])
    
    async def _generate_recommendations(
        self,
        analysis: Dict,
        security_issues: List[Dict],
        best_practices: List[Dict],
        dependencies: List[BuildDependency],
        file_type: BuildFileType
    ) -> List[Dict]:
        context = {
            'file_type': file_type.value,
            'security_issues_count': len(security_issues),
            'best_practices_issues': len(best_practices),
            'dependencies_count': len(dependencies)
        }
        
        recommendations_prompt = (
            f"Based on the build file analysis, suggest improvements:\n"
            f"{json.dumps(context, indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            recommendations_prompt
        )
        
        return response.get('recommendations', [])