import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
import toml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

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

class ConfigFormat(str, Enum):
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    ENV = "env"
    INI = "ini"

@dataclass
class ConfigIssue:
    type: str
    severity: str
    message: str
    path: str
    value: Optional[str] = None
    suggestion: Optional[str] = None
    reference: Optional[str] = None

class ConfigPatterns:
    SENSITIVE_PATTERNS = {
        r'password', r'secret', r'key', r'token',
        r'auth', r'credential', r'private',
    }
    
    ENV_PATTERNS = {
        r'\$\{.*?\}',
        r'\$[A-Za-z_][A-Za-z0-9_]*',
        r'%[A-Za-z_][A-Za-z0-9_]*%',
    }
    
    FORMAT_MAPPING = {
        '.yaml': ConfigFormat.YAML,
        '.yml': ConfigFormat.YAML,
        '.json': ConfigFormat.JSON,
        '.toml': ConfigFormat.TOML,
        '.env': ConfigFormat.ENV,
        '.ini': ConfigFormat.INI,
    }
    
    @classmethod
    def is_sensitive_key(cls, key: str) -> bool:
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in cls.SENSITIVE_PATTERNS)
    
    @classmethod
    def find_env_vars(cls, value: str) -> Set[str]:
        env_vars = set()
        for pattern in cls.ENV_PATTERNS:
            matches = re.finditer(pattern, str(value))
            env_vars.update(match.group() for match in matches)
        return env_vars
    
    @classmethod
    def get_format(cls, file_path: Path) -> Optional[ConfigFormat]:
        return cls.FORMAT_MAPPING.get(file_path.suffix.lower())

class ConfigAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={AgentCapability.CONFIGURATION},
            file_patterns={
                '*.yaml', '*.yml', '*.json', '*.toml',
                '*.env', '*.ini'
            },
            max_file_size=1024 * 1024,
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
            config_format = ConfigPatterns.get_format(file_path)
            if not config_format:
                raise AnalysisError(f"Unsupported config format: {file_path}")
            
            config_data = self._parse_config(content, config_format)
            structure_analysis = self._analyze_structure(config_data)
            security_issues = await self._security_scan(config_data, config_format)
            env_analysis = self._analyze_environment(config_data)
            recommendations = await self._generate_recommendations(
                config_data,
                structure_analysis,
                security_issues,
                env_analysis,
                config_format
            )
            
            return {
                'format': config_format.value,
                'structure': structure_analysis,
                'security_issues': security_issues,
                'environment': env_analysis,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Config analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Config analysis failed: {str(e)}") from e
    
    def _parse_config(self, content: str, format: ConfigFormat) -> Dict:
        try:
            if format == ConfigFormat.YAML:
                return yaml.safe_load(content)
            elif format == ConfigFormat.JSON:
                return json.loads(content)
            elif format == ConfigFormat.TOML:
                return toml.loads(content)
            elif format == ConfigFormat.ENV:
                return self._parse_env_file(content)
            elif format == ConfigFormat.INI:
                return self._parse_ini_file(content)
            else:
                raise AnalysisError(f"Unsupported format: {format}")
        except Exception as e:
            raise AnalysisError(f"Failed to parse {format} content: {str(e)}")
    
    def _parse_env_file(self, content: str) -> Dict[str, str]:
        result = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    result[key.strip()] = value.strip()
        return result
    
    def _parse_ini_file(self, content: str) -> Dict[str, Dict[str, str]]:
        result = {}
        current_section = None
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                result[current_section] = {}
            elif '=' in line and current_section is not None:
                key, value = line.split('=', 1)
                result[current_section][key.strip()] = value.strip()
        
        return result
    
    def _analyze_structure(self, config: Dict) -> Dict:
        stats = {
            'total_keys': 0,
            'max_depth': 0,
            'sensitive_keys': 0,
            'array_count': 0,
            'object_count': 0
        }
        
        issues = []
        
        def analyze_value(value: Any, path: str = '', depth: int = 0):
            if isinstance(value, dict):
                stats['object_count'] += 1
                stats['max_depth'] = max(stats['max_depth'], depth)
                
                for key, val in value.items():
                    current_path = f"{path}.{key}" if path else key
                    stats['total_keys'] += 1
                    
                    if ConfigPatterns.is_sensitive_key(key):
                        stats['sensitive_keys'] += 1
                        issues.append(ConfigIssue(
                            type='security',
                            severity='warning',
                            message='Potentially sensitive key found',
                            path=current_path,
                            suggestion='Consider using environment variables'
                        ))
                    
                    analyze_value(val, current_path, depth + 1)
                    
            elif isinstance(value, (list, tuple)):
                stats['array_count'] += 1
                for i, item in enumerate(value):
                    analyze_value(item, f"{path}[{i}]", depth + 1)
        
        analyze_value(config)
        return {'statistics': stats, 'issues': [issue.__dict__ for issue in issues]}
    
    async def _security_scan(self, config: Dict, format: ConfigFormat) -> List[Dict]:
        security_issues = []
        
        def scan_value(value: Any, path: str = ''):
            if isinstance(value, str):
                if any(pattern in value.lower() for pattern in ConfigPatterns.SENSITIVE_PATTERNS):
                    security_issues.append({
                        'type': 'sensitive_data',
                        'severity': 'high',
                        'path': path,
                        'message': 'Possible sensitive data in value'
                    })
            elif isinstance(value, dict):
                for key, val in value.items():
                    current_path = f"{path}.{key}" if path else key
                    scan_value(val, current_path)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    scan_value(item, f"{path}[{i}]")
        
        scan_value(config)
        
        response = await self.llm_router.route_request(
            TaskType.SECURITY_AUDIT,
            f"Analyze this {format.value} configuration for security issues:\n{json.dumps(config, indent=2)}"
        )
        
        if 'security_issues' in response:
            security_issues.extend(response['security_issues'])
        
        return security_issues
    
    def _analyze_environment(self, config: Dict) -> Dict:
        env_vars = set()
        required_vars = set()
        optional_vars = set()
        
        def find_env_vars(value: Any):
            if isinstance(value, str):
                vars = ConfigPatterns.find_env_vars(value)
                env_vars.update(vars)
                for var in vars:
                    if ':-' in var or ':?' in var:
                        optional_vars.add(var)
                    else:
                        required_vars.add(var)
            elif isinstance(value, dict):
                for val in value.values():
                    find_env_vars(val)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    find_env_vars(item)
        
        find_env_vars(config)
        return {
            'total_vars': len(env_vars),
            'required_vars': list(required_vars),
            'optional_vars': list(optional_vars),
            'all_vars': list(env_vars)
        }
    
    async def _generate_recommendations(
        self,
        config: Dict,
        structure_analysis: Dict,
        security_issues: List[Dict],
        env_analysis: Dict,
        format: ConfigFormat
    ) -> List[Dict]:
        context = {
            'format': format.value,
            'structure': structure_analysis,
            'security_issues_count': len(security_issues),
            'env_vars_count': env_analysis['total_vars']
        }
        
        response = await self.llm_router.route_request(
            TaskType.CONFIGURATION,
            f"Based on the analysis of this {format.value} configuration, "
            f"suggest improvements:\n{json.dumps(context, indent=2)}"
        )
        
        return response.get('recommendations', [])</parameter>
<parameter name="language">python</parameter>
</invoke>