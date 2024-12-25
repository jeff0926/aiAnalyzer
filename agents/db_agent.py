import json
import logging
import re
import sqlparse
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

class DatabaseType(str, Enum):
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    ORACLE = "oracle"
    MONGODB = "mongodb"
    CASSANDRA = "cassandra"

class SchemaObjectType(str, Enum):
    TABLE = "table"
    VIEW = "view"
    INDEX = "index"
    TRIGGER = "trigger"
    PROCEDURE = "procedure"
    FUNCTION = "function"
    CONSTRAINT = "constraint"
    SEQUENCE = "sequence"

@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True
    default: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    comment: Optional[str] = None

@dataclass
class Table:
    name: str
    columns: List[Column]
    primary_key: Optional[List[str]] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    indexes: Dict[str, List[str]] = field(default_factory=dict)
    constraints: Dict[str, str] = field(default_factory=dict)
    comments: Optional[str] = None

@dataclass
class Migration:
    version: str
    name: str
    up_commands: List[str]
    down_commands: List[str]
    dependencies: List[str] = field(default_factory=list)
    applied: bool = False

class DatabasePatterns:
    MIGRATION_FILE_PATTERNS = {
        r'\d{14}_\w+\.(?:sql|py|rb|js|ts|php)$',  # Timestamp-based
        r'V\d+__\w+\.sql$',  # Flyway-style
        r'\d{4}_\d{2}_\d{2}_\d{6}_\w+\.py$'  # Alembic-style
    }
    
    FILE_MAPPING = {
        '.sql': {'mysql', 'postgresql', 'sqlite', 'mssql', 'oracle'},
        '.migration': {'mysql', 'postgresql'},
        'schema.rb': {'mysql', 'postgresql'},
        'structure.sql': {'mysql', 'postgresql'},
        '.prisma': {'mysql', 'postgresql', 'sqlite'},
        '.dbml': {'mysql', 'postgresql'},
        'schema.json': {'mongodb'},
        'schema.cql': {'cassandra'}
    }
    
    SQL_PATTERNS = {
        'create_table': r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)',
        'create_view': r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([^\s(]+)',
        'create_index': r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+([^\s(]+)',
        'create_trigger': r'CREATE\s+TRIGGER\s+([^\s(]+)',
        'create_procedure': r'CREATE\s+PROCEDURE\s+([^\s(]+)',
        'create_function': r'CREATE\s+FUNCTION\s+([^\s(]+)',
        'alter_table': r'ALTER\s+TABLE\s+([^\s(]+)',
        'drop_table': r'DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([^\s;]+)'
    }
    
    COLUMN_PATTERN = (
        r'(\w+)\s+'  # Column name
        r'(\w+(?:\s*\([^)]+\))?)'  # Data type with optional size
        r'(?:\s+(?:NOT\s+)?NULL)?'  # Nullability
        r'(?:\s+DEFAULT\s+[^,)]+)?'  # Default value
        r'(?:\s+(?:PRIMARY\s+KEY|UNIQUE|CHECK\s*\([^)]+\)|REFERENCES\s+\w+(?:\s*\([^)]+\))?))?' # Constraints
    )
    
    CONSTRAINT_PATTERNS = {
        'primary_key': r'PRIMARY\s+KEY\s*\(([^)]+)\)',
        'foreign_key': r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)',
        'unique': r'UNIQUE\s*\(([^)]+)\)',
        'check': r'CHECK\s*\(([^)]+)\)'
    }
    
    INDEX_PATTERNS = {
        'btree': r'USING\s+BTREE',
        'hash': r'USING\s+HASH',
        'gin': r'USING\s+GIN',
        'gist': r'USING\s+GIST'
    }
    
    @classmethod
    def get_database_type(cls, file_path: Path) -> Optional[Set[DatabaseType]]:
        file_suffix = file_path.suffix
        file_name = file_path.name.lower()
        
        for pattern, db_types in cls.FILE_MAPPING.items():
            if pattern.startswith('.'):
                if file_suffix == pattern:
                    return {DatabaseType(dt) for dt in db_types}
            else:
                if file_name == pattern:
                    return {DatabaseType(dt) for dt in db_types}
        return None
    
    @classmethod
    def is_migration_file(cls, file_path: Path) -> bool:
        file_name = file_path.name
        return any(re.match(pattern, file_name) for pattern in cls.MIGRATION_FILE_PATTERNS)
    
    @classmethod
    def parse_column_definition(cls, column_def: str) -> Column:
        match = re.match(cls.COLUMN_PATTERN, column_def.strip(), re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid column definition: {column_def}")
            
        name, data_type = match.groups()
        nullable = 'NOT NULL' not in column_def.upper()
        
        default_match = re.search(r'DEFAULT\s+([^,)]+)', column_def, re.IGNORECASE)
        default = default_match.group(1) if default_match else None
        
        constraints = []
        for constraint_type in ['PRIMARY KEY', 'UNIQUE', 'REFERENCES', 'CHECK']:
            if constraint_type in column_def.upper():
                constraints.append(constraint_type)
        
        comment_match = re.search(r'COMMENT\s+\'([^\']+)\'', column_def, re.IGNORECASE)
        comment = comment_match.group(1) if comment_match else None
        
        return Column(
            name=name,
            data_type=data_type,
            nullable=nullable,
            default=default,
            constraints=constraints,
            comment=comment
        )
        
       class DatabaseAnalysisAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={
                AgentCapability.DATABASE,
                AgentCapability.CODE_ANALYSIS
            },
            file_patterns={
                '*.sql', '*.migration', 'schema.rb',
                'structure.sql', '*.prisma', '*.dbml',
                'schema.json', 'schema.cql'
            },
            max_file_size=10 * 1024 * 1024,  # 10MB
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
            db_types = DatabasePatterns.get_database_type(file_path)
            if not db_types:
                raise AnalysisError(f"Unsupported database file: {file_path}")
            
            # Determine file type and parse accordingly
            if file_path.suffix == '.sql':
                analysis = self._analyze_sql(content, list(db_types)[0])
            elif DatabasePatterns.is_migration_file(file_path):
                analysis = self._analyze_migration(content, file_path)
            else:
                analysis = self._analyze_schema(content, file_path)
            
            # Perform additional analyses
            performance_analysis = await self._analyze_performance(analysis)
            security_analysis = await self._analyze_security(analysis)
            best_practices = await self._check_best_practices(analysis)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                analysis,
                performance_analysis,
                security_analysis,
                best_practices
            )
            
            return {
                'database_types': [db.value for db in db_types],
                'analysis': analysis,
                'performance': performance_analysis,
                'security': security_analysis,
                'best_practices': best_practices,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Database analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Database analysis failed: {str(e)}") from e
    
    def _analyze_sql(self, content: str, db_type: DatabaseType) -> Dict:
        # Parse SQL statements
        statements = sqlparse.parse(content)
        schema_objects = []
        dependencies = []
        
        for statement in statements:
            if statement.get_type() == 'CREATE':
                obj = self._analyze_create_statement(statement)
                if obj:
                    schema_objects.append(obj)
            elif statement.get_type() == 'ALTER':
                dep = self._analyze_alter_statement(statement)
                if dep:
                    dependencies.append(dep)
        
        # Extract tables and their relationships
        tables = [obj for obj in schema_objects if obj['type'] == 'table']
        relationships = self._extract_relationships(tables)
        
        return {
            'objects': schema_objects,
            'tables': tables,
            'relationships': relationships,
            'dependencies': dependencies,
            'statistics': self._calculate_schema_statistics(schema_objects)
        }
    
    def _analyze_create_statement(self, statement) -> Optional[Dict]:
        for obj_type, pattern in DatabasePatterns.SQL_PATTERNS.items():
            if obj_type.startswith('create_'):
                match = re.search(pattern, str(statement), re.IGNORECASE)
                if match:
                    obj_name = match.group(1)
                    if 'TABLE' in str(statement).upper():
                        return self._parse_create_table(statement, obj_name)
                    return {
                        'type': obj_type.replace('create_', ''),
                        'name': obj_name,
                        'definition': str(statement)
                    }
        return None
    
    def _parse_create_table(self, statement, table_name: str) -> Dict:
        columns = []
        constraints = {}
        indexes = {}
        
        # Extract column definitions
        columns_match = re.search(r'\((.*)\)', str(statement), re.DOTALL)
        if columns_match:
            column_defs = columns_match.group(1).split(',')
            for col_def in column_defs:
                try:
                    column = DatabasePatterns.parse_column_definition(col_def)
                    columns.append(vars(column))
                except ValueError:
                    # Might be a table-level constraint
                    for const_type, pattern in DatabasePatterns.CONSTRAINT_PATTERNS.items():
                        match = re.search(pattern, col_def, re.IGNORECASE)
                        if match:
                            constraints[const_type] = match.groups()
                            break
        
        return {
            'type': 'table',
            'name': table_name,
            'columns': columns,
            'constraints': constraints,
            'indexes': indexes
        }
    
    def _analyze_migration(self, content: str, file_path: Path) -> Dict:
        # Extract version and name from filename
        version_match = re.search(r'(\d+)', file_path.stem)
        version = version_match.group(1) if version_match else '0'
        name = re.sub(r'^\d+_', '', file_path.stem)
        
        # Parse migration content
        up_commands = []
        down_commands = []
        current_section = None
        
        for line in content.splitlines():
            if '-- +up' in line.lower() or '-- +migrate up' in line.lower():
                current_section = 'up'
            elif '-- +down' in line.lower() or '-- +migrate down' in line.lower():
                current_section = 'down'
            elif line.strip() and not line.strip().startswith('--'):
                if current_section == 'up':
                    up_commands.append(line)
                elif current_section == 'down':
                    down_commands.append(line)
        
        return {
            'version': version,
            'name': name,
            'up_commands': up_commands,
            'down_commands': down_commands,
            'changes': self._analyze_migration_changes(up_commands)
        }
    
    def _analyze_migration_changes(self, commands: List[str]) -> List[Dict]:
        changes = []
        for cmd in commands:
            parsed = sqlparse.parse(cmd)[0] if cmd else None
            if parsed:
                change_type = parsed.get_type()
                if change_type in ('CREATE', 'ALTER', 'DROP'):
                    changes.append({
                        'type': change_type,
                        'object_type': self._get_object_type(parsed),
                        'object_name': self._get_object_name(parsed),
                        'command': cmd
                    })
        return changes
    
    def _analyze_schema(self, content: str, file_path: Path) -> Dict:
        schema_type = file_path.suffix.lower()
        
        if schema_type == '.prisma':
            return self._analyze_prisma_schema(content)
        elif schema_type == '.dbml':
            return self._analyze_dbml_schema(content)
        elif schema_type == '.json':
            return self._analyze_mongodb_schema(content)
        elif schema_type == '.cql':
            return self._analyze_cassandra_schema(content)
        
        raise AnalysisError(f"Unsupported schema type: {schema_type}")
    
    async def _analyze_performance(self, analysis: Dict) -> Dict:
        # Extract relevant information for performance analysis
        tables = analysis.get('tables', [])
        indexes = [idx for table in tables for idx in table.get('indexes', {}).items()]
        relationships = analysis.get('relationships', [])
        
        # Use LLM for performance insights
        prompt = (
            f"Analyze database schema for performance:\n"
            f"Tables: {json.dumps(tables, indent=2)}\n"
            f"Indexes: {json.dumps(indexes, indent=2)}\n"
            f"Relationships: {json.dumps(relationships, indent=2)}\n"
            f"Consider:\n"
            f"1. Index coverage and efficiency\n"
            f"2. Table relationships and join complexity\n"
            f"3. Data types and storage optimization\n"
            f"4. Potential bottlenecks"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            prompt
        )
        
        return response.get('performance_analysis', {
            'index_coverage': [],
            'join_complexity': [],
            'optimization_opportunities': []
        })
    
    async def _analyze_security(self, analysis: Dict) -> Dict:
        # Extract security-relevant information
        tables = analysis.get('tables', [])
        sensitive_columns = self._find_sensitive_columns(tables)
        permissions = self._extract_permissions(analysis)
        
        # Use LLM for security insights
        prompt = (
            f"Analyze database schema for security:\n"
            f"Sensitive Data: {json.dumps(sensitive_columns, indent=2)}\n"
            f"Permissions: {json.dumps(permissions, indent=2)}\n"
            f"Consider:\n"
            f"1. Data privacy and protection\n"
            f"2. Access control\n"
            f"3. SQL injection risks\n"
            f"4. Audit logging"
        )
        
        response = await self.llm_router.route_request(
            TaskType.SECURITY_AUDIT,
            prompt
        )
        
        return response.get('security_analysis', {
            'sensitive_data': sensitive_columns,
            'permissions': permissions,
            'vulnerabilities': [],
            'recommendations': []
        })
    
    async def _check_best_practices(self, analysis: Dict) -> List[Dict]:
        # Use LLM to check database best practices
        prompt = (
            f"Check database schema against best practices:\n"
            f"{json.dumps(analysis, indent=2)}\n"
            f"Consider:\n"
            f"1. Naming conventions\n"
            f"2. Primary key usage\n"
            f"3. Foreign key constraints\n"
            f"4. Indexing strategy\n"
            f"5. Data normalization"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            prompt
        )
        
        return response.get('best_practices', [])
    
    async def _generate_recommendations(
        self,
        analysis: Dict,
        performance_analysis: Dict,
        security_analysis: Dict,
        best_practices: List[Dict]
    ) -> List[Dict]:
        # Combine all analysis results for comprehensive recommendations
        context = {
            'schema_complexity': len(analysis.get('tables', [])),
            'performance_issues': len(performance_analysis.get('optimization_opportunities', [])),
            'security_issues': len(security_analysis.get('vulnerabilities', [])),
            'best_practice_violations': len(best_practices)
        }
        
        prompt = (
            f"Generate database improvement recommendations based on:\n"
            f"{json.dumps(context, indent=2)}\n"
            f"Focus on:\n"
            f"1. Schema optimization\n"
            f"2. Performance improvements\n"
            f"3. Security enhancements\n"
            f"4. Best practices compliance"
        )
        
        response = await self.llm_router.route_request(
            TaskType.CODE_ANALYSIS,
            prompt
        )
        
        return response.get('recommendations', []) 