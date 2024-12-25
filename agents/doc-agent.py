import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import markdown
from docutils.core import publish_parts

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

class DocFormat(str, Enum):
    MARKDOWN = "md"
    RST = "rst"
    TXT = "txt"
    WIKI = "wiki"
    ADOC = "adoc"

@dataclass
class DocSection:
    title: str
    content: str
    level: int
    start_line: int
    end_line: int
    subsections: List['DocSection'] = field(default_factory=list)

@dataclass
class DocQuality:
    spelling_errors: List[Dict[str, Any]]
    grammar_issues: List[Dict[str, Any]]
    broken_links: List[str]
    missing_sections: List[str]
    readability_score: float

class DocumentationPatterns:
    FORMAT_MAPPING = {
        '.md': DocFormat.MARKDOWN,
        '.markdown': DocFormat.MARKDOWN,
        '.rst': DocFormat.RST,
        '.txt': DocFormat.TXT,
        '.wiki': DocFormat.WIKI,
        '.adoc': DocFormat.ADOC
    }
    
    COMMON_SECTIONS = {
        'readme': {
            'overview',
            'installation',
            'usage',
            'configuration',
            'contributing',
            'license'
        },
        'api': {
            'endpoints',
            'authentication',
            'parameters',
            'responses',
            'errors'
        }
    }
    
    LINK_PATTERNS = {
        DocFormat.MARKDOWN: [
            r'\[([^\]]+)\]\(([^)]+)\)',
            r'<([^>]+)>'
        ],
        DocFormat.RST: [
            r'`([^`]+)`_',
            r'.. _[^:]+: ([^\s]+)'
        ]
    }
    
    CODE_BLOCK_PATTERNS = {
        DocFormat.MARKDOWN: (r'```[\s\S]*?```', r'`[^`]+`'),
        DocFormat.RST: (r'::\n\n(?:[ ]{4}[\s\S]*?)\n\n', r'``[^`]+``')
    }
    
    @classmethod
    def get_format(cls, file_path: Path) -> Optional[DocFormat]:
        return cls.FORMAT_MAPPING.get(file_path.suffix.lower())
    
    @classmethod
    def extract_links(cls, content: str, format: DocFormat) -> List[str]:
        links = []
        patterns = cls.LINK_PATTERNS.get(format, [])
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            links.extend(match.group(1) for match in matches)
        return links

class DocumentationAgent(BaseAgent):
    def _default_config(self) -> AgentConfig:
        return AgentConfig(
            capabilities={AgentCapability.DOCUMENTATION},
            file_patterns={
                '*.md', '*.markdown', '*.rst', '*.txt',
                '*.wiki', '*.adoc'
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
            doc_format = DocumentationPatterns.get_format(file_path)
            if not doc_format:
                raise AnalysisError(f"Unsupported documentation format: {file_path}")
            
            parsed_content = self._parse_content(content, doc_format)
            sections = self._extract_sections(parsed_content, doc_format)
            quality = await self._assess_quality(content, doc_format)
            coverage = self._analyze_coverage(sections, file_path.stem.lower())
            references = self._analyze_references(content, doc_format)
            recommendations = await self._generate_recommendations(
                sections,
                quality,
                coverage,
                references,
                doc_format
            )
            
            return {
                'format': doc_format.value,
                'sections': [self._section_to_dict(s) for s in sections],
                'quality': quality.__dict__,
                'coverage': coverage,
                'references': references,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Documentation analysis failed for {file_path}: {str(e)}")
            raise AnalysisError(f"Documentation analysis failed: {str(e)}") from e
    
    def _parse_content(self, content: str, format: DocFormat) -> str:
        if format == DocFormat.MARKDOWN:
            return markdown.markdown(content)
        elif format == DocFormat.RST:
            return publish_parts(content, writer_name='html')['html_body']
        return content
    
    def _extract_sections(self, content: str, format: DocFormat) -> List[DocSection]:
        sections = []
        current_section = None
        
        if format in {DocFormat.MARKDOWN, DocFormat.RST}:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('#') or line.startswith('='):
                    if current_section:
                        current_section.end_line = i - 1
                        sections.append(current_section)
                    
                    level = 1
                    if format == DocFormat.MARKDOWN:
                        level = len(re.match(r'^#+', line).group())
                        title = line.lstrip('#').strip()
                    else:
                        title = line.strip()
                        if i + 1 < len(lines):
                            underline = lines[i + 1]
                            level = 1 if '=' in underline else 2
                    
                    current_section = DocSection(
                        title=title,
                        content='',
                        level=level,
                        start_line=i,
                        end_line=i,
                        subsections=[]
                    )
                elif current_section:
                    current_section.content += line + '\n'
            
            if current_section:
                current_section.end_line = len(lines) - 1
                sections.append(current_section)
        
        return self._organize_sections(sections)
    
    def _organize_sections(self, sections: List[DocSection]) -> List[DocSection]:
        root_sections = []
        section_stack = []
        
        for section in sections:
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()
            
            if section_stack:
                section_stack[-1].subsections.append(section)
            else:
                root_sections.append(section)
            
            section_stack.append(section)
        
        return root_sections
    
    def _section_to_dict(self, section: DocSection) -> Dict:
        return {
            'title': section.title,
            'content': section.content.strip(),
            'level': section.level,
            'start_line': section.start_line,
            'end_line': section.end_line,
            'subsections': [
                self._section_to_dict(s) for s in section.subsections
            ]
        }
    
    async def _assess_quality(
        self,
        content: str,
        format: DocFormat
    ) -> DocQuality:
        # Strip code blocks to avoid false positives
        code_block_pattern, inline_code_pattern = DocumentationPatterns.CODE_BLOCK_PATTERNS.get(
            format, ('', '')
        )
        clean_content = re.sub(code_block_pattern, '', content)
        clean_content = re.sub(inline_code_pattern, '', clean_content)
        
        quality_prompt = (
            f"Analyze this documentation for quality issues:\n{clean_content}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.DOCUMENTATION,
            quality_prompt
        )
        
        quality_data = response.get('quality', {})
        return DocQuality(
            spelling_errors=quality_data.get('spelling_errors', []),
            grammar_issues=quality_data.get('grammar_issues', []),
            broken_links=quality_data.get('broken_links', []),
            missing_sections=quality_data.get('missing_sections', []),
            readability_score=quality_data.get('readability_score', 0.0)
        )
    
    def _analyze_coverage(
        self,
        sections: List[DocSection],
        doc_type: str
    ) -> Dict:
        expected_sections = DocumentationPatterns.COMMON_SECTIONS.get(
            doc_type, set()
        )
        found_sections = {
            s.title.lower() for s in sections
        }
        
        missing_sections = expected_sections - found_sections
        coverage_score = len(found_sections) / max(len(expected_sections), 1)
        
        return {
            'found_sections': list(found_sections),
            'missing_sections': list(missing_sections),
            'coverage_score': coverage_score
        }
    
    def _analyze_references(
        self,
        content: str,
        format: DocFormat
    ) -> Dict:
        links = DocumentationPatterns.extract_links(content, format)
        
        internal_links = [
            link for link in links
            if not link.startswith(('http://', 'https://'))
        ]
        external_links = [
            link for link in links
            if link.startswith(('http://', 'https://'))
        ]
        
        return {
            'total_links': len(links),
            'internal_links': internal_links,
            'external_links': external_links
        }
    
    async def _generate_recommendations(
        self,
        sections: List[DocSection],
        quality: DocQuality,
        coverage: Dict,
        references: Dict,
        format: DocFormat
    ) -> List[Dict]:
        context = {
            'format': format.value,
            'sections_count': len(sections),
            'quality_score': quality.readability_score,
            'coverage_score': coverage['coverage_score'],
            'total_links': references['total_links']
        }
        
        recommendations_prompt = (
            f"Based on the documentation analysis, suggest improvements:\n"
            f"{json.dumps(context, indent=2)}"
        )
        
        response = await self.llm_router.route_request(
            TaskType.DOCUMENTATION,
            recommendations_prompt
        )
        
        return response.get('recommendations', [])