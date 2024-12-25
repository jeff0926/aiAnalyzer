"""
Base Analysis Agent Module

This module defines the base class for all analysis agents. It provides common
functionality and interfaces that specialized agents must implement.

Features:
- Abstract interface definition
- Common utilities
- Result validation
- Error handling
- Progress tracking
- Resource management
"""

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.llm.llm_router import LLMRouter, TaskType
from utils.cache import Cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentCapability(Enum):
    """Capabilities that agents can provide."""
    CODE_ANALYSIS = "code"
    SECURITY_SCAN = "security"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "config"
    BUILD = "build"
    INFRASTRUCTURE = "infrastructure"
    DATABASE = "database"
    TEST = "test"

class AnalysisScope(Enum):
    """Scope of analysis to perform."""
    QUICK = "quick"         # Basic analysis
    STANDARD = "standard"   # Normal depth
    DEEP = "deep"          # Thorough analysis

@dataclass
class AgentConfig:
    """Configuration for analysis agents."""
    capabilities: Set[AgentCapability]
    file_patterns: Set[str]
    max_file_size: int = 1024 * 1024  # 1MB
    timeout: int = 300  # seconds
    use_cache: bool = True
    scope: AnalysisScope = AnalysisScope.STANDARD
    extra_config: Dict[str, Any] = field(default_factory=dict)

class AnalysisError(Exception):
    """Base class for analysis-related errors."""
    pass

class FileTypeError(AnalysisError):
    """Error for unsupported file types."""
    pass

class FileSizeError(AnalysisError):
    """Error for files exceeding size limit."""
    pass

class TimeoutError(AnalysisError):
    """Error for analysis timeout."""
    pass

class BaseAgent(abc.ABC):
    """
    Abstract base class for all analysis agents.
    """
    
    def __init__(
        self,
        llm_router: LLMRouter,
        config: Optional[Union[AgentConfig, Dict]] = None
    ):
        """
        Initialize the analysis agent.
        
        Args:
            llm_router: LLM routing component
            config: Agent configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            self.llm_router = llm_router
            self.config = (config if isinstance(config, AgentConfig)
                         else AgentConfig(**config) if config
                         else self._default_config())
            
            # Initialize cache if enabled
            self.cache = Cache() if self.config.use_cache else None
            
            # Analysis state
            self._current_file: Optional[Path] = None
            self._start_time: Optional[datetime] = None
            self._processed_files: Set[Path] = set()
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise ValueError(f"Invalid agent configuration: {str(e)}") from e
    
    @abc.abstractmethod
    def _default_config(self) -> AgentConfig:
        """
        Provide default configuration for the agent.
        
        Returns:
            Default agent configuration
            
        This method must be implemented by all agent subclasses.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def _analyze_content(
        self,
        content: str,
        file_path: Path,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Perform actual content analysis.
        
        Args:
            content: File content to analyze
            file_path: Path of file being analyzed
            context: Optional analysis context
            
        Returns:
            Analysis results
            
        This method must be implemented by all agent subclasses.
        """
        raise NotImplementedError
    
    def can_handle(self, file_path: Path) -> bool:
        """
        Check if agent can handle given file type.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if agent can handle file, False otherwise
        """
        import fnmatch
        return any(
            fnmatch.fnmatch(file_path.name, pattern)
            for pattern in self.config.file_patterns
        )
    
    def _validate_file(self, file_path: Path, content: str):
        """
        Validate file before analysis.
        
        Args:
            file_path: Path to validate
            content: File content
            
        Raises:
            FileTypeError: If file type is not supported
            FileSizeError: If file is too large
        """
        if not self.can_handle(file_path):
            raise FileTypeError(
                f"Agent does not support files matching {file_path.name}"
            )
        
        if len(content.encode('utf-8')) > self.config.max_file_size:
            raise FileSizeError(
                f"File size exceeds limit of {self.config.max_file_size} bytes"
            )
    
    def _check_timeout(self):
        """
        Check if analysis has exceeded timeout.
        
        Raises:
            TimeoutError: If timeout has been exceeded
        """
        if (
            self._start_time and
            (datetime.now() - self._start_time).total_seconds() >
            self.config.timeout
        ):
            raise TimeoutError(
                f"Analysis timeout exceeded ({self.config.timeout}s)"
            )
    
    async def analyze(
        self,
        file_path: Path,
        content: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze file content.
        
        Args:
            file_path: Path of file to analyze
            content: File content
            context: Optional analysis context
            
        Returns:
            Analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        try:
            self._current_file = file_path
            self._start_time = datetime.now()
            
            # Validate file
            self._validate_file(file_path, content)
            
            # Check cache
            if self.cache:
                cached = self.cache.get(str(file_path))
                if cached:
                    logger.debug(f"Using cached results for {file_path}")
                    return cached
            
            # Perform analysis
            results = await self._analyze_content(content, file_path, context)
            
            # Add metadata
            results.update({
                'file_path': str(file_path),
                'agent_type': self.__class__.__name__,
                'capabilities': [c.value for c in self.config.capabilities],
                'analysis_time': (
                    datetime.now() - self._start_time
                ).total_seconds(),
                'analysis_scope': self.config.scope.value
            })
            
            # Cache results
            if self.cache:
                self.cache.set(str(file_path), results)
            
            self._processed_files.add(file_path)
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {str(e)}")
            if isinstance(e, (FileTypeError, FileSizeError, TimeoutError)):
                raise
            raise AnalysisError(f"Analysis failed: {str(e)}") from e
            
        finally:
            self._current_file = None
            self._start_time = None
    
    async def analyze_batch(
        self,
        files: List[Tuple[Path, str]],
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Analyze multiple files.
        
        Args:
            files: List of (path, content) tuples
            context: Optional analysis context
            
        Returns:
            List of analysis results
        """
        tasks = []
        for file_path, content in files:
            if self.can_handle(file_path):
                tasks.append(self.analyze(file_path, content, context))
        
        if not tasks:
            return []
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    @property
    def processed_files(self) -> Set[Path]:
        """Get set of processed file paths."""
        return self._processed_files.copy()
    
    async def cleanup(self):
        """Cleanup agent resources."""
        if self.cache:
            self.cache.clear()
        self._processed_files.clear()
    
    def __str__(self) -> str:
        """Get string representation of agent."""
        capabilities = [c.value for c in self.config.capabilities]
        patterns = list(self.config.file_patterns)
        return (f"{self.__class__.__name__}("
                f"capabilities={capabilities}, "
                f"patterns={patterns})")
    
    def __repr__(self) -> str:
        """Get detailed string representation of agent."""
        return str(self)