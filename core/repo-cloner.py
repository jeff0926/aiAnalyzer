"""
Repository Cloner Module

This module handles repository cloning and updating using a streaming approach.
It supports both HTTPS and SSH protocols, with progress tracking and error handling.

Features:
- Stream-based cloning
- Progress tracking
- Concurrent downloads
- Shallow cloning option
- LFS support
- Submodule handling
"""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
import aiohttp
import git
from git.exc import GitCommandError, InvalidGitRepositoryError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloneMethod(Enum):
    """Supported repository cloning methods."""
    HTTPS = "https"
    SSH = "ssh"
    LOCAL = "local"

@dataclass
class CloneConfig:
    """Configuration for repository cloning."""
    shallow: bool = True
    depth: int = 1
    include_lfs: bool = True
    include_submodules: bool = True
    cleanup_on_error: bool = True
    max_retries: int = 3
    timeout: int = 600  # seconds
    
    # Authentication
    ssh_key: Optional[Path] = None
    username: Optional[str] = None
    token: Optional[str] = None

class RepositoryCloner:
    """
    Handles repository cloning and updating with streaming support.
    """
    
    def __init__(self, config: Optional[Union[CloneConfig, Dict]] = None):
        """
        Initialize the repository cloner.
        
        Args:
            config: Cloner configuration settings
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            self.config = (config if isinstance(config, CloneConfig)
                         else CloneConfig(**(config or {})))
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize cloner: {str(e)}")
            raise
        
        # Track cloned repositories
        self._cloned_repos: Set[Path] = set()
        
        # Set up authentication if provided
        self._setup_auth()
    
    def _setup_auth(self):
        """Configure Git authentication."""
        if self.config.ssh_key:
            os.environ['GIT_SSH_COMMAND'] = f'ssh -i {self.config.ssh_key}'
        
        if self.config.username and self.config.token:
            self._auth_string = f'{self.config.username}:{self.config.token}'
        else:
            self._auth_string = None
    
    def _get_clone_url(self, repo_url: str) -> str:
        """
        Get appropriate clone URL based on authentication.
        
        Args:
            repo_url: Original repository URL
            
        Returns:
            URL to use for cloning
            
        Raises:
            ValueError: If URL is invalid
        """
        parsed = urlparse(repo_url)
        
        if parsed.scheme not in {'http', 'https', 'ssh', 'file'}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        
        if self._auth_string and parsed.scheme in {'http', 'https'}:
            # Insert authentication into URL
            return f"{parsed.scheme}://{self._auth_string}@{parsed.netloc}{parsed.path}"
        
        return repo_url
    
    def _get_clone_method(self, repo_url: str) -> CloneMethod:
        """Determine appropriate clone method from URL."""
        parsed = urlparse(repo_url)
        
        if parsed.scheme == 'file':
            return CloneMethod.LOCAL
        elif parsed.scheme == 'ssh' or (
            parsed.scheme in {'http', 'https'} and 
            '@' in parsed.netloc and 
            ':' not in parsed.netloc
        ):
            return CloneMethod.SSH
        else:
            return CloneMethod.HTTPS
    
    async def _stream_clone(
        self,
        repo_url: str,
        local_path: Path,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Clone repository with streaming progress updates.
        
        Args:
            repo_url: Repository URL
            local_path: Local path for clone
            progress_callback: Optional callback for progress updates
            
        Yields:
            Progress information dictionaries
            
        Raises:
            GitCommandError: If cloning fails
        """
        clone_url = self._get_clone_url(repo_url)
        clone_method = self._get_clone_method(clone_url)
        
        logger.info(f"Cloning {repo_url} to {local_path}")
        
        # Prepare clone options
        clone_opts = {
            'progress': progress_callback,
            'recursive': self.config.include_submodules,
        }
        
        if self.config.shallow:
            clone_opts['depth'] = self.config.depth
            
        if self.config.include_lfs:
            clone_opts['enable-lfs'] = True
        
        try:
            repo = git.Repo.clone_from(
                clone_url,
                local_path,
                **clone_opts
            )
            
            self._cloned_repos.add(local_path)
            
            # Track progress
            total_objects = repo.git.count_objects()
            current_objects = 0
            
            while current_objects < total_objects:
                current_objects = len(repo.objects)
                progress = (current_objects / total_objects) * 100
                
                yield {
                    'status': 'cloning',
                    'progress': progress,
                    'current': current_objects,
                    'total': total_objects
                }
                
                await asyncio.sleep(0.1)
            
            yield {
                'status': 'complete',
                'progress': 100,
                'path': str(local_path)
            }
            
        except GitCommandError as e:
            logger.error(f"Clone failed: {str(e)}")
            
            if self.config.cleanup_on_error and local_path.exists():
                shutil.rmtree(local_path)
                
            yield {
                'status': 'error',
                'error': str(e)
            }
            raise
    
    async def clone_or_update(
        self,
        repo_url: str,
        local_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """
        Clone repository or update if it exists.
        
        Args:
            repo_url: Repository URL
            local_path: Optional local path for clone
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to cloned/updated repository
            
        Raises:
            GitCommandError: If clone/update fails
            ValueError: If parameters are invalid
        """
        if not local_path:
            repo_name = urlparse(repo_url).path.split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            local_path = Path('repos') / repo_name
        
        local_path = Path(local_path)
        
        try:
            if local_path.exists():
                # Repository exists, update it
                logger.info(f"Updating existing repository at {local_path}")
                
                try:
                    repo = git.Repo(local_path)
                    
                    # Fetch updates
                    repo.remote().fetch()
                    
                    # Reset to origin's head
                    repo.head.reset('origin/HEAD', index=True, working_tree=True)
                    
                    if self.config.include_submodules:
                        repo.submodule_update(recursive=True)
                    
                    if progress_callback:
                        progress_callback({
                            'status': 'updated',
                            'path': str(local_path)
                        })
                    
                except (GitCommandError, InvalidGitRepositoryError) as e:
                    logger.warning(
                        f"Update failed, re-cloning repository: {str(e)}"
                    )
                    # Remove failed repository
                    shutil.rmtree(local_path)
                    # Proceed to clone
            
            if not local_path.exists():
                # Clone new repository
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                retries = 0
                while retries < self.config.max_retries:
                    try:
                        async for progress in self._stream_clone(
                            repo_url,
                            local_path,
                            progress_callback
                        ):
                            if progress['status'] == 'error':
                                raise GitCommandError(
                                    'clone',
                                    progress['error']
                                )
                        break
                    except GitCommandError as e:
                        retries += 1
                        if retries >= self.config.max_retries:
                            raise
                        logger.warning(
                            f"Clone attempt {retries} failed: {str(e)}"
                        )
                        await asyncio.sleep(1)
            
            return local_path
            
        except Exception as e:
            logger.error(f"Repository clone/update failed: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up cloned repositories."""
        for repo_path in self._cloned_repos:
            if repo_path.exists():
                try:
                    shutil.rmtree(repo_path)
                except Exception as e:
                    logger.error(f"Failed to remove {repo_path}: {str(e)}")
        
        self._cloned_repos.clear()