"""
LLM Router Module

This module handles intelligent routing between local and cloud LLMs based on task requirements.
It manages model selection, caching, and fallback strategies.

Features:
- Task-based routing logic
- Privacy-aware processing
- Automatic fallback handling
- Response caching
- Cost management
- Rate limiting
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration of supported model types."""
    LOCAL_CODE = "local_code"      # CodeLlama-34b
    LOCAL_GENERAL = "local_gen"    # Llama-2-70b
    CLOUD_GPT4 = "gpt4"           # OpenAI GPT-4
    CLOUD_CLAUDE = "claude"       # Anthropic Claude
    CLOUD_AZURE = "azure"         # Azure OpenAI

class TaskType(Enum):
    """Enumeration of task types for routing decisions."""
    CODE_ANALYSIS = "code"
    SECURITY_AUDIT = "security"
    ARCHITECTURE_REVIEW = "architecture"
    DOCUMENTATION = "documentation"
    GENERAL = "general"

@dataclass
class RoutingConfig:
    """Configuration for LLM routing decisions."""
    privacy_mode: bool = False
    max_cloud_cost: float = 10.0
    cache_ttl: int = 3600  # seconds
    rate_limits: Dict[ModelType, int] = None
    fallback_strategy: str = "local"  # local, cloud, or fail
    
    def __post_init__(self):
        """Initialize default rate limits if not provided."""
        if self.rate_limits is None:
            self.rate_limits = {
                ModelType.CLOUD_GPT4: 100,     # requests per minute
                ModelType.CLOUD_CLAUDE: 120,
                ModelType.CLOUD_AZURE: 150,
                ModelType.LOCAL_CODE: 1000,
                ModelType.LOCAL_GENERAL: 800
            }

class LLMRouter:
    """
    Manages routing decisions between local and cloud LLMs based on task requirements.
    """
    
    def __init__(self, config: Optional[Union[RoutingConfig, Dict]] = None):
        """
        Initialize the LLM router.
        
        Args:
            config: Router configuration settings
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            self.config = (config if isinstance(config, RoutingConfig)
                         else RoutingConfig(**(config or {})))
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize router: {str(e)}")
            raise
            
        # Initialize rate limiting state
        self._request_counts: Dict[ModelType, List[datetime]] = {
            model: [] for model in ModelType
        }
        
        # Initialize cache
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Track costs
        self._total_cost: float = 0.0
        
        # Initialize models (placeholder for actual implementation)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize connections to local and cloud LLM services."""
        # Placeholder for actual model initialization
        # Will be implemented when adding specific model integrations
        pass
    
    def _check_rate_limit(self, model_type: ModelType) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self._request_counts[model_type] = [
            t for t in self._request_counts[model_type]
            if t > minute_ago
        ]
        
        # Check against limit
        return len(self._request_counts[model_type]) < self.config.rate_limits[model_type]
    
    def _update_rate_limit(self, model_type: ModelType):
        """Record a new request for rate limiting."""
        self._request_counts[model_type].append(datetime.now())
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """
        Get cached response if available and not expired.
        
        Args:
            cache_key: Key to look up in cache
            
        Returns:
            Cached response or None if not found/expired
        """
        if cache_key not in self._cache:
            return None
            
        timestamp = self._cache_timestamps[cache_key]
        if (datetime.now() - timestamp).seconds > self.config.cache_ttl:
            # Cache expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
            
        return self._cache[cache_key]
    
    def _cache_response(self, cache_key: str, response: Dict):
        """Cache a response with current timestamp."""
        self._cache[cache_key] = response
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _estimate_cost(self, model_type: ModelType, input_size: int) -> float:
        """
        Estimate cost of request in USD.
        
        Args:
            model_type: Type of model to use
            input_size: Size of input in tokens
            
        Returns:
            Estimated cost in USD
        """
        # Placeholder cost estimation logic
        costs = {
            ModelType.CLOUD_GPT4: 0.03,    # per 1K tokens
            ModelType.CLOUD_CLAUDE: 0.02,
            ModelType.CLOUD_AZURE: 0.02,
            ModelType.LOCAL_CODE: 0.0,
            ModelType.LOCAL_GENERAL: 0.0
        }
        
        return (input_size / 1000) * costs[model_type]
    
    def _select_model(self, task_type: TaskType, input_size: int) -> ModelType:
        """
        Select appropriate model based on task and constraints.
        
        Args:
            task_type: Type of task to process
            input_size: Size of input in tokens
            
        Returns:
            Selected model type
            
        Raises:
            RuntimeError: If no suitable model is available
        """
        # Define preferred models for each task type
        task_models = {
            TaskType.CODE_ANALYSIS: [
                ModelType.LOCAL_CODE,
                ModelType.CLOUD_GPT4,
                ModelType.CLOUD_CLAUDE
            ],
            TaskType.SECURITY_AUDIT: [
                ModelType.CLOUD_GPT4,
                ModelType.CLOUD_CLAUDE,
                ModelType.LOCAL_GENERAL
            ],
            TaskType.ARCHITECTURE_REVIEW: [
                ModelType.CLOUD_CLAUDE,
                ModelType.CLOUD_GPT4,
                ModelType.LOCAL_GENERAL
            ],
            TaskType.DOCUMENTATION: [
                ModelType.LOCAL_GENERAL,
                ModelType.CLOUD_CLAUDE,
                ModelType.LOCAL_CODE
            ],
            TaskType.GENERAL: [
                ModelType.LOCAL_GENERAL,
                ModelType.CLOUD_CLAUDE,
                ModelType.LOCAL_CODE
            ]
        }
        
        for model_type in task_models[task_type]:
            # Check privacy constraints
            if self.config.privacy_mode and model_type in {
                ModelType.CLOUD_GPT4,
                ModelType.CLOUD_CLAUDE,
                ModelType.CLOUD_AZURE
            }:
                continue
                
            # Check rate limits
            if not self._check_rate_limit(model_type):
                continue
                
            # Check cost constraints
            estimated_cost = self._estimate_cost(model_type, input_size)
            if self._total_cost + estimated_cost > self.config.max_cloud_cost:
                continue
                
            return model_type
            
        # Handle fallback strategy
        if self.config.fallback_strategy == "local":
            return ModelType.LOCAL_GENERAL
        elif self.config.fallback_strategy == "cloud":
            return ModelType.CLOUD_CLAUDE
        else:
            raise RuntimeError("No suitable model available")
    
    async def route_request(
        self,
        task_type: Union[TaskType, str],
        input_text: str,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route a request to appropriate LLM and return response.
        
        Args:
            task_type: Type of task for model selection
            input_text: Text input to process
            cache_key: Optional key for response caching
            
        Returns:
            Model response
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If processing fails
        """
        try:
            # Normalize task type
            if isinstance(task_type, str):
                task_type = TaskType(task_type)
            
            # Check cache if key provided
            if cache_key:
                cached = self._get_cached_response(cache_key)
                if cached:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached
            
            # Select model
            input_size = len(input_text.split())  # Simple token estimation
            model_type = self._select_model(task_type, input_size)
            
            # Update rate limiting
            self._update_rate_limit(model_type)
            
            # Process request (placeholder for actual implementation)
            response = await self._process_with_model(model_type, input_text)
            
            # Update cost tracking
            self._total_cost += self._estimate_cost(model_type, input_size)
            
            # Cache response if key provided
            if cache_key:
                self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Request routing failed: {str(e)}")
            raise
    
    async def _process_with_model(
        self,
        model_type: ModelType,
        input_text: str
    ) -> Dict[str, Any]:
        """
        Process request with selected model.
        
        Args:
            model_type: Type of model to use
            input_text: Text input to process
            
        Returns:
            Model response
            
        Raises:
            RuntimeError: If processing fails
        """
        # Placeholder for actual model integration
        # Will be implemented when adding specific model support
        return {
            "model": model_type.value,
            "status": "not_implemented",
            "timestamp": datetime.now().isoformat()
        }
    
    @property
    def total_cost(self) -> float:
        """Get total cost of cloud API usage."""
        return self._total_cost
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        
    async def cleanup(self):
        """Cleanup resources and connections."""
        # Will be implemented when adding specific model integrations
        pass