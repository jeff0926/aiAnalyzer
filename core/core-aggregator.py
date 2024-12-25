"""
Results Aggregator Module

This module handles the aggregation of analysis results from various agents into
a coherent final output. It combines insights from code analysis, documentation review,
security scanning, and other sources.

Features:
- Result merging and deduplication
- Conflict resolution
- Priority-based aggregation
- Cross-reference handling
- Export capabilities
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightPriority(Enum):
    """Priority levels for analysis insights."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1

@dataclass
class AnalysisInsight:
    """Individual insight from analysis agents."""
    source: str
    category: str
    priority: InsightPriority
    description: str
    file_path: Optional[str] = None
    line_numbers: List[int] = field(default_factory=list)
    related_insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class ResultsAggregator:
    """
    Aggregates and processes analysis results from multiple agents.
    """
    
    def __init__(self):
        """Initialize the results aggregator."""
        self.insights: List[AnalysisInsight] = []
        self.relationships = nx.DiGraph()
        self._processed_files: Set[str] = set()
    
    def add_insight(self, insight: Union[AnalysisInsight, Dict]):
        """
        Add a new analysis insight.
        
        Args:
            insight: Analysis insight to add
            
        Raises:
            ValueError: If insight is invalid
        """
        try:
            if isinstance(insight, dict):
                # Convert dict to AnalysisInsight
                insight['priority'] = InsightPriority[insight['priority'].upper()]
                insight = AnalysisInsight(**insight)
            
            # Check for duplicates
            if not self._is_duplicate(insight):
                self.insights.append(insight)
                
                # Add relationships
                for related_id in insight.related_insights:
                    self.relationships.add_edge(
                        insight.description,
                        related_id,
                        weight=insight.priority.value
                    )
                
                if insight.file_path:
                    self._processed_files.add(insight.file_path)
                    
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid insight data: {str(e)}")
            raise ValueError(f"Failed to add insight: {str(e)}") from e
    
    def _is_duplicate(self, insight: AnalysisInsight) -> bool:
        """
        Check if an insight is duplicate.
        
        Args:
            insight: Insight to check
            
        Returns:
            True if insight is duplicate, False otherwise
        """
        for existing in self.insights:
            if (
                existing.source == insight.source and
                existing.category == insight.category and
                existing.description == insight.description and
                existing.file_path == insight.file_path
            ):
                # Merge line numbers and related insights
                existing.line_numbers = list(set(
                    existing.line_numbers + insight.line_numbers
                ))
                existing.related_insights = list(set(
                    existing.related_insights + insight.related_insights
                ))
                # Keep highest priority
                existing.priority = max(
                    existing.priority,
                    insight.priority,
                    key=lambda p: p.value
                )
                # Update timestamp
                existing.timestamp = max(existing.timestamp, insight.timestamp)
                return True
        return False
    
    def aggregate(self, results: List[Dict]) -> Dict:
        """
        Aggregate multiple analysis results.
        
        Args:
            results: List of analysis results to aggregate
            
        Returns:
            Aggregated analysis results
            
        Raises:
            ValueError: If results are invalid
        """
        try:
            # Process each result
            for result in results:
                if 'insights' in result:
                    for insight_data in result['insights']:
                        self.add_insight(insight_data)
                
                if 'relationships' in result:
                    for rel in result['relationships']:
                        self.relationships.add_edge(
                            rel['source'],
                            rel['target'],
                            **rel.get('attributes', {})
                        )
            
            return self.get_summary()
            
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            raise ValueError(f"Failed to aggregate results: {str(e)}") from e
    
    def get_summary(self) -> Dict:
        """
        Generate summary of aggregated results.
        
        Returns:
            Dictionary containing aggregated analysis results
        """
        # Group insights by priority
        insights_by_priority = {
            priority: [] for priority in InsightPriority
        }
        
        for insight in self.insights:
            insights_by_priority[insight.priority].append({
                'source': insight.source,
                'category': insight.category,
                'description': insight.description,
                'file_path': insight.file_path,
                'line_numbers': insight.line_numbers,
                'related_insights': insight.related_insights,
                'metadata': insight.metadata,
                'timestamp': insight.timestamp.isoformat()
            })
        
        # Calculate metrics
        critical_count = len(insights_by_priority[InsightPriority.CRITICAL])
        high_count = len(insights_by_priority[InsightPriority.HIGH])
        total_files = len(self._processed_files)
        
        # Generate relationship data
        relationships = []
        for source, target, data in self.relationships.edges(data=True):
            relationships.append({
                'source': source,
                'target': target,
                'weight': data.get('weight', 1),
                'type': data.get('type', 'related')
            })
        
        return {
            'summary': {
                'total_insights': len(self.insights),
                'critical_insights': critical_count,
                'high_priority_insights': high_count,
                'files_analyzed': total_files,
                'generated_at': datetime.now().isoformat()
            },
            'insights': {
                priority.name.lower(): insights
                for priority, insights in insights_by_priority.items()
            },
            'relationships': relationships,
            'metrics': {
                'complexity_score': self._calculate_complexity_score(),
                'risk_score': self._calculate_risk_score(),
                'coverage': self._calculate_coverage()
            }
        }
    
    def _calculate_complexity_score(self) -> float:
        """Calculate overall complexity score."""
        if not self.insights:
            return 0.0
            
        # Consider number of insights and their relationships
        insight_count = len(self.insights)
        relationship_count = self.relationships.number_of_edges()
        
        # Basic complexity score
        base_score = min(100, (insight_count * 10 + relationship_count * 5) / 2)
        
        # Adjust based on critical and high priority insights
        critical_count = sum(
            1 for i in self.insights
            if i.priority == InsightPriority.CRITICAL
        )
        high_count = sum(
            1 for i in self.insights
            if i.priority == InsightPriority.HIGH
        )
        
        adjustment = (critical_count * 15 + high_count * 10) / max(insight_count, 1)
        
        return round(min(100, base_score + adjustment), 2)
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score."""
        if not self.insights:
            return 0.0
            
        # Weight insights by priority
        weights = {
            InsightPriority.CRITICAL: 10,
            InsightPriority.HIGH: 7,
            InsightPriority.MEDIUM: 4,
            InsightPriority.LOW: 2,
            InsightPriority.INFO: 1
        }
        
        weighted_sum = sum(
            weights[i.priority] for i in self.insights
        )
        max_possible = len(self.insights) * weights[InsightPriority.CRITICAL]
        
        return round((weighted_sum / max_possible) * 100, 2)
    
    def _calculate_coverage(self) -> float:
        """Calculate analysis coverage percentage."""
        if not self._processed_files:
            return 0.0
            
        # This is a placeholder - actual coverage calculation would
        # need information about total files in repository
        return round(len(self._processed_files) * 100 / max(len(self._processed_files), 1), 2)
    
    def generate_summary(self, output_path: Path):
        """
        Generate and save analysis summary.
        
        Args:
            output_path: Path to save summary
            
        Raises:
            IOError: If writing fails
        """
        try:
            summary = self.get_summary()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Summary saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save summary: {str(e)}")
            raise IOError(f"Failed to save summary: {str(e)}") from e
    
    def export_graph(self, output_path: Path):
        """
        Export relationship graph.
        
        Args:
            output_path: Path to save graph
            
        Raises:
            IOError: If export fails
        """
        try:
            # Convert graph to format suitable for visualization
            graph_data = nx.node_link_data(self.relationships)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
                
            logger.info(f"Graph exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export graph: {str(e)}")
            raise IOError(f"Failed to export graph: {str(e)}") from e
    
    def clear(self):
        """Clear all aggregated results."""
        self.insights.clear()
        self.relationships.clear()
        self._processed_files.clear()