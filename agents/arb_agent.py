"""
Graph Analysis Algorithms Module

This module provides advanced graph analysis algorithms for the knowledge graph.
It includes algorithms for:
- Impact analysis
- Dependency chains
- Coupling detection
- Modularity analysis
- Change propagation
- Architectural patterns
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms import community

from graph.graph_core import KnowledgeGraph, NodeType, EdgeType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(str, Enum):
    IMPACT = "impact"
    DEPENDENCY = "dependency"
    COUPLING = "coupling"
    MODULARITY = "modularity"
    CHANGE = "change"
    PATTERN = "pattern"

@dataclass
class ImpactAnalysis:
    """Impact analysis results."""
    affected_nodes: List[str]
    impact_paths: List[List[str]]
    severity: Dict[str, float]
    propagation_depth: Dict[str, int]
    critical_paths: List[List[str]]

@dataclass
class DependencyChain:
    """Dependency chain analysis results."""
    chains: List[List[str]]
    circular_dependencies: List[List[str]]
    dependency_depth: Dict[str, int]
    bottlenecks: List[str]
    stability_metrics: Dict[str, float]

@dataclass
class CouplingAnalysis:
    """Coupling analysis results."""
    coupled_pairs: List[Tuple[str, str]]
    coupling_strength: Dict[Tuple[str, str], float]
    high_coupling_areas: List[Set[str]]
    coupling_metrics: Dict[str, float]
    refactoring_suggestions: List[Dict]

@dataclass
class ModularityAnalysis:
    """Modularity analysis results."""
    modules: List[Set[str]]
    modularity_score: float
    cohesion_metrics: Dict[str, float]
    interface_complexity: Dict[str, float]
    improvement_suggestions: List[Dict]

class GraphAnalyzer:
    """
    Advanced graph analysis algorithms for repository insights.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        """
        Initialize the graph analyzer.
        
        Args:
            graph: Knowledge graph to analyze
        """
        self.graph = graph
        self.nx_graph = graph.graph
        
        # Cache for expensive computations
        self._betweenness_cache = None
        self._modularity_cache = None
        self._community_cache = None
    
    def analyze_impact(
        self,
        node_ids: List[str],
        max_depth: Optional[int] = None
    ) -> ImpactAnalysis:
        """
        Perform impact analysis starting from given nodes.
        
        Args:
            node_ids: Starting nodes for analysis
            max_depth: Maximum propagation depth
            
        Returns:
            Impact analysis results
            
        Raises:
            ValueError: If any node ID is not found
        """
        try:
            # Validate nodes
            for node_id in node_ids:
                if node_id not in self.nx_graph:
                    raise ValueError(f"Node not found: {node_id}")
            
            # Find affected nodes and paths
            affected_nodes = set()
            impact_paths = []
            propagation_depth = {}
            
            for node_id in node_ids:
                # Perform BFS to find affected nodes
                if max_depth:
                    bfs_tree = nx.bfs_tree(self.nx_graph, node_id, depth_limit=max_depth)
                else:
                    bfs_tree = nx.bfs_tree(self.nx_graph, node_id)
                
                affected_nodes.update(bfs_tree.nodes())
                
                # Find paths to affected nodes
                for target in bfs_tree.nodes():
                    if target != node_id:
                        paths = list(nx.all_simple_paths(
                            bfs_tree,
                            node_id,
                            target
                        ))
                        impact_paths.extend(paths)
                        propagation_depth[target] = len(paths[0]) - 1
            
            # Calculate impact severity
            severity = self._calculate_impact_severity(
                affected_nodes,
                impact_paths
            )
            
            # Identify critical paths
            critical_paths = self._identify_critical_paths(
                impact_paths,
                severity
            )
            
            return ImpactAnalysis(
                affected_nodes=list(affected_nodes),
                impact_paths=impact_paths,
                severity=severity,
                propagation_depth=propagation_depth,
                critical_paths=critical_paths
            )
            
        except Exception as e:
            logger.error(f"Impact analysis failed: {str(e)}")
            raise
    
    def _calculate_impact_severity(
        self,
        affected_nodes: Set[str],
        impact_paths: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate impact severity for affected nodes."""
        severity = {}
        
        # Get node importance metrics
        if not self._betweenness_cache:
            self._betweenness_cache = nx.betweenness_centrality(self.nx_graph)
        
        for node in affected_nodes:
            # Combine multiple factors for severity
            centrality = self._betweenness_cache[node]
            incoming = len(list(self.nx_graph.predecessors(node)))
            outgoing = len(list(self.nx_graph.successors(node)))
            
            # Calculate severity score (0-1)
            severity[node] = min(1.0, (
                centrality * 0.4 +  # Node importance
                (incoming / (incoming + outgoing + 1)) * 0.3 +  # Dependencies
                (outgoing / (incoming + outgoing + 1)) * 0.3  # Dependents
            ))
        
        return severity
    
    def _identify_critical_paths(
        self,
        paths: List[List[str]],
        severity: Dict[str, float]
    ) -> List[List[str]]:
        """Identify critical impact propagation paths."""
        path_scores = []
        
        for path in paths:
            # Calculate path criticality score
            score = sum(severity.get(node, 0) for node in path) / len(path)
            path_scores.append((score, path))
        
        # Return top 20% most critical paths
        path_scores.sort(reverse=True)
        critical_count = max(1, len(path_scores) // 5)
        return [path for _, path in path_scores[:critical_count]]
    def analyze_dependencies(
        self,
        node_ids: Optional[List[str]] = None
    ) -> DependencyChain:
        """
        Analyze dependency chains in the graph.
        
        Args:
            node_ids: Optional specific nodes to analyze
            
        Returns:
            Dependency chain analysis results
        """
        try:
            # Get relevant subgraph
            if node_ids:
                subgraph = self.graph.get_subgraph(set(node_ids))
            else:
                subgraph = self.nx_graph
            
            # Find all dependency chains
            chains = self._find_dependency_chains(subgraph)
            
            # Find circular dependencies
            circular = list(nx.simple_cycles(subgraph))
            
            # Calculate dependency depth
            depth = self._calculate_dependency_depth(subgraph)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(subgraph)
            
            # Calculate stability metrics
            stability = self._calculate_stability_metrics(subgraph)
            
            return DependencyChain(
                chains=chains,
                circular_dependencies=circular,
                dependency_depth=depth,
                bottlenecks=bottlenecks,
                stability_metrics=stability
            )
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {str(e)}")
            raise
    
    def _find_dependency_chains(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find all significant dependency chains."""
        chains = []
        nodes = list(graph.nodes())
        
        for source in nodes:
            for target in nodes:
                if source != target:
                    # Find all paths between source and target
                    try:
                        paths = list(nx.all_simple_paths(
                            graph,
                            source,
                            target,
                            cutoff=10  # Limit path length for performance
                        ))
                        if paths:
                            # Filter out redundant paths
                            unique_paths = self._filter_redundant_paths(paths)
                            chains.extend(unique_paths)
                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        continue
        
        return chains
    
    def _filter_redundant_paths(self, paths: List[List[str]]) -> List[List[str]]:
        """Filter out redundant dependency paths."""
        if not paths:
            return []
        
        # Sort paths by length
        paths.sort(key=len)
        
        # Keep paths that aren't subsets of longer paths
        unique_paths = [paths[0]]
        for path in paths[1:]:
            path_set = set(path)
            if not any(path_set.issuperset(set(existing)) for existing in unique_paths):
                unique_paths.append(path)
        
        return unique_paths
    
    def _calculate_dependency_depth(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Calculate dependency depth for each node."""
        depth = {}
        
        for node in graph.nodes():
            # Calculate longest path from this node
            try:
                paths = nx.single_source_shortest_path_length(graph, node)
                depth[node] = max(paths.values()) if paths else 0
            except (nx.NetworkXError, ValueError):
                depth[node] = 0
        
        return depth
    
    def _identify_bottlenecks(self, graph: nx.DiGraph) -> List[str]:
        """Identify dependency bottlenecks."""
        bottlenecks = []
        
        # Calculate betweenness centrality if not cached
        if not self._betweenness_cache:
            self._betweenness_cache = nx.betweenness_centrality(graph)
        
        # Calculate additional metrics
        degree = dict(graph.degree())
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())
        
        for node, centrality in self._betweenness_cache.items():
            # Node is a bottleneck if it has:
            # 1. High betweenness centrality
            # 2. High degree relative to graph size
            # 3. Both incoming and outgoing dependencies
            if (centrality > 0.4 and
                degree[node] > graph.number_of_nodes() / 5 and
                in_degree[node] > 0 and
                out_degree[node] > 0):
                
                bottlenecks.append(node)
        
        return sorted(
            bottlenecks,
            key=lambda x: self._betweenness_cache[x],
            reverse=True
        )
    
    def _calculate_stability_metrics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate stability metrics for components."""
        metrics = {}
        
        for node in graph.nodes():
            # Get dependencies
            incoming = list(graph.predecessors(node))
            outgoing = list(graph.successors(node))
            
            # Calculate base stability (0-1)
            # More outgoing dependencies = less stable
            base_stability = 1 / (1 + len(outgoing))
            
            # Adjust for various factors
            adjustments = 1.0
            
            # Circular dependencies reduce stability
            if any(node in graph.predecessors(dep) for dep in outgoing):
                adjustments *= 0.8
            
            # Many incoming dependencies suggest importance
            if len(incoming) > graph.number_of_nodes() / 10:
                adjustments *= 0.9
            
            # High betweenness suggests critical component
            if self._betweenness_cache[node] > 0.5:
                adjustments *= 0.9
            
            # Calculate final stability score
            metrics[node] = base_stability * adjustments
        
        return metrics
    
    def _analyze_dependency_impact(
        self,
        node: str,
        graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """Analyze the impact of a node's dependencies."""
        return {
            'direct_dependents': list(graph.successors(node)),
            'direct_dependencies': list(graph.predecessors(node)),
            'dependency_count': graph.in_degree(node),
            'dependent_count': graph.out_degree(node),
            'circular': any(
                node in graph.predecessors(dep)
                for dep in graph.successors(node)
            ),
            'centrality': self._betweenness_cache[node],
            'risk_score': self._calculate_dependency_risk(node, graph)
        }
    
    def _calculate_dependency_risk(
        self,
        node: str,
        graph: nx.DiGraph
    ) -> float:
        """Calculate risk score for a node's dependencies."""
        # Factors that increase risk:
        # 1. Number of dependencies
        # 2. Circular dependencies
        # 3. Critical path position
        # 4. Stability of dependencies
        
        risk_score = 0.0
        
        # Dependency count risk
        dep_count = graph.in_degree(node)
        risk_score += min(0.4, dep_count / 20)
        
        # Circular dependency risk
        if any(node in graph.predecessors(dep) for dep in graph.successors(node)):
            risk_score += 0.2
        
        # Critical path risk
        if self._betweenness_cache[node] > 0.5:
            risk_score += 0.2
        
        # Dependency stability risk
        dep_stability = 0.0
        deps = list(graph.predecessors(node))
        if deps:
            stability_metrics = self._calculate_stability_metrics(graph)
            dep_stability = sum(stability_metrics[dep] for dep in deps) / len(deps)
            risk_score += (1 - dep_stability) * 0.2
        
        return min(1.0, risk_score)