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
    
    def analyze_coupling(
        self,
        min_coupling: float = 0.5
    ) -> CouplingAnalysis:
        """
        Analyze component coupling in the graph.
        
        Args:
            min_coupling: Minimum coupling strength threshold (0-1)
            
        Returns:
            Coupling analysis results
        """
        try:
            # Calculate coupling between all node pairs
            coupled_pairs = []
            coupling_strength = {}
            
            # Get nodes by type for more accurate coupling analysis
            components = self._get_component_nodes()
            
            # Analyze coupling between components
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    strength = self._calculate_coupling_strength(comp1, comp2)
                    if strength >= min_coupling:
                        pair = (comp1, comp2)
                        coupled_pairs.append(pair)
                        coupling_strength[pair] = strength
            
            # Find highly coupled areas
            high_coupling_areas = self._find_coupled_communities()
            
            # Calculate coupling metrics
            coupling_metrics = self._calculate_coupling_metrics()
            
            # Generate refactoring suggestions
            suggestions = self._generate_coupling_suggestions(
                coupled_pairs,
                coupling_strength,
                high_coupling_areas
            )
            
            return CouplingAnalysis(
                coupled_pairs=coupled_pairs,
                coupling_strength=coupling_strength,
                high_coupling_areas=high_coupling_areas,
                coupling_metrics=coupling_metrics,
                refactoring_suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Coupling analysis failed: {str(e)}")
            raise
    
    def _get_component_nodes(self) -> List[str]:
        """Get nodes that represent architectural components."""
        component_types = {
            NodeType.COMPONENT,
            NodeType.SERVICE,
            NodeType.MODULE,
            NodeType.PACKAGE
        }
        
        return [
            node for node, data in self.nx_graph.nodes(data=True)
            if NodeType(data.get('type', '')) in component_types
        ]
    
    def _calculate_coupling_strength(
        self,
        node1: str,
        node2: str
    ) -> float:
        """
        Calculate coupling strength between two nodes.
        
        Considers:
        1. Direct dependencies
        2. Shared dependencies
        3. Shared dependents
        4. Interface complexity
        5. Data coupling
        """
        # Get node data
        node1_data = self.nx_graph.nodes[node1]
        node2_data = self.nx_graph.nodes[node2]
        
        # Calculate different coupling factors
        structural_coupling = self._calculate_structural_coupling(node1, node2)
        dependency_coupling = self._calculate_dependency_coupling(node1, node2)
        interface_coupling = self._calculate_interface_coupling(node1_data, node2_data)
        data_coupling = self._calculate_data_coupling(node1_data, node2_data)
        
        # Combine factors with weights
        coupling_strength = (
            structural_coupling * 0.4 +
            dependency_coupling * 0.3 +
            interface_coupling * 0.2 +
            data_coupling * 0.1
        )
        
        return min(1.0, coupling_strength)
    
    def _calculate_structural_coupling(
        self,
        node1: str,
        node2: str
    ) -> float:
        """Calculate structural coupling based on direct connections."""
        # Check direct dependencies
        direct = (
            self.nx_graph.has_edge(node1, node2) or
            self.nx_graph.has_edge(node2, node1)
        )
        
        # Check path lengths
        try:
            path_length = nx.shortest_path_length(self.nx_graph, node1, node2)
        except nx.NetworkXNoPath:
            path_length = float('inf')
        
        # Calculate structural coupling score
        if direct:
            return 1.0
        elif path_length < float('inf'):
            return max(0.0, 1.0 - (path_length - 1) * 0.3)
        return 0.0
    
    def _calculate_dependency_coupling(
        self,
        node1: str,
        node2: str
    ) -> float:
        """Calculate coupling based on shared dependencies."""
        # Get dependencies and dependents
        deps1 = set(self.nx_graph.predecessors(node1))
        deps2 = set(self.nx_graph.predecessors(node2))
        deps_on1 = set(self.nx_graph.successors(node1))
        deps_on2 = set(self.nx_graph.successors(node2))
        
        # Calculate shared dependencies ratio
        total_deps = len(deps1.union(deps2))
        shared_deps = len(deps1.intersection(deps2))
        deps_ratio = shared_deps / max(1, total_deps)
        
        # Calculate shared dependents ratio
        total_deps_on = len(deps_on1.union(deps_on2))
        shared_deps_on = len(deps_on1.intersection(deps_on2))
        deps_on_ratio = shared_deps_on / max(1, total_deps_on)
        
        # Combine ratios with weights
        return deps_ratio * 0.6 + deps_on_ratio * 0.4
    
    def _calculate_interface_coupling(
        self,
        node1_data: Dict,
        node2_data: Dict
    ) -> float:
        """Calculate coupling based on interface complexity."""
        # Get interface definitions
        interfaces1 = node1_data.get('interfaces', [])
        interfaces2 = node2_data.get('interfaces', [])
        
        if not interfaces1 or not interfaces2:
            return 0.0
        
        # Compare interface signatures
        shared_methods = 0
        similar_methods = 0
        total_methods = 0
        
        for i1 in interfaces1:
            total_methods += 1
            for i2 in interfaces2:
                if i1['signature'] == i2['signature']:
                    shared_methods += 1
                elif self._are_signatures_similar(i1['signature'], i2['signature']):
                    similar_methods += 1
        
        # Calculate interface coupling score
        return min(1.0, (
            (shared_methods / max(1, total_methods)) * 0.7 +
            (similar_methods / max(1, total_methods)) * 0.3
        ))
    
    def _calculate_data_coupling(
        self,
        node1_data: Dict,
        node2_data: Dict
    ) -> float:
        """Calculate coupling based on shared data structures."""
        # Get data structures
        data1 = node1_data.get('data_structures', [])
        data2 = node2_data.get('data_structures', [])
        
        if not data1 or not data2:
            return 0.0
        
        # Compare data structure similarity
        shared_structures = 0
        similar_structures = 0
        total_structures = len(data1)
        
        for d1 in data1:
            for d2 in data2:
                if d1['schema'] == d2['schema']:
                    shared_structures += 1
                elif self._are_schemas_similar(d1['schema'], d2['schema']):
                    similar_structures += 1
        
        # Calculate data coupling score
        return min(1.0, (
            (shared_structures / max(1, total_structures)) * 0.7 +
            (similar_structures / max(1, total_structures)) * 0.3
        ))
    
    def _are_signatures_similar(self, sig1: str, sig2: str) -> bool:
        """Check if two method signatures are similar."""
        # Remove whitespace and normalize
        sig1 = ''.join(sig1.split())
        sig2 = ''.join(sig2.split())
        
        # Compare parameter types and return type
        return (
            sig1.split('->')[-1] == sig2.split('->')[-1] and
            len(sig1.split('(')[1].split(',')) == len(sig2.split('(')[1].split(','))
        )
    
    def _are_schemas_similar(self, schema1: Dict, schema2: Dict) -> bool:
        """Check if two data schemas are similar."""
        # Compare field types
        fields1 = set(schema1.keys())
        fields2 = set(schema2.keys())
        
        shared_fields = len(fields1.intersection(fields2))
        total_fields = len(fields1.union(fields2))
        
        return shared_fields / total_fields > 0.7
    
    def _find_coupled_communities(self) -> List[Set[str]]:
        """Find communities of highly coupled components."""
        if not self._community_cache:
            # Create weighted graph for community detection
            weighted_graph = nx.Graph()
            
            for node in self.nx_graph.nodes():
                weighted_graph.add_node(node)
            
            # Add edges with coupling strength as weight
            for n1 in self.nx_graph.nodes():
                for n2 in self.nx_graph.nodes():
                    if n1 < n2:  # Avoid duplicate calculations
                        strength = self._calculate_coupling_strength(n1, n2)
                        if strength > 0:
                            weighted_graph.add_edge(n1, n2, weight=strength)
            
            # Use Louvain method for community detection
            self._community_cache = list(
                community.louvain_communities(weighted_graph)
            )
        
        return self._community_cache
    
    def _generate_coupling_suggestions(
        self,
        coupled_pairs: List[Tuple[str, str]],
        coupling_strength: Dict[Tuple[str, str], float],
        communities: List[Set[str]]
    ) -> List[Dict]:
        """Generate refactoring suggestions for coupled components."""
        suggestions = []
        
        # Check highly coupled pairs
        for pair in coupled_pairs:
            strength = coupling_strength[pair]
            if strength > 0.8:
                suggestions.append({
                    'type': 'merge_components',
                    'components': pair,
                    'reason': 'Extremely high coupling indicates components '
                             'should potentially be merged',
                    'priority': 'high',
                    'coupling_strength': strength
                })
            elif strength > 0.6:
                suggestions.append({
                    'type': 'extract_interface',
                    'components': pair,
                    'reason': 'High coupling suggests need for clear interface '
                             'or shared abstraction',
                    'priority': 'medium',
                    'coupling_strength': strength
                })
        
        # Check communities for refactoring opportunities
        for community_nodes in communities:
            if len(community_nodes) > 3:
                # Large highly-coupled community
                suggestions.append({
                    'type': 'refactor_community',
                    'components': list(community_nodes),
                    'reason': 'Large group of coupled components suggests need '
                             'for architectural restructuring',
                    'priority': 'high',
                    'size': len(community_nodes)
                })
        
        return suggestions
    
    def analyze_modularity(self) -> ModularityAnalysis:
        """
        Analyze system modularity and suggest improvements.
        
        Returns:
            ModularityAnalysis containing modularity metrics and suggestions
        """
        try:
            # Detect modules using community detection
            if not self._community_cache:
                self._community_cache = list(
                    community.louvain_communities(self.nx_graph.to_undirected())
                )
            modules = self._community_cache
            
            # Calculate modularity score
            modularity_score = self._calculate_modularity_score(modules)
            
            # Calculate cohesion metrics
            cohesion_metrics = self._calculate_cohesion_metrics(modules)
            
            # Analyze interface complexity
            interface_complexity = self._analyze_interface_complexity(modules)
            
            # Generate improvement suggestions
            suggestions = self._generate_modularity_suggestions(
                modules,
                cohesion_metrics,
                interface_complexity
            )
            
            return ModularityAnalysis(
                modules=modules,
                modularity_score=modularity_score,
                cohesion_metrics=cohesion_metrics,
                interface_complexity=interface_complexity,
                improvement_suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Modularity analysis failed: {str(e)}")
            raise
    
    def _calculate_modularity_score(self, modules: List[Set[str]]) -> float:
        """Calculate overall modularity score."""
        if not modules:
            return 0.0
        
        try:
            # Convert to format required by networkx
            communities = {i: module for i, module in enumerate(modules)}
            
            # Calculate Newman-Girvan modularity
            modularity = community.modularity(
                self.nx_graph.to_undirected(),
                communities.values()
            )
            
            return max(0.0, modularity)
            
        except Exception as e:
            logger.warning(f"Modularity calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_cohesion_metrics(
        self,
        modules: List[Set[str]]
    ) -> Dict[str, float]:
        """
        Calculate cohesion metrics for each module.
        
        Metrics include:
        - Internal density
        - External coupling
        - Interface complexity
        - Dependency patterns
        """
        metrics = {}
        
        for i, module in enumerate(modules):
            module_name = f"module_{i}"
            
            # Get subgraph for this module
            subgraph = self.nx_graph.subgraph(module)
            
            # Calculate internal cohesion
            internal_edges = subgraph.number_of_edges()
            possible_edges = len(module) * (len(module) - 1) / 2
            internal_density = (
                internal_edges / possible_edges if possible_edges > 0 else 0
            )
            
            # Calculate external coupling
            external_edges = sum(
                1 for n1 in module
                for n2 in self.nx_graph.neighbors(n1)
                if n2 not in module
            )
            
            # Calculate interface complexity
            interface_nodes = set(
                n1 for n1 in module
                for n2 in self.nx_graph.neighbors(n1)
                if n2 not in module
            )
            interface_complexity = len(interface_nodes) / len(module)
            
            # Combine metrics
            metrics[module_name] = {
                'size': len(module),
                'internal_density': internal_density,
                'external_coupling': external_edges / len(module),
                'interface_complexity': interface_complexity,
                'cohesion_score': self._calculate_cohesion_score(
                    internal_density,
                    external_edges / len(module),
                    interface_complexity
                )
            }
        
        return metrics
    
    def _calculate_cohesion_score(
        self,
        internal_density: float,
        external_coupling: float,
        interface_complexity: float
    ) -> float:
        """Calculate overall cohesion score from individual metrics."""
        return (
            internal_density * 0.4 +
            (1 - external_coupling) * 0.4 +
            (1 - interface_complexity) * 0.2
        )
    
    def _analyze_interface_complexity(
        self,
        modules: List[Set[str]]
    ) -> Dict[str, float]:
        """
        Analyze interface complexity between modules.
        
        Considers:
        - Number of interface points
        - Dependency patterns
        - Data flow complexity
        """
        complexity = {}
        
        for i, module1 in enumerate(modules):
            for j, module2 in enumerate(modules):
                if i < j:  # Avoid duplicate calculations
                    # Count interface points
                    interface_edges = sum(
                        1 for n1 in module1
                        for n2 in module2
                        if self.nx_graph.has_edge(n1, n2) or
                           self.nx_graph.has_edge(n2, n1)
                    )
                    
                    # Calculate complexity metrics
                    if interface_edges > 0:
                        interface_name = f"interface_{i}_{j}"
                        complexity[interface_name] = {
                            'edge_count': interface_edges,
                            'relative_size': interface_edges / min(len(module1), len(module2)),
                            'bidirectional': any(
                                self.nx_graph.has_edge(n1, n2) and
                                self.nx_graph.has_edge(n2, n1)
                                for n1 in module1
                                for n2 in module2
                            ),
                            'complexity_score': min(1.0, interface_edges / 5)
                        }
        
        return complexity
    
    def _generate_modularity_suggestions(
        self,
        modules: List[Set[str]],
        cohesion_metrics: Dict[str, float],
        interface_complexity: Dict[str, float]
    ) -> List[Dict]:
        """Generate suggestions for improving modularity."""
        suggestions = []
        
        # Check module sizes
        for module_name, metrics in cohesion_metrics.items():
            if metrics['size'] > 10 and metrics['internal_density'] < 0.3:
                suggestions.append({
                    'type': 'split_module',
                    'module': module_name,
                    'reason': 'Large module with low internal cohesion',
                    'priority': 'high',
                    'metrics': metrics
                })
            elif metrics['external_coupling'] > 0.7:
                suggestions.append({
                    'type': 'reduce_coupling',
                    'module': module_name,
                    'reason': 'High external coupling suggests potential design issues',
                    'priority': 'medium',
                    'metrics': metrics
                })
        
        # Check interface complexity
        for interface_name, metrics in interface_complexity.items():
            if metrics['complexity_score'] > 0.7:
                suggestions.append({
                    'type': 'simplify_interface',
                    'interface': interface_name,
                    'reason': 'Complex interface between modules',
                    'priority': 'medium',
                    'metrics': metrics
                })
            elif metrics['bidirectional'] and metrics['edge_count'] > 3:
                suggestions.append({
                    'type': 'refactor_bidirectional',
                    'interface': interface_name,
                    'reason': 'Complex bidirectional dependencies',
                    'priority': 'high',
                    'metrics': metrics
                })
        
        # Check overall modularity
        modularity_score = self._calculate_modularity_score(modules)
        if modularity_score < 0.3:
            suggestions.append({
                'type': 'improve_modularity',
                'reason': 'Low overall modularity score indicates potential architectural issues',
                'priority': 'high',
                'score': modularity_score,
                'recommendations': [
                    'Consider restructuring system boundaries',
                    'Identify and extract common functionality',
                    'Review dependency patterns'
                ]
            })
        
        return suggestions
    
    def analyze_change_impact(
        self,
        node_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze the impact of changing specific nodes.
        
        Args:
            node_ids: List of nodes to analyze
            
        Returns:
            Change impact analysis results
        """
        # First perform impact analysis
        impact = self.analyze_impact(node_ids)
        
        # Analyze module boundaries
        affected_modules = self._find_affected_modules(
            impact.affected_nodes
        )
        
        # Calculate change complexity
        complexity = self._calculate_change_complexity(
            impact.affected_nodes,
            affected_modules
        )
        
        # Generate change plan
        change_plan = self._generate_change_plan(
            node_ids,
            impact,
            complexity
        )
        
        return {
            'impact_analysis': impact,
            'affected_modules': affected_modules,
            'change_complexity': complexity,
            'change_plan': change_plan
        }
    
    def _find_affected_modules(
        self,
        affected_nodes: List[str]
    ) -> Dict[str, Set[str]]:
        """Find which modules are affected by changes."""
        affected_modules = {}
        
        if not self._community_cache:
            self._community_cache = list(
                community.louvain_communities(self.nx_graph.to_undirected())
            )
        
        for i, module in enumerate(self._community_cache):
            affected = module.intersection(set(affected_nodes))
            if affected:
                affected_modules[f"module_{i}"] = affected
        
        return affected_modules
    
    def _calculate_change_complexity(
        self,
        affected_nodes: List[str],
        affected_modules: Dict[str, Set[str]]
    ) -> Dict[str, float]:
        """Calculate complexity metrics for the change."""
        return {
            'scope': len(affected_nodes),
            'module_count': len(affected_modules),
            'interface_changes': sum(
                1 for n1 in affected_nodes
                for n2 in self.nx_graph.neighbors(n1)
                if n2 not in affected_nodes
            ),
            'risk_score': min(1.0, len(affected_nodes) / 20 +
                                  len(affected_modules) / 5)
        }
    
    def _generate_change_plan(
        self,
        target_nodes: List[str],
        impact: ImpactAnalysis,
        complexity: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate a structured change plan."""
        return {
            'phases': [
                {
                    'phase': 1,
                    'description': 'Prepare and validate changes',
                    'tasks': [
                        'Review and update tests for affected components',
                        'Validate interface contracts',
                        'Create change validation plan'
                    ]
                },
                {
                    'phase': 2,
                    'description': 'Implement core changes',
                    'tasks': [
                        f'Update {node}' for node in target_nodes
                    ]
                },
                {
                    'phase': 3,
                    'description': 'Propagate changes',
                    'tasks': [
                        f'Update dependent component {node}'
                        for node in impact.affected_nodes
                        if node not in target_nodes
                    ]
                },
                {
                    'phase': 4,
                    'description': 'Validate changes',
                    'tasks': [
                        'Run full test suite',
                        'Validate all affected interfaces',
                        'Perform integration testing'
                    ]
                }
            ],
            'estimated_complexity': complexity['risk_score'],
            'critical_paths': impact.critical_paths,
            'validation_focus': [
                node for node in impact.affected_nodes
                if impact.severity[node] > 0.7
            ]
        }