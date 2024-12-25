import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeType(str, Enum):
    FILE = "file"
    COMPONENT = "component"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PACKAGE = "package"
    DATABASE = "database"
    API = "api"
    SERVICE = "service"
    DEPLOYMENT = "deployment"

class EdgeType(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    DEFINES = "defines"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    INHERITS = "inherits"
    DEPLOYS = "deploys"
    CONNECTS = "connects"
    USES = "uses"

@dataclass
class Node:
    id: str
    type: NodeType
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    source: str
    target: str
    type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    bidirectional: bool = False

class KnowledgeGraph:
    """
    Core knowledge graph implementation for storing and analyzing repository insights.
    Uses NetworkX as the underlying graph data structure.
    """
    
    def __init__(self):
        """Initialize the knowledge graph."""
        # Main graph for storing all nodes and relationships
        self.graph = nx.DiGraph()
        
        # Additional graphs for specific analysis
        self.dependency_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.inheritance_graph = nx.DiGraph()
        
        # Node and edge type indices for quick lookup
        self._node_types: Dict[NodeType, Set[str]] = {
            node_type: set() for node_type in NodeType
        }
        self._edge_types: Dict[EdgeType, Set[Tuple[str, str]]] = {
            edge_type: set() for edge_type in EdgeType
        }
        
        # Cache for commonly accessed subgraphs
        self._subgraph_cache: Dict[str, nx.DiGraph] = {}
    
    def add_node(self, node: Node) -> None:
        """
        Add a node to the knowledge graph.
        
        Args:
            node: Node to add
            
        Raises:
            ValueError: If node ID already exists with different type
        """
        try:
            if node.id in self.graph:
                existing_type = self.graph.nodes[node.id]['type']
                if existing_type != node.type:
                    raise ValueError(
                        f"Node {node.id} already exists with type {existing_type}"
                    )
                # Update existing node
                self.graph.nodes[node.id].update(vars(node))
            else:
                # Add new node
                self.graph.add_node(
                    node.id,
                    **vars(node)
                )
                self._node_types[node.type].add(node.id)
            
            # Update specific graphs based on node type
            if node.type in {NodeType.FUNCTION, NodeType.CLASS, NodeType.MODULE}:
                self.call_graph.add_node(node.id, **vars(node))
            elif node.type in {NodeType.PACKAGE, NodeType.COMPONENT}:
                self.dependency_graph.add_node(node.id, **vars(node))
            
            # Clear cached subgraphs
            self._subgraph_cache.clear()
            
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {str(e)}")
            raise
    
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the knowledge graph.
        
        Args:
            edge: Edge to add
            
        Raises:
            ValueError: If source or target nodes don't exist
        """
        try:
            if not (edge.source in self.graph and edge.target in self.graph):
                raise ValueError(
                    f"Source {edge.source} or target {edge.target} node not found"
                )
            
            # Add edge to main graph
            self.graph.add_edge(
                edge.source,
                edge.target,
                **vars(edge)
            )
            self._edge_types[edge.type].add((edge.source, edge.target))
            
            if edge.bidirectional:
                self.graph.add_edge(
                    edge.target,
                    edge.source,
                    **vars(edge)
                )
                self._edge_types[edge.type].add((edge.target, edge.source))
            
            # Update specific graphs based on edge type
            if edge.type == EdgeType.CALLS:
                self.call_graph.add_edge(edge.source, edge.target, **vars(edge))
            elif edge.type == EdgeType.DEPENDS_ON:
                self.dependency_graph.add_edge(edge.source, edge.target, **vars(edge))
            elif edge.type == EdgeType.INHERITS:
                self.inheritance_graph.add_edge(edge.source, edge.target, **vars(edge))
            
            # Clear cached subgraphs
            self._subgraph_cache.clear()
            
        except Exception as e:
            logger.error(
                f"Failed to add edge {edge.source} -> {edge.target}: {str(e)}"
            )
            raise
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node data by ID."""
        return dict(self.graph.nodes[node_id]) if node_id in self.graph else None
    
    def get_edges(self, source: str, target: str = None) -> List[Dict]:
        """Get edges between nodes."""
        if target:
            return [
                dict(self.graph.edges[source, target])
            ] if self.graph.has_edge(source, target) else []
        
        return [
            dict(self.graph.edges[source, t])
            for t in self.graph.neighbors(source)
        ] if source in self.graph else []
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Dict]:
        """Get all nodes of a specific type."""
        return [
            dict(self.graph.nodes[node_id])
            for node_id in self._node_types[node_type]
        ]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Dict]:
        """Get all edges of a specific type."""
        return [
            dict(self.graph.edges[source, target])
            for source, target in self._edge_types[edge_type]
        ]
    
    def get_subgraph(self, node_ids: Set[str]) -> nx.DiGraph:
        """Get subgraph containing specified nodes."""
        cache_key = ','.join(sorted(node_ids))
        if cache_key not in self._subgraph_cache:
            self._subgraph_cache[cache_key] = self.graph.subgraph(node_ids).copy()
        return self._subgraph_cache[cache_key]
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = None
    ) -> List[List[str]]:
        """Find all paths between two nodes."""
        try:
            if max_length:
                return list(nx.all_simple_paths(
                    self.graph,
                    source,
                    target,
                    cutoff=max_length
                ))
            return list(nx.all_simple_paths(
                self.graph,
                source,
                target
            ))
        except nx.NetworkXNoPath:
            return []
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_connected_components(self) -> List[Set[str]]:
        """Get strongly connected components."""
        return list(nx.strongly_connected_components(self.graph))
    
    def get_node_centrality(self, centrality_type: str = 'degree') -> Dict[str, float]:
        """Calculate node centrality."""
        if centrality_type == 'degree':
            return nx.degree_centrality(self.graph)
        elif centrality_type == 'betweenness':
            return nx.betweenness_centrality(self.graph)
        elif centrality_type == 'closeness':
            return nx.closeness_centrality(self.graph)
        elif centrality_type == 'eigenvector':
            return nx.eigenvector_centrality(self.graph)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
    
    def get_node_clustering(self) -> Dict[str, float]:
        """Calculate clustering coefficient for each node."""
        return nx.clustering(self.graph.to_undirected())
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the graph."""
        return list(nx.simple_cycles(self.graph))
    
    def save(self, file_path: Path) -> None:
        """Save graph to file."""
        try:
            data = {
                'nodes': [
                    {
                        'id': node_id,
                        **dict(data)
                    }
                    for node_id, data in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        **dict(data)
                    }
                    for source, target, data in self.graph.edges(data=True)
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save graph to {file_path}: {str(e)}")
            raise
    
    def load(self, file_path: Path) -> None:
        """Load graph from file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Clear existing graph
            self.graph.clear()
            self.dependency_graph.clear()
            self.call_graph.clear()
            self.inheritance_graph.clear()
            self._node_types = {node_type: set() for node_type in NodeType}
            self._edge_types = {edge_type: set() for edge_type in EdgeType}
            self._subgraph_cache.clear()
            
            # Add nodes
            for node_data in data['nodes']:
                node_id = node_data.pop('id')
                self.add_node(Node(id=node_id, **node_data))
            
            # Add edges
            for edge_data in data['edges']:
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                self.add_edge(Edge(source=source, target=target, **edge_data))
                
        except Exception as e:
            logger.error(f"Failed to load graph from {file_path}: {str(e)}")
            raise
    
    def to_visualization_format(self) -> Dict:
        """Convert graph to format suitable for visualization."""
        return {
            'nodes': [
                {
                    'id': node_id,
                    'label': data.get('name', node_id),
                    'type': data.get('type'),
                    'metadata': data.get('metadata', {}),
                    'metrics': data.get('metrics', {})
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'type': data.get('type'),
                    'weight': data.get('weight', 1.0),
                    'metadata': data.get('metadata', {})
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }
    
    def get_summary(self) -> Dict:
        """Get graph summary statistics."""
        return {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'node_types': {
                node_type.value: len(nodes)
                for node_type, nodes in self._node_types.items()
            },
            'edge_types': {
                edge_type.value: len(edges)
                for edge_type, edges in self._edge_types.items()
            },
            'density': nx.density(self.graph),
            'average_degree': sum(d for n, d in self.graph.degree()) / self.graph.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(self.graph.to_undirected()),
            'strongly_connected_components': len(list(nx.strongly_connected_components(self.graph))),
            'weakly_connected_components': len(list(nx.weakly_connected_components(self.graph)))
        }