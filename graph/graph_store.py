"""
Graph Storage Module

This module handles persistence and querying of the knowledge graph.
It provides:
- Graph serialization/deserialization
- Query capabilities
- Caching
- Version tracking
- Export/import functionality
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from graph.graph_core import KnowledgeGraph, Node, Edge, NodeType, EdgeType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphQuery:
    """Query builder for graph filtering and traversal."""
    
    def __init__(self, graph: KnowledgeGraph):
        """Initialize with target graph."""
        self.graph = graph
        self._node_filters: List[callable] = []
        self._edge_filters: List[callable] = []
        self._traversal_rules: List[Dict] = []
        self._limit: Optional[int] = None
        self._order_by: Optional[str] = None
        self._descending: bool = False
    
    def filter_nodes(self, **kwargs) -> 'GraphQuery':
        """Add node property filters."""
        def node_filter(n: Dict) -> bool:
            return all(
                n.get(k) == v for k, v in kwargs.items()
            )
        self._node_filters.append(node_filter)
        return self
    
    def filter_edges(self, **kwargs) -> 'GraphQuery':
        """Add edge property filters."""
        def edge_filter(e: Dict) -> bool:
            return all(
                e.get(k) == v for k, v in kwargs.items()
            )
        self._edge_filters.append(edge_filter)
        return self
    
    def traverse(
        self,
        edge_type: Optional[EdgeType] = None,
        direction: str = 'out',
        min_depth: int = 1,
        max_depth: int = None
    ) -> 'GraphQuery':
        """Add traversal rule."""
        self._traversal_rules.append({
            'edge_type': edge_type,
            'direction': direction,
            'min_depth': min_depth,
            'max_depth': max_depth
        })
        return self
    
    def limit(self, n: int) -> 'GraphQuery':
        """Limit number of results."""
        self._limit = n
        return self
    
    def order_by(self, field: str, descending: bool = False) -> 'GraphQuery':
        """Set result ordering."""
        self._order_by = field
        self._descending = descending
        return self
    
    def execute(self) -> List[Dict]:
        """Execute query and return results."""
        # Start with all nodes
        nodes = set(self.graph.graph.nodes())
        
        # Apply node filters
        for node_filter in self._node_filters:
            nodes = {
                n for n in nodes
                if node_filter(self.graph.graph.nodes[n])
            }
        
        # Apply traversal rules
        for rule in self._traversal_rules:
            nodes = self._traverse_nodes(nodes, rule)
        
        # Get node data
        results = [
            {
                'id': node,
                **self.graph.graph.nodes[node]
            }
            for node in nodes
        ]
        
        # Apply ordering
        if self._order_by:
            results.sort(
                key=lambda x: x.get(self._order_by, ''),
                reverse=self._descending
            )
        
        # Apply limit
        if self._limit:
            results = results[:self._limit]
        
        return results
    
    def _traverse_nodes(
        self,
        start_nodes: Set[str],
        rule: Dict
    ) -> Set[str]:
        """Traverse graph according to rule."""
        result_nodes = set()
        
        for start_node in start_nodes:
            # Get connected nodes based on direction
            if rule['direction'] == 'out':
                connected = nx.descendants(self.graph.graph, start_node)
            elif rule['direction'] == 'in':
                connected = nx.ancestors(self.graph.graph, start_node)
            else:  # both
                connected = nx.descendants(self.graph.graph, start_node)
                connected.update(nx.ancestors(self.graph.graph, start_node))
            
            # Filter by edge type if specified
            if rule['edge_type']:
                connected = {
                    n for n in connected
                    if any(
                        self.graph.graph.edges[e].get('type') == rule['edge_type']
                        for e in self.graph.graph.edges(start_node)
                        if n in e
                    )
                }
            
            # Apply depth constraints
            for node in connected:
                try:
                    depth = nx.shortest_path_length(
                        self.graph.graph,
                        start_node,
                        node
                    )
                    if (depth >= rule['min_depth'] and
                        (rule['max_depth'] is None or
                         depth <= rule['max_depth'])):
                        result_nodes.add(node)
                except nx.NetworkXNoPath:
                    continue
        
        return result_nodes

class GraphStore:
    """
    Handles persistence and querying of knowledge graphs.
    """
    
    def __init__(self, storage_dir: Path):
        """
        Initialize graph store.
        
        Args:
            storage_dir: Directory for graph storage
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.versions_file = storage_dir / 'versions.json'
        self._versions = self._load_versions()
        
        # Cache settings
        self.cache_size = 5
        self._cache: Dict[str, KnowledgeGraph] = {}
    
    def save_graph(
        self,
        graph: KnowledgeGraph,
        version: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save graph to storage.
        
        Args:
            graph: Knowledge graph to save
            version: Version identifier
            metadata: Optional version metadata
        """
        try:
            # Create version directory
            version_dir = self.storage_dir / version
            version_dir.mkdir(exist_ok=True)
            
            # Save graph structure
            graph_file = version_dir / 'graph.pickle'
            with open(graph_file, 'wb') as f:
                pickle.dump(graph.graph, f)
            
            # Update version tracking
            self._versions[version] = {
                'created_at': datetime.now().isoformat(),
                'node_count': graph.graph.number_of_nodes(),
                'edge_count': graph.graph.number_of_edges(),
                'metadata': metadata or {}
            }
            self._save_versions()
            
            # Update cache
            self._cache[version] = graph
            self._prune_cache()
            
            logger.info(f"Saved graph version {version}")
            
        except Exception as e:
            logger.error(f"Failed to save graph version {version}: {str(e)}")
            raise
    
    def load_graph(self, version: str) -> KnowledgeGraph:
        """
        Load graph from storage.
        
        Args:
            version: Version to load
            
        Returns:
            Loaded knowledge graph
            
        Raises:
            ValueError: If version not found
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")
        
        try:
            # Check cache first
            if version in self._cache:
                return self._cache[version]
            
            # Load from storage
            version_dir = self.storage_dir / version
            graph_file = version_dir / 'graph.pickle'
            
            graph = KnowledgeGraph()
            with open(graph_file, 'rb') as f:
                graph.graph = pickle.load(f)
            
            # Update cache
            self._cache[version] = graph
            self._prune_cache()
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to load graph version {version}: {str(e)}")
            raise
    
    def get_versions(self) -> Dict[str, Dict]:
        """Get all version information."""
        return self._versions.copy()
    
    def delete_version(self, version: str) -> None:
        """
        Delete a graph version.
        
        Args:
            version: Version to delete
            
        Raises:
            ValueError: If version not found
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")
        
        try:
            # Remove from storage
            version_dir = self.storage_dir / version
            if version_dir.exists():
                for file in version_dir.iterdir():
                    file.unlink()
                version_dir.rmdir()
            
            # Remove from tracking
            del self._versions[version]
            self._save_versions()
            
            # Remove from cache
            self._cache.pop(version, None)
            
            logger.info(f"Deleted graph version {version}")
            
        except Exception as e:
            logger.error(f"Failed to delete version {version}: {str(e)}")
            raise
    
    def export_graph(
        self,
        version: str,
        format: str = 'json',
        output_file: Optional[Path] = None
    ) -> Optional[str]:
        """
        Export graph in specified format.
        
        Args:
            version: Version to export
            format: Export format ('json', 'graphml', 'gexf')
            output_file: Optional output file
            
        Returns:
            Exported data as string if no output file specified
        """
        graph = self.load_graph(version)
        
        try:
            if format == 'json':
                data = {
                    'nodes': [
                        {
                            'id': n,
                            **graph.graph.nodes[n]
                        }
                        for n in graph.graph.nodes()
                    ],
                    'edges': [
                        {
                            'source': u,
                            'target': v,
                            **graph.graph.edges[u, v]
                        }
                        for u, v in graph.graph.edges()
                    ]
                }
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    return json.dumps(data, indent=2)
                    
            elif format == 'graphml':
                if output_file:
                    nx.write_graphml(graph.graph, output_file)
                else:
                    from io import StringIO
                    output = StringIO()
                    nx.write_graphml(graph.graph, output)
                    return output.getvalue()
                    
            elif format == 'gexf':
                if output_file:
                    nx.write_gexf(graph.graph, output_file)
                else:
                    from io import StringIO
                    output = StringIO()
                    nx.write_gexf(graph.graph, output)
                    return output.getvalue()
                    
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(
                f"Failed to export graph version {version} as {format}: {str(e)}"
            )
            raise
    
    def import_graph(
        self,
        data: Union[str, Path],
        format: str = 'json',
        version: str = None
    ) -> KnowledgeGraph:
        """
        Import graph from external format.
        
        Args:
            data: Import data or file path
            format: Import format ('json', 'graphml', 'gexf')
            version: Optional version to save as
            
        Returns:
            Imported knowledge graph
        """
        try:
            graph = KnowledgeGraph()
            
            if format == 'json':
                if isinstance(data, Path):
                    with open(data) as f:
                        data = json.load(f)
                else:
                    data = json.loads(data)
                    
                # Create graph from JSON data
                for node in data['nodes']:
                    node_id = node.pop('id')
                    graph.graph.add_node(node_id, **node)
                    
                for edge in data['edges']:
                    source = edge.pop('source')
                    target = edge.pop('target')
                    graph.graph.add_edge(source, target, **edge)
                    
            elif format == 'graphml':
                if isinstance(data, str):
                    from io import StringIO
                    data = StringIO(data)
                graph.graph = nx.read_graphml(data)
                
            elif format == 'gexf':
                if isinstance(data, str):
                    from io import StringIO
                    data = StringIO(data)
                graph.graph = nx.read_gexf(data)
                
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # Save imported graph if version specified
            if version:
                self.save_graph(
                    graph,
                    version,
                    {'imported_format': format}
                )
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to import graph: {str(e)}")
            raise
    
    def query(self) -> GraphQuery:
        """Create new graph query."""
        return GraphQuery(self)
    
    def _load_versions(self) -> Dict[str, Dict]:
        """Load version tracking data."""
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                return json.load(f)
        return {}
    
    def _save_versions(self) -> None:
        """Save version tracking data."""
        with open(self.versions_file, 'w') as f:
            json.dump(self._versions, f, indent=2)
    
    def _prune_cache(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        while len(self._cache) > self.cache_size:
            oldest = min(
                self._cache.keys(),
                key=lambda v: self._versions[v]['created_at']
            )
            del self._cache[oldest]