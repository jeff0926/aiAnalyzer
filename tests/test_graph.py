import unittest
from graph.graph_core import KnowledgeGraph, Node, Edge
from graph.graph_algorithms import GraphAlgorithms
from graph.graph_store import GraphStore
from datetime import datetime
from pathlib import Path
import os


class TestGraphComponents(unittest.TestCase):
    def setUp(self):
        """Set up test environment with a sample graph."""
        self.graph = KnowledgeGraph()
        self.graph.add_node("1", type="component", name="Node1")
        self.graph.add_node("2", type="component", name="Node2")
        self.graph.add_edge("1", "2", type="depends_on")
        self.graph_algorithms = GraphAlgorithms(self.graph)
        self.graph_store = GraphStore("./test_cache")

    def tearDown(self):
        """Clean up test storage."""
        if Path("./test_cache").exists():
            for file in Path("./test_cache").glob("*"):
                os.remove(file)
            os.rmdir("./test_cache")

    def test_add_and_get_node(self):
        """Test adding and retrieving a node."""
        node = self.graph.get_node("1")
        self.assertIsNotNone(node)
        self.assertEqual(node["type"], "component")

    def test_add_and_get_edge(self):
        """Test adding and retrieving an edge."""
        edge = self.graph.get_edge("1", "2")
        self.assertIsNotNone(edge)
        self.assertEqual(edge["type"], "depends_on")

    def test_graph_metrics(self):
        """Test graph metrics like node and edge count."""
        metrics = self.graph_algorithms.calculate_metrics()
        self.assertEqual(metrics["node_count"], 2)
        self.assertEqual(metrics["edge_count"], 1)

    def test_centrality_analysis(self):
        """Test centrality analysis."""
        centrality = self.graph_algorithms.calculate_centrality()
        self.assertIn("1", centrality)
        self.assertIn("2", centrality)

    def test_store_and_load_graph(self):
        """Test storing and loading the graph."""
        self.graph_store.save_graph(self.graph, version="v1.0")
        loaded_graph = self.graph_store.load_graph("v1.0")
        self.assertEqual(loaded_graph.graph.number_of_nodes(), 2)
        self.assertEqual(loaded_graph.graph.number_of_edges(), 1)

    def test_version_management(self):
        """Test graph version tracking."""
        self.graph_store.save_graph(self.graph, version="v1.0")
        versions = self.graph_store.get_versions()
        self.assertIn("v1.0", versions)

    def test_query_graph(self):
        """Test querying the graph."""
        query = self.graph_store.query()
        query.filter_nodes(type="component")
        results = query.execute()
        self.assertEqual(len(results), 2)

    def test_delete_version(self):
        """Test deleting a graph version."""
        self.graph_store.save_graph(self.graph, version="v1.0")
        self.graph_store.delete_version("v1.0")
        versions = self.graph_store.get_versions()
        self.assertNotIn("v1.0", versions)

    def test_import_export_graph(self):
        """Test importing and exporting graphs."""
        export_data = self.graph_store.export_graph(version="v1.0", format="json")
        self.assertIsNotNone(export_data)

        imported_graph = self.graph_store.import_graph(export_data, format="json", version="v2.0")
        self.assertEqual(imported_graph.graph.number_of_nodes(), 2)


if __name__ == "__main__":
    unittest.main()
