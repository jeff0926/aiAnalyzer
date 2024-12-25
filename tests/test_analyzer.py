import unittest
from core.analyzer import Analyzer  # Adjust import path if necessary


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test environment and mock data."""
        self.analyzer = Analyzer()
        self.mock_data = {
            "repo_name": "Test Repository",
            "components": [
                {"name": "ComponentA", "type": "class"},
                {"name": "ComponentB", "type": "function"}
            ],
            "issues": [
                {"severity": "high", "description": "Critical bug in ComponentA"}
            ]
        }

    def test_analyze_repository(self):
        """Test repository analysis process."""
        results = self.analyzer.analyze_repository(self.mock_data)
        self.assertIn("summary", results)
        self.assertIn("issues", results)
        self.assertEqual(len(results["issues"]), 1)

    def test_generate_summary(self):
        """Test summary generation."""
        summary = self.analyzer.generate_summary(self.mock_data)
        self.assertTrue(isinstance(summary, str))
        self.assertIn("Test Repository", summary)

    def test_detect_issues(self):
        """Test issue detection."""
        issues = self.analyzer.detect_issues(self.mock_data)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["severity"], "high")

    def test_generate_knowledge_graph(self):
        """Test knowledge graph generation."""
        graph = self.analyzer.generate_knowledge_graph(self.mock_data)
        self.assertTrue("nodes" in graph)
        self.assertTrue("edges" in graph)

    def test_performance_metrics(self):
        """Test performance metrics analysis."""
        metrics = self.analyzer.performance_metrics(self.mock_data)
        self.assertTrue("execution_time" in metrics)
        self.assertGreater(metrics["execution_time"], 0)


if __name__ == "__main__":
    unittest.main()
