import unittest
from agents.base_agent import BaseAgent
from agents.code_agent import CodeAgent
from agents.config_agent import ConfigAgent
from agents.doc_agent import DocAgent
from agents.test_agent import TestAgent
from agents.infra_agent import InfraAgent
from agents.security_agent import SecurityAgent
from agents.db_agent import DBAgent
from agents.arb_agent import ARBAgent


class TestAgents(unittest.TestCase):
    def setUp(self):
        """Set up test environment and mock data."""
        self.base_agent = BaseAgent()
        self.code_agent = CodeAgent()
        self.config_agent = ConfigAgent()
        self.doc_agent = DocAgent()
        self.test_agent = TestAgent()
        self.infra_agent = InfraAgent()
        self.security_agent = SecurityAgent()
        self.db_agent = DBAgent()
        self.arb_agent = ARBAgent()
        self.mock_data = {
            "source_code": "def example_function(): return True",
            "config": {"key": "value"},
            "document": "This is a test document.",
            "test_cases": ["test_case_1", "test_case_2"],
            "infrastructure": {"type": "cloud", "provider": "AWS"},
            "security_findings": [{"severity": "critical", "description": "SQL Injection"}],
            "db_schema": {"tables": ["users", "orders"]},
        }

    def test_base_agent_execute(self):
        """Test execution of the base agent."""
        result = self.base_agent.execute(self.mock_data)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")

    def test_code_agent_analyze_code(self):
        """Test code analysis."""
        findings = self.code_agent.analyze_code(self.mock_data["source_code"])
        self.assertTrue(len(findings) > 0)

    def test_config_agent_validate_config(self):
        """Test configuration validation."""
        validation_result = self.config_agent.validate_config(self.mock_data["config"])
        self.assertTrue(validation_result["is_valid"])

    def test_doc_agent_parse_document(self):
        """Test document parsing."""
        parsed = self.doc_agent.parse_document(self.mock_data["document"])
        self.assertTrue(len(parsed) > 0)

    def test_test_agent_run_tests(self):
        """Test running test cases."""
        results = self.test_agent.run_tests(self.mock_data["test_cases"])
        self.assertEqual(len(results), len(self.mock_data["test_cases"]))

    def test_infra_agent_check_infrastructure(self):
        """Test infrastructure checks."""
        report = self.infra_agent.check_infrastructure(self.mock_data["infrastructure"])
        self.assertIn("provider", report)

    def test_security_agent_detect_vulnerabilities(self):
        """Test detection of security vulnerabilities."""
        findings = self.security_agent.detect_vulnerabilities(self.mock_data["security_findings"])
        self.assertEqual(findings[0]["severity"], "critical")

    def test_db_agent_analyze_schema(self):
        """Test database schema analysis."""
        analysis = self.db_agent.analyze_schema(self.mock_data["db_schema"])
        self.assertTrue("tables" in analysis)

    def test_arb_agent_perform_advanced_task(self):
        """Test advanced task execution by ARB agent."""
        result = self.arb_agent.perform_advanced_task(self.mock_data)
        self.assertTrue("task_summary" in result)


if __name__ == "__main__":
    unittest.main()
