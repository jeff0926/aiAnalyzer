import unittest
import tempfile
import os
from utils.cache import Cache
from utils.parser import Parser
from utils.file_utils import FileUtils
from utils.prompts import PromptUtils


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for file operations."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.temp_file, "w") as f:
            f.write("Sample content")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    # === Cache Tests ===
    def test_cache_set_and_get(self):
        cache = Cache(cache_dir=self.temp_dir.name, default_expiry=10)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_cache_expiry(self):
        cache = Cache(cache_dir=self.temp_dir.name, default_expiry=1)
        cache.set("key2", "value2")
        import time
        time.sleep(2)
        self.assertIsNone(cache.get("key2"))

    def test_cache_clear(self):
        cache = Cache(cache_dir=self.temp_dir.name)
        cache.set("key3", "value3")
        cache.clear()
        self.assertIsNone(cache.get("key3"))

    # === Parser Tests ===
    def test_parse_json(self):
        json_string = '{"key": "value"}'
        parsed = Parser.parse_json(json_string)
        self.assertEqual(parsed["key"], "value")

    def test_parse_invalid_json(self):
        with self.assertRaises(ValueError):
            Parser.parse_json("{key: value}")

    def test_dict_to_xml_and_back(self):
        data = {"root": {"child": "value"}}
        xml_string = Parser.dict_to_xml(data)
        element = Parser.parse_xml(xml_string)
        parsed_dict = Parser.xml_to_dict(element)
        self.assertEqual(parsed_dict, data)

    # === File Utils Tests ===
    def test_read_file(self):
        content = FileUtils.read_file(self.temp_file)
        self.assertEqual(content, "Sample content")

    def test_write_file(self):
        new_content = "New content"
        FileUtils.write_file(self.temp_file, new_content)
        content = FileUtils.read_file(self.temp_file)
        self.assertEqual(content, new_content)

    def test_list_files(self):
        files = FileUtils.list_files(self.temp_dir.name)
        self.assertIn(self.temp_file, files)

    # === Prompts Tests ===
    def test_build_prompt(self):
        template = "Hello, {name}!"
        variables = {"name": "Alice"}
        result = PromptUtils.build_prompt(template, variables)
        self.assertEqual(result, "Hello, Alice!")

    def test_split_long_text(self):
        text = "This is a test " * 50
        chunks = PromptUtils.split_long_text(text, max_length=100)
        self.assertTrue(len(chunks) > 1)

    def test_enrich_prompt(self):
        prompt = "Base prompt"
        context = "Additional context"
        enriched = PromptUtils.enrich_prompt(prompt, context)
        self.assertIn("Additional context", enriched)

    def test_parse_response(self):
        response = "Key1: Value1\nKey2: Value2"
        parsed = PromptUtils.parse_response(response)
        self.assertEqual(parsed["Key1"], "Value1")
        self.assertEqual(parsed["Key2"], "Value2")


if __name__ == "__main__":
    unittest.main()
