from typing import Dict, List


class PromptUtils:
    """
    Utility class for handling and formatting prompts for LLM interactions.
    """

    @staticmethod
    def build_prompt(template: str, variables: Dict[str, str]) -> str:
        """
        Build a prompt by replacing placeholders in the template with actual values.

        Args:
            template: The prompt template containing placeholders (e.g., `{key}`).
            variables: A dictionary of values to replace placeholders.

        Returns:
            A formatted prompt string.

        Raises:
            KeyError: If a placeholder in the template has no corresponding value in `variables`.
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            raise KeyError(f"Missing value for placeholder: {e}")

    @staticmethod
    def generate_task_prompt(task_type: str, details: Dict[str, str]) -> str:
        """
        Generate a task-specific prompt.

        Args:
            task_type: The type of task (e.g., "summarize", "analyze", "classify").
            details: Additional details to include in the prompt.

        Returns:
            A task-specific prompt string.
        """
        task_templates = {
            "summarize": "Please summarize the following text:\n\n{text}",
            "analyze": "Analyze the following data for patterns:\n\n{data}",
            "classify": "Classify the following input:\n\n{input}"
        }

        if task_type not in task_templates:
            raise ValueError(f"Unsupported task type: {task_type}")

        return PromptUtils.build_prompt(task_templates[task_type], details)

    @staticmethod
    def split_long_text(text: str, max_length: int = 2048) -> List[str]:
        """
        Split a long text into smaller chunks of a specified maximum length.

        Args:
            text: The long text to split.
            max_length: The maximum length of each chunk.

        Returns:
            A list of text chunks.
        """
        chunks = []
        while text:
            chunk = text[:max_length]
            split_point = max_length
            if len(text) > max_length:
                # Avoid splitting in the middle of a word
                split_point = chunk.rfind(" ") + 1 if " " in chunk else max_length
            chunks.append(text[:split_point].strip())
            text = text[split_point:].strip()
        return chunks

    @staticmethod
    def enrich_prompt(prompt: str, context: str, delimiter: str = "\n\n") -> str:
        """
        Enrich a prompt by appending additional context.

        Args:
            prompt: The original prompt.
            context: Additional context to append.
            delimiter: The delimiter to separate the prompt and context.

        Returns:
            An enriched prompt string.
        """
        return f"{prompt}{delimiter}{context}"

    @staticmethod
    def parse_response(response: str) -> Dict[str, str]:
        """
        Parse a response from the LLM into a structured dictionary.

        Args:
            response: The response string.

        Returns:
            A dictionary with parsed information.
        """
        parsed = {}
        lines = response.split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.strip()] = value.strip()
        return parsed
