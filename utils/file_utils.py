import os
from pathlib import Path
from typing import List, Optional


class FileUtils:
    """
    Utility class for file operations such as reading, writing, and validation.
    """

    @staticmethod
    def read_file(file_path: str, encoding: str = "utf-8") -> str:
        """
        Read the content of a file.

        Args:
            file_path: Path to the file.
            encoding: Encoding to use when reading the file.

        Returns:
            The content of the file as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there is an error reading the file.
        """
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading file {file_path}: {e}")

    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = "utf-8"):
        """
        Write content to a file.

        Args:
            file_path: Path to the file.
            content: Content to write to the file.
            encoding: Encoding to use when writing the file.

        Raises:
            IOError: If there is an error writing to the file.
        """
        try:
            with open(file_path, "w", encoding=encoding) as file:
                file.write(content)
        except IOError as e:
            raise IOError(f"Error writing to file {file_path}: {e}")

    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
        """
        List all files in a directory, optionally filtering by extension.

        Args:
            directory: Path to the directory.
            extension: File extension to filter by (e.g., ".txt").

        Returns:
            A list of file paths.

        Raises:
            NotADirectoryError: If the given path is not a directory.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files = [str(file) for file in dir_path.iterdir() if file.is_file()]
        if extension:
            files = [file for file in files if file.endswith(extension)]
        return files

    @staticmethod
    def create_directory(directory: str):
        """
        Create a directory if it does not already exist.

        Args:
            directory: Path to the directory.

        Raises:
            IOError: If there is an error creating the directory.
        """
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            raise IOError(f"Error creating directory {directory}: {e}")

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file exists, False otherwise.
        """
        return Path(file_path).is_file()

    @staticmethod
    def delete_file(file_path: str):
        """
        Delete a file.

        Args:
            file_path: Path to the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there is an error deleting the file.
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            file.unlink()
        except OSError as e:
            raise IOError(f"Error deleting file {file_path}: {e}")
