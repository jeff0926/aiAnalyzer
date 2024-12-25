import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union


class Parser:
    """
    Utility class for parsing and transforming data formats.
    """

    @staticmethod
    def parse_json(json_string: str) -> Dict[str, Any]:
        """
        Parse a JSON string into a Python dictionary.

        Args:
            json_string: The JSON string to parse.

        Returns:
            A dictionary representation of the JSON string.

        Raises:
            ValueError: If the JSON string is invalid.
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @staticmethod
    def to_json(data: Dict[str, Any], pretty: bool = False) -> str:
        """
        Convert a dictionary to a JSON string.

        Args:
            data: The dictionary to convert.
            pretty: Whether to pretty-print the JSON.

        Returns:
            A JSON string representation of the dictionary.
        """
        if pretty:
            return json.dumps(data, indent=4)
        return json.dumps(data)

    @staticmethod
    def parse_xml(xml_string: str) -> ET.Element:
        """
        Parse an XML string into an ElementTree element.

        Args:
            xml_string: The XML string to parse.

        Returns:
            The root element of the parsed XML.

        Raises:
            ValueError: If the XML string is invalid.
        """
        try:
            return ET.fromstring(xml_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")

    @staticmethod
    def xml_to_dict(element: ET.Element) -> Dict[str, Any]:
        """
        Convert an ElementTree element to a nested dictionary.

        Args:
            element: The root element to convert.

        Returns:
            A dictionary representation of the XML.
        """
        def element_to_dict(el: ET.Element) -> Union[Dict[str, Any], str]:
            children = list(el)
            if not children:
                return el.text or ""
            return {child.tag: element_to_dict(child) for child in children}

        return {element.tag: element_to_dict(element)}

    @staticmethod
    def dict_to_xml(data: Dict[str, Any], root_tag: str = "root") -> str:
        """
        Convert a dictionary to an XML string.

        Args:
            data: The dictionary to convert.
            root_tag: The tag for the root element.

        Returns:
            An XML string representation of the dictionary.
        """
        def dict_to_element(tag: str, value: Union[Dict[str, Any], str]) -> ET.Element:
            elem = ET.Element(tag)
            if isinstance(value, dict):
                for k, v in value.items():
                    child = dict_to_element(k, v)
                    elem.append(child)
            else:
                elem.text = str(value)
            return elem

        root = dict_to_element(root_tag, data)
        return ET.tostring(root, encoding="unicode")

