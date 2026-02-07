"""
Request parameter to config field mapping module.

Provides a decorator-based system for mapping flat request form parameters
to parser config fields. A single request param can map to multiple config fields.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.config.config import RequestConfig

# Type alias for registry entries: (ConfigClass, field_name)
ConfigFieldMapping = Tuple[Type[BaseModel], str]

# Registry storing param_name -> list of (ConfigClass, field_name) mappings
_PARAM_REGISTRY: Optional[Dict[str, List[ConfigFieldMapping]]] = None


def build_param_registry() -> Dict[str, List[ConfigFieldMapping]]:
    """
    Scan all parser configs and RequestConfig to build param -> config field mapping.

    Reads json_schema_extra["request_param"] from each field to determine
    which request parameter maps to that field.

    Returns:
        Dict mapping request param names to list of (ConfigClass, field_name) tuples
    """
    # Import here to avoid circular imports
    from app.config.config import (
        DocxParserConfig,
        ExcelParserConfig,
        ImageParserConfig,
        ParserConfig,
        PDFParserConfig,
        RequestConfig,
    )

    registry: Dict[str, List[ConfigFieldMapping]] = {}

    # All config classes to scan (including RequestConfig for request-level attributes)
    config_classes = [
        ParserConfig,
        PDFParserConfig,
        ExcelParserConfig,
        DocxParserConfig,
        ImageParserConfig,
        RequestConfig,
    ]

    for config_class in config_classes:
        for field_name, field_info in config_class.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if isinstance(extra, dict) and "request_param" in extra:
                param_name = extra["request_param"]
                if param_name not in registry:
                    registry[param_name] = []
                registry[param_name].append((config_class, field_name))

    return registry


def get_param_registry() -> Dict[str, List[ConfigFieldMapping]]:
    """
    Get the parameter registry, building it if necessary.

    Returns:
        The global parameter registry mapping request params to config fields
    """
    global _PARAM_REGISTRY
    if _PARAM_REGISTRY is None:
        _PARAM_REGISTRY = build_param_registry()
    return _PARAM_REGISTRY


def get_available_request_params() -> List[str]:
    """
    Get list of all available request parameter names.

    Returns:
        List of parameter names that can be passed in request form data
    """
    return list(get_param_registry().keys())


class RequestConfigBuilder:
    """
    Builder class for creating RequestConfig from form data.

    Uses the parameter registry to map flat form data fields to the appropriate
    parser config fields.
    """

    @classmethod
    def from_form_data(cls, form_data: Dict[str, Any]) -> "RequestConfig":
        """
        Create a RequestConfig from form data.

        Parses the flat form data and populates all parser configs based on
        the parameter registry mappings.

        Args:
            form_data: Dict of form field names to values (None values are ignored)

        Returns:
            Fully populated RequestConfig with all parser configs
        """
        # Import here to avoid circular imports
        from app.config.config import (
            DocxParserConfig,
            ExcelParserConfig,
            ImageParserConfig,
            PDFParserConfig,
            RequestConfig,
        )

        registry = get_param_registry()

        # Collect overrides for each config class
        pdf_overrides: Dict[str, Any] = {}
        excel_overrides: Dict[str, Any] = {}
        docx_overrides: Dict[str, Any] = {}
        image_overrides: Dict[str, Any] = {}
        request_overrides: Dict[str, Any] = {}

        # Map config classes to their override dicts
        config_to_overrides = {
            PDFParserConfig: pdf_overrides,
            ExcelParserConfig: excel_overrides,
            DocxParserConfig: docx_overrides,
            ImageParserConfig: image_overrides,
            RequestConfig: request_overrides,
        }

        # Process each form data field
        for param_name, value in form_data.items():
            # Skip None values (not provided in request)
            if value is None:
                continue

            # Check if this param is registered
            if param_name not in registry:
                continue

            # Apply value to all mapped config fields
            for config_class, field_name in registry[param_name]:
                # Handle base ParserConfig fields - apply to all parser configs (not RequestConfig)
                if config_class.__name__ == "ParserConfig":
                    for key, override_dict in config_to_overrides.items():
                        if key != RequestConfig:
                            override_dict[field_name] = value
                elif config_class in config_to_overrides:
                    config_to_overrides[config_class][field_name] = value

        # Create configs with overrides
        pdf_config = PDFParserConfig(**pdf_overrides)
        excel_config = ExcelParserConfig(**excel_overrides)
        docx_config = DocxParserConfig(**docx_overrides)
        image_config = ImageParserConfig(**image_overrides)

        # Build RequestConfig with all parser configs and request-level attributes
        return RequestConfig(
            pdf_config=pdf_config,
            excel_config=excel_config,
            docx_config=docx_config,
            image_config=image_config,
            **request_overrides,
        )
