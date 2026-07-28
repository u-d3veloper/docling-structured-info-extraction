from pydantic import BaseModel, Field, create_model
from typing import Any, Type

TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "list": list[Any],
}


def build_dynamic_model(schema_dict: dict) -> Type[BaseModel]:
    """
    Build a dynamic Pydantic model based on the provided schema dictionary.
    Each key in the schema dictionary represents a field name, and its value is a dictionary
    containing the field's type and an optional description.

    Args:
        schema_dict (dict): A dictionary where each key is a field name and its value is
        a dictionary with 'type' and 'description' keys.

    Returns:
        Type[BaseModel]: A dynamically created Pydantic model class with the specified fields.
    """
    fields = {}
    for field_name, field_info in schema_dict.items():
        if not isinstance(field_info, dict):
            raise ValueError(f"Invalid schema for field '{field_name}'. Expected a dictionary with 'type' and 'description'.")

        field_type = TYPE_MAPPING.get(field_info.get("type", "string"))
        if field_type is None:
            raise ValueError(
                f"Unsupported type '{field_info.get('type')}' for field '{field_name}'. "
                f"Invalid type. Supported types are: {list(TYPE_MAPPING.keys())}."
            )

        fields[field_name] = (
            field_type | None,
            Field(default=None, description=field_info.get("description", "")),
        )

    return create_model("ExtractionModel", **fields)
