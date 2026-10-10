from typing import Any
from pydantic import BaseModel

def _clean_property(schema: dict[str, Any]) -> dict[str, Any]:
    """去掉 title/default，并把 anyOf[type, null] 压平成纯 type，适配函数调用。"""
    cleaned = {k: v for k, v in schema.items() if k not in ("title", "default")}
    if "anyOf" in cleaned:
        non_null = [branch for branch in cleaned["anyOf"] if branch.get("type") != "null"]
        if non_null:
            cleaned = {**cleaned, **non_null[0]}
            cleaned.pop("anyOf", None)
    return cleaned

def to_function_schema(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """由 pydantic 模型自动生成 OpenAI/DeepSeek 兼容的 function schema。"""
    json_schema = model.model_json_schema()
    properties = {
        key: _clean_property(value)
        for key, value in json_schema.get("properties", {}).items()
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": json_schema.get("required", []),
                "additionalProperties": False,
            },
        },
    }