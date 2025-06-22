from typing import Dict, Literal

PrimitiveType = Literal["int", "float", "string", "bool", "datetime"]
SchemaDefinition = Dict[str, PrimitiveType]
Schemas = Dict[str, SchemaDefinition]
