import os
import tempfile
import yaml
import pytest
from rotab.loader import SchemaManager
from rotab.ast.context.validation_context import VariableInfo


@pytest.fixture
def tmp_schema_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Valid structured schema
        structured = {
            "name": "trans",
            "description": "test schema",
            "columns": [
                {"name": "id", "dtype": "str", "description": "id field"},
                {"name": "amount", "dtype": "int", "description": "transaction amount"},
            ],
        }
        with open(os.path.join(tmpdir, "structured.yaml"), "w") as f:
            yaml.dump(structured, f)

        # Valid simple schema
        simple = {"id": "str", "amount": "int"}
        with open(os.path.join(tmpdir, "simple.yml"), "w") as f:
            yaml.dump(simple, f)

        # Invalid schema
        invalid = ["not", "a", "dict"]
        with open(os.path.join(tmpdir, "invalid.yaml"), "w") as f:
            yaml.dump(invalid, f)

        yield tmpdir


def test_structured_schema(tmp_schema_dir):
    manager = SchemaManager(tmp_schema_dir)
    schema = manager.get_schema("structured")
    assert isinstance(schema, VariableInfo)
    assert schema.type == "dataframe"
    assert schema.columns["id"] == "str"
    assert schema.columns["amount"] == "int"


def test_simple_schema(tmp_schema_dir):
    manager = SchemaManager(tmp_schema_dir)
    schema = manager.get_schema("simple")
    assert isinstance(schema, VariableInfo)
    assert schema.columns["id"] == "str"
    assert schema.columns["amount"] == "int"


def test_invalid_schema(tmp_schema_dir):
    manager = SchemaManager(tmp_schema_dir)
    with pytest.raises(ValueError, match="Invalid schema format"):
        manager.get_schema("invalid")


def test_missing_schema(tmp_schema_dir):
    manager = SchemaManager(tmp_schema_dir)
    with pytest.raises(FileNotFoundError):
        manager.get_schema("not_found")
