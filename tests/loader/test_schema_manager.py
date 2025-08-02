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
            "path": "/test/test.csv",
            "columns": [
                {"name": "id", "dtype": "str", "description": "id field"},
                {"name": "amount", "dtype": "int", "description": "transaction amount"},
            ],
        }
        with open(os.path.join(tmpdir, "structured.yaml"), "w") as f:
            yaml.dump(structured, f)

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
    assert schema.path == "/test/test.csv"


def test_invalid_schema(tmp_schema_dir):
    manager = SchemaManager(tmp_schema_dir)
    with pytest.raises(ValueError, match="'columns' key not found"):
        manager.get_schema("invalid")


def test_missing_schema(tmp_schema_dir):
    manager = SchemaManager(tmp_schema_dir)
    with pytest.raises(FileNotFoundError):
        manager.get_schema("not_found")
