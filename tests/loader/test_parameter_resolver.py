import os
import tempfile
import yaml
import pytest
from rotab.loader import ParameterResolver


@pytest.fixture
def valid_param_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # params1.yaml
        with open(os.path.join(tmpdir, "a.yaml"), "w") as f:
            yaml.dump({"min_age": 18, "threshold": 100}, f)

        # params2.yml
        with open(os.path.join(tmpdir, "b.yml"), "w") as f:
            yaml.dump({"max_age": 65}, f)

        yield tmpdir


def test_resolve_flat_keys(valid_param_dir):
    resolver = ParameterResolver(valid_param_dir)
    assert resolver.params["min_age"] == 18
    assert resolver.params["threshold"] == 100
    assert resolver.params["max_age"] == 65
    assert resolver.resolve("${min_age}") == 18


def test_resolve_nested_structure(valid_param_dir):
    with open(os.path.join(valid_param_dir, "nested.yaml"), "w") as f:
        yaml.dump({"group": {"subkey": "value"}}, f)
    resolver = ParameterResolver(valid_param_dir)
    assert resolver.resolve("${group.subkey}") == "value"


def test_resolve_in_dict(valid_param_dir):
    resolver = ParameterResolver(valid_param_dir)
    obj = {"query": "age > ${threshold}"}
    resolved = resolver.resolve(obj)
    assert resolved["query"] == "age > 100"


def test_resolve_missing_key_raises(valid_param_dir):
    resolver = ParameterResolver(valid_param_dir)
    with pytest.raises(KeyError, match="Parameter 'not_found' not found"):
        resolver.resolve("${not_found}")


def test_duplicate_key_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.yaml"), "w") as f:
            yaml.dump({"key": "value1"}, f)
        with open(os.path.join(tmpdir, "b.yaml"), "w") as f:
            yaml.dump({"key": "value2"}, f)

        with pytest.raises(ValueError, match="Duplicate parameter key 'key' found in b.yaml"):
            ParameterResolver(tmpdir)
