import pytest
import polars as pl
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def setup_data_and_path(tmp_path_factory):
    """
    Creates a dummy CSV file and a model directory for testing.
    This fixture is run once per session and cleans up afterward.
    """
    data_dir = tmp_path_factory.mktemp("data")
    model_dir = tmp_path_factory.mktemp("models")
    data_path = data_dir / "test_data.csv"

    df_dummy = pl.DataFrame(
        {
            "yyyymm": [
                "202401",
                "202402",
                "202403",
                "202404",
                "202405",
                "202406",
                "202407",
                "202408",
                "202409",
                "202410",
            ],
            "customer_id": ["1", "2", "1", "2", "3", "1", "3", "2", "1", "3"],
            "feature_1": [10, 20, 15, 25, 30, 12, 35, 22, 18, 28],
            "feature_2": [5, 8, 6, 9, 7, 5, 9, 8, 6, 7],
            "target_1": [100, 150, 120, 180, 200, 110, 220, 160, 130, 210],
            "target_2": [20, 30, 25, 35, 40, 22, 45, 32, 28, 42],
        }
    )
    df_dummy.write_csv(data_path)

    return data_path, model_dir, df_dummy


@pytest.fixture(scope="function", autouse=True)
def clean_up_model_dir(setup_data_and_path):
    """
    Cleans up the model directory before each test to ensure a clean state.
    """
    _, model_path, _ = setup_data_and_path
    if model_path.exists():
        for child in model_path.iterdir():
            if child.is_file():
                child.unlink()
    yield
    if model_path.exists():
        for child in model_path.iterdir():
            if child.is_file():
                child.unlink()
