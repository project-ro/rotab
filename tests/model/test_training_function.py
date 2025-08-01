import numpy as np
import pytest
import polars as pl
import joblib
import lightgbm as lgb
from pathlib import Path
from rotab.core.operation.transform_funcs_polars import train_lgbm_with_optuna_multi_target


def test_function_runs_and_returns_correct_type(setup_data_and_path):
    data_path, model_path, _ = setup_data_and_path
    features = ["customer_id", "feature_1", "feature_2"]
    targets = ["target_1", "target_2"]

    results = train_lgbm_with_optuna_multi_target(
        data_path=str(data_path),
        features=features,
        targets=targets,
        split_by="timestamp",
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        n_trials=2,
        model_path=str(model_path),
    )

    assert isinstance(results, dict)
    assert len(results) == len(targets)
    for target_name in targets:
        assert target_name in results
        model, shap_df = results[target_name]
        assert isinstance(model, lgb.Booster)
        assert isinstance(shap_df, pl.DataFrame)
        assert "feature" in shap_df.columns
        assert "importance" in shap_df.columns
        assert len(shap_df) == len(features)


def test_model_files_are_saved(setup_data_and_path):
    data_path, model_path, _ = setup_data_and_path
    features = ["customer_id", "feature_1", "feature_2"]
    targets = ["target_1", "target_2"]

    train_lgbm_with_optuna_multi_target(
        data_path=str(data_path),
        features=features,
        targets=targets,
        split_by="timestamp",
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        model_path=str(model_path),
        n_trials=2,
    )

    for target_name in targets:
        model_file = Path(model_path) / f"lgbm_reg_{target_name}_model.joblib"
        assert model_file.exists()


def test_saved_model_can_predict(setup_data_and_path):
    data_path, model_path, df_dummy = setup_data_and_path
    features = ["customer_id", "feature_1", "feature_2"]
    targets = ["target_1", "target_2"]

    train_lgbm_with_optuna_multi_target(
        data_path=str(data_path),
        features=features,
        targets=targets,
        split_by="timestamp",
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        model_path=str(model_path),
        n_trials=2,
    )

    for target_name in targets:
        model_file = Path(model_path) / f"lgbm_reg_{target_name}_model.joblib"
        loaded_model = joblib.load(model_file)

        new_data_df = df_dummy.tail(1).select(features)
        new_data_np = new_data_df.to_numpy().astype(np.float32)

        prediction = loaded_model.predict(new_data_np)

        assert isinstance(prediction, np.ndarray) or isinstance(prediction, float)


@pytest.mark.parametrize("split_by", ["timestamp", "random"])
def test_split_logic_runs_without_error(setup_data_and_path, split_by):
    data_path, model_path, _ = setup_data_and_path
    features = ["customer_id", "feature_1", "feature_2"]
    targets = ["target_1"]

    results = train_lgbm_with_optuna_multi_target(
        data_path=str(data_path),
        features=features,
        targets=targets,
        split_by=split_by,
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        n_trials=2,
        model_path=str(model_path),
    )

    assert "target_1" in results


def test_invalid_split_by_raises_error(setup_data_and_path):
    data_path, model_path, _ = setup_data_and_path
    features = ["customer_id", "feature_1", "feature_2"]
    targets = ["target_1"]

    with pytest.raises(ValueError):
        train_lgbm_with_optuna_multi_target(
            data_path=str(data_path),
            features=features,
            targets=targets,
            split_by="invalid_split",
            split_by_column="yyyymm",
            timestamp_format="%Y%m",
            n_trials=2,
            model_path=str(model_path),
        )


def test_invalid_data_path_returns_empty_dict(setup_data_and_path):
    data_path, model_path, _ = setup_data_and_path
    features = ["customer_id", "feature_1"]
    targets = ["target_1"]

    results = train_lgbm_with_optuna_multi_target(
        data_path="non_existent_file.csv",
        features=features,
        targets=targets,
        split_by="timestamp",
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        model_path=str(model_path),
        n_trials=2,
    )

    assert results == {}
