import numpy as np
import pytest
import polars as pl
import joblib
import json
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


def test_one_hot_strict_mode_runs_and_saves_meta(setup_data_and_path):
    data_path, model_path, df_dummy = setup_data_and_path

    # カテゴリ列を安定に作成（末尾だけ unseen "Z"）
    n = df_dummy.height
    vals = (["A", "B", "C"] * ((n // 3) + 1))[:n]
    vals[-1] = "Z"
    df_with_cat = df_dummy.with_columns(pl.Series("category", vals, dtype=pl.Utf8))

    data_path_cat = Path(model_path) / "data_with_category.csv"
    df_with_cat.write_csv(data_path_cat)

    features = ["customer_id", "feature_1", "feature_2", "category"]
    targets = ["target_1"]

    results = train_lgbm_with_optuna_multi_target(
        data_path=str(data_path_cat),
        features=features,
        targets=targets,
        split_by="timestamp",
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        n_trials=2,
        model_path=str(model_path),
        one_hot_cols=["category"],
        one_hot_drop_first=False,
    )

    # 返り値チェック
    assert "target_1" in results
    model, shap_df = results["target_1"]
    assert isinstance(model, lgb.Booster)
    assert isinstance(shap_df, pl.DataFrame)

    # 元列は出ず、ダミー列が存在
    feats = shap_df["feature"].to_list()
    assert "category" not in feats
    assert any(f.startswith("category_") for f in feats)

    # メタファイル確認
    onehot_meta_file = Path(model_path) / "onehot_columns.json"
    assert onehot_meta_file.exists()
    meta = json.loads(onehot_meta_file.read_text())
    assert meta["mapping"]  # 何かしらマッピングがある
    # 学習（train+validate）に出てない "Z" は mapping に含まれないはず
    assert not any(col.endswith("_Z") for col in sum(meta["mapping"].values(), []))

    # モデルファイルも保存されている
    model_file = Path(model_path) / "lgbm_reg_target_1_model.joblib"
    assert model_file.exists()


def test_one_hot_strict_mode_unseen_category_is_dropped_not_crash(setup_data_and_path):
    data_path, model_path, df_dummy = setup_data_and_path

    # 末尾に unseen "Z"
    n = df_dummy.height
    vals = (["A", "B", "C"] * ((n // 3) + 1))[:n]
    vals[-1] = "Z"
    df_with_cat = df_dummy.with_columns(pl.Series("category", vals, dtype=pl.Utf8))

    data_path_cat = Path(model_path) / "data_with_category_2.csv"
    df_with_cat.write_csv(data_path_cat)

    features = ["customer_id", "feature_1", "feature_2", "category"]
    targets = ["target_1"]

    results = train_lgbm_with_optuna_multi_target(
        data_path=str(data_path_cat),
        features=features,
        targets=targets,
        split_by="timestamp",
        split_by_column="yyyymm",
        timestamp_format="%Y%m",
        n_trials=2,
        model_path=str(model_path),
        one_hot_cols=["category"],
        one_hot_drop_first=False,
    )

    model, shap_df = results["target_1"]

    feats = shap_df["feature"].to_list()
    # テストにしか出ないカテゴリ "Z" に対応するダミーは train+validate で作られない想定
    assert "category_Z" not in feats
