import pandas as pd
import pytest

from food_waste import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    fit_final_model,
    generate_demo_data,
    get_tree_feature_importance,
    train_and_compare,
    validate_dataset,
)


def test_demo_data_is_reproducible_and_has_expected_schema():
    first = generate_demo_data(rows=30, seed=7)
    second = generate_demo_data(rows=30, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == REQUIRED_COLUMNS
    assert (first[TARGET_COLUMN] >= 0).all()
    assert (first[TARGET_COLUMN] <= first["quantity_prepared_kg"] * 0.95).all()


def test_validate_dataset_excludes_missing_targets():
    data = generate_demo_data(rows=25)
    data.loc[0, TARGET_COLUMN] = None

    cleaned, notes = validate_dataset(data)

    assert len(cleaned) == 24
    assert notes == ["Excluded 1 row(s) with no waste target."]


def test_validate_dataset_rejects_missing_required_columns():
    data = generate_demo_data(rows=25).drop(columns=["food_type"])

    with pytest.raises(ValueError, match="Missing required columns: food_type"):
        validate_dataset(data)


def test_validate_dataset_rejects_negative_numeric_inputs():
    data = generate_demo_data(rows=25)
    data.loc[0, "guests"] = -1

    with pytest.raises(ValueError, match="Numeric features must be zero or greater"):
        validate_dataset(data)


def test_training_compares_all_models_and_returns_fitted_pipelines():
    data = generate_demo_data(rows=100)

    models, metrics = train_and_compare(data)

    assert set(models) == {"Linear Regression", "Decision Tree", "Random Forest"}
    assert list(metrics.columns) == ["Model", "MAE (kg)", "RMSE (kg)", "R²"]
    assert len(metrics) == 3
    assert metrics["MAE (kg)"].is_monotonic_increasing
    prediction = models[metrics.iloc[0]["Model"]].predict(data[FEATURE_COLUMNS].head(1))
    assert len(prediction) == 1


def test_training_requires_enough_rows():
    with pytest.raises(ValueError, match="At least 20 valid rows"):
        train_and_compare(generate_demo_data(rows=19))


def test_final_model_trains_on_all_available_rows_and_predicts():
    data = generate_demo_data(rows=40)
    data.loc[0, TARGET_COLUMN] = None
    model = fit_final_model(data, "Decision Tree")

    assert model.named_steps["regressor"].tree_.n_node_samples[0] == 39
    assert len(model.predict(data[FEATURE_COLUMNS].head(1))) == 1


def test_tree_feature_importance_aggregates_to_input_features():
    model = fit_final_model(generate_demo_data(rows=80), "Decision Tree")

    importance = get_tree_feature_importance(model)

    assert list(importance.columns) == ["Feature", "Importance"]
    assert set(importance["Feature"]).issubset(FEATURE_COLUMNS)
    assert importance["Importance"].is_monotonic_decreasing
    assert importance["Importance"].sum() == pytest.approx(1.0)


def test_linear_model_does_not_report_tree_importance():
    model = fit_final_model(generate_demo_data(rows=80), "Linear Regression")

    with pytest.raises(ValueError, match="tree-based models only"):
        get_tree_feature_importance(model)
