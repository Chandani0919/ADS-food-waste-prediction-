"""Data preparation and machine-learning helpers for the food-waste app."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

NUMERIC_FEATURES = [
    "guests",
    "quantity_prepared_kg",
    "purchase_history_kg",
    "price_per_kg",
]
CATEGORICAL_FEATURES = [
    "food_type",
    "event_type",
    "storage_condition",
    "season",
    "preparation_method",
    "location",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "waste_kg"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CHOICES = {
    "food_type": ["Vegetables", "Rice", "Poultry", "Seafood", "Pasta", "Dessert"],
    "event_type": ["Regular service", "Wedding", "Conference", "Holiday"],
    "storage_condition": ["Good", "Average", "Poor"],
    "season": ["Spring", "Summer", "Autumn", "Winter"],
    "preparation_method": ["Fresh", "Batch cooked", "Pre-prepared"],
    "location": ["Urban", "Suburban", "Rural"],
}


def generate_demo_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create reproducible, explicitly synthetic restaurant data for demos."""
    if rows < 1:
        raise ValueError("rows must be a positive integer")

    rng = np.random.default_rng(seed)
    guests = rng.integers(25, 301, size=rows)
    food_type = rng.choice(CHOICES["food_type"], size=rows)
    event_type = rng.choice(
        CHOICES["event_type"], size=rows, p=[0.62, 0.12, 0.16, 0.10]
    )
    storage_condition = rng.choice(
        CHOICES["storage_condition"], size=rows, p=[0.58, 0.32, 0.10]
    )
    season = rng.choice(CHOICES["season"], size=rows)
    preparation_method = rng.choice(CHOICES["preparation_method"], size=rows)
    location = rng.choice(CHOICES["location"], size=rows)
    price_by_food = {
        "Vegetables": 4.5,
        "Rice": 3.2,
        "Poultry": 9.0,
        "Seafood": 14.0,
        "Pasta": 5.5,
        "Dessert": 7.0,
    }
    food_portion = {
        "Vegetables": 0.25,
        "Rice": 0.22,
        "Poultry": 0.32,
        "Seafood": 0.28,
        "Pasta": 0.30,
        "Dessert": 0.16,
    }
    price_per_kg = np.array([price_by_food[item] for item in food_type]) * rng.uniform(
        0.8, 1.25, size=rows
    )
    purchase_history_kg = guests * rng.uniform(0.38, 0.75, size=rows)
    quantity_prepared_kg = guests * rng.uniform(0.48, 0.78, size=rows)

    event_demand = pd.Series(event_type).map(
        {"Regular service": 1.0, "Wedding": 1.12, "Conference": 0.94, "Holiday": 1.10}
    ).to_numpy()
    season_demand = pd.Series(season).map(
        {"Spring": 1.0, "Summer": 1.04, "Autumn": 0.98, "Winter": 1.06}
    ).to_numpy()
    storage_waste = pd.Series(storage_condition).map(
        {"Good": 0.01, "Average": 0.04, "Poor": 0.10}
    ).to_numpy()
    method_waste = pd.Series(preparation_method).map(
        {"Fresh": 0.015, "Batch cooked": 0.035, "Pre-prepared": 0.055}
    ).to_numpy()

    expected_consumption = (
        guests
        * np.array([food_portion[item] for item in food_type])
        * event_demand
        * season_demand
    )
    waste_kg = (
        quantity_prepared_kg
        - expected_consumption
        + quantity_prepared_kg * (storage_waste + method_waste)
        + rng.normal(0, 2.0, size=rows)
    )
    waste_kg = np.clip(waste_kg, 0, quantity_prepared_kg * 0.95)

    data = pd.DataFrame(
        {
            "food_type": food_type,
            "guests": guests,
            "event_type": event_type,
            "quantity_prepared_kg": quantity_prepared_kg.round(2),
            "storage_condition": storage_condition,
            "purchase_history_kg": purchase_history_kg.round(2),
            "season": season,
            "preparation_method": preparation_method,
            "location": location,
            "price_per_kg": price_per_kg.round(2),
            "waste_kg": waste_kg.round(2),
        }
    )
    return data[REQUIRED_COLUMNS]


def validate_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate the uploaded schema and clean unusable target rows."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    cleaned = data[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_features = cleaned[NUMERIC_FEATURES].isna().any(axis=1)
    if invalid_features.any():
        raise ValueError(
            f"{int(invalid_features.sum())} row(s) have missing or non-numeric "
            "values in numeric feature columns. Fix these values before training."
        )
    negative_values = (cleaned[NUMERIC_FEATURES] < 0).any(axis=1)
    if negative_values.any():
        raise ValueError("Numeric features must be zero or greater.")

    missing_targets = cleaned[TARGET_COLUMN].isna()
    cleaned = cleaned.loc[~missing_targets].copy()
    if (cleaned[TARGET_COLUMN] < 0).any():
        raise ValueError("Waste values must be zero or greater.")

    notes = []
    if missing_targets.any():
        notes.append(f"Excluded {int(missing_targets.sum())} row(s) with no waste target.")
    return cleaned, notes


def _make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def train_and_compare(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit three regressors on the same split and return their test metrics."""
    cleaned, _ = validate_dataset(data)
    if len(cleaned) < 20:
        raise ValueError("At least 20 valid rows are required to train and evaluate models.")
    if cleaned[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The waste target must contain at least two distinct values.")

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned[FEATURE_COLUMNS],
        cleaned[TARGET_COLUMN],
        test_size=test_size,
        random_state=random_state,
    )
    estimators: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
    }
    fitted: dict[str, Pipeline] = {}
    metric_rows = []
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[("preprocessor", _make_preprocessor()), ("regressor", estimator)]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted[name] = pipeline
        metric_rows.append(
            {
                "Model": name,
                "MAE (kg)": mean_absolute_error(y_test, predictions),
                "RMSE (kg)": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R²": r2_score(y_test, predictions),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("MAE (kg)").reset_index(drop=True)
    return fitted, metrics


def fit_final_model(data: pd.DataFrame, model_name: str) -> Pipeline:
    """Fit the selected regressor on all available labeled rows."""
    cleaned, _ = validate_dataset(data)
    estimators: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    if model_name not in estimators:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline(
        steps=[("preprocessor", _make_preprocessor()), ("regressor", estimators[model_name])]
    )
    pipeline.fit(cleaned[FEATURE_COLUMNS], cleaned[TARGET_COLUMN])
    return pipeline


def get_tree_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Aggregate a fitted tree model's encoded importances by original feature."""
    regressor = model.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        raise ValueError("Feature importance is available for tree-based models only.")

    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance_by_feature = {feature: 0.0 for feature in FEATURE_COLUMNS}
    for name, importance in zip(
        transformed_names, regressor.feature_importances_, strict=True
    ):
        encoded_name = str(name).split("__", maxsplit=1)[-1]
        feature_name = next(
            (
                feature
                for feature in FEATURE_COLUMNS
                if encoded_name == feature or encoded_name.startswith(f"{feature}_")
            ),
            None,
        )
        if feature_name is not None:
            importance_by_feature[feature_name] += float(importance)

    return (
        pd.DataFrame(
            {
                "Feature": list(importance_by_feature),
                "Importance": list(importance_by_feature.values()),
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
