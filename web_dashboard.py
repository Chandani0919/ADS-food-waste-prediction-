"""Flask server for the food-waste dashboard and machine-learning API."""

from __future__ import annotations

import argparse
from io import StringIO
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from food_waste import (
    CATEGORICAL_FEATURES,
    CHOICES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    fit_final_model,
    generate_demo_data,
    get_tree_feature_importance,
    train_and_compare,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parent
PAGE_PATHS = {
    "/": ROOT / "dashboard.html",
    "/overview": ROOT / "dashboard.html",
    "/prediction": ROOT / "prediction.html",
    "/model-evaluation": ROOT / "model_evaluation.html",
    "/dataset": ROOT / "dataset.html",
}
ASSET_PATHS = {
    "/dashboard.css": (ROOT / "dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": (ROOT / "dashboard.js", "text/javascript; charset=utf-8"),
}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MODEL_NAMES = ("Linear Regression", "Decision Tree", "Random Forest")


class DashboardState:
    """Hold the current dataset, evaluation results, and lazily fitted models."""

    def __init__(self, data: pd.DataFrame) -> None:
        self._lock = RLock()
        self._data = pd.DataFrame()
        self._metrics = pd.DataFrame()
        self._models: dict[str, Any] = {}
        self._source = ""
        self._default_model = ""
        self.replace_data(data, "Synthetic demo data")

    def replace_data(self, data: pd.DataFrame, source: str) -> None:
        cleaned, _ = validate_dataset(data)
        if len(cleaned) < 20:
            raise ValueError("At least 20 valid labeled rows are required.")
        if cleaned[TARGET_COLUMN].nunique() < 2:
            raise ValueError("The waste target must contain at least two distinct values.")

        _, metrics = train_and_compare(cleaned)
        with self._lock:
            self._data = cleaned
            self._metrics = metrics
            self._models = {}
            self._default_model = str(metrics.iloc[0]["Model"])
            self._source = source

    def _get_model(self, model_name: str) -> Any:
        if model_name not in MODEL_NAMES:
            raise ValueError("Select one of the listed regression models.")
        with self._lock:
            model = self._models.get(model_name)
            if model is None:
                model = fit_final_model(self._data, model_name)
                self._models[model_name] = model
            return model

    def summary(self, model_name: str | None = None) -> dict[str, Any]:
        with self._lock:
            data = self._data
            metrics = self._metrics
            source = self._source
            selected_model = model_name or self._default_model

            total_prepared = float(data["quantity_prepared_kg"].sum())
            total_waste = float(data[TARGET_COLUMN].sum())
            waste_by_food = (
                data.groupby("food_type")[TARGET_COLUMN]
                .mean()
                .sort_values(ascending=False)
            )
            result: dict[str, Any] = {
                "source": source,
                "rows": int(len(data)),
                "total_waste_kg": total_waste,
                "average_waste_kg": float(data[TARGET_COLUMN].mean()),
                "waste_ratio_percent": (
                    total_waste / total_prepared * 100 if total_prepared else 0.0
                ),
                "default_model": self._default_model,
                "selected_model": selected_model,
                "metrics": [
                    {
                        "model": str(row["Model"]),
                        "mae": float(row["MAE (kg)"]),
                        "rmse": float(row["RMSE (kg)"]),
                        "r2": float(row["R²"]),
                    }
                    for _, row in metrics.iterrows()
                ],
                "feature_options": {
                    feature: list(
                        dict.fromkeys(
                            data[feature].dropna().astype(str).tolist()
                            + CHOICES.get(feature, [])
                        )
                    )
                    for feature in CATEGORICAL_FEATURES
                },
                "numeric_defaults": {
                    feature: max(0.0, float(data[feature].median()))
                    for feature in NUMERIC_FEATURES
                },
                "waste_by_food": [
                    {"food_type": str(food), "average_waste_kg": float(value)}
                    for food, value in waste_by_food.items()
                ],
                "feature_importance": [],
            }

        if selected_model in {"Decision Tree", "Random Forest"}:
            importance = get_tree_feature_importance(self._get_model(selected_model))
            result["feature_importance"] = [
                {"feature": str(row["Feature"]), "importance": float(row["Importance"])}
                for _, row in importance.head(10).iterrows()
            ]
        return result

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = payload.get("model")
        if not isinstance(model_name, str) or model_name not in MODEL_NAMES:
            raise ValueError("Select a valid regression model.")

        raw_inputs = payload.get("inputs")
        if not isinstance(raw_inputs, dict):
            raise ValueError("Prediction inputs are required.")

        row: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            try:
                number = float(raw_inputs[feature])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Enter a valid number for {feature}.") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{feature} must be a finite number of zero or greater.")
            row[feature] = number

        for feature in CATEGORICAL_FEATURES:
            value = raw_inputs.get(feature)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Choose a value for {feature}.")
            row[feature] = value

        prediction = float(self._get_model(model_name).predict(pd.DataFrame([row]))[0])
        return {
            "prediction_kg": max(0.0, prediction),
            "model": model_name,
            "input_features": len(FEATURE_COLUMNS),
        }


STATE = DashboardState(generate_demo_data())

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.after_request
def add_security_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'"
    )
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error: RequestEntityTooLarge) -> tuple[Response, int]:
    return jsonify(error="Request body is too large (5 MB maximum)."), 413


for route, page_path in PAGE_PATHS.items():
    app.add_url_rule(
        route,
        endpoint=f"page_{page_path.stem}_{len(app.view_functions)}",
        view_func=lambda page_path=page_path: send_file(
            page_path, mimetype="text/html; charset=utf-8"
        ),
    )

for route, (asset_path, content_type) in ASSET_PATHS.items():
    app.add_url_rule(
        route,
        endpoint=f"asset_{route.lstrip('/').replace('.', '_')}",
        view_func=lambda asset_path=asset_path, content_type=content_type: send_file(
            asset_path, mimetype=content_type
        ),
    )


@app.get("/api/summary")
def get_summary() -> tuple[Response, int] | Response:
    try:
        return jsonify(STATE.summary(request.args.get("model")))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.get("/sample.csv")
def get_sample_csv() -> Response:
    sample = generate_demo_data(rows=30).to_csv(index=False).encode("utf-8")
    return Response(
        sample,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="food_waste_sample.csv"'},
    )


@app.post("/api/upload")
def upload_csv() -> tuple[Response, int] | Response:
    body = request.get_data(cache=False)
    if not body:
        return jsonify(error="The request body is empty."), 400
    try:
        csv_text = body.decode("utf-8-sig")
        uploaded_data = pd.read_csv(StringIO(csv_text))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return jsonify(error="The uploaded file must be a readable UTF-8 CSV."), 400

    try:
        STATE.replace_data(uploaded_data, "Uploaded CSV")
        return jsonify(STATE.summary())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/predict")
def predict() -> tuple[Response, int] | Response:
    body = request.get_data(cache=False)
    if not body:
        return jsonify(error="The request body is empty."), 400
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        return jsonify(error="Prediction request must contain valid JSON."), 400
    if not isinstance(payload, dict):
        return jsonify(error="Prediction request must be a JSON object."), 400

    try:
        return jsonify(STATE.predict(payload))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local food-waste prediction dashboard."
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    port = args.port
    print(f"Food Waste ML Dashboard running at http://127.0.0.1:{port}")
    print("This local demo server is not intended for public internet deployment.")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
