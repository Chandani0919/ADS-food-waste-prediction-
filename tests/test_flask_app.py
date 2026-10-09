import pandas as pd

from app import app
from food_waste import FEATURE_COLUMNS, generate_demo_data
import web_dashboard


def test_dashboard_pages_and_assets_are_served():
    client = app.test_client()

    for route in ("/", "/overview", "/prediction", "/model-evaluation", "/dataset"):
        response = client.get(route)
        assert response.status_code == 200
        assert response.mimetype == "text/html"

    assert client.get("/dashboard.css").mimetype == "text/css"
    assert client.get("/dashboard.js").mimetype == "text/javascript"


def test_summary_contains_model_and_dataset_data():
    response = app.test_client().get("/api/summary?model=Decision%20Tree")

    assert response.status_code == 200
    summary = response.get_json()
    assert summary["rows"] == 500
    assert len(summary["metrics"]) == 3
    assert summary["feature_importance"]


def test_prediction_endpoint_uses_selected_model():
    client = app.test_client()
    summary = client.get("/api/summary").get_json()
    inputs = {
        feature: summary["numeric_defaults"][feature]
        if feature in summary["numeric_defaults"]
        else summary["feature_options"][feature][0]
        for feature in FEATURE_COLUMNS
    }

    response = client.post(
        "/api/predict",
        json={"model": summary["default_model"], "inputs": inputs},
    )

    assert response.status_code == 200
    assert response.get_json()["prediction_kg"] >= 0


def test_upload_endpoint_accepts_csv_without_replacing_test_state(monkeypatch):
    received = {}

    def record_upload(data, source):
        received["data"] = data
        received["source"] = source

    monkeypatch.setattr(web_dashboard.STATE, "replace_data", record_upload)
    csv_data = generate_demo_data(rows=30).to_csv(index=False).encode("utf-8")

    response = app.test_client().post(
        "/api/upload", data=csv_data, content_type="text/csv; charset=utf-8"
    )

    assert response.status_code == 200
    pd.testing.assert_frame_equal(received["data"], generate_demo_data(rows=30))
    assert received["source"] == "Uploaded CSV"


def test_prediction_endpoint_returns_validation_errors_as_json():
    response = app.test_client().post("/api/predict", json={})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_sample_csv_download():
    response = app.test_client().get("/sample.csv")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "food_waste_sample.csv" in response.headers["Content-Disposition"]
