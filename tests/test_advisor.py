from __future__ import annotations

import io
from pathlib import Path

import pytest

from smart_fertilizer.app import create_app
from smart_fertilizer.domain import InputValidationError, normalize_crop, normalize_fertilizer, validate_soil_reading
from smart_fertilizer.ml_pipeline import DEFAULT_DATASET_PATH, load_and_clean_dataset, train_and_evaluate


def test_dataset_normalization_fixes_casing_and_trailing_dots():
    df = load_and_clean_dataset(DEFAULT_DATASET_PATH)
    assert len(df) == 552
    assert "Rice" in set(df["Crop_Type"])
    assert "rice" not in set(df["Crop_Type"])
    assert "Kidney Beans" in set(df["Crop_Type"])
    assert "Potassium Sulfate" in set(df["Fertilizer"])
    assert "Potassium sulfate." not in set(df["Fertilizer"])
    assert normalize_crop("kidneybeans") == "Kidney Beans"
    assert normalize_fertilizer("Potassium sulfate.") == "Potassium Sulfate"


def test_input_validation_supports_decimals_ph_area_and_bounds():
    reading = validate_soil_reading(
        {
            "temperature": 28.5,
            "humidity": 65.2,
            "moisture": 42.0,
            "soil_type": "loamy",
            "crop_type": "rice",
            "nitrogen": 70,
            "potassium": 45,
            "phosphorous": 35,
            "soil_ph": 5.2,
            "area": 2.0,
            "area_unit": "hectare",
            "crop_stage": "vegetative",
        }
    )
    assert reading.soil_type == "Loamy"
    assert reading.crop_type == "Rice"
    assert reading.soil_ph == 5.2
    assert reading.area_acres == pytest.approx(4.9421, rel=1e-3)

    with pytest.raises(InputValidationError, match="soil_ph"):
        validate_soil_reading(
            {
                "temperature": 25,
                "humidity": 60,
                "moisture": 40,
                "soil_type": "Loamy",
                "crop_type": "Wheat",
                "nitrogen": 30,
                "potassium": 10,
                "phosphorous": 10,
                "soil_ph": 12.0,
            }
        )


def test_all_farmer_and_platform_features_end_to_end(tmp_path: Path):
    bundle_path = tmp_path / "model_bundle.joblib"
    metrics_path = tmp_path / "metrics.json"
    db_path = tmp_path / "advisor.db"

    metrics = train_and_evaluate(
        dataset_path=DEFAULT_DATASET_PATH,
        bundle_path=bundle_path,
        metrics_path=metrics_path,
    )
    assert metrics["dataset_rows"] == 552
    assert metrics["unique_rows"] == 138
    assert len(metrics["benchmarks"]) == 4

    def fake_weather(url: str):
        if "geocoding-api" in url:
            return {"results": [{"name": "Thanjavur", "admin1": "Tamil Nadu", "country": "India", "latitude": 10.787, "longitude": 79.1378}]}
        return {
            "hourly": {
                "precipitation": [1.5] * 24,
                "precipitation_probability": [80] * 24,
                "temperature_2m": [29.0] * 24,
            }
        }

    app = create_app(db_path=db_path, bundle_path=bundle_path, weather_fetcher=fake_weather)
    client = app.test_client()

    # 1. Metadata, PWA Manifest & Service Worker
    meta = client.get("/api/metadata").get_json()
    assert len(meta["fertilizer_catalog"]) == 14
    assert client.get("/manifest.webmanifest").status_code == 200
    assert client.get("/sw.js").status_code == 200
    assert client.get("/static/images/farm_background.jpg").status_code == 200

    # 2. Farm Plot Creation
    plot_res = client.post("/api/plots", json={"name": "East Paddy Block", "area_acres": 2.0, "soil_type": "Clayey", "location": "Thanjavur"})
    assert plot_res.status_code == 201
    plot_id = plot_res.get_json()["id"]

    # 3. Prediction with 45 kg Urea Bag Sizing, Split Schedule, Acidic pH, & Organic Alternatives
    pred = client.post(
        "/api/predict",
        json={
            "plot_id": plot_id,
            "temperature": 28.0,
            "humidity": 54.0,
            "moisture": 41.0,
            "soil_type": "Clayey",
            "crop_type": "paddy",
            "nitrogen": 36.0,
            "potassium": 0.0,
            "phosphorous": 0.0,
            "soil_ph": 5.2,
            "area": 2.0,
            "area_unit": "acre",
            "crop_stage": "vegetative",
        },
    )
    assert pred.status_code == 200
    body = pred.get_json()
    assert body["prediction"]["recommended"] == "Urea"
    assert body["dosage_calculator"]["bag_weight_kg"] == 45.0
    assert body["dosage_calculator"]["ref_bag_price_inr"] == 266.5
    assert body["dosage_calculator"]["total_kg"] > 0
    assert sum(s["pct"] for s in body["split_schedule"]) == 100
    assert abs(sum(s["amount_kg"] for s in body["split_schedule"]) - body["dosage_calculator"]["total_kg"]) < 0.2
    assert body["diagnostics"]["ph_analysis"]["status"] == "Strongly Acidic"
    assert len(body["organic_alternatives"]) >= 2

    # 4. Out-of-Distribution / Extreme Reading Safety Warning
    ood_res = client.post(
        "/api/predict",
        json={
            "temperature": 58.0,
            "humidity": 5.0,
            "moisture": 2.0,
            "soil_type": "Sandy",
            "crop_type": "Rice",
            "nitrogen": 295.0,
            "potassium": 290.0,
            "phosphorous": 290.0,
            "persist": False,
        },
    )
    assert ood_res.status_code == 200
    ood_body = ood_res.get_json()
    assert ood_body["prediction"]["ood_flagged"] is True
    assert ood_body["prediction"]["safety_warning"] is True

    # 5. Multi-Season Plot Trend Check
    trends = client.get(f"/api/plots/{plot_id}/trends")
    assert trends.status_code == 200
    assert len(trends.get_json()["trends"]) == 1

    # 6. Weather Rain-Runoff Warning Check
    weather = client.get("/api/weather?city=Thanjavur")
    assert weather.status_code == 200
    w_json = weather.get_json()
    assert w_json["rain_risk"]["level"] == "HIGH"
    assert "delay top-dressing" in w_json["rain_risk"]["headline"].lower()

    # 7. Batch CSV Upload & CSV Download
    csv_payload = (
        "Plot_Name,Temperature,Humidity,Moisture,Soil_Type,Crop_Type,Nitrogen,Potassium,Phosphorous,Soil_pH,Area_Acres\n"
        "North Field,28,54,41,Clayey,Paddy,36,0,0,6.4,2.0\n"
        "Bad Row,abc,54,41,Clayey,Paddy,36,0,0,6.4,2.0\n"
    )
    batch_res = client.post("/api/batch-predict", json={"csv_text": csv_payload})
    assert batch_res.status_code == 200
    batch_json = batch_res.get_json()
    assert batch_json["processed_count"] == 1
    assert batch_json["error_count"] == 1
    assert batch_json["results"][0]["recommended_fertilizer"] == "Urea"

    batch_csv = client.post(
        "/api/batch-predict?format=csv",
        data={"file": (io.BytesIO(csv_payload.encode("utf-8")), "samples.csv")},
        content_type="multipart/form-data",
    )
    assert batch_csv.status_code == 200
    assert b"Urea" in batch_csv.data

    app.extensions["fertilizer_store"].close()
