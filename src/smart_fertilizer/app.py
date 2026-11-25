"""Flask REST API, Weather Advisory, Batch CSV, Plot Tracking, and PWA Dashboard."""

from __future__ import annotations

import csv
import io
import sqlite3
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from smart_fertilizer.domain import (
    InputValidationError,
    analyze_npk_and_ph,
    build_split_schedule,
    calculate_dosage_and_cost,
    get_fertilizer_profile,
    metadata_payload,
    normalize_soil,
    recommend_organic_alternatives,
    validate_soil_reading,
)
from smart_fertilizer.ml_pipeline import DEFAULT_BUNDLE_PATH, PROJECT_ROOT, FertilizerPredictor
from smart_fertilizer.storage import PredictionStore
from smart_fertilizer.weather import check_weather_advisory

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "advisor.db"


def create_app(
    db_path: str | Path = DEFAULT_DB_PATH,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    weather_fetcher: Callable[[str], dict[str, Any]] | None = None,
) -> Flask:
    """Create configured Smart Fertilizer Advisor Flask application."""
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    predictor = FertilizerPredictor(bundle_path=bundle_path)
    store = PredictionStore(db_path=db_path)

    app.extensions["fertilizer_predictor"] = predictor
    app.extensions["fertilizer_store"] = store

    def _build_full_advisory(payload: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        reading = validate_soil_reading(payload)
        prediction = predictor.predict(reading, top_k=3)
        diagnostics = analyze_npk_and_ph(reading)
        profile = get_fertilizer_profile(prediction["recommended"])
        dosage = calculate_dosage_and_cost(
            fertilizer_name=prediction["recommended"],
            reading=reading,
            diagnostics=diagnostics,
            top_alternatives=prediction["top_alternatives"],
        )
        split_schedule = build_split_schedule(reading, total_kg=dosage["total_kg"])
        organic_options = recommend_organic_alternatives(reading, diagnostics)

        if persist:
            saved = store.record(reading=reading, prediction=prediction, diagnostics=diagnostics, dosage=dosage)
            record_id = saved["id"]
            created_at = saved["created_at"]
            plot_name = saved["plot_name"]
        else:
            record_id = None
            created_at = None
            plot_name = reading.plot_name

        return {
            "record_id": record_id,
            "created_at": created_at,
            "plot_id": reading.plot_id,
            "plot_name": plot_name,
            "input": {
                **reading.to_feature_dict(),
                "soil_ph": reading.soil_ph,
                "area": reading.area,
                "area_unit": reading.area_unit,
                "area_acres": reading.area_acres,
                "crop_stage": reading.crop_stage,
            },
            "prediction": prediction,
            "fertilizer_profile": profile,
            "dosage_calculator": dosage,
            "split_schedule": split_schedule,
            "organic_alternatives": organic_options,
            "diagnostics": diagnostics,
        }

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/manifest.webmanifest")
    def manifest():
        return send_from_directory(str(PROJECT_ROOT / "static"), "manifest.webmanifest", mimetype="application/manifest+json")

    @app.get("/sw.js")
    def service_worker():
        return send_from_directory(str(PROJECT_ROOT / "static"), "sw.js", mimetype="application/javascript")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "smart-fertilizer-advisor",
                "selected_model": predictor.metrics["selected_model"],
                "dataset_rows": predictor.metrics["dataset_rows"],
            }
        )

    @app.get("/api/metadata")
    def metadata():
        payload = metadata_payload()
        payload["model_metrics"] = predictor.metrics
        return jsonify(payload)

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True)
        if body is None:
            body = request.form.to_dict()
        persist = bool(body.get("persist", True)) if isinstance(body, dict) else True
        try:
            advisory = _build_full_advisory(body, persist=persist)
            return jsonify(advisory)
        except InputValidationError as exc:
            return jsonify({"error": str(exc)}), 400

    # ----- Farm Plots & Multi-Season Tracking -----
    @app.get("/api/plots")
    def list_plots():
        return jsonify({"plots": store.list_plots()})

    @app.post("/api/plots")
    def create_plot():
        body = request.get_json(silent=True) or {}
        try:
            soil = normalize_soil(body.get("soil_type", "Loamy"))
            area = float(body.get("area_acres", 1.0))
            if area < 0.05 or area > 5000:
                raise InputValidationError("area_acres must be between 0.05 and 5000")
            plot = store.create_plot(
                name=str(body.get("name", "")),
                area_acres=round(area, 2),
                soil_type=soil,
                location=str(body.get("location", "")),
            )
            return jsonify(plot), 201
        except (InputValidationError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except sqlite3.IntegrityError:
            return jsonify({"error": "A farm plot with that name already exists"}), 409

    @app.delete("/api/plots/<int:plot_id>")
    def delete_plot(plot_id: int):
        if not store.delete_plot(plot_id):
            return jsonify({"error": "Plot not found"}), 404
        return jsonify({"deleted": True, "id": plot_id})

    @app.get("/api/plots/<int:plot_id>/trends")
    def plot_trends(plot_id: int):
        plot = store.get_plot(plot_id)
        if not plot:
            return jsonify({"error": "Plot not found"}), 404
        return jsonify({"plot": plot, "trends": store.get_plot_trends(plot_id)})

    # ----- Live Open-Meteo Rain-Runoff Check -----
    @app.get("/api/weather")
    def weather_check():
        city = request.args.get("city")
        raw_lat = request.args.get("lat")
        raw_lon = request.args.get("lon")
        try:
            lat = float(raw_lat) if raw_lat not in (None, "") else None
            lon = float(raw_lon) if raw_lon not in (None, "") else None
            kwargs: dict[str, Any] = {"city": city, "lat": lat, "lon": lon}
            if weather_fetcher is not None:
                kwargs["fetcher"] = weather_fetcher
            result = check_weather_advisory(**kwargs)
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Weather service unavailable: {exc}"}), 502

    # ----- Batch CSV Prediction for Labs & Co-ops -----
    @app.post("/api/batch-predict")
    def batch_predict():
        csv_text = ""
        if "file" in request.files:
            csv_text = request.files["file"].read().decode("utf-8-sig", errors="replace")
        else:
            body = request.get_json(silent=True) or {}
            csv_text = str(body.get("csv_text") or "")

        if not csv_text.strip():
            return jsonify({"error": "Provide a CSV file upload or non-empty csv_text"}), 400

        reader = csv.DictReader(io.StringIO(csv_text))
        rows_out: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for idx, raw_row in enumerate(reader, 1):
            norm_row = {k.strip().lower().replace(" ", "_"): v for k, v in raw_row.items() if k}
            if "temparature" in norm_row and "temperature" not in norm_row:
                norm_row["temperature"] = norm_row["temparature"]
            try:
                adv = _build_full_advisory(norm_row, persist=False)
                rows_out.append(
                    {
                        "row": idx,
                        "plot_name": adv["plot_name"] or f"Sample #{idx}",
                        "soil_type": adv["input"]["Soil_Type"],
                        "crop_type": adv["input"]["Crop_Type"],
                        "area_acres": adv["input"]["area_acres"],
                        "soil_ph": adv["input"]["soil_ph"],
                        "recommended_fertilizer": adv["prediction"]["recommended"],
                        "confidence_pct": adv["prediction"]["confidence_pct"],
                        "total_kg": adv["dosage_calculator"]["total_kg"],
                        "bags_to_buy": adv["dosage_calculator"]["recommended_bags_to_buy"],
                        "estimated_cost_inr": adv["dosage_calculator"]["estimated_cost_inr"],
                        "deficient_nutrients": ",".join(adv["diagnostics"]["deficient_keys"]) or "None",
                        "safety_warning": adv["prediction"]["safety_warning"],
                    }
                )
            except InputValidationError as exc:
                errors.append({"row": idx, "error": str(exc)})

        if request.args.get("format") == "csv":
            buf = io.StringIO()
            fieldnames = [
                "row",
                "plot_name",
                "soil_type",
                "crop_type",
                "area_acres",
                "soil_ph",
                "recommended_fertilizer",
                "confidence_pct",
                "total_kg",
                "bags_to_buy",
                "estimated_cost_inr",
                "deficient_nutrients",
                "safety_warning",
            ]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
            return Response(
                buf.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=batch_fertilizer_recommendations.csv"},
            )

        return jsonify(
            {
                "processed_count": len(rows_out),
                "error_count": len(errors),
                "results": rows_out,
                "errors": errors,
            }
        )

    # ----- History -----
    @app.get("/api/history")
    def history():
        limit = request.args.get("limit", default=50, type=int)
        plot_id = request.args.get("plot_id", default=None, type=int)
        return jsonify({"items": store.list_recent(limit=limit, plot_id=plot_id)})

    @app.delete("/api/history/<int:record_id>")
    def delete_history(record_id: int):
        if not store.delete(record_id):
            return jsonify({"error": "Prediction record not found"}), 404
        return jsonify({"deleted": True, "id": record_id})

    @app.get("/api/history/export.csv")
    def export_history_csv():
        plot_id = request.args.get("plot_id", default=None, type=int)
        csv_text = store.export_csv(plot_id=plot_id)
        return Response(
            csv_text,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=fertilizer_predictions.csv"},
        )

    return app
