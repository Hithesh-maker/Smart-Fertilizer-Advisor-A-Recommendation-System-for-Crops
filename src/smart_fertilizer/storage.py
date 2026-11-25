"""SQLite persistence repository for farm plots, multi-season logs, and CSV export."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smart_fertilizer.domain import SoilReading


class PredictionStore:
    """Thread-safe SQLite repository storing farm plot profiles & fertilizer logs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS plots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                area_acres REAL NOT NULL DEFAULT 1.0,
                soil_type TEXT NOT NULL DEFAULT 'Loamy',
                location TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                plot_id INTEGER NULL REFERENCES plots(id) ON DELETE SET NULL,
                plot_name TEXT NOT NULL DEFAULT '',
                area_acres REAL NOT NULL DEFAULT 1.0,
                crop_stage TEXT NOT NULL DEFAULT 'sowing',
                soil_ph REAL NOT NULL DEFAULT 6.5,
                soil_type TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                moisture REAL NOT NULL,
                nitrogen REAL NOT NULL,
                potassium REAL NOT NULL,
                phosphorous REAL NOT NULL,
                recommended_fertilizer TEXT NOT NULL,
                confidence_pct REAL NOT NULL,
                bags_required REAL NOT NULL DEFAULT 1.0,
                estimated_cost_inr REAL NOT NULL DEFAULT 0.0,
                model_used TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL
            );
            """
        )
        # Forward-compatible migration if opening an older DB
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(predictions)").fetchall()}
        migrations = {
            "plot_id": "INTEGER NULL",
            "plot_name": "TEXT NOT NULL DEFAULT ''",
            "area_acres": "REAL NOT NULL DEFAULT 1.0",
            "crop_stage": "TEXT NOT NULL DEFAULT 'sowing'",
            "soil_ph": "REAL NOT NULL DEFAULT 6.5",
            "bags_required": "REAL NOT NULL DEFAULT 1.0",
            "estimated_cost_inr": "REAL NOT NULL DEFAULT 0.0",
        }
        for col, col_def in migrations.items():
            if col not in cols:
                self.conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_def}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ----- Farm Plots -----
    def list_plots(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT p.*, COUNT(pr.id) AS test_count
            FROM plots p
            LEFT JOIN predictions pr ON pr.plot_id = p.id
            GROUP BY p.id
            ORDER BY p.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def create_plot(self, name: str, area_acres: float = 1.0, soil_type: str = "Loamy", location: str = "") -> dict[str, Any]:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Plot name is required")
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO plots (name, area_acres, soil_type, location, created_at) VALUES (?, ?, ?, ?, ?)",
            (cleaned, float(area_acres), soil_type, location.strip(), now),
        )
        self.conn.commit()
        return dict(self.conn.execute("SELECT * FROM plots WHERE id = ?", (cur.lastrowid,)).fetchone())

    def get_plot(self, plot_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM plots WHERE id = ?", (plot_id,)).fetchone()
        return dict(row) if row else None

    def delete_plot(self, plot_id: int) -> bool:
        self.conn.execute("UPDATE predictions SET plot_id = NULL WHERE plot_id = ?", (plot_id,))
        cur = self.conn.execute("DELETE FROM plots WHERE id = ?", (plot_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def get_plot_trends(self, plot_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, created_at, crop_type, crop_stage, soil_ph, nitrogen, phosphorous, potassium,
                   recommended_fertilizer, confidence_pct, bags_required, estimated_cost_inr
            FROM predictions
            WHERE plot_id = ?
            ORDER BY id ASC
            """,
            (plot_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----- Predictions -----
    def record(
        self,
        reading: SoilReading,
        prediction: dict[str, Any],
        diagnostics: dict[str, Any],
        dosage: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        resolved_plot_name = reading.plot_name
        if reading.plot_id and not resolved_plot_name:
            plot = self.get_plot(reading.plot_id)
            if plot:
                resolved_plot_name = plot["name"]

        cur = self.conn.execute(
            """
            INSERT INTO predictions (
                created_at, plot_id, plot_name, area_acres, crop_stage, soil_ph,
                soil_type, crop_type, temperature, humidity, moisture,
                nitrogen, potassium, phosphorous, recommended_fertilizer,
                confidence_pct, bags_required, estimated_cost_inr, model_used, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                reading.plot_id,
                resolved_plot_name,
                reading.area_acres,
                reading.crop_stage,
                reading.soil_ph,
                reading.soil_type,
                reading.crop_type,
                reading.temperature,
                reading.humidity,
                reading.moisture,
                reading.nitrogen,
                reading.potassium,
                reading.phosphorous,
                prediction["recommended"],
                prediction["confidence_pct"],
                dosage["exact_bags"],
                dosage["estimated_cost_inr"],
                prediction["model_used"],
                json.dumps(diagnostics),
            ),
        )
        self.conn.commit()
        return self.get_by_id(int(cur.lastrowid))  # type: ignore[return-value]

    def get_by_id(self, record_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["diagnostics"] = json.loads(item.pop("diagnostics_json"))
        return item

    def list_recent(self, limit: int = 50, plot_id: int | None = None) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 200))
        if plot_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM predictions WHERE plot_id = ? ORDER BY id DESC LIMIT ?",
                (plot_id, safe_limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["diagnostics"] = json.loads(item.pop("diagnostics_json"))
            results.append(item)
        return results

    def delete(self, record_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def export_csv(self, limit: int = 200, plot_id: int | None = None) -> str:
        rows = self.list_recent(limit=limit, plot_id=plot_id)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "created_at",
                "plot_name",
                "area_acres",
                "crop_stage",
                "soil_ph",
                "soil_type",
                "crop_type",
                "temperature",
                "humidity",
                "moisture",
                "nitrogen",
                "potassium",
                "phosphorous",
                "recommended_fertilizer",
                "confidence_pct",
                "bags_required",
                "estimated_cost_inr",
                "model_used",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["id"],
                    r["created_at"],
                    r["plot_name"],
                    r["area_acres"],
                    r["crop_stage"],
                    r["soil_ph"],
                    r["soil_type"],
                    r["crop_type"],
                    r["temperature"],
                    r["humidity"],
                    r["moisture"],
                    r["nitrogen"],
                    r["potassium"],
                    r["phosphorous"],
                    r["recommended_fertilizer"],
                    r["confidence_pct"],
                    r["bags_required"],
                    r["estimated_cost_inr"],
                    r["model_used"],
                ]
            )
        return buf.getvalue()
