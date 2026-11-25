"""Live Open-Meteo weather lookup and rain-runoff fertilizer advisory check."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Callable

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _default_http_get_json(url: str, timeout: float = 8.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "SmartFertilizerAdvisor/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def evaluate_rain_risk(precip_24h_mm: float, max_prob_pct: float) -> dict[str, str]:
    """Classify rainfall runoff risk for top-dressed fertilizers."""
    if precip_24h_mm >= 10.0 or max_prob_pct >= 65.0:
        return {
            "level": "HIGH",
            "headline": "Rain expected within 24h — delay top-dressing",
            "advice": (
                f"Forecast shows {precip_24h_mm:.1f} mm precipitation ({max_prob_pct:.0f}% max probability) over the next 24h. "
                "Delay Urea or surface N application to prevent leaching and runoff loss."
            ),
        }
    if precip_24h_mm >= 2.0 or max_prob_pct >= 40.0:
        return {
            "level": "MODERATE",
            "headline": "Light showers possible — apply and incorporate carefully",
            "advice": (
                f"Forecast shows {precip_24h_mm:.1f} mm ({max_prob_pct:.0f}% probability). "
                "Light moisture aids incorporation, but avoid heavy doses on sloped fields."
            ),
        }
    return {
        "level": "LOW",
        "headline": "Clear 24h window for fertilizer application",
        "advice": (
            f"Minimal rain expected ({precip_24h_mm:.1f} mm, {max_prob_pct:.0f}% probability). "
            "Safe window to apply and lightly irrigate if soil moisture is below 35%."
        ),
    }


def check_weather_advisory(
    city: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    fetcher: Callable[[str], dict[str, Any]] = _default_http_get_json,
) -> dict[str, Any]:
    """Fetch 24h weather forecast from Open-Meteo (free, no API key) and assess runoff risk."""
    resolved_name = city.strip() if city else "Field Coordinates"

    if lat is None or lon is None:
        if not city or not city.strip():
            raise ValueError("Provide either 'city' or both 'lat' and 'lon'")
        geo_url = f"{GEOCODE_URL}?{urllib.parse.urlencode({'name': city.strip(), 'count': 1, 'language': 'en', 'format': 'json'})}"
        geo = fetcher(geo_url)
        results = geo.get("results") or []
        if not results:
            raise ValueError(f"Could not find location '{city}'")
        top = results[0]
        lat = float(top["latitude"])
        lon = float(top["longitude"])
        admin = top.get("admin1") or ""
        country = top.get("country") or ""
        resolved_name = ", ".join(part for part in (top.get("name"), admin, country) if part)

    forecast_qs = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,precipitation_probability,temperature_2m",
            "forecast_days": 2,
            "timezone": "auto",
        }
    )
    forecast = fetcher(f"{FORECAST_URL}?{forecast_qs}")
    hourly = forecast.get("hourly") or {}
    precip_list = [float(x or 0.0) for x in (hourly.get("precipitation") or [])[:24]]
    prob_list = [float(x or 0.0) for x in (hourly.get("precipitation_probability") or [])[:24]]
    temp_list = [float(x) for x in (hourly.get("temperature_2m") or [])[:24] if x is not None]

    precip_24h_mm = round(sum(precip_list), 2)
    max_prob_pct = round(max(prob_list) if prob_list else 0.0, 1)
    avg_temp_c = round(sum(temp_list) / len(temp_list), 1) if temp_list else None
    risk = evaluate_rain_risk(precip_24h_mm, max_prob_pct)

    return {
        "location": resolved_name,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "next_24h_precip_mm": precip_24h_mm,
        "max_rain_probability_pct": max_prob_pct,
        "avg_temp_24h_c": avg_temp_c,
        "rain_risk": risk,
        "source": "Open-Meteo",
    }
