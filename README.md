# Smart Fertilizer Advisor v2 — SSNM Field Advisor

Production-ready precision agriculture & Site-Specific Nutrient Management (SSNM) web platform built with **Flask**, **scikit-learn**, **SQLite**, and an offline-capable **PWA** dashboard.

## Features Implemented

### 1. Farmer Decision Tools
- **Acreage Dosage, Bag Count & Cost Calculator (`domain.py`):** Converts Acres/Hectares (`1 ha = 2.47105 ac`), sizes bags accurately (`45 kg` for neem-coated Urea at `₹266.50` Govt MRP; `50 kg` for DAP at `₹1,350` capped MRP & indicative NBS blends with custom dealer-price override), and surfaces lower-cost alternatives.
- **Crop Split-Application Timeline (`build_split_schedule`):** 3-stage application schedule (Basal at Sowing, Active Vegetative, Reproductive/Fruit Fill) tailored by crop group (`cereal`, `cash`, `legume_oilseed`, `horticulture`).
- **Soil pH Lockout & Live Rain-Runoff Warning (`analyze_npk_and_ph`, `weather.py`):** Evaluates acidic `< 5.5` (lime advisory) and alkaline `> 8.2` (gypsum/ammonia-loss advisory) pH lockouts plus live 24h Open-Meteo precipitation checks via city name or GPS.
- **Organic & Bio-Fertilizer Substitution Guide:** Pairs detected N, P, K deficiencies with *Azospirillum*/*Rhizobium*, *PSB*, *KMB*, *FYM*/vermicompost, and green manure guidance.
- **Multilingual UI & Voice Readout:** Live language selector (**English, Tamil, Hindi, Telugu, Kannada, Malayalam**) with `Web Speech API` readout.
- **WhatsApp & SMS Prescription Share:** One-click formatted summary sharing and print-optimized prescription slip.

### 2. Platform & ML Reliability Features
- **Farm Plot Profiles & Multi-Season N-P-K Tracking (`storage.py`):** SQLite `plots` + `predictions` tables with an SVG multi-season N-P-K trend chart (`/api/plots/<id>/trends`).
- **Batch CSV Soil-Lab Upload (`/api/batch-predict`):** Upload farmer/lab CSV files and download enriched fertilizer/bag/deficiency spreadsheets.
- **Out-of-Distribution (OOD) & Low-Confidence Guardrails (`ml_pipeline.py`):** Calibrated 1-NN Euclidean distance in `StandardScaler` space (`p99 = 2.17`) and `< 55%` confidence safety banners.
- **Offline-Ready PWA Mode (`/manifest.webmanifest`, `/sw.js`):** Service Worker asset caching plus `localStorage` offline submission queue with automatic sync on reconnect.

## Quick Start

```powershell
cd C:\Users\Hithesh\.gemini\antigravity\scratch\smart-fertilizer-advisor
.\.venv\Scripts\python.exe -m smart_fertilizer.ml_pipeline
.\.venv\Scripts\python.exe run.py --port 5055
.\.venv\Scripts\python.exe -m pytest -v
```
