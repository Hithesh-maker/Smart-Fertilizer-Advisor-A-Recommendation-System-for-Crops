"""Domain normalization, input validation, dosage/cost calculation, split schedules, pH, and organic advisory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

VALID_SOILS = ("Black", "Clayey", "Loamy", "Red", "Sandy")
VALID_STAGES = ("sowing", "vegetative", "reproductive")
VALID_AREA_UNITS = ("acre", "hectare")
ACRES_PER_HECTARE = 2.47105

CROP_ALIASES = {
    "barley": "Barley",
    "coffee": "Coffee",
    "cotton": "Cotton",
    "ground nuts": "Ground Nuts",
    "groundnuts": "Ground Nuts",
    "kidney beans": "Kidney Beans",
    "kidneybeans": "Kidney Beans",
    "maize": "Maize",
    "millets": "Millets",
    "oil seeds": "Oil Seeds",
    "oilseeds": "Oil Seeds",
    "orange": "Orange",
    "paddy": "Paddy",
    "pomegranate": "Pomegranate",
    "pulses": "Pulses",
    "rice": "Rice",
    "sugarcane": "Sugarcane",
    "tobacco": "Tobacco",
    "watermelon": "Watermelon",
    "wheat": "Wheat",
}

VALID_CROPS = tuple(sorted(set(CROP_ALIASES.values())))

# Standard Indian fertilizer bag sizes & verified subsidized / NBS indicative MRPs (INR)
# Sources: Dept. of Fertilizers India (Urea 45 kg ₹266.50 inclusive of neem coating/GST; DAP 50 kg ₹1,350; MOP/NPK indicative NBS averages).
FERTILIZER_PROFILES: dict[str, dict[str, Any]] = {
    "Urea": {
        "npk_ratio": "46-0-0",
        "focus": "High Nitrogen",
        "bag_weight_kg": 45.0,
        "ref_price_inr": 266.50,
        "price_basis": "Govt-notified 45 kg neem-coated MRP (~₹266.50)",
        "base_kg_per_acre": 48.0,
        "dosage_hint": "Split across basal/vegetative/panicle stages; avoid application in standing floodwater or before heavy rain.",
        "notes": "Ideal when soil nitrogen is deficient while P and K reserves remain adequate.",
    },
    "DAP": {
        "npk_ratio": "18-46-0",
        "focus": "High Phosphorous + Starter Nitrogen",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1350.0,
        "price_basis": "Capped 50 kg MRP (₹1,350)",
        "base_kg_per_acre": 50.0,
        "dosage_hint": "Apply as basal dose near the root zone at sowing/transplanting.",
        "notes": "Supports early root development and tillering in phosphorus-deficient soils.",
    },
    "TSP": {
        "npk_ratio": "0-46-0",
        "focus": "Concentrated Phosphorous",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1250.0,
        "price_basis": "Indicative 50 kg market reference (check bag MRP)",
        "base_kg_per_acre": 45.0,
        "dosage_hint": "Incorporate into soil prior to planting when nitrogen is already sufficient.",
        "notes": "Delivers high phosphorus without raising soil nitrogen.",
    },
    "Superphosphate": {
        "npk_ratio": "0-16-0 (+Ca, S)",
        "focus": "Moderate Phosphorous & Sulfur",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 500.0,
        "price_basis": "Indicative 50 kg SSP NBS reference (check bag MRP)",
        "base_kg_per_acre": 75.0,
        "dosage_hint": "Apply pre-planting in sulfur-responsive oilseed, pulse, or neutral/alkaline soils.",
        "notes": "Supplies phosphorus alongside calcium (21%) and sulfur (11%).",
    },
    "Potassium Sulfate": {
        "npk_ratio": "0-0-50 (+18S)",
        "focus": "Chloride-Free Potassium",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 2400.0,
        "price_basis": "Indicative 50 kg SOP reference (check bag MRP)",
        "base_kg_per_acre": 35.0,
        "dosage_hint": "Preferred potash source for chloride-sensitive crops (pomegranate, orange, tobacco).",
        "notes": "Improves fruit quality, drought resilience, and sugar translocation.",
    },
    "Potassium Chloride": {
        "npk_ratio": "0-0-60",
        "focus": "Concentrated Potassium (MOP)",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1700.0,
        "price_basis": "Indicative 50 kg NBS MOP range (~₹1,650–₹1,750)",
        "base_kg_per_acre": 30.0,
        "dosage_hint": "Broadcast before sowing or split with nitrogen on cereals; avoid on saline soils.",
        "notes": "Cost-effective potash booster when N and P levels are balanced.",
    },
    "28-28": {
        "npk_ratio": "28-28-0",
        "focus": "Balanced Nitrogen & Phosphorous",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1600.0,
        "price_basis": "Indicative 50 kg NBS complex reference (check bag MRP)",
        "base_kg_per_acre": 50.0,
        "dosage_hint": "Effective basal or early top-dress blend where soil K is sufficient.",
        "notes": "Drives simultaneous canopy growth and root establishment.",
    },
    "20-20": {
        "npk_ratio": "20-20-0",
        "focus": "Moderate N-P Blend",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1480.0,
        "price_basis": "Indicative 50 kg NBS 20-20-0-13 range (~₹1,410–₹1,550)",
        "base_kg_per_acre": 55.0,
        "dosage_hint": "Use for steady vegetative and root nutrition in loamy/clayey soils.",
        "notes": "Balanced N-P availability with reduced nitrogen leaching risk.",
    },
    "17-17-17": {
        "npk_ratio": "17-17-17",
        "focus": "Equal Complete NPK",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1550.0,
        "price_basis": "Indicative 50 kg complex reference (check bag MRP)",
        "base_kg_per_acre": 55.0,
        "dosage_hint": "General-purpose basal dressing across diverse grains and vegetables.",
        "notes": "Restores multi-nutrient soil equilibrium uniformly.",
    },
    "15-15-15": {
        "npk_ratio": "15-15-15",
        "focus": "Balanced Complete NPK",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1450.0,
        "price_basis": "Indicative 50 kg complex reference (check bag MRP)",
        "base_kg_per_acre": 60.0,
        "dosage_hint": "Apply in split doses for perennial orchards and row crops.",
        "notes": "Gentle balanced ratio for maintaining moderate nutrient reserves.",
    },
    "14-35-14": {
        "npk_ratio": "14-35-14",
        "focus": "Phosphorous-Heavy Complete NPK",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1750.0,
        "price_basis": "Indicative 50 kg NBS complex reference (check bag MRP)",
        "base_kg_per_acre": 50.0,
        "dosage_hint": "Apply at planting for crops needing strong early root vigor.",
        "notes": "Combines high P availability with starter N and K.",
    },
    "14-14-14": {
        "npk_ratio": "14-14-14",
        "focus": "Controlled Complete NPK",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1420.0,
        "price_basis": "Indicative 50 kg complex reference (check bag MRP)",
        "base_kg_per_acre": 60.0,
        "dosage_hint": "Steady maintenance formula for fruits and horticultural crops.",
        "notes": "Balances N-P-K release without burning sensitive roots.",
    },
    "10-26-26": {
        "npk_ratio": "10-26-26",
        "focus": "High Phosphorous & Potassium",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1850.0,
        "price_basis": "Indicative 50 kg NBS range (~₹1,810–₹1,980)",
        "base_kg_per_acre": 50.0,
        "dosage_hint": "Apply during reproductive, tuber, or fruit-filling stages.",
        "notes": "Ideal when nitrogen is adequate but P and K require replenishment.",
    },
    "10-10-10": {
        "npk_ratio": "10-10-10",
        "focus": "Light Balanced NPK",
        "bag_weight_kg": 50.0,
        "ref_price_inr": 1250.0,
        "price_basis": "Indicative 50 kg maintenance reference (check bag MRP)",
        "base_kg_per_acre": 55.0,
        "dosage_hint": "Maintenance top-dress for soils near optimal fertility.",
        "notes": "Low-concentration formula preventing over-fertilization.",
    },
}

DEFAULT_NPK_TARGET = {"N": (20, 45), "P": (15, 40), "K": (15, 40)}

CROP_NPK_OVERRIDES: dict[str, dict[str, tuple[int, int]]] = {
    "Rice": {"N": (55, 100), "P": (25, 50), "K": (30, 60)},
    "Paddy": {"N": (30, 55), "P": (15, 35), "K": (10, 30)},
    "Wheat": {"N": (30, 55), "P": (15, 35), "K": (10, 30)},
    "Sugarcane": {"N": (30, 60), "P": (20, 45), "K": (20, 50)},
    "Cotton": {"N": (25, 50), "P": (15, 35), "K": (15, 40)},
    "Coffee": {"N": (60, 110), "P": (20, 45), "K": (25, 50)},
}

CROP_GROUPS: dict[str, str] = {
    "Rice": "cereal",
    "Paddy": "cereal",
    "Wheat": "cereal",
    "Maize": "cereal",
    "Barley": "cereal",
    "Millets": "cereal",
    "Cotton": "cash",
    "Sugarcane": "cash",
    "Tobacco": "cash",
    "Pulses": "legume_oilseed",
    "Ground Nuts": "legume_oilseed",
    "Kidney Beans": "legume_oilseed",
    "Oil Seeds": "legume_oilseed",
    "Pomegranate": "horticulture",
    "Orange": "horticulture",
    "Watermelon": "horticulture",
    "Coffee": "horticulture",
}

STAGE_SPLITS_BY_GROUP: dict[str, list[dict[str, Any]]] = {
    "cereal": [
        {"stage": "sowing", "label": "Basal at Sowing / Transplanting", "timing": "Day 0", "pct": 40, "action": "Incorporate basal dose into moist root zone before sowing/transplanting."},
        {"stage": "vegetative", "label": "Active Tillering / Vegetative", "timing": "Days 25–30", "pct": 35, "action": "Side-dress or broadcast onto drained moist field for canopy growth."},
        {"stage": "reproductive", "label": "Panicle Initiation / Booting", "timing": "Days 50–60", "pct": 25, "action": "Final split to boost grain filling and panicle weight."},
    ],
    "cash": [
        {"stage": "sowing", "label": "Basal Furrow Placement", "timing": "Day 0", "pct": 35, "action": "Apply deep in furrows below seed/sett placement."},
        {"stage": "vegetative", "label": "Grand Growth / Square Formation", "timing": "Days 30–45", "pct": 40, "action": "Apply around root zone followed by light earthing up and irrigation."},
        {"stage": "reproductive", "label": "Boll / Cane Maturation", "timing": "Days 65–80", "pct": 25, "action": "Support boll retention / sucrose accumulation; avoid late excess N."},
    ],
    "legume_oilseed": [
        {"stage": "sowing", "label": "Basal Starter Application", "timing": "Day 0", "pct": 60, "action": "Legumes fix atmospheric N; prioritize basal P/K and sulfur at sowing."},
        {"stage": "vegetative", "label": "Pre-Flowering / Pegging", "timing": "Days 20–30", "pct": 25, "action": "Light supplementary feeding during branching and nodulation."},
        {"stage": "reproductive", "label": "Pod / Seed Filling", "timing": "Days 45–55", "pct": 15, "action": "Support pod fill and oil synthesis without stimulating excess foliage."},
    ],
    "horticulture": [
        {"stage": "sowing", "label": "Pre-Flush / Basal Ring Dressing", "timing": "Start of Season", "pct": 35, "action": "Apply in drip/ring basin 30 cm away from trunk and mulch well."},
        {"stage": "vegetative", "label": "New Flush & Canopy Expansion", "timing": "Days 30–40", "pct": 30, "action": "Support shoot vigor before flower bud differentiation."},
        {"stage": "reproductive", "label": "Fruit Set & Sizing", "timing": "Days 60–90", "pct": 35, "action": "Boost K/P availability for fruit sugar, firmness, and shelf life."},
    ],
}

PRESET_SAMPLES: list[dict[str, Any]] = [
    {
        "label": "Paddy • Clayey (High N Need)",
        "temperature": 28.0,
        "humidity": 54.0,
        "moisture": 41.0,
        "soil_type": "Clayey",
        "crop_type": "Paddy",
        "nitrogen": 36.0,
        "potassium": 0.0,
        "phosphorous": 0.0,
        "soil_ph": 6.4,
        "area": 2.0,
        "area_unit": "acre",
        "crop_stage": "vegetative",
    },
    {
        "label": "Cotton • Black Soil (P-Boost)",
        "temperature": 31.5,
        "humidity": 62.0,
        "moisture": 44.0,
        "soil_type": "Black",
        "crop_type": "Cotton",
        "nitrogen": 14.0,
        "potassium": 10.0,
        "phosphorous": 38.0,
        "soil_ph": 7.7,
        "area": 3.0,
        "area_unit": "acre",
        "crop_stage": "sowing",
    },
    {
        "label": "Rice • Loamy Soil",
        "temperature": 25.0,
        "humidity": 84.0,
        "moisture": 32.0,
        "soil_type": "Loamy",
        "crop_type": "Rice",
        "nitrogen": 66.0,
        "potassium": 59.0,
        "phosphorous": 36.0,
        "soil_ph": 6.2,
        "area": 1.5,
        "area_unit": "acre",
        "crop_stage": "sowing",
    },
    {
        "label": "Pomegranate • Red Soil",
        "temperature": 27.0,
        "humidity": 58.0,
        "moisture": 40.0,
        "soil_type": "Red",
        "crop_type": "Pomegranate",
        "nitrogen": 18.0,
        "potassium": 28.0,
        "phosphorous": 24.0,
        "soil_ph": 5.3,
        "area": 1.0,
        "area_unit": "hectare",
        "crop_stage": "reproductive",
    },
]


class InputValidationError(ValueError):
    """Raised when incoming agronomic telemetry violates domain bounds."""


@dataclass(frozen=True)
class SoilReading:
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    nitrogen: float
    potassium: float
    phosphorous: float
    soil_ph: float = 6.5
    area: float = 1.0
    area_unit: str = "acre"
    crop_stage: str = "sowing"
    plot_id: int | None = None
    plot_name: str = ""
    custom_bag_price_inr: float | None = None

    @property
    def area_acres(self) -> float:
        return round(self.area * ACRES_PER_HECTARE, 4) if self.area_unit == "hectare" else round(self.area, 4)

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "Temperature": self.temperature,
            "Humidity": self.humidity,
            "Moisture": self.moisture,
            "Soil_Type": self.soil_type,
            "Crop_Type": self.crop_type,
            "Nitrogen": self.nitrogen,
            "Potassium": self.potassium,
            "Phosphorous": self.phosphorous,
        }


def normalize_soil(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().split()).title()
    if cleaned not in VALID_SOILS:
        raise InputValidationError(f"Unsupported soil_type '{value}'. Expected one of: {', '.join(VALID_SOILS)}")
    return cleaned


def normalize_crop(value: str) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    if raw in CROP_ALIASES:
        return CROP_ALIASES[raw]
    compact = raw.replace(" ", "")
    if compact in CROP_ALIASES:
        return CROP_ALIASES[compact]
    raise InputValidationError(f"Unsupported crop_type '{value}'. Expected one of: {', '.join(VALID_CROPS)}")


def normalize_fertilizer(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().rstrip(".").split())
    if cleaned.lower() == "potassium sulfate":
        return "Potassium Sulfate"
    if cleaned.lower() == "potassium chloride":
        return "Potassium Chloride"
    return cleaned


def _parse_bounded_float(raw: Any, field: str, min_val: float, max_val: float) -> float:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise InputValidationError(f"Field '{field}' is required")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Field '{field}' must be numeric") from exc
    if math.isnan(val) or math.isinf(val) or not (min_val <= val <= max_val):
        raise InputValidationError(f"Field '{field}' must be between {min_val} and {max_val}")
    return round(val, 2)


def validate_soil_reading(payload: dict[str, Any]) -> SoilReading:
    """Validate and coerce incoming JSON/form sensor & field management readings."""
    if not isinstance(payload, dict):
        raise InputValidationError("Request payload must be a JSON object")

    raw_ph = payload.get("soil_ph", payload.get("ph"))
    soil_ph = _parse_bounded_float(6.5 if raw_ph in (None, "") else raw_ph, "soil_ph", 3.5, 9.5)

    raw_area = payload.get("area", payload.get("area_acres"))
    area = _parse_bounded_float(1.0 if raw_area in (None, "") else raw_area, "area", 0.05, 5000.0)

    raw_unit = str(payload.get("area_unit") or "acre").strip().lower()
    if raw_unit not in VALID_AREA_UNITS:
        raise InputValidationError(f"Unsupported area_unit '{raw_unit}'. Use 'acre' or 'hectare'")

    raw_stage = str(payload.get("crop_stage") or "sowing").strip().lower()
    if raw_stage not in VALID_STAGES:
        raise InputValidationError(f"Unsupported crop_stage '{raw_stage}'. Use sowing, vegetative, or reproductive")

    raw_price = payload.get("custom_bag_price_inr")
    custom_price: float | None = None
    if raw_price not in (None, ""):
        custom_price = _parse_bounded_float(raw_price, "custom_bag_price_inr", 10.0, 20000.0)

    raw_plot_id = payload.get("plot_id")
    plot_id: int | None = None
    if raw_plot_id not in (None, ""):
        try:
            plot_id = int(raw_plot_id)
        except (TypeError, ValueError) as exc:
            raise InputValidationError("Field 'plot_id' must be an integer") from exc

    return SoilReading(
        temperature=_parse_bounded_float(payload.get("temperature", payload.get("temp")), "temperature", -10.0, 60.0),
        humidity=_parse_bounded_float(payload.get("humidity", payload.get("humid")), "humidity", 0.0, 100.0),
        moisture=_parse_bounded_float(payload.get("moisture", payload.get("mois")), "moisture", 0.0, 100.0),
        soil_type=normalize_soil(payload.get("soil_type", payload.get("soil"))),
        crop_type=normalize_crop(payload.get("crop_type", payload.get("crop"))),
        nitrogen=_parse_bounded_float(payload.get("nitrogen", payload.get("nitro")), "nitrogen", 0.0, 300.0),
        potassium=_parse_bounded_float(payload.get("potassium", payload.get("pota")), "potassium", 0.0, 300.0),
        phosphorous=_parse_bounded_float(payload.get("phosphorous", payload.get("phos")), "phosphorous", 0.0, 300.0),
        soil_ph=soil_ph,
        area=area,
        area_unit=raw_unit,
        crop_stage=raw_stage,
        plot_id=plot_id,
        plot_name=str(payload.get("plot_name") or "").strip()[:80],
        custom_bag_price_inr=custom_price,
    )


def analyze_npk_and_ph(reading: SoilReading) -> dict[str, Any]:
    """Evaluate soil N-P-K targets, soil pH nutrient lockout chemistry, and weather hazards."""
    targets = CROP_NPK_OVERRIDES.get(reading.crop_type, DEFAULT_NPK_TARGET)
    nutrients: dict[str, Any] = {}
    deficient_keys: list[str] = []
    for key, val in (("N", reading.nitrogen), ("P", reading.phosphorous), ("K", reading.potassium)):
        low, high = targets[key]
        status = "Deficient" if val < low else ("Excess" if val > high else "Optimal")
        if status == "Deficient":
            deficient_keys.append(key)
        nutrients[key] = {"value": val, "target_min": low, "target_max": high, "status": status}

    alerts: list[str] = []
    if reading.moisture < 25:
        alerts.append("Low soil moisture (<25%): irrigate lightly before fertilizer application to prevent root scorch.")
    elif reading.moisture > 75:
        alerts.append("High soil moisture (>75%): split nitrogen application to minimize leaching runoff.")
    if reading.temperature > 38:
        alerts.append("High ambient temperature (>38°C): apply nitrogen during early morning or evening to curb volatilization.")

    # Soil pH lockout analysis
    ph = reading.soil_ph
    if ph < 5.5:
        ph_status = "Strongly Acidic"
        ph_advice = (
            "Acidic soil (<5.5) reduces Phosphorus availability (Fe/Al fixation) and suppresses microbial N fixation. "
            "Apply agricultural lime (CaCO3, typically 200–400 kg/acre based on buffer pH) 2–3 weeks before sowing."
        )
        alerts.append(f"pH {ph} ({ph_status}): risk of Phosphorus lockout — lime amendment advised before heavy P/N dosing.")
    elif ph < 6.0:
        ph_status = "Slightly Acidic"
        ph_advice = "Mild acidity (5.5–5.9). Suitable for tea/coffee/potatoes; monitor Ca/Mg and P uptake for cereals."
    elif ph <= 7.5:
        ph_status = "Optimal Availability"
        ph_advice = "Optimal pH window (6.0–7.5) for balanced N, P, K, and micronutrient absorption."
    elif ph <= 8.2:
        ph_status = "Moderately Alkaline"
        ph_advice = "Alkaline pH (7.6–8.2) slows Zinc, Iron, and Phosphorus uptake. Favor band-placed P + organic compost."
    else:
        ph_status = "Strongly Alkaline / Sodic"
        ph_advice = (
            "High pH (>8.2) causes Phosphorus precipitation and severe Urea ammonia volatilization. "
            "Incorporate gypsum (CaSO4·2H2O) + FYM/green manure and bury/irrigate nitrogen promptly."
        )
        alerts.append(f"pH {ph} ({ph_status}): risk of ammonia volatilization & P tie-up — consider gypsum + soil incorporation.")

    return {
        "nutrients": nutrients,
        "deficient_keys": deficient_keys,
        "ph_analysis": {"ph": ph, "status": ph_status, "advice": ph_advice},
        "alerts": alerts,
    }


def get_fertilizer_profile(name: str) -> dict[str, Any]:
    norm = normalize_fertilizer(name)
    profile = FERTILIZER_PROFILES.get(
        norm,
        {
            "npk_ratio": "Custom Blend",
            "focus": "Targeted Soil Nutrition",
            "bag_weight_kg": 50.0,
            "ref_price_inr": 1500.0,
            "price_basis": "Indicative 50 kg market reference",
            "base_kg_per_acre": 50.0,
            "dosage_hint": "Follow local soil-testing lab split dosage guidelines.",
            "notes": "Matched by soil-climate classification pipeline.",
        },
    )
    return {"name": norm, **profile}


def calculate_dosage_and_cost(
    fertilizer_name: str,
    reading: SoilReading,
    diagnostics: dict[str, Any],
    top_alternatives: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute realistic dosage (kg/acre, total kg), bag count (45/50 kg), cost (INR), and cheaper alternative."""
    prof = get_fertilizer_profile(fertilizer_name)
    base_rate = float(prof["base_kg_per_acre"])

    # Adjust dosage +/- 15% based on how many primary nutrients are deficient
    def_count = len(diagnostics.get("deficient_keys", []))
    factor = 0.85 if def_count == 0 else (1.0 if def_count == 1 else 1.15)
    kg_per_acre = round(base_rate * factor, 1)
    total_kg = round(kg_per_acre * reading.area_acres, 1)

    bag_weight = float(prof["bag_weight_kg"])
    exact_bags = round(total_kg / bag_weight, 2)
    bags_to_buy = max(1, math.ceil(exact_bags)) if total_kg >= bag_weight * 0.5 else round(exact_bags, 2)
    unit_price = reading.custom_bag_price_inr if reading.custom_bag_price_inr is not None else float(prof["ref_price_inr"])
    estimated_cost_inr = round(exact_bags * unit_price, 2)
    bag_purchase_cost_inr = round(math.ceil(exact_bags) * unit_price, 2)

    lower_cost_alt = None
    for alt in top_alternatives or []:
        alt_name = alt.get("fertilizer")
        if not alt_name or alt_name == prof["name"]:
            continue
        alt_prof = get_fertilizer_profile(alt_name)
        alt_kg = round(float(alt_prof["base_kg_per_acre"]) * factor * reading.area_acres, 1)
        alt_bags = round(alt_kg / float(alt_prof["bag_weight_kg"]), 2)
        alt_cost = round(alt_bags * float(alt_prof["ref_price_inr"]), 2)
        if alt_cost < estimated_cost_inr and (lower_cost_alt is None or alt_cost < lower_cost_alt["estimated_cost_inr"]):
            lower_cost_alt = {
                "fertilizer": alt_prof["name"],
                "npk_ratio": alt_prof["npk_ratio"],
                "confidence_pct": alt.get("confidence_pct", 0.0),
                "total_kg": alt_kg,
                "bag_weight_kg": alt_prof["bag_weight_kg"],
                "exact_bags": alt_bags,
                "estimated_cost_inr": alt_cost,
                "savings_inr": round(estimated_cost_inr - alt_cost, 2),
            }

    return {
        "area_acres": reading.area_acres,
        "input_area": reading.area,
        "area_unit": reading.area_unit,
        "kg_per_acre": kg_per_acre,
        "total_kg": total_kg,
        "bag_weight_kg": bag_weight,
        "exact_bags": exact_bags,
        "recommended_bags_to_buy": math.ceil(exact_bags),
        "ref_bag_price_inr": unit_price,
        "price_basis": "Custom user dealer price" if reading.custom_bag_price_inr else prof["price_basis"],
        "estimated_cost_inr": estimated_cost_inr,
        "bag_purchase_cost_inr": bag_purchase_cost_inr,
        "lower_cost_alternative": lower_cost_alt,
    }


def build_split_schedule(reading: SoilReading, total_kg: float) -> list[dict[str, Any]]:
    """Build crop-group split-application timeline summing exactly to total_kg."""
    group = CROP_GROUPS.get(reading.crop_type, "cereal")
    template = STAGE_SPLITS_BY_GROUP[group]
    schedule: list[dict[str, Any]] = []
    running = 0.0
    for idx, step in enumerate(template):
        if idx == len(template) - 1:
            stage_kg = round(total_kg - running, 1)
        else:
            stage_kg = round(total_kg * (step["pct"] / 100.0), 1)
            running += stage_kg
        schedule.append(
            {
                **step,
                "amount_kg": stage_kg,
                "is_current_stage": step["stage"] == reading.crop_stage,
            }
        )
    return schedule


def recommend_organic_alternatives(reading: SoilReading, diagnostics: dict[str, Any]) -> list[dict[str, str]]:
    """Provide accurate bio-fertilizer and organic soil pairings matching detected N/P/K deficiencies."""
    deficient = set(diagnostics.get("deficient_keys", []))
    recs: list[dict[str, str]] = []
    group = CROP_GROUPS.get(reading.crop_type, "cereal")

    if "N" in deficient or not deficient:
        if group == "legume_oilseed":
            recs.append(
                {
                    "target": "Nitrogen (N)",
                    "option": "Rhizobium Culture + Vermicompost",
                    "rate": "200 g/acre seed inoculation + 1–1.5 tonnes/acre compost",
                    "benefit": "Enhances biological root-nodule N fixation cleanly for legumes/pulses.",
                }
            )
        else:
            recs.append(
                {
                    "target": "Nitrogen (N)",
                    "option": "Azospirillum / Azotobacter + Neem-Cake / FYM",
                    "rate": "2 kg/acre soil mix + 2 tonnes/acre well-rotted FYM",
                    "benefit": "Supplies slow-release organic N and cuts chemical Urea leaching losses.",
                }
            )

    if "P" in deficient:
        recs.append(
            {
                "target": "Phosphorous (P)",
                "option": "Phosphate Solubilizing Bacteria (PSB) + Enriched Compost",
                "rate": "2 kg/acre mixed with 50 kg compost",
                "benefit": "Solubilizes bound soil phosphates (especially valuable in high/low pH soils).",
            }
        )

    if "K" in deficient:
        recs.append(
            {
                "target": "Potassium (K)",
                "option": "Potash Mobilizing Bacteria (KMB) + Wood Ash / Coir Compost",
                "rate": "2 kg/acre KMB + 100–150 kg/acre wood ash",
                "benefit": "Improves drought tolerance, stalk strength, and fruit/grain quality.",
            }
        )

    if reading.crop_type in ("Rice", "Paddy", "Sugarcane"):
        recs.append(
            {
                "target": "Soil Organic Matter",
                "option": "Green Manure (Dhaincha / Sunhemp) or Azolla",
                "rate": "Plough in before transplanting (45 days growth)",
                "benefit": "Adds 20–30 kg/acre natural nitrogen and improves soil water retention.",
            }
        )
    return recs


def metadata_payload() -> dict[str, Any]:
    return {
        "soils": list(VALID_SOILS),
        "crops": list(VALID_CROPS),
        "stages": list(VALID_STAGES),
        "area_units": list(VALID_AREA_UNITS),
        "fertilizers": sorted(FERTILIZER_PROFILES.keys()),
        "fertilizer_catalog": [{"name": k, **v} for k, v in FERTILIZER_PROFILES.items()],
        "presets": PRESET_SAMPLES,
        "bounds": {
            "temperature": [-10, 60],
            "humidity": [0, 100],
            "moisture": [0, 100],
            "nitrogen": [0, 300],
            "potassium": [0, 300],
            "phosphorous": [0, 300],
            "soil_ph": [3.5, 9.5],
            "area": [0.05, 5000],
        },
    }
