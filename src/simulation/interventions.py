"""Define intervention parameter deltas and climate-fit metadata."""

from __future__ import annotations


def _intervention(
    name: str,
    *,
    cost_usd: int,
    description: str,
    category: str,
    search_tags: list[str],
    deltas: dict,
) -> dict:
    return {
        "name": name,
        "cost_usd": int(cost_usd),
        "description": description,
        "category": category,
        "search_tags": list(search_tags),
        "deltas": deltas,
    }


INTERVENTIONS = {
    "mangrove_restoration": _intervention(
        "Coastal Mangrove Restoration",
        cost_usd=100_000_000,
        category="coastal_nature",
        search_tags=["mangroves", "coast", "storm surge", "blue carbon", "shoreline"],
        description="Restores mangrove buffers that cool humid coasts, reduce storm-surge volatility, and absorb wave energy.",
        deltas={
            "temp_reduction": 0.80,
            "temp_volatility_reduction": 0.15,
            "temp_momentum_reduction": 0.10,
            "precip_volatility_reduction": 0.50,
            "precip_momentum_reduction": 0.25,
            "soil_moisture_boost": 0.10,
            "tail_risk_reduction": 0.30,
            "coastal_only": True,
            "coastal_factor_threshold": 0.60,
            "kg_allow_codes": ["Af", "Am", "Aw", "As", "Cfa", "Cwa"],
            "kg_block_codes": ["BWh", "BWk", "EF", "ET"],
        },
    ),
    "regenerative_agriculture": _intervention(
        "Regenerative Agriculture",
        cost_usd=100_000_000,
        category="agriculture",
        search_tags=["soil health", "cropland", "drought resilience", "farming"],
        description="Improves soil structure, moisture retention, and crop resilience across productive landscapes.",
        deltas={
            "temp_reduction": 0.30,
            "temp_volatility_reduction": 0.10,
            "precip_volatility_reduction": 0.25,
            "precip_momentum_reduction": 0.10,
            "soil_moisture_boost": 0.20,
            "gdp_boost_factor": 1.15,
            "tail_risk_reduction": 0.15,
            "kg_block_codes": ["BWh", "BWk", "EF", "ET"],
        },
    ),
    "drip_irrigation": _intervention(
        "Precision Drip Irrigation",
        cost_usd=100_000_000,
        category="water_management",
        search_tags=["irrigation", "water efficiency", "semi-arid", "desert farming"],
        description="Targets water-stressed farms with efficient irrigation to stabilize yields under chronic heat and dryness.",
        deltas={
            "temp_reduction": 0.10,
            "precip_momentum_reduction": 0.08,
            "soil_moisture_boost": 0.22,
            "gdp_boost_factor": 1.10,
            "tail_risk_reduction": 0.12,
            "kg_allow_codes": ["BWh", "BWk", "BSh", "BSk", "Csa", "Csb", "Cwa", "Aw"],
        },
    ),
    "agroforestry_belts": _intervention(
        "Agroforestry Shelterbelts",
        cost_usd=100_000_000,
        category="agriculture",
        search_tags=["agroforestry", "tree belts", "windbreak", "crop resilience"],
        description="Adds tree cover to farms to cool fields, retain moisture, and reduce exposure to heat and wind stress.",
        deltas={
            "temp_reduction": 0.45,
            "temp_volatility_reduction": 0.12,
            "precip_volatility_reduction": 0.12,
            "soil_moisture_boost": 0.14,
            "gdp_boost_factor": 1.08,
            "tail_risk_reduction": 0.13,
            "kg_allow_codes": ["Af", "Am", "Aw", "Cfa", "Cfb", "Cwa", "Cwb", "Dfa", "Dfb"],
            "kg_block_codes": ["BWh", "BWk", "EF"],
        },
    ),
    "peatland_rewetting": _intervention(
        "Peatland Rewetting",
        cost_usd=100_000_000,
        category="wetlands",
        search_tags=["peat", "wetlands", "rewetting", "fire prevention", "carbon"],
        description="Raises water tables in wet organic soils to reduce fire risk, cooling spikes, and carbon loss.",
        deltas={
            "temp_reduction": 0.35,
            "temp_volatility_reduction": 0.10,
            "precip_volatility_reduction": 0.10,
            "soil_moisture_boost": 0.25,
            "tail_risk_reduction": 0.18,
            "kg_allow_codes": ["Af", "Am", "Cfb", "Cfc", "Dfb", "Dfc", "ET"],
            "kg_block_codes": ["BWh", "BWk"],
        },
    ),
    "cool_roofs": _intervention(
        "Urban Cool Roofs",
        cost_usd=100_000_000,
        category="urban_heat",
        search_tags=["cool roofs", "urban heat", "reflective surfaces", "cities"],
        description="Deploys high-albedo roofs to suppress urban heat loading and reduce short-term thermal volatility.",
        deltas={
            "temp_reduction": 0.55,
            "temp_volatility_reduction": 0.14,
            "temp_momentum_reduction": 0.08,
            "tail_risk_reduction": 0.10,
            "kg_block_codes": ["EF"],
        },
    ),
    "floodplain_reconnection": _intervention(
        "Floodplain Reconnection",
        cost_usd=100_000_000,
        category="flood_management",
        search_tags=["floodplain", "river", "flood control", "overflow storage"],
        description="Reconnects rivers to natural floodplains to dampen peak runoff and reduce cascade-prone flood shocks.",
        deltas={
            "temp_reduction": 0.12,
            "precip_volatility_reduction": 0.28,
            "precip_momentum_reduction": 0.18,
            "soil_moisture_boost": 0.10,
            "tail_risk_reduction": 0.17,
            "kg_allow_codes": ["Af", "Am", "Aw", "Cfa", "Cfb", "Cwa", "Dfa", "Dfb"],
        },
    ),
    "coral_reef_restoration": _intervention(
        "Coral Reef Restoration",
        cost_usd=100_000_000,
        category="coastal_nature",
        search_tags=["coral", "reef", "coastal defense", "marine habitat"],
        description="Rebuilds reef structure to break wave energy and reduce coastal storm losses in warm marine climates.",
        deltas={
            "temp_reduction": 0.18,
            "precip_volatility_reduction": 0.22,
            "tail_risk_reduction": 0.16,
            "coastal_only": True,
            "coastal_factor_threshold": 0.70,
            "kg_allow_codes": ["Af", "Am", "Aw", "As", "Cfa", "Cwa"],
            "kg_block_codes": ["BWk", "EF", "ET"],
        },
    ),
    "dune_restoration": _intervention(
        "Coastal Dune Restoration",
        cost_usd=100_000_000,
        category="coastal_defense",
        search_tags=["dunes", "sand", "coastal erosion", "shore protection"],
        description="Restores dune systems in dry coasts to cut erosion and buffer wind- and wave-driven hazard spikes.",
        deltas={
            "temp_volatility_reduction": 0.08,
            "precip_volatility_reduction": 0.16,
            "tail_risk_reduction": 0.12,
            "coastal_only": True,
            "coastal_factor_threshold": 0.60,
            "kg_allow_codes": ["BWh", "BWk", "BSh", "BSk", "Csa", "Csb"],
        },
    ),
    "watershed_reforestation": _intervention(
        "Watershed Reforestation",
        cost_usd=100_000_000,
        category="forestry",
        search_tags=["reforestation", "uplands", "watershed", "runoff control"],
        description="Rebuilds upper-basin tree cover to stabilize runoff, cool slopes, and improve moisture retention downstream.",
        deltas={
            "temp_reduction": 0.40,
            "temp_volatility_reduction": 0.12,
            "precip_volatility_reduction": 0.20,
            "soil_moisture_boost": 0.12,
            "tail_risk_reduction": 0.18,
            "kg_allow_codes": ["Am", "Aw", "Cfa", "Cfb", "Cwa", "Cwb", "Dfa", "Dfb", "Dwa", "Dwb"],
            "kg_block_codes": ["BWh", "BWk", "EF"],
        },
    ),
    "rainwater_harvesting": _intervention(
        "Rainwater Harvesting Networks",
        cost_usd=100_000_000,
        category="water_management",
        search_tags=["rainwater", "storage", "cisterns", "water security"],
        description="Captures episodic rainfall to bridge dry periods in climates with seasonal or chronic water stress.",
        deltas={
            "precip_momentum_reduction": 0.12,
            "soil_moisture_boost": 0.16,
            "gdp_boost_factor": 1.05,
            "tail_risk_reduction": 0.10,
            "kg_allow_codes": ["BWh", "BWk", "BSh", "BSk", "Aw", "Csa", "Csb", "Cwa"],
        },
    ),
    "drought_resistant_crops": _intervention(
        "Drought-Resistant Crops",
        cost_usd=100_000_000,
        category="agriculture",
        search_tags=["drought crops", "seed systems", "heat-tolerant agriculture"],
        description="Shifts crop systems toward heat- and moisture-stress tolerance in dry and strongly seasonal climates.",
        deltas={
            "temp_volatility_reduction": 0.06,
            "precip_momentum_reduction": 0.10,
            "soil_moisture_boost": 0.08,
            "gdp_boost_factor": 1.09,
            "tail_risk_reduction": 0.11,
            "kg_allow_codes": ["BWh", "BWk", "BSh", "BSk", "Aw", "Csa", "Csb", "Cwa", "Dwa", "Dwb"],
        },
    ),
    "saline_soil_rehabilitation": _intervention(
        "Saline Soil Rehabilitation",
        cost_usd=100_000_000,
        category="agriculture",
        search_tags=["salinity", "soil remediation", "delta farming", "coastal agriculture"],
        description="Recovers salt-affected farmland in dry deltas and monsoon coasts to restore productivity and infiltration.",
        deltas={
            "temp_reduction": 0.08,
            "soil_moisture_boost": 0.11,
            "gdp_boost_factor": 1.07,
            "tail_risk_reduction": 0.10,
            "kg_allow_codes": ["BSh", "BSk", "Aw", "Am", "Csa", "Cwa", "Cfa"],
            "kg_block_codes": ["EF", "ET"],
        },
    ),
    "check_dams": _intervention(
        "Check Dams and Micro-Catchments",
        cost_usd=100_000_000,
        category="water_management",
        search_tags=["check dams", "micro-catchments", "runoff capture", "erosion control"],
        description="Slows flash runoff in dry uplands to reduce erosion, recharge soils, and smooth hydrologic shocks.",
        deltas={
            "precip_volatility_reduction": 0.14,
            "precip_momentum_reduction": 0.12,
            "soil_moisture_boost": 0.14,
            "tail_risk_reduction": 0.11,
            "kg_allow_codes": ["BSh", "BSk", "Cwa", "Cwb", "Dwa", "Dwb"],
        },
    ),
    "cool_pavements": _intervention(
        "Cool Pavements",
        cost_usd=100_000_000,
        category="urban_heat",
        search_tags=["cool pavement", "urban heat island", "streets", "reflective materials"],
        description="Uses reflective paving materials to reduce urban heat accumulation and day-to-day thermal extremes.",
        deltas={
            "temp_reduction": 0.42,
            "temp_volatility_reduction": 0.10,
            "tail_risk_reduction": 0.08,
            "kg_block_codes": ["EF"],
        },
    ),
    "early_warning_systems": _intervention(
        "Climate Early Warning Systems",
        cost_usd=100_000_000,
        category="preparedness",
        search_tags=["early warning", "alerts", "preparedness", "evacuation"],
        description="Cuts realized losses by improving detection, communication, and response before climate shocks fully cascade.",
        deltas={
            "tail_risk_reduction": 0.09,
        },
    ),
    "wetland_restoration": _intervention(
        "Inland Wetland Restoration",
        cost_usd=100_000_000,
        category="wetlands",
        search_tags=["wetlands", "marsh", "flood storage", "habitat"],
        description="Restores inland wetlands that moderate floods, store water, and reduce hydrologic volatility in humid basins.",
        deltas={
            "temp_reduction": 0.18,
            "precip_volatility_reduction": 0.24,
            "soil_moisture_boost": 0.18,
            "tail_risk_reduction": 0.16,
            "kg_allow_codes": ["Af", "Am", "Aw", "Cfa", "Cfb", "Cwa", "Cwb", "Dfa", "Dfb"],
            "kg_block_codes": ["BWh", "BWk"],
        },
    ),
    "slope_stabilization": _intervention(
        "Slope Stabilization and Terracing",
        cost_usd=100_000_000,
        category="land_management",
        search_tags=["terracing", "landslides", "mountains", "slope stabilization"],
        description="Stabilizes steep, high-rainfall terrain to reduce runoff bursts and landslide-linked cascade failures.",
        deltas={
            "precip_volatility_reduction": 0.18,
            "soil_moisture_boost": 0.08,
            "tail_risk_reduction": 0.14,
            "kg_allow_codes": ["Cfb", "Cfc", "Cwb", "Dfb", "Dfc", "Dwb", "Am", "Aw"],
        },
    ),
    "riparian_buffers": _intervention(
        "Riparian Buffer Corridors",
        cost_usd=100_000_000,
        category="water_management",
        search_tags=["riparian", "river buffers", "erosion", "water quality"],
        description="Vegetated river corridors reduce erosion, cool channels, and smooth flood impacts in riverine systems.",
        deltas={
            "temp_reduction": 0.12,
            "precip_volatility_reduction": 0.16,
            "soil_moisture_boost": 0.09,
            "tail_risk_reduction": 0.13,
            "kg_allow_codes": ["Af", "Am", "Aw", "Cfa", "Cfb", "Cwa", "Cwb", "Dfa", "Dfb"],
        },
    ),
    "green_stormwater_networks": _intervention(
        "Green Stormwater Networks",
        cost_usd=100_000_000,
        category="urban_water",
        search_tags=["stormwater", "bioswales", "urban flooding", "green infrastructure"],
        description="Uses distributed urban green infrastructure to absorb intense rainfall and reduce city flood pulses.",
        deltas={
            "temp_reduction": 0.18,
            "precip_volatility_reduction": 0.22,
            "soil_moisture_boost": 0.10,
            "tail_risk_reduction": 0.14,
            "kg_allow_codes": ["Am", "Aw", "As", "Cfa", "Cfb", "Cwa"],
        },
    ),
    "managed_aquifer_recharge": _intervention(
        "Managed Aquifer Recharge",
        cost_usd=100_000_000,
        category="water_management",
        search_tags=["aquifer recharge", "groundwater", "water banking", "drought"],
        description="Stores excess runoff underground so dry climates can draw on groundwater during persistent stress periods.",
        deltas={
            "precip_momentum_reduction": 0.14,
            "soil_moisture_boost": 0.18,
            "gdp_boost_factor": 1.06,
            "tail_risk_reduction": 0.13,
            "kg_allow_codes": ["BWh", "BWk", "BSh", "BSk", "Aw", "Csa", "Csb", "Cwa"],
        },
    ),
    "seagrass_restoration": _intervention(
        "Seagrass Meadow Restoration",
        cost_usd=100_000_000,
        category="coastal_nature",
        search_tags=["seagrass", "marine meadows", "coastal habitat", "blue carbon"],
        description="Rebuilds seagrass meadows that attenuate wave energy, trap sediments, and stabilize warm shallow coasts.",
        deltas={
            "temp_reduction": 0.10,
            "precip_volatility_reduction": 0.18,
            "tail_risk_reduction": 0.13,
            "coastal_only": True,
            "coastal_factor_threshold": 0.65,
            "kg_allow_codes": ["Af", "Am", "Aw", "As", "Cfa", "Cwa", "BSh"],
            "kg_block_codes": ["BWk", "EF", "ET"],
        },
    ),
    "firebreak_landscaping": _intervention(
        "Firebreak Landscaping",
        cost_usd=100_000_000,
        category="fire_management",
        search_tags=["firebreaks", "wildfire", "fuel management", "drylands"],
        description="Reduces fuel continuity in dry climates where heat and drought can trigger abrupt wildfire-driven losses.",
        deltas={
            "temp_volatility_reduction": 0.05,
            "soil_moisture_boost": 0.04,
            "tail_risk_reduction": 0.12,
            "kg_allow_codes": ["BSh", "BSk", "Csa", "Csb", "Csc", "Dsa", "Dsb"],
        },
    ),
    "shade_tree_corridors": _intervention(
        "Urban Shade Tree Corridors",
        cost_usd=100_000_000,
        category="urban_heat",
        search_tags=["street trees", "shade", "heat adaptation", "urban greening"],
        description="Plants urban tree corridors that lower heat exposure, cool surfaces, and soften thermal momentum.",
        deltas={
            "temp_reduction": 0.48,
            "temp_volatility_reduction": 0.12,
            "soil_moisture_boost": 0.05,
            "tail_risk_reduction": 0.10,
            "kg_allow_codes": ["Af", "Am", "Aw", "As", "Cfa", "Cfb", "Cwa", "Cwb"],
            "kg_block_codes": ["BWh", "BWk", "EF"],
        },
    ),
    "living_breakwaters": _intervention(
        "Living Breakwaters",
        cost_usd=100_000_000,
        category="coastal_defense",
        search_tags=["breakwaters", "hybrid coast", "storm surge", "shoreline resilience"],
        description="Deploys hybrid reef and breakwater systems that soften coastal wave attack in exposed shorelines.",
        deltas={
            "precip_volatility_reduction": 0.24,
            "precip_momentum_reduction": 0.10,
            "tail_risk_reduction": 0.18,
            "coastal_only": True,
            "coastal_factor_threshold": 0.70,
            "kg_allow_codes": ["Af", "Am", "Aw", "As", "BSh", "Cfa", "Cwa", "Csa"],
            "kg_block_codes": ["BWk", "EF", "ET"],
        },
    ),
    "permeable_surfaces": _intervention(
        "Permeable Surface Retrofits",
        cost_usd=100_000_000,
        category="urban_water",
        search_tags=["permeable pavement", "infiltration", "urban runoff", "flooding"],
        description="Increases infiltration in built areas to reduce runoff spikes and improve local water retention.",
        deltas={
            "temp_reduction": 0.12,
            "precip_volatility_reduction": 0.15,
            "soil_moisture_boost": 0.08,
            "tail_risk_reduction": 0.09,
            "kg_block_codes": ["EF"],
        },
    ),
}


def climate_fit_summary(intervention: dict) -> str:
    """Return a compact human-readable summary of KG compatibility rules."""
    deltas = intervention.get("deltas", {})
    allow_codes = list(deltas.get("kg_allow_codes", []))
    allow_prefixes = list(deltas.get("kg_allow_prefixes", []))
    block_codes = list(deltas.get("kg_block_codes", []))
    block_prefixes = list(deltas.get("kg_block_prefixes", []))

    parts: list[str] = []
    if allow_codes:
        parts.append("best in " + ", ".join(allow_codes))
    elif allow_prefixes:
        parts.append("best in groups " + ", ".join(allow_prefixes))

    blocked: list[str] = []
    if block_codes:
        blocked.extend(block_codes)
    if block_prefixes:
        blocked.extend(block_prefixes)
    if blocked:
        parts.append("avoid " + ", ".join(blocked))

    if deltas.get("coastal_only"):
        parts.append("coastal cells only")

    return "; ".join(parts) if parts else "broad climate applicability"


def build_search_description(key: str, intervention: dict) -> str:
    """Create a semantic-search-friendly text block for an intervention."""
    tags = ", ".join(intervention.get("search_tags", []))
    climate_text = climate_fit_summary(intervention)
    return (
        f"{intervention['name']}. {intervention['description']} "
        f"Category: {intervention.get('category', 'general')}. "
        f"Keywords: {tags}. Climate fit: {climate_text}. "
        f"Intervention key: {key}."
    )

