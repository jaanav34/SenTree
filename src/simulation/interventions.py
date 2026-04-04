"""
Define intervention parameter deltas.

Each delta maps to a feature in the 11-feature vector:
  [temp, precip, temp_vol, temp_mom, precip_vol, precip_mom,
   gdp, pop, soil_moisture, coastal_factor, tail_risk_score]

Mangrove restoration is backed by literature showing:
  - 20-50% reduction in wave energy / storm surge volatility
  - 0.3-1.0 C coastal cooling via evapotranspiration
  - Carbon sequestration offsets

Regenerative agriculture:
  - 10-25% improvement in soil moisture retention
  - 5-15% GDP uplift from improved yields
  - Reduced drought sensitivity
"""

INTERVENTIONS = {
    'mangrove_restoration': {
        'name': 'Coastal Mangrove Restoration',
        'cost_usd': 1_000_000_000,
        'deltas': {
            'temp_reduction': 0.80,               # C — coastal cooling
            'temp_volatility_reduction': 0.15,     # 15% — thermal buffering
            'temp_momentum_reduction': 0.10,       # 10% — slows warming rate
            'precip_volatility_reduction': 0.50,   # 50% — storm surge buffer
            'precip_momentum_reduction': 0.25,     # 25% — stabilizes precip
            'soil_moisture_boost': 0.10,           # mangrove root systems
            'tail_risk_reduction': 0.30,           # 30% — direct tail-risk mitigation
            'coastal_only': True,
            # Prefer coastal-factor mask when available; fallback lon threshold kept for demo.
            'coastal_factor_threshold': 0.60,
            'coastal_lon_threshold': 120,
        },
        'description': 'Mangrove buffer reduces storm surge and precipitation volatility along coastlines',
    },
    'regenerative_agriculture': {
        'name': 'Regenerative Agriculture',
        'cost_usd': 1_000_000_000,
        'deltas': {
            'temp_reduction': 0.30,               # C — albedo + evapotranspiration
            'temp_volatility_reduction': 0.10,     # 10% — microclimate stabilization
            'precip_volatility_reduction': 0.25,   # 25% — soil moisture buffering
            'precip_momentum_reduction': 0.10,     # 10% — drought resilience
            'soil_moisture_boost': 0.20,           # 20% — core mechanism
            'gdp_boost_factor': 1.15,             # 15% — improved yields
            'tail_risk_reduction': 0.15,           # 15% — moderate risk reduction
            'coastal_only': False,
        },
        'description': 'Soil health improvements buffer against drought and boost agricultural output',
    },
}
