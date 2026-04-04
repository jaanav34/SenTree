"""Define intervention parameter deltas."""

INTERVENTIONS = {
    'mangrove_restoration': {
        'name': 'Coastal Mangrove Restoration',
        'cost_usd': 1_000_000_000,
        'deltas': {
            # Slightly stronger deltas so synthetic runs visibly differ.
            'precip_volatility_reduction': 0.50,
            'precip_momentum_reduction': 0.25,
            'temp_reduction': 0.80,
            'coastal_only': True,
            'coastal_lon_threshold': 120,
        },
        'description': 'Mangrove buffer reduces storm surge and precipitation volatility along coastlines',
    },
    'regenerative_agriculture': {
        'name': 'Regenerative Agriculture',
        'cost_usd': 1_000_000_000,
        'deltas': {
            'precip_volatility_reduction': 0.25,
            'precip_momentum_reduction': 0.10,
            'temp_reduction': 0.30,
            'gdp_boost_factor': 1.15,
            'coastal_only': False,
        },
        'description': 'Soil health improvements buffer against drought and boost agricultural output',
    },
}
