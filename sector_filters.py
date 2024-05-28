# sector_filters.py

# Sector-specific thresholds for filtering based on Alpha Vantage data
sector_thresholds = {
    'Technology': {
        'MarketCapitalization': 10e9,  # Tech companies with significant market presence
        'EBITDA': 500e6,  # High EBITDA to indicate profitability
        'PERatio': [10, 50],  # Wide P/E range due to growth potential
        'EPS': 1,  # Positive EPS for profitability
        'Beta': 2,  # Higher beta acceptable due to sector volatility
    },
    'Healthcare': {
        'MarketCapitalization': 5e9,
        'EBITDA': 200e6,
        'PERatio': [15, 40],  # P/E range to accommodate both growth and value companies
        'EPS': 0.5,  # Positive EPS, though lower threshold due to long R&D cycles
        'Beta': 1.5,  # Moderate beta reflecting sector stability and growth
    },
    'Financial': {
        'MarketCapitalization': 20e9,  # Financial institutions often have large market caps
        'EBITDA': 1e9,  # High EBITDA due to the nature of financial operations
        'PERatio': [5, 20],  # Lower P/E range due to different business models
        'EPS': 2,  # Higher EPS to reflect strong earnings
        'Beta': 1.5,  # Slightly higher beta due to market sensitivity
    },
    # Default values for sectors not explicitly defined
    'default': {
        'MarketCapitalization': 2e9,
        'EBITDA': 100e6,
        'PERatio': [10, 30],  # General P/E range for a broad market
        'EPS': 1,
        'Beta': 1.5,
    }
}

# Example adjustments for market conditions (these could be dynamic based on market analysis)
market_condition_adjustments = {
    'PERatio': -5,  # Tighten P/E ratio in overvalued markets
    'Beta': 0.1,  # Slightly increase beta tolerance in stable markets
}

# Exporting the settings
def get_sector_thresholds():
    return sector_thresholds

def get_market_condition_adjustments():
    return market_condition_adjustments
