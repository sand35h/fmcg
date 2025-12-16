"""
Synthetic FMCG dataset generator for Supply & Demand Forecasting system.
Enhanced for Final Year Project with UK-specific features.

Features:
1. 20+ years of historical data (2004-2024)
2. UK-specific holidays (Christmas, Easter, Black Friday, Bank Holidays, etc.)
3. Concept Drift (Simulating brand/price changes over time)
4. Data Quality Issues (Simulating weekly Sell-In data for 'Traditional' channel)
5. Inventory Waste (Simulating spoilage for 'Perishable' SKUs)
6. Weather patterns (UK temperate climate simulation)
7. Competitor activity simulation
8. Lead time and shipment data
9. Safety stock recommendations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# SETUP LOGGING
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
YEARS = 20  # 20+ years as per project brief
START_DATE = "2004-01-01"
FREQ = 'D'
N_SKUS = 50
N_LOCS = 20
CHANNELS = ['ModernTrade', 'Traditional', 'Ecommerce']
REGIONS = ['Greater London', 'South East', 'North West', 'Midlands', 'Scotland', 'Wales']
SAVE_OUTPUT = True
OUTPUT_DIR = "output"
np.random.seed(SEED)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    logging.info(f"Directory ensured: {path}")


def daterange(start, end):
    return pd.date_range(start=start, end=end, freq=FREQ)


# =============================================================================
# UK-SPECIFIC HOLIDAY CALENDAR
# =============================================================================

def build_festival_calendar(start_date, end_date):
    """
    Generates UK-specific holiday and retail event dates with varying dates by year.
    Includes major holidays and retail events that significantly impact FMCG demand.
    """
    logging.info("Building UK-specific holiday calendar.")
    festivals = []
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    
    for y in range(start_year, end_year + 1):
        # UK Bank Holidays and major retail events
        uk_holidays = [
            # New Year's Day (January 1)
            (1, 1, "New_Years_Day", 1.4),
            # Valentine's Day (February 14) - Major retail event
            (2, 14, "Valentines_Day", 1.6),
            # Mother's Day UK (varies - usually mid-March to early April)
            (3, np.random.randint(10, 31), "Mothers_Day", 1.8),
            # Easter (varies - March/April)
            (4, np.random.randint(1, 25), "Easter_Sunday", 2.0),
            # Early May Bank Holiday (first Monday of May)
            (5, np.random.randint(1, 7), "May_Bank_Holiday", 1.5),
            # Spring Bank Holiday (last Monday of May)
            (5, np.random.randint(25, 31), "Spring_Bank_Holiday", 1.4),
            # Father's Day (third Sunday of June)
            (6, np.random.randint(15, 21), "Fathers_Day", 1.5),
            # Summer Bank Holiday (last Monday of August) - Scotland early Aug
            (8, np.random.randint(25, 31), "Summer_Bank_Holiday", 1.6),
            # Halloween (October 31)
            (10, 31, "Halloween", 1.7),
            # Bonfire Night / Guy Fawkes (November 5)
            (11, 5, "Bonfire_Night", 1.4),
            # Black Friday (last Friday of November) - Major retail event
            (11, np.random.randint(23, 29), "Black_Friday", 2.5),
            # Cyber Monday (Monday after Black Friday)
            (11, np.random.randint(26, 30), "Cyber_Monday", 1.8),
            # Christmas Eve (December 24)
            (12, 24, "Christmas_Eve", 2.2),
            # Christmas Day (December 25)
            (12, 25, "Christmas_Day", 2.5),
            # Boxing Day (December 26) - Major sales
            (12, 26, "Boxing_Day", 2.3),
            # New Year's Eve (December 31)
            (12, 31, "New_Years_Eve", 1.8),
        ]
        
        for m, d, name, uplift in uk_holidays:
            try:
                # Adjust day to be within valid range
                d = min(d, 28 if m == 2 else (30 if m in [4, 6, 9, 11] else 31))
                dt = datetime(y, m, d)
                if pd.Timestamp(dt) >= pd.Timestamp(start_date) and pd.Timestamp(dt) <= pd.Timestamp(end_date):
                    festivals.append({
                        'date': pd.Timestamp(dt),
                        'festival': name,
                        'demand_multiplier': uplift,
                        'festival_duration_days': 7 if name in ['Christmas_Day', 'Easter_Sunday', 'Black_Friday'] else 3
                    })
            except Exception as e:
                logging.warning(f"Could not create date for {name} in year {y}: {e}")
                continue
    
    fest_df = pd.DataFrame(festivals)
    logging.info(f"Holiday calendar built with {len(fest_df)} holiday dates.")
    return fest_df


# =============================================================================
# WEATHER SIMULATION (UK Temperate Climate)
# =============================================================================

def generate_weather_data(start_date, end_date):
    """
    Generates daily weather data simulating UK's temperate climate patterns.
    Optimized with vectorized NumPy operations.
    """
    logging.info("Generating UK weather simulation data (optimized).")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Vectorized month and day_of_year extraction
    months = dates.month.values
    day_of_year = dates.dayofyear.values
    
    # Vectorized temperature calculation
    base_temp = 10 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp = base_temp + np.random.normal(0, 3, n_days)
    
    # Vectorized rainfall - create masks for seasons
    is_summer = np.isin(months, [6, 7, 8])
    is_autumn_winter = np.isin(months, [10, 11, 12, 1, 2])
    is_spring = ~is_summer & ~is_autumn_winter
    
    # Rainfall probabilities and amounts by season
    rainfall_prob = np.where(is_summer, 0.35, np.where(is_autumn_winter, 0.55, 0.45))
    rainfall_scale = np.where(is_summer, 8, np.where(is_autumn_winter, 12, 10))
    
    # Generate rainfall (vectorized)
    rain_occurs = np.random.rand(n_days) < rainfall_prob
    rainfall_amount = np.where(rain_occurs, np.random.exponential(rainfall_scale), 0)
    
    # Vectorized humidity
    is_high_humidity_month = np.isin(months, [11, 12, 1, 2])
    humidity = 75 + 10 * is_high_humidity_month + np.random.normal(0, 8, n_days)
    humidity = np.clip(humidity, 40, 100)
    
    # Vectorized weather demand factor
    weather_demand_factor = np.ones(n_days)
    weather_demand_factor = np.where(temp > 20, 1.15, weather_demand_factor)
    weather_demand_factor = np.where(temp < 5, 1.10, weather_demand_factor)
    weather_demand_factor = np.where(rainfall_amount > 20, weather_demand_factor * 0.92, weather_demand_factor)
    
    # Vectorized is_winter
    is_winter = np.isin(months, [12, 1, 2]).astype(int)
    
    weather_df = pd.DataFrame({
        'date': dates,
        'temperature_c': np.round(temp, 1),
        'rainfall_mm': np.round(rainfall_amount, 1),
        'humidity_pct': np.round(humidity, 1),
        'weather_demand_factor': np.round(weather_demand_factor, 3),
        'is_winter': is_winter
    })
    
    logging.info(f"Weather data generated for {len(weather_df)} days.")
    return weather_df


# =============================================================================
# MACROECONOMIC INDICATORS
# =============================================================================

def generate_monthly_macro(start_date, end_date):
    """Generates slow-moving macroeconomic indicators for UK."""
    logging.info("Generating monthly macroeconomic data.")
    idx = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # UK GDP growth: ~1-3% annually with some volatility
    gdp_growth = 0.02 + np.cumsum(np.random.normal(0, 0.001, len(idx)))
    gdp_growth = np.clip(gdp_growth, -0.02, 0.05)  # UK realistic bounds
    
    # CPI (inflation): UK targets ~2%
    cpi = 100 + np.cumsum(np.random.normal(0.15, 0.3, len(idx)))
    
    # Consumer confidence: UK typically ranges 85-115
    consumer_conf = 100 + np.random.normal(0, 6, len(idx))
    consumer_conf = np.clip(consumer_conf, 80, 120)
    
    # Fuel prices (affects distribution costs) - UK petrol prices
    fuel_price = 100 + np.cumsum(np.random.normal(0.15, 0.8, len(idx)))
    fuel_price = np.clip(fuel_price, 80, 180)
    
    # Exchange rate (GBP to USD) - affects imported goods
    exchange_rate = 1.30 + np.cumsum(np.random.normal(0, 0.01, len(idx)))
    exchange_rate = np.clip(exchange_rate, 1.10, 1.60)
    
    macro_df = pd.DataFrame({
        'month': idx,
        'gdp_growth': gdp_growth.round(4),
        'cpi_index': cpi.round(2),
        'consumer_confidence': consumer_conf.round(2),
        'fuel_price_index': fuel_price.round(2),
        'exchange_rate_gbp_usd': exchange_rate.round(4)
    })
    logging.info("Macro data generation complete.")
    return macro_df


# =============================================================================
# COMPETITOR ACTIVITY SIMULATION
# =============================================================================

def generate_competitor_activity(start_date, end_date, n_skus):
    """Generates competitor promotional activity and pricing. Optimized with vectorization."""
    logging.info("Generating competitor activity data (optimized).")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Vectorized competitor promotions (5% chance)
    promo_occurs = np.random.rand(n_days) < 0.05
    comp_promo_intensity = np.where(promo_occurs, np.random.uniform(0.1, 0.3, n_days), 0)
    
    # Vectorized competitor pricing pressure
    comp_price_pressure = 1.0 + np.random.normal(0, 0.02, n_days)
    
    comp_df = pd.DataFrame({
        'date': dates,
        'competitor_promo_intensity': np.round(comp_promo_intensity, 3),
        'competitor_price_pressure': np.round(comp_price_pressure, 3)
    })
    
    logging.info(f"Competitor activity data generated for {len(comp_df)} days.")
    return comp_df


# =============================================================================
# MASTER DATA GENERATION
# =============================================================================

def create_sku_master(n_skus):
    """Generates SKU metadata including cost, base price, and shelf life."""
    logging.info(f"Creating {n_skus} SKU master records.")
    
    # Nepal-specific brands and categories
    brands = [f"Brand_{chr(65+i)}" for i in range(min(10, max(5, n_skus//5)))]
    categories = ["DAIRY", "BEVERAGES", "SNACKS", "HOMECARE", "PERSONALCARE", "NOODLES", "BISCUITS"]
    pack_sizes = ['200ml', '250ml', '500ml', '1L', '2L', '50g', '100g', '250g', '500g', '1kg']
    materials = ['TETRA', 'PET', 'POUCH', 'GLASS', 'BOX', 'SACHET']
    
    # Shelf life categories
    shelf_life_cats = {
        'PERISHABLE': 14,       # Dairy, fresh items
        'SEMI_PERISHABLE': 90,  # Some beverages, snacks
        'STABLE': 365           # Homecare, dry goods
    }
    
    # ABC classification probabilities
    abc_classes = ['A', 'B', 'C']
    abc_probs = [0.2, 0.3, 0.5]  # 20% A-class (fast movers), 30% B, 50% C
    
    skus = []
    for i in range(n_skus):
        category = np.random.choice(categories)
        
        # Set shelf life based on category
        if category in ['DAIRY']:
            life_category = 'PERISHABLE'
        elif category in ['BEVERAGES', 'SNACKS', 'BISCUITS']:
            life_category = np.random.choice(['SEMI_PERISHABLE', 'PERISHABLE'], p=[0.8, 0.2])
        else:
            life_category = 'STABLE'
        
        base_price = round(float(np.random.uniform(20, 500)), 2)  # NPR pricing
        
        skus.append({
            'sku_id': f"SKU_{i+1:04d}",
            'sku_name': f"{np.random.choice(brands)}_{category[:3]}_{i+1}",
            'brand': np.random.choice(brands),
            'category': category,
            'segment': np.random.choice(['Premium', 'Mid-Range', 'Economy']),
            'pack_size': np.random.choice(pack_sizes),
            'material': np.random.choice(materials),
            'base_price': base_price,
            'cost': round(base_price * np.random.uniform(0.5, 0.75), 2),
            'mrp': round(base_price * np.random.uniform(1.1, 1.3), 2),
            'shelf_life_days': shelf_life_cats[life_category],
            'life_category': life_category,
            'abc_class': np.random.choice(abc_classes, p=abc_probs),
            'min_order_qty': np.random.choice([6, 12, 24, 48]),
            'lead_time_days': np.random.randint(2, 14),
            'is_seasonal': np.random.choice([0, 1], p=[0.7, 0.3])
        })
    
    sku_master = pd.DataFrame(skus)
    logging.info("SKU master creation complete.")
    return sku_master


def create_location_master(n_locs):
    """Generates Location metadata for UK regions."""
    logging.info(f"Creating {n_locs} Location master records.")
    
    # UK cities and their characteristics (city, region, population_multiplier, income_multiplier)
    uk_cities = [
        ('London', 'Greater London', 1.5, 1.5),
        ('Birmingham', 'Midlands', 1.2, 1.1),
        ('Manchester', 'North West', 1.3, 1.2),
        ('Leeds', 'North West', 1.1, 1.1),
        ('Glasgow', 'Scotland', 1.1, 1.0),
        ('Liverpool', 'North West', 1.0, 1.0),
        ('Bristol', 'South East', 0.95, 1.15),
        ('Sheffield', 'Midlands', 0.9, 0.95),
        ('Edinburgh', 'Scotland', 0.9, 1.2),
        ('Cardiff', 'Wales', 0.85, 1.0),
        ('Newcastle', 'North West', 0.8, 0.95),
        ('Nottingham', 'Midlands', 0.75, 0.9),
        ('Southampton', 'South East', 0.7, 1.05),
        ('Leicester', 'Midlands', 0.7, 0.9),
        ('Brighton', 'South East', 0.65, 1.1),
        ('Oxford', 'South East', 0.6, 1.25),
        ('Cambridge', 'South East', 0.55, 1.3),
        ('Aberdeen', 'Scotland', 0.5, 1.1),
        ('Swansea', 'Wales', 0.45, 0.85),
        ('Belfast', 'North West', 0.5, 0.9),
    ]
    
    locs = []
    for i in range(n_locs):
        if i < len(uk_cities):
            city, region, pop_mult, income_mult = uk_cities[i]
        else:
            city = f"City_{i+1}"
            region = np.random.choice(REGIONS)
            pop_mult = np.random.uniform(0.3, 0.8)
            income_mult = np.random.uniform(0.5, 0.9)
        
        locs.append({
            'location_id': f"LOC_{i+1:03d}",
            'city': city,
            'region': region,
            'location_type': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], p=[0.4, 0.4, 0.2]),
            'population': int(1e5 * pop_mult * np.random.uniform(0.8, 1.5)),
            'avg_income_index': round(income_mult * np.random.uniform(0.9, 1.1), 2),
            'distribution_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], p=[0.3, 0.4, 0.3]),
            'storage_capacity_units': int(np.random.uniform(5000, 50000)),
            'cold_storage_available': np.random.choice([0, 1], p=[0.3, 0.7])
        })
    
    loc_master = pd.DataFrame(locs)
    logging.info("Location master creation complete.")
    return loc_master


# =============================================================================
# CONCEPT DRIFT FUNCTION
# =============================================================================

def apply_concept_drift(sku_master, dates):
    """
    Simulates concept drift by adjusting base price and brand/category over time.
    Optimized: Creates yearly snapshots and expands them in bulk instead of day-by-day.
    """
    logging.info("Applying concept drift (optimized).")
    
    initial_state = sku_master[['sku_id', 'base_price', 'brand', 'category']].copy()
    current_state = initial_state.copy()
    yearly_snapshots = []
    
    years = sorted(dates.year.unique())
    
    for year in years:
        # 1. Price Inflation (UK: ~4-8% annually)
        if year > years[0]:
            inflation_rate = np.random.uniform(1.04, 1.08)
            current_state['base_price'] = current_state['base_price'] * inflation_rate
        
        # 2. Occasional rebranding (every 5 years) - vectorized
        if (year - years[0]) % 5 == 0 and year != years[0]:
            logging.info(f"Simulating rebranding event in year {year}.")
            rebrand_mask = np.random.rand(len(current_state)) < 0.15
            new_brands = [f"NewBrand_{np.random.randint(100)}" for _ in range(rebrand_mask.sum())]
            current_state.loc[rebrand_mask, 'brand'] = new_brands
        
        # Store yearly snapshot with year marker
        snapshot = current_state.copy()
        snapshot['year'] = year
        yearly_snapshots.append(snapshot)
    
    # Create year-to-snapshot lookup
    yearly_df = pd.concat(yearly_snapshots, ignore_index=True)
    
    # Create dates DataFrame with year
    dates_df = pd.DataFrame({'date': dates, 'year': dates.year})
    
    # Merge to expand - one merge instead of day-by-day loop
    drift_df = dates_df.merge(yearly_df, on='year', how='left')
    drift_df = drift_df.drop(columns=['year'])
    
    logging.info("Concept drift simulation complete.")
    return drift_df[['date', 'sku_id', 'base_price', 'brand', 'category']]


# =============================================================================
# CORE DEMAND SIMULATOR
# =============================================================================

def simulate_demand(sku_master, loc_master, start_date, end_date, fest_df, weather_df, macro_df, comp_df):
    """
    Simulates daily time series data for all SKU-Location combinations.
    OPTIMIZED: Pre-computed lookups, vectorized operations, efficient inventory simulation.
    """
    dates = daterange(start_date, end_date)
    N = len(dates)
    n_combinations = len(sku_master) * len(loc_master)
    logging.info(f"Starting demand simulation (optimized) for {n_combinations} time series over {N} days.")
    
    # Generate time-dependent master data (drift)
    drift_df = apply_concept_drift(sku_master, dates)
    
    # Precompute time-based effects (vectorized)
    day_of_year = dates.dayofyear.values
    weekly = dates.dayofweek.values
    
    yearly_season = 1 + 0.20 * np.sin(2 * np.pi * day_of_year / 365.25)
    weekly_season = 1 + 0.12 * np.where(weekly >= 5, 0.25, -0.05)
    
    # Macro effect - vectorized lookup
    month_starts = dates.to_period('M').to_timestamp()
    macro_df_indexed = macro_df.set_index('month')['consumer_confidence']
    consumer_conf_ts = month_starts.map(lambda x: macro_df_indexed.get(x, 100)).values.astype(float)
    mean_conf = np.nanmean(consumer_conf_ts)
    macro_effect_ts = 1 + (consumer_conf_ts - mean_conf) / mean_conf * 0.08
    
    # Weather effect
    weather_effect = weather_df.set_index('date')['weather_demand_factor'].reindex(dates).fillna(1.0).values
    
    # Competitor effect
    comp_effect_base = comp_df.set_index('date')['competitor_promo_intensity'].reindex(dates).fillna(0).values
    
    # PRE-COMPUTE FESTIVAL EFFECTS as a date lookup (major optimization)
    logging.info("Pre-computing festival effects...")
    festival_multiplier = np.ones(N)
    dates_arr = dates.values
    for _, fest_row in fest_df.iterrows():
        fest_date = pd.Timestamp(fest_row['date'])
        multiplier = fest_row['demand_multiplier']
        duration = fest_row['festival_duration_days']
        start_dt = fest_date - pd.Timedelta(days=duration)
        end_dt = fest_date + pd.Timedelta(days=duration)
        mask = (dates >= start_dt) & (dates <= end_dt)
        festival_multiplier[mask] *= multiplier
    
    # Create date-to-index mapping for fast lookups
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    # Build full cartesian product
    full_df = pd.MultiIndex.from_product(
        [dates, sku_master['sku_id'], loc_master['location_id']],
        names=['date', 'sku_id', 'location_id']
    ).to_frame(index=False)
    
    full_df = full_df.merge(drift_df, on=['date', 'sku_id'], how='left')
    full_df = full_df.merge(
        sku_master.drop(columns=['base_price', 'brand', 'category']),
        on='sku_id', how='left'
    )
    full_df = full_df.merge(loc_master, on='location_id', how='left')
    full_df['weekday'] = full_df['date'].dt.weekday
    
    # Pre-compute SKU lookups
    sku_lookup = sku_master.set_index('sku_id').to_dict('index')
    loc_lookup = loc_master.set_index('location_id').to_dict('index')
    abc_popularity = {'A': 2.0, 'B': 1.0, 'C': 0.5}
    
    logging.info("Simulating promotions and calculating demand (optimized)...")
    
    final_rows = []
    total_groups = n_combinations
    processed = 0
    
    for (sku_id, loc_id), group in full_df.groupby(['sku_id', 'location_id']):
        group = group.copy()
        sku_data = sku_lookup[sku_id]
        loc_data = loc_lookup[loc_id]
        life_category = sku_data['life_category']
        abc_class = sku_data['abc_class']
        
        n = len(group)
        
        # Promotions - vectorized random intervals
        promo_flag = np.zeros(n, dtype=np.int32)
        num_promos = np.random.poisson(4 * YEARS)
        if num_promos > 0:
            starts = np.random.randint(0, max(1, n - 14), size=num_promos)
            lengths = np.random.randint(3, 14, size=num_promos)
            for start, length in zip(starts, lengths):
                promo_flag[start:min(start + length, n)] = 1
        
        # Pricing - fully vectorized
        base_price_ts = group['base_price'].values
        promo_depth_vals = np.random.uniform(0.70, 0.90, n)
        promo_depth = np.where(promo_flag == 1, promo_depth_vals, 1.0)
        noise = np.random.normal(0, 0.015, n)
        price_ts = np.clip(base_price_ts * promo_depth * (1 + noise), 1, None)
        
        # Demand components
        sku_popularity = abc_popularity[abc_class]
        location_factor = loc_data['avg_income_index']
        price_sensitivity = np.random.uniform(-1.5, -0.3)
        
        trend_factor = np.random.uniform(-0.05, 0.15)
        sku_trend = 1 + np.linspace(-0.1, 0.3, n) * trend_factor
        
        # Use pre-computed arrays (already matching length since all groups have same N)
        current_yearly_season = yearly_season
        current_weekly_season = weekly_season
        current_macro_effect = macro_effect_ts
        current_weather_effect = weather_effect
        current_comp_effect = comp_effect_base
        current_festival = festival_multiplier  # Pre-computed!
        
        # Base demand - fully vectorized
        mean_demand = 800 * sku_popularity * location_factor * current_yearly_season * current_weekly_season * sku_trend
        
        # Price effect
        price_effect = np.power(price_ts / base_price_ts, price_sensitivity)
        
        # Competitor effect
        comp_impact = 1 - current_comp_effect * 0.5
        
        # Combined expected demand (using pre-computed festival)
        expected = mean_demand * price_effect * comp_impact * current_festival * current_macro_effect * current_weather_effect
        
        # Random shocks - vectorized
        shock = np.ones(n)
        num_shocks = np.random.poisson(2 * YEARS)
        if num_shocks > 0:
            shock_starts = np.random.randint(0, n, size=num_shocks)
            shock_durations = np.random.randint(5, 30, size=num_shocks)
            shock_magnitudes = np.random.uniform(0.6, 1.5, size=num_shocks)
            for s, dur, mag in zip(shock_starts, shock_durations, shock_magnitudes):
                shock[s:min(s + dur, n)] *= mag
        expected *= shock
        
        # Realize demand (Poisson) - vectorized
        realized_demand = np.random.poisson(lam=np.maximum(1, expected)).astype(np.int32)
        
        # Inventory simulation - optimized with pre-allocated arrays
        population = loc_data['population']
        opening_stock_base = int(800 * sku_popularity * (population / 1e5))
        opening_stock_arr = np.zeros(n, dtype=np.int32)
        closing_stock_arr = np.zeros(n, dtype=np.int32)
        incoming = (np.random.poisson(4, n) * 25).astype(np.int32)
        
        current_stock = max(0, opening_stock_base + np.random.randint(-100, 100))
        fulfilled_demand_arr = np.zeros(n, dtype=np.int32)
        returns = (realized_demand * np.random.uniform(0.0, 0.008, n)).astype(np.int32)
        waste_arr = np.zeros(n, dtype=np.int32)
        stockout_flag = np.zeros(n, dtype=np.int32)
        
        # Pre-generate waste factors for perishable/semi-perishable
        waste_factors = np.random.uniform(0.03, 0.12, n) if life_category == 'PERISHABLE' else \
                       (np.random.uniform(0.01, 0.03, n) * (np.random.rand(n) < 0.05) if life_category == 'SEMI_PERISHABLE' else np.zeros(n))
        
        # Inventory loop (kept but optimized - this is inherently sequential due to stock dependencies)
        for i in range(n):
            opening_stock_arr[i] = current_stock
            available = current_stock + incoming[i]
            
            if realized_demand[i] > available:
                fulfilled_demand_arr[i] = available
                stockout_flag[i] = 1
            else:
                fulfilled_demand_arr[i] = realized_demand[i]
            
            stock_before_waste = available - fulfilled_demand_arr[i] + returns[i]
            waste = int(stock_before_waste * waste_factors[i])
            waste_arr[i] = waste
            current_stock = max(0, stock_before_waste - waste)
            closing_stock_arr[i] = current_stock
        
        # Calculate metrics - vectorized pandas operations
        fulfilled_series = pd.Series(fulfilled_demand_arr)
        avg30 = fulfilled_series.rolling(30, min_periods=1).mean().values
        mos = np.where(avg30 > 0, closing_stock_arr / avg30, np.nan)
        
        # Safety stock recommendation
        demand_std = fulfilled_series.rolling(30, min_periods=7).std().fillna(0).values
        lead_time = sku_data['lead_time_days']
        service_factor = 1.65
        safety_stock = (service_factor * demand_std * np.sqrt(lead_time)).astype(np.int32)
        
        # Reorder point
        avg_daily_demand = fulfilled_series.rolling(7, min_periods=1).mean().values
        reorder_point = (avg_daily_demand * lead_time + safety_stock).astype(np.int32)
        
        # Assign channel
        channel = np.random.choice(CHANNELS, size=n)
        
        # Build final columns
        group['channel'] = channel
        group['price'] = np.round(price_ts, 2)
        group['promo_flag'] = promo_flag
        group['promo_depth'] = np.round(np.where(promo_flag == 1, promo_depth, 0), 2)
        group['comp_promo_flag'] = (current_comp_effect > 0).astype(np.int32)
        group['expected_demand'] = np.round(expected).astype(np.int32)
        group['actual_demand_sellout'] = fulfilled_demand_arr
        group['actual_demand'] = fulfilled_demand_arr.copy()
        group['unfulfilled_demand'] = realized_demand - fulfilled_demand_arr
        group['opening_stock'] = opening_stock_arr
        group['incoming_stock'] = incoming
        group['closing_stock'] = closing_stock_arr
        group['returns'] = returns
        group['waste_spoiled'] = waste_arr
        group['stockout_flag'] = stockout_flag
        group['mos'] = np.round(mos, 2)
        group['festival_uplift'] = np.round(current_festival, 2)
        group['weather_factor'] = current_weather_effect
        group['macro_factor'] = current_macro_effect
        group['safety_stock'] = safety_stock
        group['reorder_point'] = reorder_point
        group['life_category'] = life_category
        group['abc_class'] = abc_class
        group['cost'] = sku_data['cost']
        
        # Sell-in logic for Traditional channel - vectorized where possible
        weekday_arr = group['weekday'].values
        is_traditional = channel == 'Traditional'
        is_monday = weekday_arr == 0
        
        # Reset non-Monday Traditional to 0
        trad_non_monday_mask = is_traditional & ~is_monday
        group.loc[trad_non_monday_mask, 'actual_demand'] = 0
        
        # For Monday Traditional, compute weekly sum
        monday_trad_mask = is_traditional & is_monday
        monday_indices = np.where(monday_trad_mask)[0]
        for idx in monday_indices:
            start_idx = max(0, idx - 6)
            sellin_sum = fulfilled_demand_arr[start_idx:idx + 1].sum()
            group.iloc[idx, group.columns.get_loc('actual_demand')] = sellin_sum
        
        final_rows.append(group)
        
        processed += 1
        if processed % 100 == 0:
            logging.info(f"Processed {processed}/{total_groups} SKU-location combinations...")
    
    df = pd.concat(final_rows, ignore_index=True)
    df = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
    df = df.drop(columns=['actual_demand_sellout', 'weekday'])
    
    logging.info(f"Demand simulation complete. Total rows: {len(df)}")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_derived_features(df):
    """Adds time features, lags, rolling averages, and elasticity proxies."""
    logging.info("Starting feature engineering.")
    
    # Time features
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['day_of_week'] = df['date'].dt.weekday
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    gb_loc = df.groupby(['sku_id', 'location_id'])
    gb_channel = df.groupby(['sku_id', 'location_id', 'channel'])
    
    # Lag features
    df['loc_lag_1'] = gb_loc['actual_demand'].shift(1)
    df['loc_lag_7'] = gb_loc['actual_demand'].shift(7)
    df['loc_lag_14'] = gb_loc['actual_demand'].shift(14)
    df['loc_lag_30'] = gb_loc['actual_demand'].shift(30)
    
    # Rolling means
    df['loc_rolling_7_mean'] = gb_loc['actual_demand'].shift(1).rolling(7, min_periods=1).mean()
    df['loc_rolling_30_mean'] = gb_loc['actual_demand'].shift(1).rolling(30, min_periods=1).mean()
    df['loc_rolling_7_std'] = gb_loc['actual_demand'].shift(1).rolling(7, min_periods=1).std()
    
    # Channel-level features
    df['chan_lag_1'] = gb_channel['actual_demand'].shift(1)
    df['chan_rolling_7_mean'] = gb_channel['actual_demand'].shift(1).rolling(7, min_periods=1).mean()
    
    # Elasticity proxy
    df['pct_price_change'] = gb_loc['price'].pct_change().fillna(0)
    df['pct_demand_change'] = gb_loc['actual_demand'].pct_change().fillna(0)
    df['lag_pct_price_change'] = gb_loc['pct_price_change'].shift(1).fillna(0)
    df['elasticity_proxy'] = df['pct_demand_change'] / df['lag_pct_price_change'].replace(0, np.nan)
    df['elasticity_proxy'] = df['elasticity_proxy'].replace([np.inf, -np.inf], np.nan)
    
    # Promotion features
    df['days_since_promo'] = gb_loc['promo_flag'].apply(
        lambda s: s.cumsum() - s.cumsum().where(s == 1).ffill().fillna(0)
    ).values
    df['promo_run'] = gb_loc['promo_flag'].apply(
        lambda s: s.groupby((s == 0).cumsum()).transform('count') * s
    ).values
    
    # YoY comparison
    df['demand_yoy'] = gb_loc['actual_demand'].shift(365)
    df['demand_yoy_pct_change'] = (df['actual_demand'] - df['demand_yoy']) / df['demand_yoy'].replace(0, np.nan)
    
    # Fill NAs
    fillna_cols = ['loc_lag_1', 'loc_lag_7', 'loc_lag_14', 'loc_lag_30', 'chan_lag_1']
    df[fillna_cols] = df[fillna_cols].fillna(0).astype(int)
    
    fillna_mean_cols = ['loc_rolling_7_mean', 'loc_rolling_30_mean', 'chan_rolling_7_mean', 'loc_rolling_7_std']
    df[fillna_mean_cols] = df[fillna_mean_cols].fillna(0)
    
    df['days_since_promo'] = df['days_since_promo'].fillna(0).astype(int)
    df['promo_run'] = df['promo_run'].fillna(0).astype(int)
    
    # Drop intermediate columns
    df = df.drop(columns=['pct_price_change', 'pct_demand_change', 'lag_pct_price_change', 'demand_yoy'])
    
    logging.info("Feature engineering complete.")
    return df


# =============================================================================
# SHIPMENT & LEAD TIME DATA
# =============================================================================

def generate_shipment_data(daily_df, sku_master, loc_master):
    """Generates shipment records between suppliers, DCs, and stores. Optimized version."""
    logging.info("Generating shipment data (optimized).")
    
    # Pre-compute SKU lookup
    sku_lookup = sku_master.set_index('sku_id')[['lead_time_days', 'min_order_qty']].to_dict('index')
    
    # Filter to only rows with incoming stock
    incoming_df = daily_df[daily_df['incoming_stock'] > 0][['sku_id', 'location_id', 'date', 'incoming_stock']].copy()
    
    if len(incoming_df) == 0:
        logging.info("No incoming stock records found.")
        return pd.DataFrame()
    
    # Vectorized lead time lookup
    incoming_df['lead_time'] = incoming_df['sku_id'].map(lambda x: sku_lookup[x]['lead_time_days'])
    
    # Vectorized ship date calculation with random variation
    n_records = len(incoming_df)
    lead_time_variation = np.random.randint(-2, 3, size=n_records)
    incoming_df['ship_date'] = incoming_df['date'] - pd.to_timedelta(incoming_df['lead_time'] + lead_time_variation, unit='D')
    
    # Calculate actual lead time and on-time flag
    incoming_df['lead_time_actual'] = (incoming_df['date'] - incoming_df['ship_date']).dt.days
    incoming_df['on_time'] = (incoming_df['lead_time_actual'] <= incoming_df['lead_time']).astype(int)
    
    # Build final DataFrame
    ship_df = pd.DataFrame({
        'shipment_id': [f"SHIP_{i+1:08d}" for i in range(n_records)],
        'sku_id': incoming_df['sku_id'].values,
        'location_id': incoming_df['location_id'].values,
        'ship_date': incoming_df['ship_date'].values,
        'receipt_date': incoming_df['date'].values,
        'quantity': incoming_df['incoming_stock'].values,
        'lead_time_actual': incoming_df['lead_time_actual'].values,
        'lead_time_expected': incoming_df['lead_time'].values,
        'on_time': incoming_df['on_time'].values
    })
    
    logging.info(f"Shipment data generated: {len(ship_df)} records.")
    return ship_df


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def generate_dataset(n_skus=N_SKUS, n_locs=N_LOCS, start_date=START_DATE, years=YEARS):
    """Main function to run the generation process."""
    end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=years) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    logging.info(f"{'='*60}")
    logging.info(f"FMCG Dataset Generation Started")
    logging.info(f"{'='*60}")
    logging.info(f"Configuration: {n_skus} SKUs, {n_locs} Locations, {years} Years")
    logging.info(f"Date Range: {start_date} to {end_date}")
    
    try:
        # Master data
        sku_master = create_sku_master(n_skus)
        loc_master = create_location_master(n_locs)
        
        # External data
        fest_df = build_festival_calendar(start_date, end_date)
        weather_df = generate_weather_data(start_date, end_date)
        macro_df = generate_monthly_macro(start_date, end_date)
        comp_df = generate_competitor_activity(start_date, end_date, n_skus)
        
        # Core simulation
        daily_df = simulate_demand(sku_master, loc_master, start_date, end_date, fest_df, weather_df, macro_df, comp_df)
        daily_df = add_derived_features(daily_df)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.reset_index(drop=True)
        
        # Shipment data
        ship_df = generate_shipment_data(daily_df, sku_master, loc_master)
        
        if SAVE_OUTPUT:
            ensure_dir(OUTPUT_DIR)
            logging.info(f"Saving output files to {OUTPUT_DIR}...")
            
            sku_master.to_csv(os.path.join(OUTPUT_DIR, 'sku_master.csv'), index=False)
            loc_master.to_csv(os.path.join(OUTPUT_DIR, 'location_master.csv'), index=False)
            daily_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_timeseries.csv'), index=False)
            fest_df.to_csv(os.path.join(OUTPUT_DIR, 'festival_calendar.csv'), index=False)
            weather_df.to_csv(os.path.join(OUTPUT_DIR, 'weather_data.csv'), index=False)
            macro_df.to_csv(os.path.join(OUTPUT_DIR, 'monthly_macro.csv'), index=False)
            comp_df.to_csv(os.path.join(OUTPUT_DIR, 'competitor_activity.csv'), index=False)
            ship_df.to_csv(os.path.join(OUTPUT_DIR, 'shipment_data.csv'), index=False)
            
            logging.info("All output files saved successfully.")
        
    except Exception as e:
        logging.critical(f"Critical error during generation: {e}", exc_info=True)
        return None, None, None, None, None, None, None, None
    
    logging.info(f"{'='*60}")
    logging.info("Dataset Generation Complete!")
    logging.info(f"{'='*60}")
    
    return sku_master, loc_master, daily_df, fest_df, weather_df, macro_df, comp_df, ship_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = generate_dataset()
    
    if results[2] is not None:
        sku_master, loc_master, daily_df, fest_df, weather_df, macro_df, comp_df, ship_df = results
        
        print("\n" + "="*60)
        print("DATASET GENERATION SUMMARY")
        print("="*60)
        print(f"SKUs Generated: {len(sku_master)}")
        print(f"Locations Generated: {len(loc_master)}")
        print(f"Time Series Rows (daily): {len(daily_df):,}")
        print(f"Date Range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
        print(f"Festival Events: {len(fest_df)}")
        print(f"Weather Records: {len(weather_df):,}")
        print(f"Shipment Records: {len(ship_df):,}")
        print("\n--- Sample Daily Data ---")
        print(daily_df[['date', 'sku_id', 'location_id', 'channel', 'abc_class', 'price', 
                        'actual_demand', 'closing_stock', 'stockout_flag', 'safety_stock']].head(15))
        print("\n--- Sample Festivals ---")
        print(fest_df.head(10))
        print("\n--- Sample Weather ---")
        print(weather_df.head(10))
    else:
        print("Generation failed. Check logs for details.")