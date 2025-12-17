"""
Enhanced FMCG Dataset Generator v2.0 for AI Forecasting Platform
Optimized for production-grade demand forecasting with realistic patterns.

NEW FEATURES:
1. Realistic UK holiday calendar (actual date rules)
2. SKU lifecycle (launches, retirements, product ramp-up)
3. Strategic promotion campaigns (seasonal, not random)
4. External shocks (recession, pandemic, supply disruptions)
5. Cross-SKU effects (substitution & cannibalization)
6. Supply constraints (variable lead times, shortages)
7. Demographic evolution (population & income growth)
8. Negative Binomial demand (overdispersion)
9. Memory-optimized for Kaggle (Parquet output)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from dateutil.easter import easter

# SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CONFIG
SEED = 42
YEARS = 20
START_DATE = "2004-01-01"
N_SKUS = 50
N_LOCS = 20
CHANNELS = ['ModernTrade', 'Traditional', 'Ecommerce']
REGIONS = ['Greater London', 'South East', 'North West', 'Midlands', 'Scotland', 'Wales']
OUTPUT_DIR = "output"
np.random.seed(SEED)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =============================================================================
# 1. REALISTIC UK HOLIDAY CALENDAR
# =============================================================================

def build_realistic_uk_calendar(start_date, end_date):
    """Generates UK holidays using ACTUAL calendar rules, not random dates."""
    logging.info("Building realistic UK holiday calendar...")
    
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    festivals = []
    
    for year in range(start_year, end_year + 1):
        # Fixed dates
        fixed_holidays = [
            (datetime(year, 1, 1), "New_Years_Day", 1.4, 3),
            (datetime(year, 2, 14), "Valentines_Day", 1.6, 3),
            (datetime(year, 10, 31), "Halloween", 1.7, 5),
            (datetime(year, 11, 5), "Bonfire_Night", 1.4, 3),
            (datetime(year, 12, 24), "Christmas_Eve", 2.2, 7),
            (datetime(year, 12, 25), "Christmas_Day", 2.5, 10),
            (datetime(year, 12, 26), "Boxing_Day", 2.3, 7),
            (datetime(year, 12, 31), "New_Years_Eve", 1.8, 5),
        ]
        
        # Easter (moveable - based on lunar calendar)
        easter_date = easter(year)
        fixed_holidays.append((easter_date, "Easter_Sunday", 2.0, 7))
        
        # Mothers Day UK (4th Sunday of Lent = 3 weeks before Easter)
        mothers_day = easter_date - timedelta(weeks=3)
        fixed_holidays.append((mothers_day, "Mothers_Day", 1.8, 5))
        
        # Fathers Day UK (3rd Sunday of June)
        june_1 = datetime(year, 6, 1)
        days_to_sunday = (6 - june_1.weekday()) % 7
        first_sunday = june_1 + timedelta(days=days_to_sunday)
        fathers_day = first_sunday + timedelta(weeks=2)
        fixed_holidays.append((fathers_day, "Fathers_Day", 1.5, 3))
        
        # Bank Holidays (specific Mondays)
        # Early May Bank Holiday (1st Monday of May)
        may_1 = datetime(year, 5, 1)
        days_to_monday = (7 - may_1.weekday()) % 7
        early_may_bh = may_1 + timedelta(days=days_to_monday)
        fixed_holidays.append((early_may_bh, "May_Bank_Holiday", 1.5, 3))
        
        # Spring Bank Holiday (last Monday of May)
        may_31 = datetime(year, 5, 31)
        days_back_to_monday = (may_31.weekday() - 0) % 7
        spring_bh = may_31 - timedelta(days=days_back_to_monday)
        fixed_holidays.append((spring_bh, "Spring_Bank_Holiday", 1.4, 3))
        
        # Summer Bank Holiday (last Monday of August)
        aug_31 = datetime(year, 8, 31)
        days_back = (aug_31.weekday() - 0) % 7
        summer_bh = aug_31 - timedelta(days=days_back)
        fixed_holidays.append((summer_bh, "Summer_Bank_Holiday", 1.6, 3))
        
        # Black Friday (4th Friday of November)
        nov_1 = datetime(year, 11, 1)
        days_to_friday = (4 - nov_1.weekday()) % 7
        first_friday = nov_1 + timedelta(days=days_to_friday)
        black_friday = first_friday + timedelta(weeks=3)
        fixed_holidays.append((black_friday, "Black_Friday", 2.5, 7))
        
        # Cyber Monday (Monday after Black Friday)
        cyber_monday = black_friday + timedelta(days=3)
        fixed_holidays.append((cyber_monday, "Cyber_Monday", 1.8, 3))
        
        for dt, name, uplift, duration in fixed_holidays:
            if pd.Timestamp(dt) >= pd.Timestamp(start_date) and pd.Timestamp(dt) <= pd.Timestamp(end_date):
                festivals.append({
                    'date': pd.Timestamp(dt),
                    'festival': name,
                    'demand_multiplier': uplift,
                    'festival_duration_days': duration
                })
    
    fest_df = pd.DataFrame(festivals)
    logging.info(f"Generated {len(fest_df)} realistic UK holiday dates.")
    return fest_df


# =============================================================================
# 2. SKU LIFECYCLE MANAGEMENT
# =============================================================================

def create_sku_master_with_lifecycle(n_skus, start_date, end_date):
    """Creates SKUs with birth/retirement dates and lifecycle phases."""
    logging.info("Creating SKU master with lifecycle...")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    total_days = (end_dt - start_dt).days
    
    # Real-world UK FMCG Examples
    real_products = {
        "DAIRY": [
            ("Muller Corner Yogurt", "Muller", "175g"), ("Cheddar Cheese Block", "Cathedral City", "350g"),
            ("Semi Skimmed Milk", "Yeo Valley", "2L"), ("Salted Butter", "Lurpak", "250g"),
            ("Oat Milk Barista", "Oatly", "1L")
        ],
        "BEVERAGES": [
            ("Coca Cola Original", "Coca-Cola", "1.5L"), ("Diet Coke", "Coca-Cola", "330ml"),
            ("Nescafe Gold Blend", "Nestle", "200g"), ("Tropicana Orange Juice", "Tropicana", "900ml"),
            ("Yorkshire Tea Bags", "Taylors", "160ct"), ("Red Bull Energy", "Red Bull", "250ml")
        ],
        "SNACKS": [
            ("Dairy Milk Bar", "Cadbury", "110g"), ("Walkers Cheese & Onion", "Walkers", "32g"),
            ("Pringles Sour Cream", "Kellogg's", "200g"), ("McVities Digestives", "McVities", "400g"),
            ("KitKat 4 Finger", "Nestle", "45g")
        ],
        "HOMECARE": [
            ("Fairy Liquid Lemon", "P&G", "433ml"), ("Persil Bio Capsules", "Unilever", "30ct"),
            ("Toilet Tissue 9 Roll", "Andrex", "9pk"), ("Domestos Bleach", "Unilever", "750ml")
        ],
        "PERSONALCARE": [
            ("Dove Beauty Bar", "Unilever", "2pk"), ("Colgate Total Toothpaste", "Colgate", "75ml"),
            ("Nivea Men Deodorant", "Beiersdorf", "150ml"), ("Head & Shoulders Shampoo", "P&G", "500ml")
        ],
        "NOODLES": [
            ("Pot Noodle Chicken", "Unilever", "90g"), ("Super Noodles BBQ", "Batchelors", "100g")
        ],
        "BISCUITS": [
            ("Jammie Dodgers", "Burtons", "140g"), ("Oreo Original", "Mondelez", "154g")
        ]
    }
    
    skus = []
    active_skus = int(n_skus * 0.7)
    new_launches = n_skus - active_skus
    
    flat_products = []
    for cat, items in real_products.items():
        for name, brand, size in items:
            flat_products.append({'cat': cat, 'name': name, 'brand': brand, 'size': size})
    
    # Cycle through real list if n_skus > len(products)
    for i in range(n_skus):
        template = flat_products[i % len(flat_products)]
        category = template['cat']
        
        # Add slight variation if repeating
        suffix = "" if i < len(flat_products) else f" v{i//len(flat_products)+1}"
        sku_name = template['name'] + suffix
        brand = template['brand']
        pack_size = template['size']
        
        # Lifecycle dates
        if i < active_skus:
            # Existing SKUs at start
            birth_date = start_dt
            retirement_prob = 0.3  # 30% chance of retirement during period
        else:
            # New launches during period
            birth_date = start_dt + timedelta(days=np.random.randint(365, total_days - 365))
            retirement_prob = 0.1  # Lower retirement for newer products
        
        # Retirement date
        if np.random.rand() < retirement_prob:
            days_after_birth = np.random.randint(730, total_days - (birth_date - start_dt).days)
            retirement_date = birth_date + timedelta(days=days_after_birth)
            if retirement_date > end_dt:
                retirement_date = None
        else:
            retirement_date = None
        
        # Shelf life
        if category == 'DAIRY':
            life_cat = 'PERISHABLE'
            shelf_life = 14
        elif category in ['BEVERAGES', 'SNACKS']:
            life_cat = 'SEMI_PERISHABLE'
            shelf_life = 90
        else:
            life_cat = 'STABLE'
            shelf_life = 365
        
        base_price = round(float(np.random.uniform(20, 500)), 2)
        
        skus.append({
            'sku_id': f"SKU_{i+1:04d}",
            'sku_name': sku_name,
            'brand': brand,
            'category': category,
            'segment': np.random.choice(['Premium', 'Mid-Range', 'Economy']),
            'pack_size': pack_size,
            'base_price': base_price,
            'cost': round(base_price * np.random.uniform(0.5, 0.75), 2),
            'shelf_life_days': shelf_life,
            'life_category': life_cat,
            'abc_class': np.random.choice(['A', 'B', 'C'], p=[0.2, 0.3, 0.5]),
            'min_order_qty': np.random.choice([6, 12, 24, 48]),
            'lead_time_days': np.random.randint(2, 14),
            'birth_date': birth_date,
            'retirement_date': retirement_date,
            'launch_phase_days': 180,  # 6 months to reach maturity
            'decline_phase_days': 90 if retirement_date else 0
        })
    
    sku_master = pd.DataFrame(skus)
    logging.info(f"Created {len(sku_master)} SKUs with lifecycle (Active: {active_skus}, Launches: {new_launches})")
    return sku_master


def calculate_lifecycle_factor(date, sku_data):
    """Calculate demand multiplier based on product lifecycle stage."""
    birth = pd.Timestamp(sku_data['birth_date'])
    retirement = pd.Timestamp(sku_data['retirement_date']) if pd.notna(sku_data['retirement_date']) else None
    date = pd.Timestamp(date)
    
    # Before birth
    if date < birth:
        return 0.0
    
    # Launch phase (ramp up from 0.2 to 1.0)
    days_since_birth = (date - birth).days
    if days_since_birth < sku_data['launch_phase_days']:
        progress = days_since_birth / sku_data['launch_phase_days']
        return 0.2 + 0.8 * progress  # Ramp from 20% to 100%
    
    # Decline phase (if retiring)
    if retirement:
        days_to_retirement = (retirement - date).days
        if days_to_retirement < sku_data['decline_phase_days']:
            progress = days_to_retirement / sku_data['decline_phase_days']
            return 0.3 + 0.7 * progress  # Decline to 30%
        elif date > retirement:
            return 0.0
    
    # Mature phase
    return 1.0


# =============================================================================
# 3. CROSS-SKU EFFECTS (Substitution & Cannibalization)
# =============================================================================

def build_sku_similarity_matrix(sku_master):
    """Creates similarity matrix for substitution effects."""
    logging.info("Building SKU similarity matrix...")
    
    sku_ids = sku_master['sku_id'].values
    n = len(sku_ids)
    similarity = np.zeros((n, n))
    
    for i, sku1 in sku_master.iterrows():
        for j, sku2 in sku_master.iterrows():
            if i == j:
                similarity[i, j] = 0
                continue
            
            score = 0.0
            # Same category = high substitution
            if sku1['category'] == sku2['category']:
                score += 0.6
            # Same brand = some cannibalization
            if sku1['brand'] == sku2['brand']:
                score += 0.3
            # Similar price = substitutes
            price_diff = abs(sku1['base_price'] - sku2['base_price']) / max(sku1['base_price'], sku2['base_price'])
            if price_diff < 0.2:
                score += 0.2
            
            similarity[i, j] = min(score, 1.0)
    
    return similarity, sku_ids


# =============================================================================
# 4. EXTERNAL SHOCKS
# =============================================================================

def generate_external_shocks(start_date, end_date):
    """Simulates major economic/supply shocks."""
    logging.info("Generating external shock events...")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    shocks = [
        {
            'name': 'Financial_Crisis',
            'start': pd.Timestamp('2008-09-01'),
            'end': pd.Timestamp('2009-12-31'),
            'demand_impact': 0.85,  # 15% demand drop
            'supply_impact': 1.0,
            'type': 'recession'
        },
        {
            'name': 'Brexit_Uncertainty',
            'start': pd.Timestamp('2016-06-24'),
            'end': pd.Timestamp('2017-06-30'),
            'demand_impact': 0.92,
            'supply_impact': 1.15,  # 15% longer lead times
            'type': 'political'
        },
        {
            'name': 'COVID_Pandemic',
            'start': pd.Timestamp('2020-03-15'),
            'end': pd.Timestamp('2021-06-30'),
            'demand_impact': 1.25,  # Panic buying
            'supply_impact': 1.30,  # Supply chain disruption
            'type': 'pandemic'
        },
        {
            'name': 'Supply_Chain_Crisis',
            'start': pd.Timestamp('2021-09-01'),
            'end': pd.Timestamp('2022-03-31'),
            'demand_impact': 1.05,
            'supply_impact': 1.40,  # Severe delays
            'type': 'supply'
        }
    ]
    
    # Filter to date range
    valid_shocks = [s for s in shocks if s['start'] >= start_dt and s['start'] <= end_dt]
    
    shock_df = pd.DataFrame(valid_shocks)
    logging.info(f"Generated {len(shock_df)} major shock events.")
    return shock_df


def get_shock_multiplier(date, shock_df, factor_type='demand'):
    """Returns shock impact multiplier for a given date."""
    date = pd.Timestamp(date)
    multiplier = 1.0
    
    for _, shock in shock_df.iterrows():
        if shock['start'] <= date <= shock['end']:
            if factor_type == 'demand':
                multiplier *= shock['demand_impact']
            elif factor_type == 'supply':
                multiplier *= shock['supply_impact']
    
    return multiplier


# =============================================================================
# 5. DEMOGRAPHIC EVOLUTION
# =============================================================================

def create_location_master_with_evolution(n_locs, start_date, end_date):
    """Creates locations with demographic growth trends."""
    logging.info("Creating location master with demographic evolution...")
    
    uk_cities = [
        ('London', 'Greater London', 1.5, 1.5, 0.02),  # High growth
        ('Birmingham', 'Midlands', 1.2, 1.1, 0.01),
        ('Manchester', 'North West', 1.3, 1.2, 0.015),
        ('Leeds', 'North West', 1.1, 1.1, 0.01),
        ('Glasgow', 'Scotland', 1.1, 1.0, 0.005),
        ('Liverpool', 'North West', 1.0, 1.0, 0.008),
        ('Bristol', 'South East', 0.95, 1.15, 0.012),
        ('Sheffield', 'Midlands', 0.9, 0.95, 0.005),
        ('Edinburgh', 'Scotland', 0.9, 1.2, 0.01),
        ('Cardiff', 'Wales', 0.85, 1.0, 0.006),
    ]
    
    locs = []
    for i in range(n_locs):
        if i < len(uk_cities):
            city, region, pop_mult, income_mult, growth_rate = uk_cities[i]
        else:
            city = f"City_{i+1}"
            region = np.random.choice(REGIONS)
            pop_mult = np.random.uniform(0.4, 0.8)
            income_mult = np.random.uniform(0.7, 1.0)
            growth_rate = np.random.uniform(-0.005, 0.015)
        
        locs.append({
            'location_id': f"LOC_{i+1:03d}",
            'city': city,
            'region': region,
            'location_type': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], p=[0.4, 0.4, 0.2]),
            'population_base': int(1e5 * pop_mult),
            'income_index_base': income_mult,
            'annual_growth_rate': growth_rate,  # Population growth
            'income_growth_rate': growth_rate * 0.5,  # Income grows slower
            'storage_capacity_units': int(np.random.uniform(5000, 50000)),
            'cold_storage_available': np.random.choice([0, 1], p=[0.3, 0.7])
        })
    
    loc_master = pd.DataFrame(locs)
    logging.info(f"Created {len(loc_master)} locations with demographic trends.")
    return loc_master


def calculate_demographic_factor(date, loc_data, start_date):
    """Calculate demographic multiplier based on years elapsed."""
    years_elapsed = (pd.Timestamp(date) - pd.Timestamp(start_date)).days / 365.25
    
    pop_growth = (1 + loc_data['annual_growth_rate']) ** years_elapsed
    income_growth = (1 + loc_data['income_growth_rate']) ** years_elapsed
    
    return pop_growth * income_growth


# =============================================================================
# 6. STRATEGIC PROMOTION CAMPAIGNS
# =============================================================================

def generate_strategic_promotions(start_date, end_date, fest_df):
    """Generates realistic promotion campaigns aligned with seasons/holidays."""
    logging.info("Generating strategic promotion campaigns...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    promo_calendar = pd.DataFrame({'date': dates, 'promo_season': 0})
    
    # Mark promotion seasons
    for _, fest in fest_df.iterrows():
        if fest['festival'] in ['Christmas_Day', 'Black_Friday', 'Easter_Sunday']:
            # Major promo period: 2 weeks before event
            promo_start = fest['date'] - pd.Timedelta(days=14)
            promo_end = fest['date']
            mask = (promo_calendar['date'] >= promo_start) & (promo_calendar['date'] <= promo_end)
            promo_calendar.loc[mask, 'promo_season'] = 1
    
    # Quarterly campaigns (mid-March, June, Sept, December)
    for year in dates.year.unique():
        for month in [3, 6, 9, 12]:
            campaign_start = pd.Timestamp(f'{year}-{month:02d}-01')
            campaign_end = campaign_start + pd.Timedelta(days=14)
            mask = (promo_calendar['date'] >= campaign_start) & (promo_calendar['date'] < campaign_end)
            promo_calendar.loc[mask, 'promo_season'] = 1
    
    logging.info(f"Strategic promo calendar created: {promo_calendar['promo_season'].sum()} promo days.")
    return promo_calendar


# =============================================================================
# 7. WEATHER, MACRO & COMPETITOR (REUSE FROM ORIGINAL)
# =============================================================================

def generate_competitor_activity(start_date, end_date):
    """Generates competitor promotional activity and pricing."""
    logging.info("Generating competitor activity data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Competitor promotions (5% chance)
    promo_occurs = np.random.rand(n_days) < 0.05
    comp_promo_intensity = np.where(promo_occurs, np.random.uniform(0.1, 0.3, n_days), 0)
    
    # Competitor pricing pressure (1.0 = Parity, >1.0 = They are cheap)
    comp_price_pressure = 1.0 + np.random.normal(0, 0.02, n_days)
    
    comp_df = pd.DataFrame({
        'date': dates,
        'competitor_promo_intensity': np.round(comp_promo_intensity, 3),
        'competitor_price_pressure': np.round(comp_price_pressure, 3)
    })
    
    return comp_df

def generate_weather_data(start_date, end_date):
    """
    UK weather simulation mirroring Public API fields (OpenWeather/MetOffice).
    Generates: avg/min/max temp, precipitation, humidity, wind speed.
    """
    logging.info("Generating realistic API-style weather data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    months = dates.month.values
    day_of_year = dates.dayofyear.values
    
    # 1. Temperature (Sine wave + random)
    base_temp = 10 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    avg_temp = base_temp + np.random.normal(0, 3, n_days)
    
    # Min/Max spread (correlated with season, wider in summer)
    temp_spread = np.where(np.isin(months, [6, 7, 8]), 
                          np.random.normal(10, 2, n_days), # Summer spread
                          np.random.normal(6, 1.5, n_days)) # Winter spread
    
    max_temp = avg_temp + (temp_spread / 2)
    min_temp = avg_temp - (temp_spread / 2)
    
    # 2. Rain & Snow
    is_summer = np.isin(months, [6, 7, 8])
    rainfall_prob = np.where(is_summer, 0.35, 0.50)
    rain_occurs = np.random.rand(n_days) < rainfall_prob
    precip_mm = np.where(rain_occurs, np.random.exponential(5), 0)
    
    # 3. Humidity (UK is humid, usually 70-90%)
    humidity = 80 - (max_temp - 15) + np.random.normal(0, 5, n_days)
    humidity = np.clip(humidity, 40, 100)
    
    # 4. Wind Speed (km/h)
    wind_speed = np.random.gamma(5, 3, n_days) # Skewed, avg around 15 km/h
    
    # 5. Internal Demand Factor (The "Truth" for simulation)
    # High temp -> Boost beverages/ice cream
    # Rain -> Drop implementation
    weather_demand_factor = np.ones(n_days)
    weather_demand_factor = np.where(max_temp > 22, 1.15, weather_demand_factor) # Hot days
    weather_demand_factor = np.where(max_temp < 5, 1.05, weather_demand_factor)  # Cold days (soup etc)
    weather_demand_factor = np.where(precip_mm > 5.0, weather_demand_factor * 0.95, weather_demand_factor) # Heavy rain
    
    weather_df = pd.DataFrame({
        'date': dates,
        'avg_temp_c': np.round(avg_temp, 1),
        'min_temp_c': np.round(min_temp, 1),
        'max_temp_c': np.round(max_temp, 1),
        'precipitation_mm': np.round(precip_mm, 1),
        'avg_humidity_pct': np.round(humidity, 1),
        'wind_speed_kmh': np.round(wind_speed, 1),
        'weather_demand_factor': np.round(weather_demand_factor, 3) # Latent feature for simulation
    })
    
    return weather_df


def generate_monthly_macro(start_date, end_date):
    """UK macroeconomic indicators."""
    idx = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    gdp_growth = 0.02 + np.cumsum(np.random.normal(0, 0.001, len(idx)))
    gdp_growth = np.clip(gdp_growth, -0.02, 0.05)
    
    cpi = 100 + np.cumsum(np.random.normal(0.15, 0.3, len(idx)))
    consumer_conf = 100 + np.random.normal(0, 6, len(idx))
    
    macro_df = pd.DataFrame({
        'month': idx,
        'gdp_growth': gdp_growth.round(4),
        'cpi_index': cpi.round(2),
        'consumer_confidence': consumer_conf.round(2)
    })
    
    return macro_df


# =============================================================================
# 8. ENHANCED DEMAND SIMULATION
# =============================================================================

def simulate_demand_v2(sku_master, loc_master, start_date, end_date, fest_df, weather_df, 
                       macro_df, shock_df, promo_calendar, comp_df):
    """Enhanced demand simulation with all improvements."""
    logging.info("Starting enhanced demand simulation v2.0...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    N = len(dates)
    
    # Build similarity matrix
    similarity_matrix, sku_ids = build_sku_similarity_matrix(sku_master)
    sku_to_idx = {sku: i for i, sku in enumerate(sku_ids)}
    
    # Precompute festival effects
    festival_multiplier = np.ones(N)
    for _, fest in fest_df.iterrows():
        fest_date = pd.Timestamp(fest['date'])
        multiplier = fest['demand_multiplier']
        duration = fest['festival_duration_days']
        start_dt = fest_date - pd.Timedelta(days=duration)
        end_dt = fest_date + pd.Timedelta(days=duration)
        mask = (dates >= start_dt) & (dates <= end_dt)
        festival_multiplier[mask] *= multiplier
    
    # Precompute time effects
    day_of_year = dates.dayofyear.values
    weekly = dates.dayofweek.values
    yearly_season = 1 + 0.20 * np.sin(2 * np.pi * day_of_year / 365.25)
    weekly_season = 1 + 0.12 * np.where(weekly >= 5, 0.25, -0.05)
    
    # Weather effect
    weather_effect = weather_df.set_index('date')['weather_demand_factor'].reindex(dates).fillna(1.0).values
    
    # Competitor effect (New)
    try:
        comp_promo = comp_df.set_index('date')['competitor_promo_intensity'].reindex(dates).fillna(0).values
        comp_price = comp_df.set_index('date')['competitor_price_pressure'].reindex(dates).fillna(1.0).values
    except Exception as e:
        logging.warning(f"Competitor data error: {e}. Using defaults.")
        comp_promo = np.zeros(N)
        comp_price = np.ones(N)

    # Macro effect
    month_starts = dates.to_period('M').to_timestamp()
    macro_indexed = macro_df.set_index('month')['consumer_confidence']
    consumer_conf_ts = month_starts.map(lambda x: macro_indexed.get(x, 100)).values.astype(float)
    macro_effect_ts = 1 + (consumer_conf_ts - 100) / 100 * 0.08
    
    # Shock effects
    shock_demand = np.array([get_shock_multiplier(d, shock_df, 'demand') for d in dates])
    shock_supply = np.array([get_shock_multiplier(d, shock_df, 'supply') for d in dates])
    
    # Promo calendar
    promo_season_flags = promo_calendar.set_index('date')['promo_season'].reindex(dates).fillna(0).values
    
    # Build full cartesian (sample for memory efficiency if needed)
    full_df = pd.MultiIndex.from_product(
        [dates, sku_master['sku_id'], loc_master['location_id']],
        names=['date', 'sku_id', 'location_id']
    ).to_frame(index=False)
    
    full_df = full_df.merge(sku_master, on='sku_id', how='left')
    full_df = full_df.merge(loc_master, on='location_id', how='left')
    
    logging.info(f"Processing {len(full_df):,} rows...")
    
    # Lookups
    sku_lookup = sku_master.set_index('sku_id').to_dict('index')
    loc_lookup = loc_master.set_index('location_id').to_dict('index')
    abc_popularity = {'A': 2.0, 'B': 1.0, 'C': 0.5}
    
    final_rows = []
    processed = 0
    total_groups = len(sku_master) * len(loc_master)
    
    # Store cross-SKU promo impacts
    cross_sku_impacts = {}
    
    for (sku_id, loc_id), group in full_df.groupby(['sku_id', 'location_id']):
        group = group.copy().reset_index(drop=True)
        sku_data = sku_lookup[sku_id]
        loc_data = loc_lookup[loc_id]
        n = len(group)
        
        # Lifecycle factor
        lifecycle_factors = np.array([calculate_lifecycle_factor(d, sku_data) for d in group['date']])
        
        # Demographic evolution
        demo_factors = np.array([calculate_demographic_factor(d, loc_data, start_date) for d in group['date']])
        
        # Strategic promotions
        promo_flag = np.zeros(n, dtype=np.int32)
        is_promo_season = promo_season_flags.copy()
        
        # Apply promotions during strategic windows
        promo_indices = np.where(is_promo_season == 1)[0]
        if len(promo_indices) > 0:
            # Random selection: promote 30% of promo season days
            n_promo_days = int(len(promo_indices) * 0.3)
            if n_promo_days > 0:
                selected = np.random.choice(promo_indices, size=n_promo_days, replace=False)
                promo_flag[selected] = 1
        
        # Price with promotions
        base_price_ts = np.full(n, sku_data['base_price'])
        promo_depth = np.where(promo_flag == 1, np.random.uniform(0.70, 0.85), 1.0)
        price_ts = base_price_ts * promo_depth
        
        # Base demand
        sku_popularity = abc_popularity[sku_data['abc_class']]
        mean_demand = (800 * sku_popularity * 
                      demo_factors * 
                      yearly_season * 
                      weekly_season * 
                      lifecycle_factors)
        
        # Price elasticity
        price_effect = np.power(price_ts / base_price_ts, -1.2)
        
        # Competitor Impact (New)
        # If competitor pressure > 1 (cheaper), we lose demand
        comp_impact = 1.0 - (comp_promo * 0.4) # Promo steals 40%
        comp_impact *= np.clip(1.0 - (comp_price - 1.0) * 2.0, 0.5, 1.5) # Price sensitivity to competitors
        
        # Combined expected demand
        expected = (mean_demand * 
                   price_effect * 
                   festival_multiplier * 
                   macro_effect_ts * 
                   weather_effect * 
                   comp_impact *
                   shock_demand)
        
        # Cross-SKU effects (substitution from competing SKUs)
        cross_effect = np.ones(n)
        if sku_id in cross_sku_impacts:
            for other_sku, impact_array in cross_sku_impacts[sku_id]:
                similarity = similarity_matrix[sku_to_idx[sku_id], sku_to_idx[other_sku]]
                cross_effect *= (1 - similarity * 0.3 * impact_array)  # Max 30% cannibalization
        
        expected *= cross_effect
        
        # Store this SKU's promo impact for cross-SKU effects
        if sku_id not in cross_sku_impacts:
            cross_sku_impacts[sku_id] = []
        cross_sku_impacts[sku_id].append((sku_id, promo_flag))
        
        # Negative Binomial demand (overdispersion)
        # Fix: Ensure expected is always positive to avoid invalid p values
        expected_safe = np.maximum(expected, 0.1)  # Minimum expected demand of 0.1
        
        r = 10  # Dispersion parameter
        p = r / (r + expected_safe)
        
        # Clip p to valid range [0, 1] to handle any edge cases
        p = np.clip(p, 0.0001, 0.9999)
        
        realized_demand = np.random.negative_binomial(r, p).astype(np.int32)
        realized_demand = np.maximum(realized_demand, 0)
        
        # Inventory simulation with supply constraints
        opening_stock_base = int(500 * sku_popularity * demo_factors[0])
        current_stock = max(0, opening_stock_base)
        
        opening_stock_arr = np.zeros(n, dtype=np.int32)
        closing_stock_arr = np.zeros(n, dtype=np.int32)
        fulfilled_arr = np.zeros(n, dtype=np.int32)
        stockout_arr = np.zeros(n, dtype=np.int32)
        waste_arr = np.zeros(n, dtype=np.int32)
        
        # Lead time with shocks
        base_lead_time = sku_data['lead_time_days']
        lead_time_ts = (base_lead_time * shock_supply).astype(int)
        
        # Incoming stock (constrained by supply shocks)
        incoming = (np.random.poisson(3, n) * 25 / shock_supply).astype(np.int32)
        
        for i in range(n):
            opening_stock_arr[i] = current_stock
            available = current_stock + incoming[i]
            
            if realized_demand[i] > available:
                fulfilled_arr[i] = available
                stockout_arr[i] = 1
            else:
                fulfilled_arr[i] = realized_demand[i]
            
            # Waste for perishables
            if sku_data['life_category'] == 'PERISHABLE':
                waste = int(current_stock * np.random.uniform(0.05, 0.15))
            else:
                waste = 0
            
            waste_arr[i] = waste
            current_stock = max(0, available - fulfilled_arr[i] - waste)
            closing_stock_arr[i] = current_stock
        
        # Assign results
        group['channel'] = np.random.choice(CHANNELS, size=n)
        group['price'] = np.round(price_ts, 2)
        group['promo_flag'] = promo_flag
        group['actual_demand'] = fulfilled_arr
        group['opening_stock'] = opening_stock_arr
        group['closing_stock'] = closing_stock_arr
        group['stockout_flag'] = stockout_arr
        group['waste_spoiled'] = waste_arr
        group['lead_time_actual'] = lead_time_ts
        
        final_rows.append(group)
        
        processed += 1
        if processed % 100 == 0:
            logging.info(f"Processed {processed}/{total_groups} combinations...")
    
    df = pd.concat(final_rows, ignore_index=True)
    df = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
    
    logging.info(f"Demand simulation complete. Total rows: {len(df):,}")
    return df


# =============================================================================
# 9. FEATURE ENGINEERING
# =============================================================================

def add_forecasting_features(df):
    """Adds features optimized for ML/DL forecasting."""
    logging.info("Adding forecasting features...")
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Lag features
    gb = df.groupby(['sku_id', 'location_id'])
    for lag in [1, 7, 14, 30]:
        df[f'demand_lag_{lag}'] = gb['actual_demand'].shift(lag).fillna(0).astype(int)
    
    # Rolling features
    for window in [7, 14, 30]:
        df[f'demand_roll_mean_{window}'] = gb['actual_demand'].shift(1).rolling(window, min_periods=1).mean().fillna(0)
        df[f'demand_roll_std_{window}'] = gb['actual_demand'].shift(1).rolling(window, min_periods=1).std().fillna(0)
    
    # Promotion features
    df['days_since_promo'] = gb['promo_flag'].apply(
        lambda s: s.cumsum() - s.cumsum().where(s == 1).ffill().fillna(0)
    ).values.astype(int)
    
    # Price features
    df['price_vs_base'] = df['price'] / df['base_price']
    df['price_change_pct'] = gb['price'].pct_change().fillna(0)
    
    logging.info("Feature engineering complete.")
    return df


# =============================================================================
# 10. MAIN ORCHESTRATION
# =============================================================================

def generate_enhanced_dataset(n_skus=N_SKUS, n_locs=N_LOCS, start_date=START_DATE, years=YEARS):
    """Main generation function with all enhancements."""
    end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=years) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    logging.info("="*60)
    logging.info("ENHANCED FMCG DATASET GENERATOR V2.0")
    logging.info("="*60)
    logging.info(f"Config: {n_skus} SKUs, {n_locs} Locations, {years} Years")
    logging.info(f"Date Range: {start_date} to {end_date}")
    
    # Generate all components
    sku_master = create_sku_master_with_lifecycle(n_skus, start_date, end_date)
    loc_master = create_location_master_with_evolution(n_locs, start_date, end_date)
    fest_df = build_realistic_uk_calendar(start_date, end_date)
    weather_df = generate_weather_data(start_date, end_date)
    macro_df = generate_monthly_macro(start_date, end_date)
    shock_df = generate_external_shocks(start_date, end_date)
    promo_calendar = generate_strategic_promotions(start_date, end_date, fest_df)
    comp_df = generate_competitor_activity(start_date, end_date)
    
    # Core simulation
    daily_df = simulate_demand_v2(sku_master, loc_master, start_date, end_date,
                                   fest_df, weather_df, macro_df, shock_df, promo_calendar, comp_df)
    daily_df = add_forecasting_features(daily_df)
    
    # Save outputs
    ensure_dir(OUTPUT_DIR)
    logging.info(f"Saving to {OUTPUT_DIR}...")
    
    # Save as Parquet (optimized for Kaggle)
    sku_master.to_parquet(os.path.join(OUTPUT_DIR, 'sku_master.parquet'), index=False)
    loc_master.to_parquet(os.path.join(OUTPUT_DIR, 'location_master.parquet'), index=False)
    daily_df.to_parquet(os.path.join(OUTPUT_DIR, 'daily_timeseries.parquet'), index=False)
    fest_df.to_parquet(os.path.join(OUTPUT_DIR, 'festival_calendar.parquet'), index=False)
    shock_df.to_parquet(os.path.join(OUTPUT_DIR, 'external_shocks.parquet'), index=False)
    macro_df.to_parquet(os.path.join(OUTPUT_DIR, 'macro_indicators.parquet'), index=False)
    weather_df.to_parquet(os.path.join(OUTPUT_DIR, 'weather_data.parquet'), index=False)
    comp_df.to_parquet(os.path.join(OUTPUT_DIR, 'competitor_activity.parquet'), index=False)
    
    # Also save CSVs for compatibility
    daily_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_timeseries.csv'), index=False)
    
    logging.info("="*60)
    logging.info("GENERATION COMPLETE!")
    logging.info("="*60)
    
    return sku_master, loc_master, daily_df, fest_df, shock_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = generate_enhanced_dataset()
    
    if results:
        sku_master, loc_master, daily_df, fest_df, shock_df = results
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"SKUs: {len(sku_master)} (with lifecycle)")
        print(f"Locations: {len(loc_master)} (with demographics)")
        print(f"Time Series Rows: {len(daily_df):,}")
        print(f"Date Range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
        print(f"Holidays: {len(fest_df)} (realistic UK calendar)")
        print(f"External Shocks: {len(shock_df)}")
        print(f"\nNew Product Launches: {(sku_master['birth_date'] > pd.to_datetime(START_DATE)).sum()}")
        print(f"Retired Products: {sku_master['retirement_date'].notna().sum()}")
        print(f"Total Stockouts: {daily_df['stockout_flag'].sum():,}")
        print(f"Avg Daily Demand: {daily_df['actual_demand'].mean():.1f}")
        
        print("\n--- Sample Data ---")
        print(daily_df[['date', 'sku_id', 'location_id', 'price', 'promo_flag', 
                        'actual_demand', 'stockout_flag']].head(10))
        
        print("\n--- External Shocks ---")
        print(shock_df[['name', 'start', 'end', 'demand_impact', 'supply_impact']])