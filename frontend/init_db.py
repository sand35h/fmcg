import sqlite3
import pandas as pd
from pathlib import Path
import random

DB_PATH = Path("inventory.db")
DATA_DIR = Path("data")

def init_db():
    print("Initializing Inventory Database...")
    
    # Connect
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            sku_id TEXT,
            location_id TEXT,
            current_stock INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (sku_id, location_id)
        )
    """)
    
    # Load Master Data
    sku_path = DATA_DIR / "sku_master.parquet"
    loc_path = DATA_DIR / "location_master.parquet"
    
    if not sku_path.exists() or not loc_path.exists():
        print("Error: Master data files not found in data/ directory.")
        return

    sku_df = pd.read_parquet(sku_path)
    loc_df = pd.read_parquet(loc_path)
    
    print(f"Seeding {len(sku_df)} SKUs x {len(loc_df)} Locations...")
    
    # Efficient Batch Insert
    data = []
    for _, sku in sku_df.iterrows():
        for _, loc in loc_df.iterrows():
            # Initial stock logic: Between 50 and 500
            initial_stock = random.randint(50, 500)
            data.append((sku['sku_id'], loc['location_id'], initial_stock))
            
    cursor.executemany(
        "INSERT OR REPLACE INTO inventory (sku_id, location_id, current_stock) VALUES (?, ?, ?)",
        data
    )
    
    conn.commit()
    conn.close()
    print("Database initialization complete.")

if __name__ == "__main__":
    init_db()
