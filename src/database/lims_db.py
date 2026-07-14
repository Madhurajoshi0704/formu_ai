import os
import sqlite3
import random
import pandas as pd
from typing import Dict, Any, List, Tuple

DB_PATH = "lims_experiments.db"

def init_lims_db(db_path: str = DB_PATH) -> str:
    """Initialize the SQLite LIMS database with normalized tables and synthetic formulation logs."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 1. Create Experiments master table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_name TEXT NOT NULL,
        date TEXT NOT NULL,
        researcher_name TEXT NOT NULL
    );
    """)
    
    # 2. Create Formulations composition table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS formulations (
        experiment_id INTEGER PRIMARY KEY,
        surfactant_pct REAL NOT NULL,
        co_surfactant_pct REAL NOT NULL,
        thickener_pct REAL NOT NULL,
        active_ingredient_pct REAL NOT NULL,
        water_pct REAL NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    );
    """)
    
    # 3. Create Process Parameters table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS process_parameters (
        experiment_id INTEGER PRIMARY KEY,
        mixing_temp_c REAL NOT NULL,
        shear_rate_rpm REAL NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    );
    """)
    
    # 4. Create Physical Measurements table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS physical_measurements (
        experiment_id INTEGER PRIMARY KEY,
        viscosity_cp REAL NOT NULL,
        ph REAL NOT NULL,
        stability_days INTEGER NOT NULL,
        foam_volume_ml REAL NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    
    # Seed data if tables are empty
    cursor.execute("SELECT COUNT(*) FROM experiments")
    if cursor.fetchone()[0] == 0:
        seed_synthetic_experiments(conn, count=150)
        
    conn.close()
    return os.path.abspath(db_path)

def seed_synthetic_experiments(conn: sqlite3.Connection, count: int = 150):
    """Generate realistic physical formulations and corresponding measured quality indicators."""
    cursor = conn.cursor()
    researchers = ["Madhura Joshi", "Dr. A. Sen", "Sarah Jenkins", "Rajesh Patel"]
    
    for i in range(count):
        # 1. Generate formulation percentages (Must sum to 100%)
        surfactant = round(random.uniform(8.0, 16.0), 2)
        co_surfactant = round(random.uniform(1.5, 5.5), 2)
        thickener = round(random.uniform(0.1, 3.0), 2)
        active = round(random.uniform(0.1, 2.5), 2)
        water = round(100.0 - (surfactant + co_surfactant + thickener + active), 2)
        
        # 2. Process parameters
        temp = round(random.uniform(40.0, 85.0), 1)
        shear = round(random.uniform(500.0, 2500.0), 0)
        
        # 3. Simulate physical properties based on formulation equations + noise
        # Viscosity (cP): positive correlation with thickener & surfactant, negative with high temp
        base_visc = (thickener * 2500) + (surfactant * 150) - ((temp - 50) * 15)
        viscosity = max(100.0, round(base_visc + random.uniform(-200, 200), 1))
        
        # pH: influenced slightly by active ingredient
        ph = max(4.0, min(8.5, round(6.2 - (active * 0.4) + random.uniform(-0.3, 0.3), 2)))
        
        # Emulsion stability (days): high mixing temp or high thickener ratio reduces stability. Optimal temp is 60C.
        stability_loss_temp = abs(temp - 60) * 0.8
        stability_loss_thickener = max(0.0, thickener - 2.5) * 50
        base_stability = 365 - stability_loss_temp - stability_loss_thickener
        stability = max(5, int(base_stability + random.randint(-20, 20)))
        
        # Foam volume (mL): driven by primary and co-surfactants, slightly reduced by mixing temp
        base_foam = (surfactant * 35) + (co_surfactant * 20) - (temp * 0.2)
        foam = max(50.0, round(base_foam + random.uniform(-25, 25), 1))
        
        # Write to tables
        exp_name = f"EXP-SHAMPOO-{2026}-{i+1:03d}"
        date_str = f"2026-06-{random.randint(1, 30):02d}"
        res_name = random.choice(researchers)
        
        cursor.execute(
            "INSERT INTO experiments (experiment_name, date, researcher_name) VALUES (?, ?, ?)",
            (exp_name, date_str, res_name)
        )
        exp_id = cursor.lastrowid
        
        cursor.execute(
            "INSERT INTO formulations VALUES (?, ?, ?, ?, ?, ?)",
            (exp_id, surfactant, co_surfactant, thickener, active, water)
        )
        
        cursor.execute(
            "INSERT INTO process_parameters VALUES (?, ?, ?)",
            (exp_id, temp, shear)
        )
        
        cursor.execute(
            "INSERT INTO physical_measurements VALUES (?, ?, ?, ?, ?)",
            (exp_id, viscosity, ph, stability, foam)
        )
        
    conn.commit()

def get_lims_dataframe(db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieve full merged experiments dataset for training and analysis."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            e.id as experiment_id, e.experiment_name, e.date, e.researcher_name,
            f.surfactant_pct, f.co_surfactant_pct, f.thickener_pct, f.active_ingredient_pct, f.water_pct,
            p.mixing_temp_c, p.shear_rate_rpm,
            m.viscosity_cp, m.ph, m.stability_days, m.foam_volume_ml
        FROM experiments e
        JOIN formulations f ON e.id = f.experiment_id
        JOIN process_parameters p ON e.id = p.experiment_id
        JOIN physical_measurements m ON e.id = m.experiment_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def run_custom_query(query: str, params: Tuple = (), db_path: str = DB_PATH) -> List[Tuple]:
    """Execute raw query on database and return raw tuples (useful for Agent executor)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        # Fetch columns too
        columns = [col[0] for col in cursor.description] if cursor.description else []
        return columns, results
    except Exception as e:
        return ["Error"], [(str(e),)]
    finally:
        conn.close()
