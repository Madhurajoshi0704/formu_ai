import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple
import random

from src.database.lims_db import get_lims_dataframe, init_lims_db

MODEL_PATH = "formulation_simulator.pkl"

class FormulationSimulator:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.input_cols = [
            'surfactant_pct', 'co_surfactant_pct', 'thickener_pct', 
            'active_ingredient_pct', 'water_pct', 'mixing_temp_c', 'shear_rate_rpm'
        ]
        self.output_cols = ['viscosity_cp', 'ph', 'stability_days', 'foam_volume_ml']

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train a multi-output RandomForestRegressor to predict physical properties from composition."""
        X = df[self.input_cols]
        y = df[self.output_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Single RandomForestRegressor handles multi-output arrays natively in Scikit-Learn
        self.model = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate R^2 scores for each output column
        r2_scores = {}
        for idx, col in enumerate(self.output_cols):
            preds = self.model.predict(X_test)
            col_y_test = y_test.iloc[:, idx]
            col_preds = preds[:, idx]
            
            # Simple R2 calculation
            u = ((col_y_test - col_preds) ** 2).sum()
            v = ((col_y_test - col_y_test.mean()) ** 2).sum()
            r2_scores[col] = float(1 - u/v) if v != 0 else 0.0
            
        # Serialize model
        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, self.input_cols, self.output_cols), f)
            
        return r2_scores

    def load_model(self):
        """Load trained model weights from disk."""
        if not os.path.exists(self.model_path):
            # Database check & lazy train
            db_path = "lims_experiments.db"
            if not os.path.exists(db_path):
                init_lims_db(db_path)
            df = get_lims_dataframe(db_path)
            self.train(df)
            
        with open(self.model_path, "rb") as f:
            self.model, self.input_cols, self.output_cols = pickle.load(f)

    def predict(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Predict quality properties for a given list of ingredients and mixing metrics."""
        if self.model is None:
            self.load_model()
            
        df_input = pd.DataFrame([inputs])[self.input_cols]
        prediction = self.model.predict(df_input)[0]
        
        return {col: float(val) for col, val in zip(self.output_cols, prediction)}

    def optimize_formulation(self, target_viscosity: float, target_ph: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        AI Formulation Design: Evaluates 2000 random candidate combinations
        to find the one that matches target viscosity and pH with minimum cost.
        """
        if self.model is None:
            self.load_model()

        best_inputs = None
        best_predictions = None
        min_loss = float('inf')

        # Grid-search simulated candidate loops
        for _ in range(2500):
            surfactant = round(random.uniform(8.0, 16.0), 2)
            co_surfactant = round(random.uniform(1.5, 5.5), 2)
            thickener = round(random.uniform(0.1, 3.0), 2)
            active = round(random.uniform(0.1, 2.5), 2)
            water = round(100.0 - (surfactant + co_surfactant + thickener + active), 2)
            
            temp = round(random.uniform(40.0, 85.0), 2)
            shear = round(random.uniform(500.0, 2500.0), 2)
            
            candidate_inputs = {
                'surfactant_pct': surfactant,
                'co_surfactant_pct': co_surfactant,
                'thickener_pct': thickener,
                'active_ingredient_pct': active,
                'water_pct': water,
                'mixing_temp_c': temp,
                'shear_rate_rpm': shear
            }
            
            preds = self.predict(candidate_inputs)
            
            # Loss: MSE distance to targets (values scaled)
            loss = (((preds['viscosity_cp'] - target_viscosity) / 5000.0) ** 2) + \
                   (((preds['ph'] - target_ph) / 2.0) ** 2)
            
            if loss < min_loss:
                min_loss = loss
                best_inputs = candidate_inputs
                best_predictions = preds

        return best_inputs, best_predictions
