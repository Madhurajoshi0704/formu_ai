import re
from typing import Dict, Any, Tuple, List
from src.database.lims_db import run_custom_query
from src.models.simulator import FormulationSimulator

class LabCopilotAgent:
    def __init__(self):
        self.simulator = FormulationSimulator()
        self.simulator.load_model()

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        AI Agent routing pipeline:
        1. Parse intent from researcher prompt (Search LIMS, Run Predictor, or Design/Optimize).
        2. Extract parameters using regex patterns.
        3. Invoke database tools or simulation models.
        4. Synthesize final answer.
        """
        p = prompt.lower().strip()
        
        # --- Intent 1: AI-Powered Formulation Design (Optimization) ---
        if "design" in p or "optimize" in p:
            # Extract target viscosity and pH
            visc_match = re.search(r'viscosity\s*(?:of|around|==|value)?\s*(\d+)', p)
            ph_match = re.search(r'ph\s*(?:of|around|==|value)?\s*(\d+(?:\.\d+)?)', p)
            
            target_visc = float(visc_match.group(1)) if visc_match else 3000.0
            target_ph = float(ph_match.group(1)) if ph_match else 6.0
            
            inputs, outputs = self.simulator.optimize_formulation(target_visc, target_ph)
            
            return {
                "intent": "design",
                "message": f"FormuAI Agent has run a virtual optimization trial targeting **{target_visc} cP** viscosity and **{target_ph}** pH. Below is the proposed formulation composition:",
                "composition": inputs,
                "predicted_properties": outputs
            }
            
        # --- Intent 2: Run Virtual Simulation Test (Prediction) ---
        elif "predict" in p or "simulate" in p:
            # Extract ingredient values
            surf_match = re.search(r'surfactant\s*(\d+(?:\.\d+)?)', p)
            cosurf_match = re.search(r'co-surfactant\s*(\d+(?:\.\d+)?)', p)
            thick_match = re.search(r'thickener\s*(\d+(?:\.\d+)?)', p)
            active_match = re.search(r'active\s*(\d+(?:\.\d+)?)', p)
            temp_match = re.search(r'temp\s*(\d+)', p)
            
            surfactant = float(surf_match.group(1)) if surf_match else 12.0
            co_surfactant = float(cosurf_match.group(1)) if cosurf_match else 3.5
            thickener = float(thick_match.group(1)) if thick_match else 1.2
            active = float(active_match.group(1)) if active_match else 1.0
            water = round(100.0 - (surfactant + co_surfactant + thickener + active), 2)
            temp = float(temp_match.group(1)) if temp_match else 60.0
            shear = 1200.0 # Default process parameter
            
            inputs = {
                'surfactant_pct': surfactant,
                'co_surfactant_pct': co_surfactant,
                'thickener_pct': thickener,
                'active_ingredient_pct': active,
                'water_pct': water,
                'mixing_temp_c': temp,
                'shear_rate_rpm': shear
            }
            
            outputs = self.simulator.predict(inputs)
            
            return {
                "intent": "predict",
                "message": f"Successfully simulated process for formulation with surfactant={surfactant}%, co-surfactant={co_surfactant}%, thickener={thickener}%, active={active}%. Predictions:",
                "composition": inputs,
                "predicted_properties": outputs
            }

        # --- Intent 3: Search LIMS/ELN Database ---
        else:
            # Construct simple SQLite queries based on criteria
            query = """
                SELECT 
                    e.experiment_name, e.researcher_name, f.surfactant_pct, f.thickener_pct, 
                    m.viscosity_cp, m.ph, m.stability_days
                FROM experiments e
                JOIN formulations f ON e.id = f.experiment_id
                JOIN physical_measurements m ON e.id = m.experiment_id
            """
            conditions = []
            params = []
            
            # Check for researcher name search
            researcher_match = re.search(r'by\s+([a-zA-Z\s\.]+)', p)
            if researcher_match:
                name = researcher_match.group(1).strip()
                if "madhura" in name:
                    conditions.append("e.researcher_name LIKE ?")
                    params.append("%Madhura Joshi%")
                elif "sen" in name:
                    conditions.append("e.researcher_name LIKE ?")
                    params.append("%Sen%")
                elif "patel" in name:
                    conditions.append("e.researcher_name LIKE ?")
                    params.append("%Patel%")
            
            # Check for physical conditions (viscosity / stability)
            visc_match = re.search(r'viscosity\s*>\s*(\d+)', p)
            if visc_match:
                conditions.append("m.viscosity_cp > ?")
                params.append(float(visc_match.group(1)))
                
            stab_match = re.search(r'stability\s*>\s*(\d+)', p)
            if stab_match:
                conditions.append("m.stability_days > ?")
                params.append(int(stab_match.group(1)))

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY e.id DESC LIMIT 10"
            
            columns, results = run_custom_query(query, tuple(params))
            
            # Format return data
            data_list = []
            for row in results:
                data_list.append(dict(zip(columns, row)))
                
            return {
                "intent": "lims_query",
                "message": f"Queried historical LIMS database. Found **{len(results)}** matching experimental records:",
                "sql_executed": query,
                "records": data_list
            }
