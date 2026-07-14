import os
import pytest
import sqlite3

TEST_DB = "test_lims_experiments.db"
TEST_MODEL = "test_formulation_simulator.pkl"

from src.database.lims_db import init_lims_db, get_lims_dataframe, run_custom_query
from src.models.simulator import FormulationSimulator
from src.agent.lab_copilot import LabCopilotAgent

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_test_files():
    """Ensure clean slate for test runs, and purge testing DB/Model files afterwards."""
    yield
    for f in [TEST_DB, TEST_MODEL]:
        if os.path.exists(f):
            os.remove(f)

def test_database_init_and_seeding():
    db_path = init_lims_db(TEST_DB)
    assert os.path.exists(db_path)
    
    # Query database and assert counts
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM experiments")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 150

    # Retrieve pandas dataframe
    df = get_lims_dataframe(TEST_DB)
    assert len(df) == 150
    assert 'viscosity_cp' in df.columns
    assert 'thickener_pct' in df.columns

def test_simulator_training_and_inference():
    df = get_lims_dataframe(TEST_DB)
    sim = FormulationSimulator(model_path=TEST_MODEL)
    
    # Train
    r2_scores = sim.train(df)
    assert 'viscosity_cp' in r2_scores
    assert r2_scores['viscosity_cp'] > 0.5  # Model fits data trends accurately

    # Inference
    sample_inputs = {
        'surfactant_pct': 12.0,
        'co_surfactant_pct': 3.5,
        'thickener_pct': 1.2,
        'active_ingredient_pct': 1.0,
        'water_pct': 82.3,
        'mixing_temp_c': 60.0,
        'shear_rate_rpm': 1200.0
    }
    preds = sim.predict(sample_inputs)
    assert preds['viscosity_cp'] > 0.0
    assert 4.0 <= preds['ph'] <= 8.5

    # Optimization target search
    best_inputs, best_outputs = sim.optimize_formulation(4500.0, 6.0)
    assert best_inputs is not None
    assert abs(best_outputs['ph'] - 6.0) < 1.0

def test_agent_processor():
    # Setup agent targeting test database
    agent = LabCopilotAgent()
    # Mock databases/model path
    agent.simulator = FormulationSimulator(model_path=TEST_MODEL)
    agent.simulator.load_model()

    # Query Intent Test
    res_query = agent.process_prompt("Show experiments by Madhura")
    assert res_query["intent"] == "lims_query"
    assert len(res_query["records"]) > 0
    assert "Madhura" in res_query["records"][0]["researcher_name"]

    # Predict Intent Test
    res_predict = agent.process_prompt("Predict formulation for surfactant 14.5 and thickener 2.2")
    assert res_predict["intent"] == "predict"
    assert res_predict["composition"]["surfactant_pct"] == 14.5
    assert res_predict["composition"]["thickener_pct"] == 2.2
    assert "predicted_properties" in res_predict

    # Design/Optimize Intent Test
    res_design = agent.process_prompt("Design shampoo with viscosity of 4000 and ph of 5.5")
    assert res_design["intent"] == "design"
    assert "viscosity" in res_design["message"]
    assert "predicted_properties" in res_design
