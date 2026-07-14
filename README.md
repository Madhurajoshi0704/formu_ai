# FormuAI: AI-Powered Formulation Optimizer & Digital Lab Agent

FormuAI is a digital R&D intelligence platform designed to accelerate formulation design and automate lab experimentation workflows in FMCG (Fast-Moving Consumer Goods) R&D laboratories. Built in Python using Streamlit, Scikit-Learn, SQLite, and Plotly, it predicts physical product metrics (viscosity, pH, foam stability) to replace physical trial-and-error, provides an NLP database agent mapping natural language queries to SQL commands, and delivers interactive visual analytics dashboards.

---

## 🏗️ System Architecture

```
                  +-----------------------------------+
                  |           R&D Chemist             |
                  +-----------------+-----------------+
                                    |
                                    v (Streamlit UI)
                  +-----------------------------------+
                  |          Lab Copilot Agent        |
                  +--------+-----------------+--------+
                           |                 |
         +-----------------+                 +-----------------+
         | NLP Query Router                                    | Predict / Optimize
         v                                                     v
   +-----+-----+                                         +-----+-----+
   | LIMS SQL  |                                         |  Multi-   |
   | Database  |                                         | Output ML |
   | (SQLite)  |                                         | Regressor |
   +-----------+                                         +-----------+
```

---

## ✨ Key Features

1. **AI-Powered Formulation Simulator**: Replaces physical laboratory measurements. Adjusting ingredient concentration ratios (surfactants, co-surfactants, thickeners, actives) generates instant physical property predictions (viscosity, pH, emulsion stability, foam volume) in under 10ms.
2. **AI Lab Copilot Agent**: Translates natural language R&D instructions (e.g. *"Show experiments run by Madhura Joshi"* or *"Design a formula with viscosity 4500 and pH 5.5"*) into database SQL commands or simulator runs.
3. **LIMS & ELN Data Management**: Integrates a normalized relational SQLite database storing historical experiments, process parameters, and measurements, simulating enterprise systems.
4. **Power BI-Style Visual Analytics**: Interactive Plotly charts illustrating property distributions, correlation matrices, and ingredient-viscosity trends to support data storytelling.

---

## 📁 Project Structure

```
formu_ai/
├── src/
│   ├── app.py               # Streamlit web application & dashboards
│   ├── database/
│   │   ├── lims_db.py       # LIMS database creation & mock data seeder
│   ├── models/
│   │   └── simulator.py     # Multi-output Random Forest regressor & optimizer
│   └── agent/
│       └── lab_copilot.py   # NLP query router & sql executor
├── tests/
│   └── test_formu_ai.py     # Unit and integration test suite
├── requirements.txt         # Python project dependencies
└── README.md                # System documentation
```

---

## 🚀 Getting Started

### Local Setup & Execution
To spin up the web dashboard locally:

```bash
# 1. Navigate to directory
cd /Users/harshitgautam/.gemini/antigravity/scratch/formu_ai

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the Streamlit web server
streamlit run src/app.py
```

- **Local Address**: `http://localhost:8501`

### Running Tests
To run the automated validation test suite:
```bash
pytest -v
```

---

## 📝 Suggested Resume Bullet Points for Digital R&D Role

- **Engineered an AI-powered Formulation Simulator** using Scikit-Learn MultiOutput Random Forests to predict conditioning shampoo properties (viscosity, pH, foam volume) from ingredient compositions, reducing physical trial-and-error cycles.
- **Developed a natural language LIMS database agent** translating R&D chemist prompts into SQL queries and simulation runs, automating experimental data retrieval from SQLite.
- **Designed a Power BI-style R&D analytics dashboard** using Streamlit and Plotly to visualize ingredient correlation matrices and process parameter distributions.
