# 🚧 TunnelAI-Lab  
### From Reactive to Predictive Tunnel Operations

TunnelAI-Lab is a research-oriented simulation and AI framework for predictive tunnel operations.

The goal of this project is to move from reactive tunnel control toward predictive, AI-assisted risk monitoring and event forecasting — inspired by real-world tunnel infrastructures (e.g., ASFINAG STI architecture).

---

## 🎯 Project Vision

Modern road tunnels are safety-critical cyber-physical systems.  
Current control systems are primarily deterministic and rule-based.

TunnelAI-Lab demonstrates how:

- 📊 Time-series forecasting
- 🔥 Event prediction
- ⚠️ Dynamic risk estimation
- 🧠 Multi-task Transformer models
- 🔒 Constraint-aware safety layers

can be integrated into a predictive tunnel management framework.

---

## 🏗 Architecture

```

TunnelAI-Lab/
│
├── tags/ # OPC-UA style tag definitions
│ └── tags.yaml
│
├── sim/ # Digital Twin Simulation
│ ├── traffic_model.py
│ ├── emission_model.py
│ ├── event_generator.py
│
├── streaming/ # Data streaming & recording
│ ├── opcua_mock_server.py
│ ├── recorder.py
│ ├── run_record.py
│ └── run_batch_record.py
│
├── models/ # AI Models
│ ├── backbone/
│ │ ├── transformer.py
│ │ ├── lstm.py
│ ├── heads/
│ │ ├── forecasting.py
│ │ ├── event.py
│ │ ├── risk.py
│
├── constraint_layer/
│ ├── sti_rules.py
│
├── evaluation/
│ ├── metrics.py
│ ├── robustness.py
│
└── ui/
├── dashboard.py

````

---

## 🔬 Core Components

### 1️⃣ Simulation Layer
A simplified digital twin of a road tunnel:
- Traffic flow
- Emissions (e.g., CO, visibility)
- Event generation (e.g., fire, congestion)

### 2️⃣ Streaming Layer
Simulates real-time data flow via:
- OPC-UA mock server
- Scenario execution
- CSV recording for training

### 3️⃣ AI Layer
Multi-task learning framework:
- Long-term time-series forecasting
- Event classification
- Risk estimation

### 4️⃣ Constraint Layer
Encodes deterministic safety logic (STI-like rule enforcement).

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Moji-bit/TunnelAI-Lab.git
cd TunnelAI-Lab
````

### 2. Create Environment (recommended: Conda)

```bash
conda create -n tunnelai python=3.10
conda activate tunnelai
pip install -r requirements.txt
```

If no `requirements.txt` exists yet:

```bash
pip install numpy pandas matplotlib streamlit pyyaml torch
```

---

## ▶️ Running the Simulation

### Run single scenario

```bash
python streaming/run_record.py --scenario scenarios/example.json --out data/raw/output.csv
```

### Run batch of scenarios

```bash
python streaming/run_batch_record.py
```

---

## 📊 Launch Dashboard

```bash
streamlit run apps/ui/dashboard.py
```

---

## 🧠 Research Use Case

TunnelAI-Lab is designed to answer:

* Can AI predict risk evolution before threshold violation?
* How robust are models under sensor noise?
* Can deterministic STI logic be combined with probabilistic AI?

---

## 📈 Evaluation

Metrics available in:

```
evaluation/metrics.py
evaluation/robustness.py
```

Includes:

* Forecasting error
* Event detection accuracy
* False alarm rate
* Sensitivity analysis

---

## 🛡 Safety Philosophy

AI does not replace deterministic safety logic.

Instead:

AI → Predicts trends
Rules → Enforce hard safety constraints

Hybrid architecture = Predictive + Safe.

---

## 📌 Status

🚧 Research Prototype
🎓 Bachelor Thesis Framework
🧪 Actively developed

---

## 📜 License

MIT License (to be added)

---

## 👤 Author

Mojtaba Akhundzadeh
BSc AI & Machine Learning
Predictive Tunnel Operations Research
