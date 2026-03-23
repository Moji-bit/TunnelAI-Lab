"""ui/dashboard.py

Single-app mode notice.

The interactive workflow has been consolidated into `apps/ai_page.py`.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

st.set_page_config(page_title="TunnelAI-Lab – Unified App", layout="wide")
st.title("🚇 TunnelAI-Lab – Unified App")
st.info(
    "Die Dashboard-Funktionen wurden in die AI-Page integriert. "
    "Bitte nutze ab jetzt nur noch `apps/ai_page.py` für Szenario-Generierung, "
    "Dataset-Building, Training, Evaluation und Model-Tests."
)

st.markdown("### Was jetzt in der AI-Page verfügbar ist")
st.markdown(
    "- Scenario Generation (1 Basis-Szenario × N Seeds)\n"
    "- Dataset Builder\n"
    "- Training + Experiment Settings\n"
    "- Layer Inspection\n"
    "- Evaluation\n"
    "- Model Test auf Simulation"
)

st.code("streamlit run apps/ai_page.py")
