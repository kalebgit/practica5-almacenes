"""Código principal del dashboard (Streamlit).

Ejecutar desde la raíz del proyecto:
    streamlit run visualization/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dashboard import charts, data_loader, metrics  # noqa: E402,F401

st.set_page_config(page_title="Consumo Eléctrico Doméstico", layout="wide")
st.title("Consumo Eléctrico Doméstico")
