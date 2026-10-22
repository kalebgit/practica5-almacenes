"""
Rutas estaticas de archivos del proyecto de consumo electrico domestico.
"""

from pathlib import Path

RUTA_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_DATA = RUTA_PROYECTO / "data"
RUTA_DATA_RAW = RUTA_DATA / "data-raw"
RUTA_DATA_PROCESSED = RUTA_DATA / "data-processed"

# Dato original de UCI: no se modifica nunca.
RUTA_ARCHIVO_CRUDO = RUTA_DATA_RAW / "household_power_consumption.txt"

# Los CSV no guardan tipos: al leerlos usar try_parse_dates=True para que
# la columna datetime regrese como datetime y no como texto.
# Paso 2: limpio (tipos, dedup, faltantes) en unidades originales.
RUTA_ARCHIVO_LIMPIO = RUTA_DATA_PROCESSED / "consumo_limpio.csv"
# Paso 3: variables numericas estandarizadas (z-score) + columna de conjunto.
RUTA_ARCHIVO_NORMALIZADO = RUTA_DATA_PROCESSED / "consumo_normalizado.csv"
# Paso 6: dataset final (sin redundantes y con componentes principales).
RUTA_ARCHIVO_PREPARADO = RUTA_DATA_PROCESSED / "consumo_preparado.csv"

# Parametros guardados para escalar datos nuevos, revertir y reproducir.
RUTA_PARAMS_ESCALADOR = RUTA_DATA_PROCESSED / "scaler_params.csv"
RUTA_PARAMS_PCA = RUTA_DATA_PROCESSED / "pca_params.csv"

# Graficas (heatmap, scree plot, loadings, EDA). Fuera de visualization/,
# que contiene unicamente el codigo del dashboard.
RUTA_GRAPHS = RUTA_PROYECTO / "graphs"

# Paso 8 (opcional): comparacion de muestras contra la poblacion completa.
RUTA_MUESTREO_COMPARACION = RUTA_DATA_PROCESSED / "muestreo_comparacion.csv"
