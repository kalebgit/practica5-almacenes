"""Paso 3: normalización z-score de las variables numéricas.

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.cleaning.normalization
"""

import sys
from pathlib import Path

import polars as pl
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_LIMPIO, RUTA_ARCHIVO_NORMALIZADO, RUTA_PARAMS_ESCALADOR  # noqa: E402
from src.cleaning.preprocessing import PROPORCION_TRAIN, VARIABLES_NUMERICAS  # noqa: E402

pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(160)


def ejecutar_normalizacion() -> pl.DataFrame:
    """Normaliza las variables numéricas y guarda el dataset resultante."""
    df = pl.read_csv(RUTA_ARCHIVO_LIMPIO, try_parse_dates=True)

    # Corte cronológico: el primer 80 % es entrenamiento (pasado), el resto prueba (futuro).
    corte = int(df.height * PROPORCION_TRAIN)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < corte).then(pl.lit("train")).otherwise(pl.lit("test")).alias("conjunto")
    )
    print(f"La prueba inicia el {df['datetime'][corte]}")

    train = df.filter((pl.col("conjunto") == "train") & ~pl.col("medicion_faltante"))
    escalador = StandardScaler().set_output(transform="polars").fit(train.select(VARIABLES_NUMERICAS))
    df = df.with_columns(escalador.transform(df.select(VARIABLES_NUMERICAS)).with_columns(pl.all().fill_nan(None)))

    medido = df.filter(~pl.col("medicion_faltante")).group_by("conjunto")
    print("Media:", medido.agg(pl.col(VARIABLES_NUMERICAS).mean().round(3)).sort("conjunto", descending=True))
    print("Std:", medido.agg(pl.col(VARIABLES_NUMERICAS).std().round(3)).sort("conjunto", descending=True))

    pl.DataFrame({"variable": VARIABLES_NUMERICAS, "media": escalador.mean_, "std": escalador.scale_}).write_csv(
        RUTA_PARAMS_ESCALADOR
    )
    df.write_csv(RUTA_ARCHIVO_NORMALIZADO)
    print(f"Guardado: {RUTA_ARCHIVO_NORMALIZADO.name} ({df.height:,} filas) y {RUTA_PARAMS_ESCALADOR.name}")
    return df


if __name__ == "__main__":
    ejecutar_normalizacion()
