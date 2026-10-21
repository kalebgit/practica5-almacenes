"""Paso 1 y 2: carga, inspección inicial y preprocesamiento (normalización cualitativa,
deduplicación, casteo de tipos y valores faltantes)."""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_CRUDO  # noqa: E402


def cargar_crudo() -> pl.DataFrame:
    """Lee el archivo original tal como lo publica UCI.

    - separador ';'
    - los faltantes vienen como '?' (nota 2 de UCI), por eso null_values="?"
    - Date y Time se leen como texto; el casteo a datetime es parte del Paso 2
    """
    return pl.read_csv(
        RUTA_ARCHIVO_CRUDO,
        separator=";",
        null_values="?",
        schema_overrides={"Date": pl.String, "Time": pl.String},
    )


def agregar_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """Une Date (dd/mm/yyyy) y Time (hh:mm:ss) en una sola columna datetime."""
    return df.with_columns(
        pl.concat_str("Date", "Time", separator=" ")
        .str.to_datetime("%d/%m/%Y %H:%M:%S")
        .alias("datetime")
    )
