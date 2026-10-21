"""Paso 1 y 2: carga, inspección inicial y preprocesamiento (normalización cualitativa,
deduplicación, casteo de tipos y valores faltantes).

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.cleaning.preprocessing
"""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_CRUDO, RUTA_ARCHIVO_LIMPIO  # noqa: E402

pl.Config.set_tbl_rows(20)

# Variables numéricas ya con nombres normalizados (minúsculas).
VARIABLES_NUMERICAS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

# Rachas de hasta esta duración se interpolan; las más largas se marcan sin imputar.
LIMITE_INTERPOLACION_MIN = 60

# Corte cronológico entrenamiento/prueba (se reutiliza en el Paso 3).
PROPORCION_TRAIN = 0.8

def seccion(titulo: str) -> None:
    """Imprime un título de sección con formato."""
    print(f"\n=== {titulo} ===")


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


def normalizar_nombres(df: pl.DataFrame) -> pl.DataFrame:
    """Normalización cualitativa: el dataset no tiene campos de texto con valores
    (solo Date y Time, que se castean), así que se normalizan los nombres de las
    columnas: minúsculas y sin espacios."""
    return df.rename(lambda c: c.strip().lower().replace(" ", "_"))


def castear_tipos(df: pl.DataFrame) -> pl.DataFrame:
    """Casteo de tipos: las 7 variables numéricas se leen como texto, se convierten a float64 y se descartan las columnas Date y Time, se sustituye por datetime."""
    df = df.rename({"date": "Date", "time": "Time"})
    df = agregar_datetime(df).drop("Date", "Time")
    return df.select("datetime", pl.col(VARIABLES_NUMERICAS).cast(pl.Float64))


def deduplicar(df: pl.DataFrame) -> pl.DataFrame:
    """Criterio: datetime es la llave de la serie (un registro por minuto).
    Si un minuto se repite, se conserva la primera aparición."""
    seccion("Deduplicación")
    antes = df.height
    df = df.unique(subset="datetime", keep="first", maintain_order=True)
    print(f"Filas antes: {antes:,}   después: {df.height:,}   eliminadas: {antes - df.height:,}")
    return df


def medir_rachas(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega `largo_racha`: duración en minutos de la racha sin medición a la que
    pertenece cada fila (0 si la fila tiene medición). Basta con revisar una
    variable porque las 7 faltan a la vez (inspección, §5)."""
    nulo = pl.col("global_active_power").is_null()
    return (
        df.with_columns((nulo != nulo.shift(1)).cum_sum().alias("bloque"))
        .with_columns(
            pl.when(nulo).then(pl.len().over("bloque")).otherwise(0).alias("largo_racha")
        )
        .drop("bloque")
    )


def tratar_faltantes(df: pl.DataFrame) -> pl.DataFrame:
    """Estrategia híbrida:
    - rachas <= LIMITE_INTERPOLACION_MIN: interpolación lineal en el tiempo
      (marca `interpolado`).
    - rachas más largas: se dejan nulas (marca `medicion_faltante`); los pasos
      que necesitan filas completas (escalador, Pearson, PCA) las excluyen.
    """
    corta = pl.col("largo_racha").is_between(1, LIMITE_INTERPOLACION_MIN)
    larga = pl.col("largo_racha") > LIMITE_INTERPOLACION_MIN
    return df.with_columns(
        [
            pl.when(corta).then(pl.col(c).interpolate_by("datetime")).otherwise(pl.col(c)).alias(c)
            for c in VARIABLES_NUMERICAS
        ]
        + [corta.alias("interpolado"), larga.alias("medicion_faltante")]
    )

def ejecutar_preprocesamiento() -> pl.DataFrame:
    """Paso 2 completo: devuelve el dataset limpio y lo guarda en CSV."""
    df = normalizar_nombres(cargar_crudo())
    print("Columnas normalizadas:", df.columns)
    df = castear_tipos(df)
    print("Tipos tras el casteo:", dict(df.schema))
    df = deduplicar(df)

    df_limpio = tratar_faltantes(medir_rachas(df)).drop("largo_racha")
    seccion("Valores faltantes")
    print(f"Filas interpoladas (rachas <= {LIMITE_INTERPOLACION_MIN} min):   {df_limpio['interpolado'].sum():,}")
    print(f"Filas sin interpolar (rachas > {LIMITE_INTERPOLACION_MIN} min):  {df_limpio['medicion_faltante'].sum():,}")

    RUTA_ARCHIVO_LIMPIO.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.write_csv(RUTA_ARCHIVO_LIMPIO)
    print(f"\nGuardado: {RUTA_ARCHIVO_LIMPIO} ({df_limpio.height:,} filas, {df_limpio.width} columnas)")
    return df_limpio

if __name__ == "__main__":
    ejecutar_preprocesamiento()
