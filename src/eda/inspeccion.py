"""
Paso 1: inspección inicial del dataset original.

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.eda.inspeccion
"""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_CRUDO  # noqa: E402
from src.cleaning.preprocessing import agregar_datetime, cargar_crudo  # noqa: E402

pl.Config.set_tbl_rows(20)

# Variables numéricas de interés para la inspección. 
VARIABLES_NUMERICAS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

def seccion(titulo: str) -> None:
    """Imprime un título de sección con formato."""
    print(f"\n=== {titulo} ===")

def tamano(df: pl.DataFrame) -> None:
    """Imprime el tamaño del DataFrame."""
    seccion("1. Tamaño")
    tam = RUTA_ARCHIVO_CRUDO.stat().st_size
    # UCI reporta MiB2^20 bytes)
    print(f"Archivo: {RUTA_ARCHIVO_CRUDO.name} ({tam / 1e6:.1f} MB = {tam / 2**20:.1f} MiB)")
    print(f"Filas: {df.height:,}   Columnas: {df.width}")
    print("Tipos detectados al leer:")
    for nombre, tipo in df.schema.items():
        print(f"  {nombre:<22} {tipo}")


def periodo(df: pl.DataFrame) -> None:
    """Imprime el periodo de tiempo cubierto por el dataset."""
    seccion("2. Periodo")
    inicio, fin = df["datetime"].min(), df["datetime"].max()
    duracion = fin - inicio
    print(f"Inicio: {inicio}   Fin: {fin}")
    print(f"Duración: {duracion.days:,} días")


def frecuencia_y_orden(df: pl.DataFrame) -> None:
    """Imprime la frecuencia de los datos y si están ordenados cronológicamente."""
    seccion("3. Frecuencia y orden")
    pasos = df["datetime"].diff().drop_nulls().value_counts()
    print("Diferencias entre registros consecutivos (paso: conteo):")
    print(pasos) # cada 1 minuto
    print(f"¿Ordenado cronológicamente?: {df['datetime'].is_sorted()}")


def rejilla_y_duplicados(df: pl.DataFrame) -> None:
    """Imprime la rejilla temporal esperada (minutos entre el primer y último registro)
    y el número de duplicados."""
    seccion("4. Rejilla temporal y duplicados")
    inicio, fin = df["datetime"].min(), df["datetime"].max()
    minutos_esperados = int((fin - inicio).total_seconds() // 60) + 1
    # datos completos sin duplicados
    print(f"Minutos entre inicio y fin (inclusive): {minutos_esperados:,}")
    print(f"Filas en el archivo:                    {df.height:,}")
    print(f"Minutos sin fila:                       {minutos_esperados - df.height:,}")
    print(f"Filas duplicadas completas:             {df.is_duplicated().sum():,}")
    print(f"Timestamps (datetime) repetidos:        {df['datetime'].is_duplicated().sum():,}")


def faltantes(df: pl.DataFrame) -> None:
    """Imprime información sobre los valores faltantes en las variables numéricas."""
    seccion("5. Valores faltantes")
    print("Nulos por columna:")
    print(df.select(VARIABLES_NUMERICAS).null_count())

    # filas con al menos un nulo y filas con todas las variables nulas
    algun_nulo = pl.any_horizontal(pl.col(VARIABLES_NUMERICAS).is_null())
    todos_nulos = pl.all_horizontal(pl.col(VARIABLES_NUMERICAS).is_null())
    n_alguno = df.select(algun_nulo.sum()).item()
    n_todos = df.select(todos_nulos.sum()).item()
    print(f"Filas con al menos un nulo: {n_alguno:,} ({n_alguno / df.height:.2%})")
    print(f"Filas con todas las variables nulas: {n_todos:,}")

    # Bloque de minutos consecutivos sin medición.
    rachas = (
        df.select("datetime", algun_nulo.alias("nulo"))
        .with_columns((pl.col("nulo") != pl.col("nulo").shift(1)).cum_sum().alias("bloque"))
        .filter(pl.col("nulo"))
        .group_by("bloque")
        .agg(pl.col("datetime").first().alias("inicio"), pl.len().alias("minutos"))
    )
    print(f"\nBloques de minutos consecutivos sin medición: {rachas.height}")
    for limite in (1, 5, 60, 1440):
        corto = rachas.filter(pl.col("minutos") <= limite)
        print(f"  rachas <= {limite:>4} min: {corto.height:>3}  ({corto['minutos'].sum():,} filas)")
    largo = rachas.filter(pl.col("minutos") > 60)
    print(f"  rachas  > 60 min: {largo.height:>3}  ({largo['minutos'].sum():,} filas)")
    # Concentración: qué parte de las filas faltantes cae en rachas cortas vs largas.
    corto = rachas.filter(pl.col("minutos") <= 60)
    print("Concentración de las filas faltantes:")
    print(
        f"  {corto.height / rachas.height:.1%} de las rachas (<= 60 min) -> "
        f"{corto['minutos'].sum() / n_alguno:.1%} de las filas faltantes"
    )
    print(
        f"  {largo.height / rachas.height:.1%} de las rachas (> 60 min)  -> "
        f"{largo['minutos'].sum() / n_alguno:.1%} de las filas faltantes"
    )
    print("\nRachas más largas:")
    top10 = rachas.sort("minutos", descending=True).select("inicio", "minutos").head(10)
    # equivalente en dias 
    top10 = top10.with_columns((pl.col("minutos") / 1440).alias("dias"))
    print(top10)

    print("\nFilas sin medición por año:")
    print(
        df.group_by(pl.col("datetime").dt.year().alias("anio"))
        .agg(pl.len().alias("filas"), algun_nulo.sum().alias("sin_medicion"))
        .with_columns((pl.col("sin_medicion") / pl.col("sin_medicion").sum()).alias("pct_del_total"))
        .sort("anio")
    )


def reparto_energia(df: pl.DataFrame) -> None:
    seccion("6. Reparto de la energía (filas con medición)")
    medido = df.drop_nulls(VARIABLES_NUMERICAS)
    # global_active_power está en kW promedio del minuto: kW * 1000 / 60 = Wh en ese minuto.
    total_wh = (medido["Global_active_power"] * 1000 / 60).sum()
    print(f"Energía total: {total_wh / 1000:,.0f} kWh")
    print(f"Global active power promedio: {medido['Global_active_power'].mean():.2f} kW")
    for col, nombre in (
        ("Sub_metering_1", "cocina"),
        ("Sub_metering_2", "lavado"),
        ("Sub_metering_3", "calentador/AC"),
    ):
        print(f"  {nombre:<14} {medido[col].sum() / total_wh:6.1%}")
    resto = total_wh - medido.select(pl.sum_horizontal(pl.col("^Sub_metering_.*$")).sum()).item()
    print(f"  {'resto (UCI)':<14} {resto / total_wh:6.1%}")


def escalas(df: pl.DataFrame) -> None:
    """Imprime medidas de tendencia central y dispersión de las variables numéricas."""
    seccion("7. Escalas de las variables numéricas")
    print(df.select(VARIABLES_NUMERICAS).describe())


def ejecutar_inspeccion() -> None:
    """Ejecuta la inspección inicial del dataset original (crudo)."""
    df = agregar_datetime(cargar_crudo()) # agrega columna datetime a partir de Date y Time
    tamano(df)
    periodo(df)
    frecuencia_y_orden(df)
    rejilla_y_duplicados(df)
    faltantes(df)
    reparto_energia(df)
    escalas(df)

if __name__ == "__main__":
    ejecutar_inspeccion()
