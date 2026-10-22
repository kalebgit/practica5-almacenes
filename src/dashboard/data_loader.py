"""Carga del dataset procesado para el dashboard.

Lee `consumo_limpio.csv` (unidades reales: kW y Wh) y lo agrega por hora, para
que el dashboard trabaje con ~35 mil filas en vez de ~2 millones de minutos.
No modifica ningún archivo.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_LIMPIO  # noqa: E402

# Lugar de la casa -> columna de energía (kWh) en la tabla por hora.
# Los submedidores según la ficha de UCI; "resto" con su fórmula (nota 1).
LUGARES = {
    "Cocina": "cocina_kwh",
    "Cuarto de lavado": "lavado_kwh",
    "Calentador y AC": "calentador_ac_kwh",
    "Resto de la casa": "resto_kwh",
}


def energia_por_minuto(df: pl.DataFrame) -> pl.DataFrame:
    """Energía de cada minuto en Wh: total y por lugar.

    global_active_power es la potencia promedio del minuto en kW, así que
    kW * 1000 / 60 = Wh consumidos en ese minuto (nota 1 de UCI)."""
    total = pl.col("global_active_power") * 1000 / 60
    return df.with_columns(
        total.alias("energia_wh"),
        pl.col("sub_metering_1").alias("cocina_wh"),
        pl.col("sub_metering_2").alias("lavado_wh"),
        pl.col("sub_metering_3").alias("calentador_ac_wh"),
        (total - pl.col("sub_metering_1") - pl.col("sub_metering_2") - pl.col("sub_metering_3")).alias("resto_wh"),
    )


def cargar_minutos() -> pl.DataFrame:
    """El dataset limpio tal cual: un minuto por fila, en unidades reales."""
    return pl.read_csv(RUTA_ARCHIVO_LIMPIO, try_parse_dates=True)


def cargar_por_hora(minutos: pl.DataFrame | None = None) -> pl.DataFrame:
    """Una fila por hora: energía (kWh) total y por lugar, potencia media y
    máxima (kW) y minutos con y sin medición. Los minutos de apagón
    (`medicion_faltante`) no suman energía; se cuentan aparte."""
    minutos = energia_por_minuto(cargar_minutos() if minutos is None else minutos)
    return (
        minutos.group_by(pl.col("datetime").dt.truncate("1h").alias("hora"))
        .agg(
            (pl.col("energia_wh").sum() / 1000).alias("energia_kwh"),
            *[(pl.col(c.replace("_kwh", "_wh")).sum() / 1000).alias(c) for c in LUGARES.values()],
            pl.col("global_active_power").mean().alias("potencia_media_kw"),
            pl.col("global_active_power").max().alias("potencia_max_kw"),
            (~pl.col("medicion_faltante")).sum().alias("minutos_medidos"),
            pl.col("medicion_faltante").sum().alias("minutos_sin_medicion"),
        )
        .sort("hora")
        .with_columns(
            pl.col("hora").dt.date().alias("fecha"),
            pl.col("hora").dt.hour().alias("hora_del_dia"),
            pl.col("hora").dt.weekday().alias("dia_semana"),  # 1 = lunes ... 7 = domingo
        )
    )


def correlacion(minutos: pl.DataFrame, inicio, fin) -> pl.DataFrame:
    """Pearson entre las 7 variables medidas, con los minutos con medición del
    rango. Se usan las 7 (no las 6 del dataset preparado) para que se vea por
    qué se descartó global_intensity."""
    variables = [c for c in minutos.columns if c not in ("datetime", "interpolado", "medicion_faltante")]
    medidos = minutos.filter(pl.col("datetime").dt.date().is_between(inicio, fin) & ~pl.col("medicion_faltante"))
    matriz = medidos.select(variables).corr()
    return matriz.insert_column(0, pl.Series("variable_1", variables)).unpivot(
        index="variable_1", variable_name="variable_2", value_name="r"
    )


if __name__ == "__main__":
    # Revisión rápida: python -m src.dashboard.data_loader
    por_hora = cargar_por_hora()
    print(por_hora.head())
    print(f"Horas: {por_hora.height:,}   Energía total: {por_hora['energia_kwh'].sum():,.0f} kWh")
    print({lugar: round(por_hora[c].sum() / por_hora["energia_kwh"].sum() * 100, 1) for lugar, c in LUGARES.items()})
