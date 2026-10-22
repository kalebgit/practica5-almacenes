"""Cálculo de métricas y KPIs del dashboard.

Todas las funciones reciben la tabla por hora de `data_loader.cargar_por_hora`
(ya filtrada por fechas) y la lista de lugares elegidos en el filtro. La
energía de la selección es la suma de esos lugares; con los cuatro, es la
energía total de la casa.

Potencia media = energía (kWh) / horas con medición: así los apagones no
cuentan como horas de consumo cero.
"""

from datetime import date

import polars as pl

from src.dashboard.data_loader import LUGARES

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Nivel de agregación del filtro -> intervalo de polars.
AGREGACIONES = {"Hora": "1h", "Día": "1d", "Semana": "1w", "Mes": "1mo"}


def filtrar(por_hora: pl.DataFrame, inicio, fin, lugares: list[str]) -> pl.DataFrame:
    """Rango de fechas (inclusive) y columna `energia_kwh_sel` con la energía
    de los lugares elegidos."""
    return por_hora.filter(pl.col("fecha").is_between(inicio, fin)).with_columns(
        pl.sum_horizontal(LUGARES[lugar] for lugar in lugares).alias("energia_kwh_sel")
    )


def potencia_media(energia_kwh: pl.Expr, minutos: pl.Expr) -> pl.Expr:
    """kW promedio = kWh / horas medidas."""
    return energia_kwh.sum() / (minutos.sum() / 60)


def kpis(df: pl.DataFrame, lugares: list[str]) -> dict:
    """Los 4 KPIs del wireframe y el % de minutos sin medición (aviso)."""
    pico = matriz_hora_dia(df).sort("potencia_kw", descending=True).row(0, named=True)
    ranking = reparto_por_lugar(df, lugares)
    minutos = df["minutos_medidos"].sum() + df["minutos_sin_medicion"].sum()
    return {
        "energia_kwh": df["energia_kwh_sel"].sum(),
        "potencia_media_kw": df.select(potencia_media(pl.col("energia_kwh_sel"), pl.col("minutos_medidos"))).item(),
        "pico_dia": DIAS[pico["dia_semana"] - 1],
        "pico_hora": pico["hora_del_dia"],
        "pico_kw": pico["potencia_kw"],
        "lugar_top": ranking["lugar"][0],
        "lugar_top_pct": ranking["pct"][0],
        "pct_sin_medicion": df["minutos_sin_medicion"].sum() / minutos * 100 if minutos else 0.0,
    }


def reparto_por_lugar(df: pl.DataFrame, lugares: list[str]) -> pl.DataFrame:
    """Ranking: energía (kWh) y % de cada lugar elegido, de mayor a menor."""
    energia = {lugar: df[LUGARES[lugar]].sum() for lugar in lugares}
    total = sum(energia.values())
    return pl.DataFrame(
        {
            "lugar": list(energia),
            "energia_kwh": list(energia.values()),
            "pct": [e / total * 100 if total else 0.0 for e in energia.values()],
        }
    ).sort("energia_kwh", descending=True)


def matriz_hora_dia(df: pl.DataFrame) -> pl.DataFrame:
    """Potencia media (kW) de la selección por día de la semana y hora del día."""
    return (
        df.group_by("dia_semana", "hora_del_dia")
        .agg(potencia_media(pl.col("energia_kwh_sel"), pl.col("minutos_medidos")).alias("potencia_kw"))
        .sort("dia_semana", "hora_del_dia")
    )


def perfil_diario(df: pl.DataFrame) -> pl.DataFrame:
    """Potencia media (kW) de la selección por hora del día, entre semana
    (lunes a viernes) y fin de semana (sábado y domingo)."""
    tipo = pl.when(pl.col("dia_semana") <= 5).then(pl.lit("Entre semana")).otherwise(pl.lit("Fin de semana"))
    return (
        df.with_columns(tipo.alias("tipo_dia"))
        .group_by("tipo_dia", "hora_del_dia")
        .agg(potencia_media(pl.col("energia_kwh_sel"), pl.col("minutos_medidos")).alias("potencia_kw"))
        .sort("tipo_dia", "hora_del_dia")
    )


def _restar_anio(dia: date) -> date:
    """Misma fecha un año antes (el 29 de febrero pasa al 28)."""
    try:
        return dia.replace(year=dia.year - 1)
    except ValueError:
        return dia.replace(year=dia.year - 1, day=28)


def rango_anio_anterior(inicio: date, fin: date, primera: date) -> tuple[date, date] | None:
    """El mismo rango un año antes, o None si cae antes del inicio de los datos."""
    previo = (_restar_anio(inicio), _restar_anio(fin))
    return previo if previo[0] >= primera else None


def cambio_pct(actual: float, anterior: float) -> float | None:
    """Cambio porcentual respecto al periodo anterior."""
    return (actual / anterior - 1) * 100 if anterior else None


def serie_temporal(df: pl.DataFrame, agregacion: str, lugares: list[str]) -> pl.DataFrame:
    """Energía (kWh) por periodo, total de la selección y por lugar, más los
    minutos sin medición del periodo (para marcar apagones)."""
    return (
        df.sort("hora")
        .group_by_dynamic("hora", every=AGREGACIONES[agregacion])
        .agg(
            pl.col("energia_kwh_sel").sum(),
            *[pl.col(LUGARES[lugar]).sum().alias(lugar) for lugar in lugares],
            pl.col("minutos_sin_medicion").sum(),
        )
        .rename({"hora": "periodo"})
    )


if __name__ == "__main__":
    # Revisión rápida: python -m src.dashboard.metrics
    from datetime import date

    from src.dashboard.data_loader import cargar_por_hora

    todos = list(LUGARES)
    df = filtrar(cargar_por_hora(), date(2006, 12, 16), date(2010, 11, 26), todos)
    print(kpis(df, todos))
    print(reparto_por_lugar(df, todos))
    print(serie_temporal(df, "Mes", todos).head(3))
    print(perfil_diario(df).head(3))
    print(rango_anio_anterior(date(2008, 1, 1), date(2008, 12, 31), date(2006, 12, 16)))
