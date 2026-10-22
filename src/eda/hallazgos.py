"""Cifras que respaldan los hallazgos del dashboard (diapositivas 4 a 6).

Usa los mismos módulos que el dashboard (src/dashboard), así que los números
coinciden con lo que se ve al explorarlo con el periodo completo.

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.eda.hallazgos
"""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.cleaning.preprocessing import seccion  # noqa: E402
from src.dashboard import metrics  # noqa: E402
from src.dashboard.data_loader import LUGARES, cargar_por_hora  # noqa: E402

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(160)

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
INVIERNO = [12, 1, 2]
VERANO = [6, 7, 8]


def potencia(expr_energia: pl.Expr = pl.col("energia_kwh_sel")) -> pl.Expr:
    """kW promedio = kWh / horas medidas (los apagones no cuentan como cero)."""
    return metrics.potencia_media(expr_energia, pl.col("minutos_medidos"))


def en_el_tiempo(df: pl.DataFrame) -> None:
    seccion("1. ¿Cuánta energía y cómo cambia en el tiempo?")
    anual = (
        df.group_by(pl.col("hora").dt.year().alias("anio"))
        .agg(pl.col("energia_kwh_sel").sum().round(0).alias("energia_kwh"), potencia().round(3).alias("potencia_kw"),
             pl.col("fecha").n_unique().alias("dias"))
        .sort("anio")
    )
    print("Por año (2006 y 2010 incompletos; comparar con la potencia media):")
    print(anual)

    mensual = (
        df.group_by(pl.col("hora").dt.month().alias("mes"))
        .agg(potencia().round(3).alias("potencia_kw"))
        .sort("mes")
        .with_columns(pl.col("mes").map_elements(lambda m: MESES[m - 1], return_dtype=pl.String).alias("nombre"))
    )
    print("\nPotencia media por mes del año (todos los años juntos):")
    print(mensual)
    alto, bajo = mensual.sort("potencia_kw").row(-1, named=True), mensual.sort("potencia_kw").row(0, named=True)
    print(f"Mes más alto: {alto['nombre']} ({alto['potencia_kw']} kW); más bajo: {bajo['nombre']} "
          f"({bajo['potencia_kw']} kW); cociente {alto['potencia_kw'] / bajo['potencia_kw']:.2f}")

    inv = df.filter(pl.col("hora").dt.month().is_in(INVIERNO)).select(potencia()).item()
    ver = df.filter(pl.col("hora").dt.month().is_in(VERANO)).select(potencia()).item()
    print(f"\nInvierno (dic-feb): {inv:.3f} kW   Verano (jun-ago): {ver:.3f} kW   cociente {inv / ver:.2f}")

    agostos = (
        df.filter(pl.col("hora").dt.month() == 8)
        .group_by(pl.col("hora").dt.year().alias("anio"))
        .agg(pl.col("energia_kwh_sel").sum().round(0).alias("energia_kwh"), potencia().round(3).alias("potencia_kw"),
             pl.col("minutos_sin_medicion").sum().alias("min_sin_medicion"))
        .sort("anio")
    )
    print("\nAgosto de cada año:")
    print(agostos)


def por_lugar(df: pl.DataFrame) -> None:
    seccion("2. ¿Qué lugar consume más?")
    print(metrics.reparto_por_lugar(df, list(LUGARES)).with_columns(pl.col("energia_kwh", "pct").round(1)))

    print("\n% de la energía por lugar, invierno (dic-feb) contra verano (jun-ago):")
    filas = []
    for nombre, meses in (("invierno", INVIERNO), ("verano", VERANO)):
        sub = df.filter(pl.col("hora").dt.month().is_in(meses))
        total = sub["energia_kwh_sel"].sum()
        filas.append({"estacion": nombre, **{l: round(sub[c].sum() / total * 100, 1) for l, c in LUGARES.items()}})
    print(pl.DataFrame(filas))


def picos(df: pl.DataFrame) -> None:
    seccion("3. ¿A qué hora y qué día hay picos?")
    perfil = metrics.perfil_diario(df).with_columns(pl.col("potencia_kw").round(3))
    ancho = perfil.pivot(on="tipo_dia", index="hora_del_dia", values="potencia_kw").sort("hora_del_dia")
    print(ancho)
    for tipo in ("Entre semana", "Fin de semana"):
        p = perfil.filter(pl.col("tipo_dia") == tipo)
        manana = p.filter(pl.col("hora_del_dia") < 12).sort("potencia_kw").row(-1, named=True)
        noche = p.filter(pl.col("hora_del_dia") >= 17).sort("potencia_kw").row(-1, named=True)
        valle = p.sort("potencia_kw").row(0, named=True)
        print(f"{tipo}: pico mañana {manana['hora_del_dia']} h ({manana['potencia_kw']} kW), "
              f"pico noche {noche['hora_del_dia']} h ({noche['potencia_kw']} kW), "
              f"valle {valle['hora_del_dia']} h ({valle['potencia_kw']} kW), "
              f"pico/valle {noche['potencia_kw'] / valle['potencia_kw']:.1f}")

    franja = pl.col("hora_del_dia").is_between(19, 22)
    noche = df.filter(franja)
    total = noche["energia_kwh_sel"].sum()
    print("\n% de la energía por lugar entre 19 y 22 h:")
    print({l: round(noche[c].sum() / total * 100, 1) for l, c in LUGARES.items()})
    print(f"Las 4 horas de 19 a 22 h concentran el {total / df['energia_kwh_sel'].sum():.1%} de la energía "
          f"(17 % si fuera parejo)")

    matriz = metrics.matriz_hora_dia(df).sort("potencia_kw", descending=True).head(5)
    print("\nLas 5 celdas día x hora con más potencia:")
    print(matriz.with_columns(pl.col("dia_semana").map_elements(lambda d: metrics.DIAS[d - 1], return_dtype=pl.String),
                              pl.col("potencia_kw").round(3)))


def ejecutar_hallazgos() -> None:
    por_hora = cargar_por_hora()
    df = metrics.filtrar(por_hora, por_hora["fecha"].min(), por_hora["fecha"].max(), list(LUGARES))
    en_el_tiempo(df)
    por_lugar(df)
    picos(df)


if __name__ == "__main__":
    ejecutar_hallazgos()
