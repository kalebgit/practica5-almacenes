"""Paso 4: matriz de correlación de Pearson y selección de variables.

Usa solo entrenamiento con medición: la decisión de qué descartar es parte del
ajuste, igual que el escalador. Pearson no cambia con el z-score, así que da lo
mismo usar el CSV normalizado o el limpio.

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.reduction.correlation
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_NORMALIZADO, RUTA_GRAPHS, RUTA_PARAMS_ESCALADOR  # noqa: E402
from src.cleaning.preprocessing import VARIABLES_NUMERICAS, seccion  # noqa: E402
from src.eda.distribuciones import COLOR_DATOS, COLOR_TEXTO, UNIDADES  # noqa: E402

pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_rows(25)
pl.Config.set_tbl_width_chars(160)

UMBRAL = 0.8  # teoría: |r| > 0.8 "muy fuerte -> candidata a redundancia"
MUESTRA = 20_000  # puntos por diagrama de dispersión (1.6 M no se distinguen)
SEMILLA = 2026
COLOR_CONTEXTO = "#9a9a96"

# Decisión del Paso 4 (salida "Descartar:" de este script); la usan el PCA y el dataset final.
VARIABLES_DESCARTADAS = ["global_intensity"]


def cargar_train() -> pl.DataFrame:
    return (
        pl.read_csv(RUTA_ARCHIVO_NORMALIZADO)
        .filter((pl.col("conjunto") == "train") & ~pl.col("medicion_faltante"))
        .select(VARIABLES_NUMERICAS)
    )


def tabla_pares(df: pl.DataFrame) -> pl.DataFrame:
    """r de cada par, prueba H0: rho = 0 (valor p) e intervalo de confianza 95 %
    (transformación de Fisher). Es redundante si todo el intervalo supera el umbral."""
    filas = []
    for a, b in combinations(VARIABLES_NUMERICAS, 2):
        res = stats.pearsonr(df[a].to_numpy(), df[b].to_numpy())
        ic = res.confidence_interval(0.95)
        filas.append({"var_1": a, "var_2": b, "r": res.statistic, "ic_inf": ic.low, "ic_sup": ic.high, "p_valor": res.pvalue})
    return (
        pl.DataFrame(filas)
        .with_columns((pl.min_horizontal(pl.col("ic_inf").abs(), pl.col("ic_sup").abs()) > UMBRAL).alias("redundante"))
        .sort(pl.col("r").abs(), descending=True)
    )


def elegir_descartes(corr: np.ndarray, pares: pl.DataFrame) -> list[str]:
    """De cada par redundante se descarta la variable con mayor |r| medio con
    las demás (criterio de la teoría): es la que más información repite."""
    seccion("|r| medio de cada variable con las demás")
    media = dict(zip(VARIABLES_NUMERICAS, (np.abs(corr).sum(axis=0) - 1) / (len(corr) - 1)))
    for var, m in sorted(media.items(), key=lambda kv: -kv[1]):
        print(f"  {var:<22} {m:.3f}")
    return [max((a, b), key=media.get) for a, b in pares.filter("redundante").select("var_1", "var_2").rows()]


def graficar_heatmap(corr: np.ndarray, pares: pl.DataFrame) -> None:
    """Triángulo inferior: azul negativa, gris ~0, rojo positiva."""
    mapa = LinearSegmentedColormap.from_list("divergente", ["#2a78d6", "#f0efec", "#e34948"])
    m = np.where(np.tril(np.ones_like(corr, dtype=bool), k=-1), corr, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(m, cmap=mapa, vmin=-1, vmax=1)
    for i, j in zip(*np.tril_indices_from(corr, k=-1)):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                fontweight="bold" if abs(corr[i, j]) > UMBRAL else "normal", color=COLOR_TEXTO)
    ax.set_xticks(range(len(corr)), VARIABLES_NUMERICAS, rotation=35, ha="right")
    ax.set_yticks(range(len(corr)), VARIABLES_NUMERICAS)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="r de Pearson")
    top = pares.row(0, named=True)
    ax.set(xlabel="Variable", ylabel="Variable")
    ax.set_title(f"Matriz de correlación de Pearson (entrenamiento)\n"
                 f"Único par con |r| > {UMBRAL}: {top['var_1']} y {top['var_2']} (r = {top['r']:.3f})",
                 color=COLOR_TEXTO, loc="left")
    fig.tight_layout()
    fig.savefig(RUTA_GRAPHS / "correlacion_heatmap.png", dpi=150)
    plt.close(fig)


def graficar_dispersion(df: pl.DataFrame, pares: pl.DataFrame) -> None:
    """Los 3 pares con mayor |r|, en unidades originales (se revierte el z-score).
    El par redundante en color, los demás en gris como contexto."""
    params = pl.read_csv(RUTA_PARAMS_ESCALADOR)
    original = df.with_columns(pl.col(v) * s + m for v, m, s in params.rows()).sample(MUESTRA, seed=SEMILLA)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, p in zip(axes, pares.head(3).iter_rows(named=True)):
        ax.scatter(original[p["var_1"]], original[p["var_2"]], s=3, alpha=0.25,
                   color=COLOR_DATOS if p["redundante"] else COLOR_CONTEXTO)
        ax.set_xlabel(f"{p['var_1']} ({UNIDADES[p['var_1']]})")
        ax.set_ylabel(f"{p['var_2']} ({UNIDADES[p['var_2']]})")
        ax.set_title(f"r = {p['r']:.3f}  IC 95 % [{p['ic_inf']:.3f}, {p['ic_sup']:.3f}]"
                     + ("\nredundante" if p["redundante"] else f"\nno supera {UMBRAL}"))
    fig.suptitle("Diagramas de dispersión de los tres pares con mayor correlación\n"
                 f"Muestra aleatoria de {MUESTRA:,} minutos de entrenamiento, en unidades originales",
                 color=COLOR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RUTA_GRAPHS / "correlacion_dispersion.png", dpi=150)
    plt.close(fig)


def ejecutar_correlacion() -> list[str]:
    """Paso 4 completo: devuelve las variables a descartar."""
    df = cargar_train()
    print(f"Filas de entrenamiento con medición: {df.height:,}")
    corr = df.corr().to_numpy()

    seccion("Matriz de correlación de Pearson")
    print(df.corr().with_columns(pl.all().round(3)).insert_column(0, pl.Series("", VARIABLES_NUMERICAS)))

    seccion(f"Pares: prueba de hipótesis e IC 95 % (umbral |r| > {UMBRAL})")
    pares = tabla_pares(df)
    print(pares.with_columns(pl.col("r", "ic_inf", "ic_sup").round(4)))

    descartes = elegir_descartes(corr, pares)
    print(f"\nDescartar: {descartes}")
    print(f"Conservar: {[v for v in VARIABLES_NUMERICAS if v not in descartes]}")

    RUTA_GRAPHS.mkdir(parents=True, exist_ok=True)
    graficar_heatmap(corr, pares)
    graficar_dispersion(df, pares)
    return descartes


if __name__ == "__main__":
    ejecutar_correlacion()
