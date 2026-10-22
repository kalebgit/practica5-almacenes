"""Distribución de las variables numéricas

Usa el dataset limpio del Paso 2, solo filas con medición (sin las rachas largas).

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.eda.distribuciones
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_LIMPIO, RUTA_GRAPHS  # noqa: E402
from src.cleaning.preprocessing import VARIABLES_NUMERICAS  # noqa: E402

pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(160)

COLOR_DATOS = "#2a78d6"
COLOR_NORMAL = "#eb6834"
COLOR_TEXTO = "#52514e"

# Unidades según la ficha de UCI.
UNIDADES = {
    "global_active_power": "kW",
    "global_reactive_power": "kW",
    "voltage": "V",
    "global_intensity": "A",
    "sub_metering_1": "Wh",
    "sub_metering_2": "Wh",
    "sub_metering_3": "Wh",
}

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLOR_TEXTO,
        "axes.labelcolor": COLOR_TEXTO,
        "xtick.color": COLOR_TEXTO,
        "ytick.color": COLOR_TEXTO,
        "axes.titlesize": 10,
        "font.size": 9,
    }
)

def cargar_medidos() -> pl.DataFrame:
    """Dataset limpio sin las rachas largas (no tienen valores que describir)."""
    df = pl.read_csv(RUTA_ARCHIVO_LIMPIO, try_parse_dates=True)
    return df.filter(~pl.col("medicion_faltante"))


def resumen(df: pl.DataFrame) -> pl.DataFrame:
    """Medidas de tendencia central y dispersión de las variables numéricas."""
    medidas = df.select(VARIABLES_NUMERICAS)
    moda = medidas.select(pl.all().mode().sort().first()).insert_column(0, pl.Series("statistic", ["moda"]))
    return pl.concat([medidas.describe(), moda]).with_columns(
        pl.col("statistic").replace({"mean": "media", "50%": "mediana"})
    )


def porcentaje_atipicos(df: pl.DataFrame) -> dict[str, float]:
    """Porcentaje de valores atípicos según el criterio IQR (Q1 - 1.5·IQR, Q3 + 1.5·IQR)."""
    pct = {}
    for var in VARIABLES_NUMERICAS:
        q1, q3 = df[var].quantile(0.25), df[var].quantile(0.75)
        pct[var] = df[var].is_between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)).not_().mean() * 100
    return pct


def efecto_escalado(df: pl.DataFrame, pct: dict[str, float]) -> pl.DataFrame:
    """Compara min-max, (x - min) / (max - min), contra z-score, (x - media) / std:
    la desviación estándar que deja cada uno y el rango que resulta con z-score."""
    filas = []
    for var in VARIABLES_NUMERICAS:
        x = df[var]
        minmax = (x - x.min()) / (x.max() - x.min())
        z = (x - x.mean()) / x.std()
        filas.append(
            {
                "variable": var,
                "pct_atipicos_iqr": round(pct[var], 1),
                "std_minmax": round(minmax.std(), 3),
                "z_min": round(z.min(), 2),
                "z_max": round(z.max(), 2),
                "std_z": round(z.std(), 3),
            }
        )
    return pl.DataFrame(filas).sort("pct_atipicos_iqr", descending=True)


def graficar_histogramas(df: pl.DataFrame) -> Path:
    """Histograma (densidad) de cada variable con la normal teórica encima."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, var in zip(axes.flat, VARIABLES_NUMERICAS):
        x = df[var].to_numpy()
        # norm.fit estima la media y la desviación de los datos.
        mu, sigma = stats.norm.fit(x)
        malla = np.linspace(x.min(), x.max(), 400)
        ax.hist(x, bins=100, density=True, color=COLOR_DATOS, label="Datos")
        ax.plot(malla, stats.norm.pdf(malla, mu, sigma), color=COLOR_NORMAL, lw=2, label="Normal (misma media y desv.)")
        ax.set_title(f"{var}\nasimetría {stats.skew(x):.2f}")
        ax.set_ylabel("Densidad")
        
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False)
    fig.suptitle("Distribución de las variables numéricas", color=COLOR_TEXTO)
    fig.tight_layout()
    ruta = RUTA_GRAPHS / "distribuciones_histogramas.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def graficar_boxplots(df: pl.DataFrame, pct: dict[str, float]) -> None:
    """Boxplot horizontal de cada variable con el % de atípicos (IQR) en la etiqueta.
    No son comparables entre variables porque las unidades son distintas."""
    orden = sorted(pct, key=pct.get, reverse=True)

    fig, axes = plt.subplots(len(orden), 1, figsize=(10, 9))
    for ax, var in zip(axes, orden):
        ax.boxplot(
            df[var].to_numpy(), orientation="horizontal", widths=0.6,
            boxprops={"color": "#9a9a96"}, whiskerprops={"color": "#9a9a96"}, capprops={"color": "#9a9a96"},
            medianprops={"color": COLOR_TEXTO}, flierprops={"marker": "o", "markersize": 2, "markerfacecolor": COLOR_NORMAL, "markeredgecolor": "none", "alpha": 0.3},
        )
        ax.set_yticks([1], [f"{var}\n{pct[var]:.1f} % atípicos"])
        ax.set_xlabel(UNIDADES[var], loc="right")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        f"{orden[0]} tiene {pct[orden[0]]:.1f} % de atípicos; {orden[-1]} solo {pct[orden[-1]]:.1f} %",
        color=COLOR_TEXTO, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(RUTA_GRAPHS / "distribuciones_boxplots.png", dpi=150)
    plt.close(fig)


def ejecutar_distribuciones() -> None:
    df = cargar_medidos()
    print(resumen(df))
    pct = porcentaje_atipicos(df)
    print("\nAtípicos (IQR) y efecto de min-max vs z-score:")
    print(efecto_escalado(df, pct))
    RUTA_GRAPHS.mkdir(parents=True, exist_ok=True)
    graficar_histogramas(df)
    graficar_boxplots(df, pct)

if __name__ == "__main__":
    ejecutar_distribuciones()
