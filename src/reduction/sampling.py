"""Paso 8 (opcional): reducción de numerosidad por muestreo.

Se compara qué tan representativa es cada
muestra de ~1% de los minutos con medición, contra la población completa,
usando la media y desviación estándar de `global_active_power` (kW).

El dataset preparado (Paso 6) no se toca: este script solo lee
`consumo_limpio.csv` y guarda la tabla comparativa y la gráfica.

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.reduction.sampling
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import RUTA_ARCHIVO_LIMPIO, RUTA_GRAPHS, RUTA_MUESTREO_COMPARACION  # noqa: E402
from src.cleaning.preprocessing import seccion  # noqa: E402
from src.eda.distribuciones import COLOR_DATOS, COLOR_NORMAL, COLOR_TEXTO  # noqa: E402

pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(160)

VARIABLE = "global_active_power"
SEMILLA = 2026
FRACCION = 0.01  # ~1 % de los minutos, para que las muestras sean comparables entre sí

# Un día tiene 1440 minutos: con ese periodo el sistemático siempre cae en la
# misma hora del día
K_SESGADO = 1440
# 1427 es primo y no es múltiplo de 60 ni de 1440: no coincide con ningún ciclo
# del dataset (día, hora), así que no debería repetir siempre la misma hora.
K_SIN_SESGO = 1427

COLOR_SESGO = "#eb6834"  # mismo tono que COLOR_NORMAL en distribuciones.py, para la muestra que sale mal


def cargar_medidos() -> pl.DataFrame:
    """Dataset limpio sin las rachas largas (igual criterio que distribuciones.py:
    no tiene sentido muestrear minutos sin medición)."""
    return pl.read_csv(RUTA_ARCHIVO_LIMPIO, try_parse_dates=True).filter(~pl.col("medicion_faltante"))


def aleatorio_simple(df: pl.DataFrame) -> pl.DataFrame:
    """Cada minuto entra a lo más una vez; todos con la misma probabilidad."""
    return df.sample(fraction=FRACCION, seed=SEMILLA)


def aleatorio_con_reemplazo(df: pl.DataFrame) -> pl.DataFrame:
    """Bootstrap: un mismo minuto puede salir varias veces. Sirve para estimar
    la variabilidad de un estadístico, no para explorar más rápido (eso ya lo
    hace el aleatorio sin reemplazo), pero se incluye para comparar las dos
    variantes que pide la teoría."""
    return df.sample(fraction=FRACCION, seed=SEMILLA, with_replacement=True)


def sistematico(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Un minuto cada k posiciones, empezando en un punto aleatorio (para no
    depender de que la fila 0 sea comparable a cualquier otra)."""
    inicio = int(pl.Series([SEMILLA]).to_numpy()[0]) % k
    return df.gather_every(k, offset=inicio)


def estratificado_por_mes(df: pl.DataFrame) -> pl.DataFrame:
    """Misma fracción de cada mes, para conservar la estacionalidad que ya se
    documentó en el EDA (invierno vs. verano)."""
    return (
        df.with_columns(pl.col("datetime").dt.strftime("%Y-%m").alias("mes"))
        .filter(pl.int_range(pl.len()).shuffle(seed=SEMILLA).over("mes") < (pl.len() * FRACCION).over("mes"))
        .drop("mes")
    )


def comparar(poblacion: pl.DataFrame, muestras: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Tabla tipo la de la guía: n, media y desviación de cada muestra contra
    la población completa, más el % de filas repetidas (solo el bootstrap
    puede repetir un mismo minuto)."""
    filas = [{
        "muestra": "población",
        "n": poblacion.height,
        "media_kw": poblacion[VARIABLE].mean(),
        "std_kw": poblacion[VARIABLE].std(),
        "pct_repetidos": 0.0,
    }]
    for nombre, muestra in muestras.items():
        filas.append({
            "muestra": nombre,
            "n": muestra.height,
            "media_kw": muestra[VARIABLE].mean(),
            "std_kw": muestra[VARIABLE].std(),
            "pct_repetidos": muestra["datetime"].is_duplicated().mean() * 100,
        })
    tabla = pl.DataFrame(filas).with_columns(pl.col("media_kw", "std_kw", "pct_repetidos").round(3))
    base = tabla.row(0, named=True)
    return tabla.with_columns(
        ((pl.col("media_kw") / base["media_kw"] - 1) * 100).round(1).alias("sesgo_media_pct")
    )


def graficar_comparacion(poblacion: pl.DataFrame, muestras: dict[str, pl.DataFrame]) -> None:
    """Densidad de la población (gris, de fondo) contra cada muestra, una por panel."""
    fig, axes = plt.subplots(1, len(muestras), figsize=(4 * len(muestras), 4), sharey=True)
    for ax, (nombre, muestra) in zip(axes, muestras.items()):
        # El sistemático sesgado (k = 1440) se resalta en naranja; los demás en azul.
        color = COLOR_SESGO if nombre == f"sistemático (k={K_SESGADO})" else COLOR_DATOS
        ax.hist(poblacion[VARIABLE], bins=60, density=True, color="#d8d7d0", label="Población")
        ax.hist(muestra[VARIABLE], bins=60, density=True, color=color, histtype="step", lw=2, label="Muestra")
        ax.set_title(nombre, fontsize=10)
        ax.set_xlabel(f"{VARIABLE} (kW)")
    axes[0].set_ylabel("Densidad")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Distribución de la potencia activa: población contra cada técnica de muestreo\n"
                 f"Muestras de ~{FRACCION:.0%} de los {poblacion.height:,} minutos con medición",
                 color=COLOR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    RUTA_GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_GRAPHS / "muestreo_comparacion.png", dpi=150)
    plt.close(fig)


def ejecutar_muestreo() -> pl.DataFrame:
    df = cargar_medidos()
    print(f"Minutos con medición: {df.height:,}")

    muestras = {
        "aleatorio simple": aleatorio_simple(df),
        "aleatorio con reemplazo": aleatorio_con_reemplazo(df),
        f"sistemático (k={K_SESGADO})": sistematico(df, K_SESGADO),
        f"sistemático (k={K_SIN_SESGO})": sistematico(df, K_SIN_SESGO),
        "estratificado por mes": estratificado_por_mes(df),
    }

    seccion("Comparación contra la población")
    tabla = comparar(df, muestras)
    print(tabla)

    bootstrap = tabla.filter(pl.col("muestra") == "aleatorio con reemplazo").row(0, named=True)
    sesgado = tabla.filter(pl.col("muestra") == f"sistemático (k={K_SESGADO})").row(0, named=True)
    sin_sesgo = tabla.filter(pl.col("muestra") == f"sistemático (k={K_SIN_SESGO})").row(0, named=True)
    print(f"\nAleatorio con reemplazo: {bootstrap['pct_repetidos']:.1f} % de los minutos salieron más de una vez "
          f"(con fracción de {FRACCION:.0%} se repite poco; el bootstrap se nota más con muestras pequeñas).")
    print(f"k = {K_SESGADO} coincide con el ciclo diario (1440 min): sesgo de {sesgado['sesgo_media_pct']:+.1f} % en la media.")
    print(f"k = {K_SIN_SESGO} no coincide con ningún ciclo: sesgo de {sin_sesgo['sesgo_media_pct']:+.1f} %.")

    graficar_comparacion(df, muestras)
    RUTA_MUESTREO_COMPARACION.parent.mkdir(parents=True, exist_ok=True)
    tabla.write_csv(RUTA_MUESTREO_COMPARACION)
    print(f"\nGuardado: {RUTA_MUESTREO_COMPARACION.name}")
    return tabla


if __name__ == "__main__":
    ejecutar_muestreo()
