"""Paso 5 y 6: PCA sobre las variables que quedan tras el Paso 4 y dataset preparado.

El PCA se ajusta solo con entrenamiento medido (igual que el escalador) y se
aplica a todas las filas con medición; las rachas largas quedan con PC nulos.

Este script se ejecuta desde la raíz del proyecto con:
    python -m src.reduction.pca
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.rutas import (  # noqa: E402
    RUTA_ARCHIVO_NORMALIZADO,
    RUTA_ARCHIVO_PREPARADO,
    RUTA_GRAPHS,
    RUTA_PARAMS_ESCALADOR,
    RUTA_PARAMS_PCA,
)
from src.cleaning.preprocessing import VARIABLES_NUMERICAS, seccion  # noqa: E402
from src.eda.distribuciones import COLOR_DATOS, COLOR_NORMAL, COLOR_TEXTO  # noqa: E402
from src.reduction.correlation import COLOR_CONTEXTO, MUESTRA, SEMILLA, VARIABLES_DESCARTADAS  # noqa: E402

pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(160)

VARIABLES_PCA = [v for v in VARIABLES_NUMERICAS if v not in VARIABLES_DESCARTADAS]
VARIANZA_OBJETIVO = 0.80  

# Qué mide cada submedidor según la ficha de UCI.
APARATOS = {"sub_metering_1": "cocina", "sub_metering_2": "lavado", "sub_metering_3": "calentador / AC"}


def verificar_estandarizacion(train: pl.DataFrame) -> None:
    seccion("1. Variables estandarizadas (entrenamiento)")
    print(pl.concat([train.mean(), train.std()]).with_columns(pl.all().round(3))
          .insert_column(0, pl.Series("estadistico", ["media", "std"])))


def tabla_varianza(pca: PCA) -> pl.DataFrame:
    seccion("2. Varianza explicada por componente")
    tabla = pl.DataFrame({
        "componente": [f"PC{i + 1}" for i in range(pca.n_components_)],
        "valor_propio": pca.explained_variance_.round(3),
        "pct_varianza": (pca.explained_variance_ratio_ * 100).round(1),
        "pct_acumulada": (pca.explained_variance_ratio_.cumsum() * 100).round(1),
    })
    print(tabla)
    return tabla


def elegir_k(pca: PCA) -> int:
    """Compara los tres criterios de la teoría y usa la varianza acumulada."""
    seccion("3. Número de componentes")
    acumulada = pca.explained_variance_ratio_.cumsum()
    k = int(np.argmax(acumulada >= VARIANZA_OBJETIVO)) + 1
    print(f"Kaiser (valor propio > 1):         {(pca.explained_variance_ > 1).sum()} componente(s)")
    print(f"Varianza acumulada >= {VARIANZA_OBJETIVO:.0%}:      {k} componentes ({acumulada[k - 1]:.1%})")
    print(f"Se conservan: {k}")
    return k


def graficar_scree(pca: PCA, k: int) -> None:
    """Valor propio por componente (los retenidos en color) y varianza acumulada."""
    pcs = np.arange(1, pca.n_components_ + 1)
    acumulada = pca.explained_variance_ratio_.cumsum() * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(pcs, pca.explained_variance_, color=[COLOR_DATOS if p <= k else COLOR_CONTEXTO for p in pcs])
    axes[0].axhline(1, color=COLOR_NORMAL, ls="--", lw=1)
    axes[0].text(pcs[-1] + 0.4, 1, "Kaiser (λ = 1)", va="bottom", ha="right", color=COLOR_NORMAL)
    axes[0].set(title="Gráfica de sedimentación (scree plot)", xlabel="Componente principal", ylabel="Valor propio (λ)")
    axes[1].plot(pcs, acumulada, marker="o", color=COLOR_CONTEXTO)
    axes[1].plot(pcs[:k], acumulada[:k], marker="o", color=COLOR_DATOS, lw=2)
    axes[1].axhline(VARIANZA_OBJETIVO * 100, color=COLOR_NORMAL, ls="--", lw=1)
    axes[1].annotate(f"{acumulada[k - 1]:.1f} %", (k, acumulada[k - 1]), xytext=(-12, 10), textcoords="offset points", ha="right")
    axes[1].set(title="Varianza explicada acumulada", xlabel="Número de componentes conservados",
                ylabel="Varianza acumulada (%)", ylim=(0, 105))
    for ax in axes:
        ax.set_xticks(pcs)
    fig.suptitle("Varianza explicada por los componentes principales\n"
                 f"Criterio: varianza acumulada >= {VARIANZA_OBJETIVO:.0%}; se conservan {k} de "
                 f"{pca.n_components_} componentes ({acumulada[k - 1]:.1f} %)",
                 color=COLOR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RUTA_GRAPHS / "pca_scree.png", dpi=150)
    plt.close(fig)


def tabla_loadings(pca: PCA, k: int) -> pl.DataFrame:
    """Loadings de los k retenidos y del primero descartado (PC{k+1}), para
    justificar por qué se deja fuera."""
    n = min(k + 1, pca.n_components_)
    seccion(f"4. Loadings: PC1 a PC{k} retenidos, PC{n} descartado")
    loadings = pl.DataFrame(pca.components_[:n].T.round(3), schema=[f"PC{i + 1}" for i in range(n)])
    loadings = loadings.insert_column(0, pl.Series("variable", VARIABLES_PCA))
    print(loadings)
    return loadings

def graficar_loadings(pca: PCA, k: int) -> None:
    """Mapa de calor de los loadings: rojo positivo, azul negativo. Incluye el
    primer componente descartado (PC{k+1}), separado por una línea.
    En negritas los pesos con |loading| >= 0.4 (los que dan nombre al componente)."""
    mapa = LinearSegmentedColormap.from_list("divergente", ["#2a78d6", "#f0efec", "#e34948"])
    n = min(k + 1, pca.n_components_)
    carga = pca.components_[:n].T
    etiquetas = [f"{v} ({APARATOS[v]})" if v in APARATOS else v for v in VARIABLES_PCA]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(carga, cmap=mapa, vmin=-1, vmax=1, aspect="auto")
    for i, j in np.ndindex(carga.shape):
        ax.text(j, i, f"{carga[i, j]:.2f}", ha="center", va="center", color=COLOR_TEXTO,
                fontweight="bold" if abs(carga[i, j]) >= 0.4 else "normal")
    ax.set_xticks(range(n), [f"PC{i + 1}\n{r:.1%}" + ("\n(descartado)" if i >= k else "")
                             for i, r in enumerate(pca.explained_variance_ratio_[:n])])
    ax.axvline(k - 0.5, color=COLOR_TEXTO, lw=2)
    ax.set_yticks(range(len(VARIABLES_PCA)), etiquetas)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="loading")
    ax.set(xlabel="Componente principal (% de varianza explicada)", ylabel="Variable original (z-score)")
    ax.set_title(f"Loadings de los componentes principales (PC1 a PC{k} retenidos, PC{n} descartado)\n"
                 "En negritas |loading| >= 0.4", color=COLOR_TEXTO, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RUTA_GRAPHS / "pca_loadings.png", dpi=150)
    plt.close(fig)

def proyectar(pca: PCA, df: pl.DataFrame, k: int) -> pl.DataFrame:
    """PC1..PCk para cada fila con medición; nulos en las rachas largas."""
    x = df.select(VARIABLES_PCA).to_numpy()
    medido = ~np.isnan(x).any(axis=1)
    pcs = np.full((len(x), k), np.nan)
    pcs[medido] = pca.transform(x[medido])[:, :k]
    return df.with_columns(pl.from_numpy(pcs, schema=[f"PC{i + 1}" for i in range(k)]).with_columns(pl.all().fill_nan(None)))


def graficar_proyeccion(pca: PCA, df: pl.DataFrame) -> None:
    """PC1 vs PC2 coloreado por el submedidor con más consumo en ese minuto
    (en Wh originales; gris si los tres marcan 0)."""
    params = {v: (m, s) for v, m, s in pl.read_csv(RUTA_PARAMS_ESCALADOR).rows()}
    sub = ["sub_metering_1", "sub_metering_2", "sub_metering_3"]
    muestra = (
        df.filter((pl.col("conjunto") == "train") & ~pl.col("medicion_faltante"))
        .sample(MUESTRA, seed=SEMILLA)
        .with_columns((pl.col(v) * params[v][1] + params[v][0]).round(0) for v in sub)
    )
    nombres = {"ninguno": "submedidores en 0", "sub_metering_1": "cocina", "sub_metering_2": "lavado",
               "sub_metering_3": "calentador / AC"}
    colores = {"ninguno": COLOR_CONTEXTO, "sub_metering_1": "#1baf7a", "sub_metering_2": COLOR_NORMAL,
               "sub_metering_3": COLOR_DATOS}
    dominante = np.where(muestra.select(sub).to_numpy().max(axis=1) == 0, "ninguno",
                         np.array(sub)[muestra.select(sub).to_numpy().argmax(axis=1)])
    fig, ax = plt.subplots(figsize=(8, 6))
    # Grupos grandes primero para que no tapen a los pequeños.
    for clave in sorted(colores, key=lambda c: -(dominante == c).sum()):
        sel = dominante == clave
        ax.scatter(muestra["PC1"].to_numpy()[sel], muestra["PC2"].to_numpy()[sel], s=4, alpha=0.4,
                   color=colores[clave], label=f"{nombres[clave]} ({sel.mean():.0%})")
    r1, r2 = pca.explained_variance_ratio_[:2]
    ax.set(xlabel=f"PC1: nivel de consumo total ({r1:.1%} de la varianza)",
           ylabel=f"PC2: lavado (+) frente a calentador / AC (−) ({r2:.1%})")
    ax.legend(frameon=False, markerscale=3, title="Submedidor con más consumo")
    ax.set_title("Proyección de los minutos en los dos primeros componentes principales\n"
                 f"Muestra de {MUESTRA:,} minutos de entrenamiento, coloreada por el submedidor con mayor consumo",
                 color=COLOR_TEXTO, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RUTA_GRAPHS / "pca_proyeccion.png", dpi=150)
    plt.close(fig)


def guardar(pca: PCA, df: pl.DataFrame) -> None:
    """Paso 6: dataset preparado (sin descartadas, con PCs) y parámetros del PCA."""
    seccion("6. Dataset preparado")
    preparado = df.drop(VARIABLES_DESCARTADAS)
    preparado.write_csv(RUTA_ARCHIVO_PREPARADO)
    pl.DataFrame(pca.components_.T, schema=[f"PC{i + 1}" for i in range(pca.n_components_)]).insert_column(
        0, pl.Series("variable", VARIABLES_PCA)
    ).write_csv(RUTA_PARAMS_PCA)
    print(f"Guardado: {RUTA_ARCHIVO_PREPARADO.name} ({preparado.height:,} filas)")
    print(f"Columnas: {preparado.columns}")
    print(f"Guardado: {RUTA_PARAMS_PCA.name} (loadings de los {pca.n_components_} componentes)")


def ejecutar_pca() -> pl.DataFrame:
    df = pl.read_csv(RUTA_ARCHIVO_NORMALIZADO, try_parse_dates=True)
    train = df.filter((pl.col("conjunto") == "train") & ~pl.col("medicion_faltante")).select(VARIABLES_PCA)
    print(f"Variables del PCA ({len(VARIABLES_PCA)}): {VARIABLES_PCA}")

    verificar_estandarizacion(train)
    pca = PCA().fit(train.to_numpy())
    tabla_varianza(pca)
    k = elegir_k(pca)
    tabla_loadings(pca, k)

    RUTA_GRAPHS.mkdir(parents=True, exist_ok=True)
    graficar_scree(pca, k)
    graficar_loadings(pca, k)
    df = proyectar(pca, df, k)
    seccion("5. Proyección en 2D")
    print("Guardada en graphs/pca_proyeccion.png")
    graficar_proyeccion(pca, df)
    guardar(pca, df)
    return df


if __name__ == "__main__":
    ejecutar_pca()
