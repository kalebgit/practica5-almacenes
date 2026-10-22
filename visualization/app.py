"""
Dashboard (Streamlit).

Ejecutar desde la raíz del proyecto:
    streamlit run visualization/app.py

Distribución según el wireframe (Storytelling, diapositiva 32): filtros arriba,
4 KPIs, tendencia y ranking, mapa de calor de picos, tabla de detalle y una
sección plegable con distribución y correlaciones.
"""

import sys
from pathlib import Path

import polars as pl
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dashboard import charts, data_loader, metrics  # noqa: E402

st.set_page_config(page_title="Consumo Eléctrico Doméstico", layout="wide")


@st.cache_data(show_spinner="Cargando datos...")
def cargar() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Se lee una sola vez; los filtros trabajan sobre la copia en memoria."""
    minutos = data_loader.cargar_minutos()
    return minutos, data_loader.cargar_por_hora(minutos)


minutos, por_hora = cargar()
primera, ultima = por_hora["fecha"].min(), por_hora["fecha"].max()

st.title("Consumo eléctrico doméstico")
st.caption("Un hogar en Sceaux, Francia · mediciones por minuto de diciembre de 2006 a noviembre de 2010 · fuente: UCI")

# --- Filtros: fecha · agregación · lugar (segmento) ---
f1, f2, f3 = st.columns([2, 1, 3])
rango = f1.date_input("Rango de fechas", (primera, ultima), min_value=primera, max_value=ultima)
agregacion = f2.selectbox("Agregación", list(metrics.AGREGACIONES), index=3)
lugares = f3.multiselect("Lugar de la casa", list(data_loader.LUGARES), default=list(data_loader.LUGARES))

if len(rango) != 2 or not lugares:
    st.info("Elige una fecha de inicio y una de fin, y al menos un lugar.")
    st.stop()
inicio, fin = rango
df = metrics.filtrar(por_hora, inicio, fin, lugares)

# --- KPIs, comparados con el mismo rango del año anterior (contexto) ---
k = metrics.kpis(df, lugares)
previo = metrics.rango_anio_anterior(inicio, fin, primera)
ka = metrics.kpis(metrics.filtrar(por_hora, *previo, lugares), lugares) if previo else None


def delta(clave: str) -> str | None:
    """Texto del cambio contra el año anterior, o None si no hay con qué comparar."""
    cambio = metrics.cambio_pct(k[clave], ka[clave]) if ka else None
    return f"{cambio:+.1f} % vs. año anterior" if cambio is not None else None


c1, c2, c3, c4 = st.columns(4)
c1.metric("Energía consumida", f"{k['energia_kwh']:,.0f} kWh", delta("energia_kwh"), delta_color="off")
c2.metric("Potencia media", f"{k['potencia_media_kw']:.2f} kW", delta("potencia_media_kw"), delta_color="off")
c3.metric("Hora pico", f"{k['pico_dia']} {k['pico_hora']}:00 h",
          f"{k['pico_kw']:.2f} kW" + (f" (año anterior: {ka['pico_dia']} {ka['pico_hora']}:00 h)" if ka else ""),
          delta_color="off", delta_arrow="off")
c4.metric("Lugar que más consume", k["lugar_top"],
          f"{k['lugar_top_pct']:.1f} % de la energía" + (f" (año anterior: {ka['lugar_top']})" if ka else ""),
          delta_color="off", delta_arrow="off")

notas = []
if previo:
    notas.append(f"Comparación contra el mismo rango un año antes ({previo[0]:%d/%m/%Y} a {previo[1]:%d/%m/%Y}).")
else:
    notas.append(f"Sin comparación anual: el rango un año antes queda antes del inicio de los datos ({primera:%d/%m/%Y}).")
if k["pct_sin_medicion"] > 0:
    notas.append(f"{k['pct_sin_medicion']:.2f} % de los minutos del rango no tienen medición (apagones); "
                 "no suman energía y no cuentan para la potencia media.")
st.caption(" ".join(notas))

# --- Tendencia y ranking ---
serie = metrics.serie_temporal(df, agregacion, lugares)
t1, t2 = st.columns([3, 2])
t1.altair_chart(charts.tendencia(serie, agregacion), width="stretch")
t2.altair_chart(charts.ranking(metrics.reparto_por_lugar(df, lugares)), width="stretch")

# --- Picos: perfil diario y hora x día ---
p1, p2 = st.columns([2, 3])
p1.altair_chart(charts.perfil_diario(metrics.perfil_diario(df)), width="stretch")
p2.altair_chart(charts.mapa_calor(metrics.matriz_hora_dia(df)), width="stretch")

# --- Tabla de detalle ---
st.subheader("Detalle por periodo")
tabla = serie.rename({"periodo": agregacion, "energia_kwh_sel": "Energía (kWh)",
                      "minutos_sin_medicion": "Minutos sin medición"})
st.dataframe(tabla, width="stretch", hide_index=True,
             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tabla.columns
                            if tabla[c].dtype == pl.Float64})
st.download_button("Descargar tabla (CSV)", tabla.write_csv(), file_name=f"consumo_{agregacion.lower()}.csv")

# --- Sección plegable: distribución y correlaciones ---
with st.expander("Ver distribución y correlaciones entre variables"):
    d1, d2 = st.columns(2)
    d1.altair_chart(charts.distribucion(df), width="stretch")
    d2.altair_chart(charts.correlacion(data_loader.correlacion(minutos, inicio, fin)), width="stretch")
