"""Creación de gráficas del dashboard (Altair: tooltips al pasar el ratón).

Paleta azul y verde validada para daltonismo (separación entre colores y
luminosidad). Cada lugar tiene siempre el mismo color; el gris es contexto.
Los tonos claros tienen poco contraste sobre blanco, por eso las barras llevan
etiqueta directa con el valor.

Cada función recibe `titulo`: por defecto describe la gráfica; el título final
(la conclusión, según storytelling) se escribe en `app.py` al analizar los datos.
"""

import altair as alt
import polars as pl

from src.dashboard.metrics import DIAS

# La tabla por hora tiene ~35 mil filas; Altair limita a 5 mil por defecto.
alt.data_transformers.disable_max_rows()

AZUL = "#1f5fa8"
VERDE = "#00843d"
GRIS = "#9a9a96"
TEXTO = "#52514e"

COLORES_LUGAR = {
    "Cocina": "#1f5fa8",
    "Cuarto de lavado": "#3fa9dc",
    "Calentador y AC": "#00843d",
    "Resto de la casa": "#8cc63f",
}


# Fechas en español (por defecto Vega escribe los meses en inglés).
LOCALE_ES = {
    "time": {
        "dateTime": "%A, %e de %B de %Y, %X", "date": "%d/%m/%Y", "time": "%H:%M:%S", "periods": ["AM", "PM"],
        "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
        "shortDays": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
        "months": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                   "octubre", "noviembre", "diciembre"],
        "shortMonths": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
    },
    "number": {"decimal": ".", "thousands": ",", "grouping": [3], "currency": ["$", ""]},
}


def _estilo(chart: alt.Chart, titulo: str, subtitulo: str = "") -> alt.Chart:
    """Sin bordes ni cuadrícula pesada (sin chartjunk); título alineado a la izquierda."""
    return (
        chart.properties(title=alt.TitleParams(titulo, subtitle=subtitulo, anchor="start", color=TEXTO,
                                               subtitleColor=GRIS, fontSize=16, subtitleFontSize=12))
        .configure(locale=LOCALE_ES)
        .configure_view(stroke=None)
        .configure_axis(gridColor="#e1e0d9", domainColor="#c3c2b7", labelColor=TEXTO, titleColor=TEXTO)
        .configure_legend(labelColor=TEXTO, titleColor=TEXTO)
    )


def tendencia(serie: pl.DataFrame, agregacion: str,
              titulo: str = "Energía consumida en el tiempo") -> alt.Chart:
    """Línea de energía (kWh) por periodo. Los periodos con apagón se marcan con
    un punto gris (su energía está subestimada)."""
    base = alt.Chart(serie).encode(
        x=alt.X("periodo:T", title=agregacion),
        y=alt.Y("energia_kwh_sel:Q", title=f"Energía por {agregacion.lower()} (kWh)"),
        tooltip=[alt.Tooltip("periodo:T", title=agregacion),
                 alt.Tooltip("energia_kwh_sel:Q", title="Energía (kWh)", format=",.1f"),
                 alt.Tooltip("minutos_sin_medicion:Q", title="Minutos sin medición")],
    )
    linea = base.mark_line(color=AZUL, strokeWidth=2)
    apagones = base.transform_filter("datum.minutos_sin_medicion > 0").mark_point(color=GRIS, filled=True, size=40)
    return _estilo(alt.layer(linea, apagones), titulo, "Puntos grises: periodos con minutos sin medición")


def ranking(reparto: pl.DataFrame, titulo: str = "Energía por lugar de la casa") -> alt.Chart:
    """Barras horizontales de mayor a menor, con el % escrito al final de cada barra."""
    base = alt.Chart(reparto).encode(
        # Orden explícito: `reparto` ya viene de mayor a menor (sort="-x" se pierde al combinar capas).
        y=alt.Y("lugar:N", sort=reparto["lugar"].to_list(), title=None),
        x=alt.X("energia_kwh:Q", title="Energía (kWh)"),
        tooltip=[alt.Tooltip("lugar:N", title="Lugar"),
                 alt.Tooltip("energia_kwh:Q", title="Energía (kWh)", format=",.0f"),
                 alt.Tooltip("pct:Q", title="% del total", format=".1f")],
    )
    barras = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("lugar:N", scale=alt.Scale(domain=list(COLORES_LUGAR), range=list(COLORES_LUGAR.values())),
                        legend=None)
    )
    etiquetas = (base.transform_calculate(etiqueta="format(datum.pct, '.1f') + ' %'")
                 .mark_text(align="left", dx=4, color=TEXTO).encode(text="etiqueta:N"))
    return _estilo(alt.layer(barras, etiquetas), titulo, "Porcentaje de la energía de los lugares elegidos")


def mapa_calor(matriz: pl.DataFrame, titulo: str = "Potencia media por hora y día de la semana") -> alt.Chart:
    """Celdas día x hora; más oscuro = más potencia (una sola tonalidad, de claro a oscuro)."""
    datos = matriz.with_columns(pl.col("dia_semana").map_elements(lambda d: DIAS[d - 1], return_dtype=pl.String)
                                .alias("dia"))
    chart = alt.Chart(datos).mark_rect().encode(
        x=alt.X("hora_del_dia:O", title="Hora del día"),
        y=alt.Y("dia:N", sort=DIAS, title=None),
        color=alt.Color("potencia_kw:Q", title="kW", scale=alt.Scale(range=["#e8f4ec", VERDE, "#00502a"])),
        tooltip=[alt.Tooltip("dia:N", title="Día"), alt.Tooltip("hora_del_dia:O", title="Hora"),
                 alt.Tooltip("potencia_kw:Q", title="Potencia media (kW)", format=".2f")],
    )
    return _estilo(chart, titulo, "Promedio de todas las semanas del rango elegido")


COLORES_TIPO_DIA = {"Entre semana": AZUL, "Fin de semana": VERDE}


def perfil_diario(perfil: pl.DataFrame, titulo: str = "Perfil diario de la potencia media") -> alt.Chart:
    """Una línea por tipo de día (0 a 23 h), con leyenda y el nombre escrito al
    final de cada línea."""
    color = alt.Color("tipo_dia:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=list(COLORES_TIPO_DIA), range=list(COLORES_TIPO_DIA.values())))
    base = alt.Chart(perfil).encode(
        x=alt.X("hora_del_dia:Q", title="Hora del día", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickCount=12)),
        y=alt.Y("potencia_kw:Q", title="Potencia media (kW)"),
        color=color,
        tooltip=[alt.Tooltip("tipo_dia:N", title="Tipo de día"), alt.Tooltip("hora_del_dia:Q", title="Hora"),
                 alt.Tooltip("potencia_kw:Q", title="Potencia media (kW)", format=".2f")],
    )
    lineas = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
    nombres = base.transform_filter("datum.hora_del_dia == 23").mark_text(align="left", dx=6).encode(text="tipo_dia:N")
    return _estilo(alt.layer(lineas, nombres), titulo, "Lunes a viernes contra sábado y domingo, en el rango elegido")


def correlacion(pares: pl.DataFrame, titulo: str = "Correlación de Pearson entre las variables medidas") -> alt.Chart:
    """Matriz de Pearson: azul negativa, gris ~0, rojo positiva (mismos colores
    que graphs/correlacion_heatmap.png). Valor escrito en cada celda."""
    orden = pares["variable_1"].unique(maintain_order=True).to_list()
    base = alt.Chart(pares).encode(
        x=alt.X("variable_2:N", sort=orden, title=None, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("variable_1:N", sort=orden, title=None),
        tooltip=[alt.Tooltip("variable_1:N", title="Variable"), alt.Tooltip("variable_2:N", title="Variable"),
                 alt.Tooltip("r:Q", title="r", format=".3f")],
    )
    celdas = base.mark_rect().encode(
        color=alt.Color("r:Q", title="r", scale=alt.Scale(domain=[-1, 0, 1], range=["#2a78d6", "#f0efec", "#e34948"]))
    )
    valores = base.mark_text(color=TEXTO, fontSize=11).encode(text=alt.Text("r:Q", format=".2f"))
    return _estilo(alt.layer(celdas, valores), titulo, "Minutos con medición del rango elegido; |r| > 0.8 = redundante")


def distribucion(por_hora: pl.DataFrame, titulo: str = "Distribución de la potencia media por hora") -> alt.Chart:
    """Histograma de la potencia media de cada hora con medición completa."""
    datos = por_hora.filter(pl.col("minutos_sin_medicion") == 0).with_columns(
        (pl.col("energia_kwh_sel") / (pl.col("minutos_medidos") / 60)).alias("potencia_kw")
    ).select("potencia_kw")
    chart = alt.Chart(datos).mark_bar(color=AZUL).encode(
        x=alt.X("potencia_kw:Q", bin=alt.Bin(maxbins=60), title="Potencia media de la hora (kW)"),
        y=alt.Y("count():Q", title="Número de horas"),
        tooltip=[alt.Tooltip("count():Q", title="Horas")],
    )
    return _estilo(chart, titulo, "Solo horas sin minutos faltantes")
