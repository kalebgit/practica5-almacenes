# Práctica 5: Normalización, Reducción de Dimensionalidad y Dashboard

Curso de Almacenes y Minería de Datos — Facultad de Ciencias, UNAM.

## Descripción del proyecto

Preparación de una serie de tiempo de consumo eléctrico doméstico para el
entrenamiento posterior de modelos de regresión y agrupamiento: preprocesamiento
(casteo, deduplicación, faltantes), normalización numérica, reducción de
dimensionalidad (correlación de Pearson y PCA) y un dashboard en Streamlit para
explorar los datos y comunicar hallazgos.

## Dataset

**Individual Household Electric Power Consumption** — UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>

Las cifras de esta sección se reproducen con `python -m src.eda.inspeccion`; lo
que proviene de la ficha de UCI (origen, unidades, tareas) está en la liga anterior.

### Origen

Mediciones de un solo hogar en Sceaux, Francia (7 km de París), donadas a UCI
el 29 de agosto de 2012 por Georges Hébrail y Alice Bérard. Licencia CC BY 4.0.

### Características

| Característica | Valor |
|---|---|
| Tipo | Multivariado, series de tiempo, variables reales |
| Área temática | Física y Química |
| Tareas asociadas | Regresión, agrupamiento (*clustering*) |
| Registros | 2,075,259 (uno por minuto) |
| Columnas | 9: `Date`, `Time` y 7 variables numéricas |
| Archivo | `household_power_consumption.txt`, separado por `;`, 133.0 MB (126.8 MiB) |
| Faltantes | 25,979 filas (1.25 %), marcadas con `?` (UCI dice "no" en su tabla de variables, pero su nota reconoce ≈ 1.25 %) |

| Variable | Descripción | Unidad |
|---|---|---|
| `Date`, `Time` | Fecha (`dd/mm/yyyy`) y hora (`hh:mm:ss`) de la medición | — |
| `Global_active_power` | Potencia activa global, promedio por minuto | kW |
| `Global_reactive_power` | Potencia reactiva global, promedio por minuto | kW |
| `Voltage` | Voltaje, promedio por minuto | V |
| `Global_intensity` | Intensidad de corriente global, promedio por minuto | A |
| `Sub_metering_1` | Energía activa de la cocina (lavavajillas, horno, microondas) | Wh |
| `Sub_metering_2` | Energía activa del cuarto de lavado (lavadora, secadora, refrigerador, luz) | Wh |
| `Sub_metering_3` | Energía activa del calentador de agua y el aire acondicionado | Wh |

La energía del resto del hogar (no cubierta por los submedidores) se obtiene como
`global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3` (Wh).

### Naturaleza temporal

- Periodo: 2006-12-16 17:24 a 2010-11-26 21:02 (1,441 días ≈ 47.3 meses).
- Frecuencia fija de 1 minuto, sin huecos en la rejilla temporal: los faltantes
  son filas con `?`, no minutos ausentes.
- El orden de los registros importa: la partición, la normalización y el muestreo
  deben respetar la cronología.

### Descarga

El archivo original supera los límites de GitHub, por lo que no se incluye en el repositorio.

1. Descargar el `.zip` desde la liga anterior.
2. Descomprimirlo y colocar `household_power_consumption.txt` en `data/data-raw/`.

## Estructura de carpetas

```
proyecto/
|-- config/
|   |-- rutas.py             # Rutas centralizadas del proyecto
|-- data/
|   |-- data-raw/            # Datos originales
|   |-- data-processed/      # Datos ya limpios
|-- docs/
|   |-- decisiones.md        # Bitácora de decisiones
|-- graphs/                  # Scripts de gráficas
|-- notebooks/               # Notebooks exploratorios (no productivos)
|-- src/
|   |-- cleaning/            # Preprocesamiento y Normalización
|   |-- eda/                 # Análisis exploratorio de datos
|   |-- reduction/           # Correlación de Pearson y PCA
|   |-- dashboard/           # Módulos auxiliares del dashboard
|   |-- pipeline.py          # Ejecuta todo el procesamiento 
|-- visualization/
|   |-- app.py               # Código principal del dashboard
|-- README.md
|-- requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Procesamiento de los datos

Desde la raíz del proyecto, con el crudo ya en `data/data-raw/`:

```bash
python -m src.pipeline
```

Genera en `data/data-processed/` el dataset preparado y los parámetros del
escalador y del PCA, y en `graphs/` las gráficas.

## Ejecución del dashboard

```bash
streamlit run visualization/app.py
```

## Resumen de decisiones

### Preprocesamiento
<!-- TODO -->

### Normalización numérica
<!-- TODO -->

### Correlación de Pearson
<!-- TODO -->

### PCA
<!-- TODO -->
