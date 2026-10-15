# Práctica 5: Normalización, Reducción de Dimensionalidad y Dashboard

Curso de Almacenes y Minería de Datos — Facultad de Ciencias, UNAM.

## Descripción del proyecto

<!-- TODO -->

## Dataset

**Individual Household Electric Power Consumption** — UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>

- Tipo: multivariado, series de tiempo
- Área: Física y Química
- Tareas asociadas: regresión, agrupamiento

### Descarga

El archivo original supera los límites de GitHub, por lo que no se incluye en el repositorio.

1. Descargar el `.zip` desde la liga anterior.
2. Descomprimirlo y colocar `household_power_consumption.txt` en `data/data-raw/`.

## Estructura de carpetas

```
proyecto/
|-- data/
|   |-- data-raw/            # dato original, sin modificar
|   |-- data-processed/      # dato preparado
|-- src/
|   |-- cleaning/            # preprocesamiento y normalizacion
|   |-- reduction/           # correlacion de Pearson y PCA
|   |-- dashboard/           # modulos auxiliares del dashboard
|-- visualization/
|   |-- app.py               # codigo principal del dashboard
|-- README.md
|-- requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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
>>>>>>> c83014b (feat(): estructura inicial)
