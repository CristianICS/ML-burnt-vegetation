# Aprendizaje Automático para la Clasificación de la Vegetación Post-Incendio en Aragón

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17297183.svg)](https://doi.org/10.5281/zenodo.17297183)

El marco presentado integra **imágenes multiespectrales Landsat**, **variables topográficas y edáficas**, y **clasificadores de aprendizaje automático** (Random Forest y Support Vector Machines) para clasificar la vegetación forestal y sus estados asociados de matorral quemado en Aragón, España.

Todos los flujos de trabajo son totalmente reproducibles y han sido publicados en [Zenodo](https://doi.org/10.5281/zenodo.17123378).

## Características

* Preprocesamiento de datos Landsat y datos auxiliares
* Construcción de conjuntos de entrenamiento a partir de fuentes manuales, basadas en inventarios y automáticas
* Selección de covariables mediante enfoques de correlación, VIF y PCA
* Entrenamiento de modelos con Random Forest y Support Vector Machine
* Estrategias de remuestreo para el desbalanceo de clases (submuestreo aleatorio, enlaces de Tomek, SMOTE, SMOTENC)
* Evaluación de modelos con diferentes métricas de error (e.g. Kappa)
* Análisis de la importancia de variables con valores SHAP
* Evaluación de la autocorrelación espacial mediante el índice de Moran (Moran's I)

## Imágenes Landsat

Las imágenes Landsat se procesaron para crear una serie armonizada entre las constelaciones Landsat y Sentinel-2. Existen imágenes desde 1984 hasta 2023. El procedimiento de cálculo es costoso, por lo que solo se disponen de datos para las zonas quemadas de Aragón.

> [!NOTE]
> No se han incluido en este repositorio debido a limitaciones de espacio.

Se creó un módulo completo de utilidades en el código compartido para gestionarlas ([tile.py](Python/utils/tile.py)). Estas imágenes se produjeron utilizando los métodos descritos en:

```bibtex
@article{alvesImpactImageAcquisition2022,
  title = {Impact of Image Acquisition Lag-Time on Monitoring Short-Term Postfire Spectral Dynamics in Tropical Savannas: The {{Campos Amaz{\^o}nicos Fire Experiment}}},
  author = {Alves, Daniel Borini and Fidelis, Alessandra and {P{\'e}rez-Cabello}, Fernando and Alvarado, Swanni T. and Conciani, Dhemerson Estev{\~a}o and Cambraia, Bruno Contursi and Silveira, Ant{\^o}nio Laffayete Pires Da and Silva, Thiago Sanna Freire},
  year = {2022},
  journal = {Journal of Applied Remote Sensing},
  volume = {16},
  number = {03},
  doi = {10.1117/1.JRS.16.034507}
}
```

> Alves, D., Fidelis, A., Pérez-Cabello, F., Alvarado, S.T., Estevão Conciani, D., Contursi Cambraia, B., Laffayete Pires da Silveira, A., Freire, T.S., 2022. Impacto del desfase temporal en la adquisición de imágenes en el monitoreo de la dinámica espectral post-incendio a corto plazo en sabanas tropicales: el Experimento de Incendios de Campos Amazônicos. J. Appl. Remote Sens. 16, 1–51. [https://doi.org/10.1117/1.jrs.16.034507](https://doi.org/10.1117/1.jrs.16.034507)

## Conjunto de etiquetas

Se ha obtenido en distintos años de la serie temporal de imágenes Landsat. Por este motivo, algunas etiquetas mantienen su posición a lo largo de varios años. Sin embargo, si han sido observadas en varios años tendrán dos valores, uno por cada año, obteniendo variables predictoras procedentes de las imágenes de la serie harmonizada en las fechas disponibles.

### Vegetación rala

Todos los conjuntos de etiquetas utilizados contienen datos de zonas con un porcentaje elevado de suelo desnudo mezclado con vegetación rala o dispersa. Se utilizan los códigos de python `Python/03_extract_soil.py` y `Python/03_filter_soil.py`, habiendo obtenido previamente los datos necesarios del SIOSE y del MDE (códigos [01_download_dem.py](Python/01_download_dem.py) y [02_download_siose.py](02_download_siose.py) respectivamente)

> [!NOTE]
> Se necesita una cuenta de [Google Earth Engine](https://earthengine.google.com/) para poder obtener la imagen de iluminación, necesaria para filtrar los puntos de suelo en función de las sombras.

Procedimiento:

1. Seleccionar los años del SIOSE (2005, 2009, 2011 y 2014)
2. Crear un compuesto con imágenes Landsat entre el 1 de junio y el 31 de julio en cada año con información del SIOSE disponible.
3. Seleccionar los píxeles con un valor de NDVI comprendido entre 0.08 y 0.15. Estos serán los píxeles candidatos que podrían ser vegetación rala.
4. Filtrar los puntos por distancia, eliminando aquellos que tienen vecinos a menos de 200 metros.
5. Obtener una imagen con el coseno del ángulo de incidencia local (*IL* o imágen de iluminación) utilizando los datos de Azimuth y Elevación solar promedio de las escenas utilizadas en el compuesto del NDVI.
6. Seleccionar de entre los píxeles candidatos aquellos con un valor IL superior a 0,7.
7. Abrir la escena del SIOSE correspondiente, descargada del servicio WMS de lDEE. Se seleccionan aquellos píxeles candidatos sobre categorías de ocupación del suelo "roquedo" y "suelo desnudo".

### Datos IFN

El conjunto de datos contiene información del Inventario Forestal Nacional Español (IFN) 2, 3 y 4 dentro del área de estudio en Aragón. Los datos se extrajeron utilizando el módulo de *utils* [`ifn.py`](Python/utils/ifn.py).

> [!IMPORTANT]
> El conjunto de datos proporcionado no incluye coordenadas geográficas porque las ubicaciones precisas de los datos del Inventario Forestal Nacional están protegidas.

## Variables predictoras adicionales

La clasificación regional de la vegetación a una escala espacial de 30 metros (píxeles Landsat) presenta problemas complejos. Las firmas de la vegetación se mezclan en píxeles tan grandes, por lo que es más difícil diferenciar entre tipos similares de vegetación. Es por ello que se han utilizado variables con información adicional con el objetivo de complementar la información espectral.

Se dividen en dos tipos. En primer lugar, las obtenidas a partir del Modelo Digital de Elevaciones obtenido de los servicios WCS del PNOA. Se selecciona el ráster de 25 metros de resolución espacial en EPSG:25830. La extracción se realiza utilizando las zonas que cuentan con imágenes Landsat disponibles.

De las alturas se derivan:

* [Pendientes](https://gdal.org/en/stable/programs/gdal_raster_slope.html), en grados.
* [Orientaciones](https://gdal.org/en/stable/programs/gdal_raster_aspect.html), entre 0 y 360 grados. Representan el acimut (*azimuth*) hacia el cual se orientan las pendientes. El valor será `nodata` si la pendiente es 0.
  * 0: Norte
  * 90: Este
  * 180: Sur
  * 270: Oeste
* Sombras ([*Hillshade*](https://gdal.org/en/stable/programs/gdal_raster_hillshade.html)), utilizando un acimut de 180 grados (sur) y una elevación de 45 grados (sol de mediodía).

Además, se ha obtenido la variable ácido/básico, capa de información binaria derivada del mapa geológico de Aragón (1993). La división entre rocas básicas y ácidas se realiza con el objetivo de mejorar la clasificación de la especie *Pinus pinaster*, con preferencia por sustratos básicos. A los códigos seleccionados como rocas básicas se les asigna un 1, y a los clasificados como ácidos un 2.

## Flujo de trabajo

1. `Python/01_download_dem.py`
   Descarga los datos de elevación del área de estudio y crea las variables predictoras relacionadas.

2. `Python/02_download_siose.py`
   Guarda la información SIOSE (datos de cobertura del suelo de España) sobre el área de estudio.

3. `Python/03_extract_soil.py` y `Python/03_filter_soil.py`
   Extraen automáticamente etiquetas de la clase de vegetación dispersa/suelo desnudo.

4. `Python/04_create_dataset.py`
   Combina datos de digitalización manual, vegetación dispersa/suelo desnudo e IFN en un único archivo y añade las variables predictoras.

5. `Notebooks/inspect_predictors.ipynb`
   Realiza análisis de datos para eliminar valores atípicos y seleccionar los mejores conjuntos de predictores.

   > [!IMPORTANT]
   > Las reglas y filtros para eliminar datos inválidos del conjunto de datos están incluidos en el módulo de *utils* [`model.py`](Python/utils/model.py), que se utiliza para llevar a cabo la fase de entrenamiento.

6. `Python/05_train_models.py`
   Entrena todos los modelos por cada combinación de variables y preprocesos definidos, guardando las estadísticas.

7. `Notebooks/inspect_models.ipynb`
   Revisa las estadísticas de la fase de entrenamiento de los modelos.

8. `Python/06_spatial_autocorr.py`
   Calcula el índice de Moran (Moran's I) para evaluar la autocorrelación espacial en el conjunto de datos.

## Instalación de entornos

Con conda, crea el entorno `classification`:

```bash
conda env create --file=requirements.yml
```

## Resultados del entrenamiento

Los archivos de registro que contienen los resultados de la fase de entrenamiento para todas las secuencias de entrenamiento se almacenan en la [carpeta de logs](results/logs/v0.1). Cada subcarpeta contiene los datos utilizados en el cuaderno [`inspect_models.ipynb`](Notebooks/inspect_models.ipynb) para seleccionar los mejores modelos.
