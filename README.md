# Machine Learning for Post-Fire Vegetation Classification in Aragón

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17297183.svg)](https://doi.org/10.5281/zenodo.17297183) -->

The presented framework integrates **Landsat multispectral imagery**, **topographic and edaphic variables**, and **machine-learning classifiers** (Random Forest and Support Vector Machines) to classify forest vegetation and its associated burned-shrub states across Aragón, Spain.

All pipelines are fully reproducible and have been published on [Zenodo](https://doi.org/10.5281/zenodo.17123378).

## Features

* Preprocessing of Landsat and ancillary data
* Construction of training datasets from manual, inventory-based, and automatic sources
* Covariate selection using correlation, VIF, and PCA approaches
* Model training with Random Forest and Support Vector Machine
* Resampling strategies for class imbalance (Random Undersampling, Tomek Links, SMOTE, SMOTENC)
* Model evaluation with balanced accuracy, Cohen's kappa, producer's and user's accuracy
* Variable importance analysis with SHAP values
* Assessment of spatial autocorrelation using Moran'sI

## Landsat images

Landsat images were processed to create a harmonized series between the Landsat and Sentinel-2 constellations.

> [!NOTE]
> They have not been included in this repository due to space constraints.

A whole utils module was created in the shared code to handle them ([tile.py](Python/utils/tile.py)). These images were produced using the methods described in:

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

> Alves, D., Fidelis, A., Pérez-Cabello, F., Alvarado, S.T., Estevão Conciani, D., Contursi Cambraia, B., Laffayete Pires da Silveira, A., Freire, T.S., 2022. Impact of image acquisition lag-time on monitoring short-term postfire spectral dynamics in tropical savannas: the Campos Amazônicos Fire Experiment. J. Appl. Remote Sens. 16, 1–51. [https://doi.org/10.1117/1.jrs.16.034507](https://doi.org/10.1117/1.jrs.16.034507)

## IFN Data

The dataset contains data from the Spanish National Forest Inventory (IFN by its Spanish name) 2, 3, and 4 from the study area in Aragón. The data was extracted using the *utils* module [`ifn.py`](Python/utils/ifn.py).

> [!IMPORTANT]
> The provided dataset does not include geographic coordinates because the precise locations of the National Forest Inventory data are protected

## Workflow

1. `Python/01_download_dem.py`
   Download the elevation data for the study area and create the related predictor variables.

2. `Python/02_download_siose.py`
   Save the SIOSE (Spanish Land Cover data) information for the study area.

3. `Python/03_extract_soil.py` and `Python/03_filter_soil.py`
   Automatically extract labels from the sparse vegetation class.

4. `Python/04_create_dataset.py`
   Combine data from manual digitization, sparse vegetation, and IFN into a single file, and add the predictor variables.

5. `Notebooks/inspect_predictors.ipynb`
   Perform data analysis to remove outliers and select the best predictor sets.

   > [!IMPORTANT]
   > The rules and filters for removing invalid data from the dataset are included int the *utils* module [`model.py`](Python/utils/model.py), which is used to perform the training phase.

6. `Python/05_train_models.py`
   Train each model pipeline and save the statistics.

7. `Notebooks/inspect_models.ipynb`
   Review the statistics from the model training phase.

8. `Python/06_spatial_autocorr.py`
   Compute Moran's I to assess spatial autocorrelation in the dataset.

## Install environments

With conda, create the `classification` environment:

```bash
conda env create --file=requirements.yml
```

## Training results

The log files containing the results of the training phase for all defined pipelines are stored in the [log folder](results/logs/v0.1). Each subfolder contains the data used in the [`inspect_models.ipynb`](Notebooks/inspect_models.ipynb) notebook to select the best model pipelines.
