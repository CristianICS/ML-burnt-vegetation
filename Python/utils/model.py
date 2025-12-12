from collections import Counter
from typing import Union, List, Dict, Any
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.preprocessing import FunctionTransformer  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

from imblearn.under_sampling import RandomUnderSampler  # type: ignore
from imblearn.under_sampling import TomekLinks  # type: ignore
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC  # type: ignore
from imblearn.pipeline import Pipeline as imbPipe  # type: ignore

from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import balanced_accuracy_score  # type: ignore
from sklearn.metrics import cohen_kappa_score  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore

from matplotlib import pyplot as plt  # type: ignore

import re
import pickle

import geopandas as gpd
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
# import warnings

pd.options.mode.copy_on_write = True

class Dataset:
    """
    Prepare and transform the labeled dataset prior to model training.
    """

    def __init__(self, labels_dataset_path, label_codes_path, version, mtype):
        """
        Preprocess the label data to drive the training phase.
        
        1) Read labels + join code table (drop problematic ESPE_rc col).
        2) Filter to a dataset "version" (rule-based query).
        3) Reduce majority class with bg_undersampling().
        4) Fix nodata (negatives -> 0) in seasonal and global predictors.
        5) Compute NDVI per season and drop outliers.
        6) Create PCA components for seasonal and global groups.

        :labels_dataset_path: Location of labels with predictor vars.
        :label_codes_path: Location of the csv with the label codes.
        :verion: Number of dataset version to use.
        :mtype: Model type to use (spring_summer | summer_long)
        """
        # Version-based filters for selecting samples
        dat_queries = {1: "(source == 'Digitized') | (source == 'SIOSE')"}
        dat_queries[2] = f"(FCC > 20) & (Ocu1 > 4) | ({dat_queries[1]})"
        dat_queries[3] = f"(FCC > 20) & (Ocu1 > 6) | ({dat_queries[1]})"

        # Store the desired model type.
        av_types = ["summer_long", "spring_summer"]
        if mtype in av_types:
            self.model_type = mtype
        else: 
            raise ValueError(f"Provided 'mtype' is not in {av_types}")

        if self.model_type == "spring_summer":
            self.define_predictor_groups_spring_summer()
        elif self.model_type == "summer_long":
            self.define_predictor_groups_summer_long()

        # Load labeled points and the code table (avoid ESPE_rc issues)
        dataset = gpd.read_file(labels_dataset_path, ignore_geometry=True)
        label_codes = pd.read_csv(label_codes_path)
        # Try to convert stringified numeric columns back to numeric
        for col in dataset.select_dtypes(include="object").columns:
            try:
                dataset[col] = pd.to_numeric(dataset[col])
            except:
                pass

        # Keep 'code_ifn' out (to avoid dropping valid data via dropna)
        label_codes = label_codes.loc[:, label_codes.columns != 'code_ifn']
        dataset = dataset.join(label_codes.set_index("code_v1"), on="ESPE")

        # Remove specific classes (Pinus uncinata groups)
        cls_query = "(code_v1_reclass != 5) & (code_v1_reclass != 15)"
        dataset.query(cls_query, inplace=True)

        # Apply version-specific filtering
        dataset.query(dat_queries[version], inplace=True)

        # Reduce very frequent class (bare ground)
        dataset = self.bg_undersampling(dataset)
        # Test the new digitized soil labels.
        # dataset.query("source != 'SIOSE'", inplace=True)

        # Identify predictor columns: seasonal vars and global covariates
        global_pred = ["dem", "shadow", "slope", "acibasi"]
        if self.model_type == "spring_summer":
            pred_vars = (
                dataset.columns.str.endswith("summer")
                | dataset.columns.str.endswith("spring")
                | dataset.columns.isin(global_pred)
            )
        elif self.model_type == "summer_long":
            pred_vars = (
                dataset.columns.str.endswith("summerlong")
                | dataset.columns.isin(global_pred)
            )

        # Replace negative nodata-like values with 0 in predictors
        dataset.loc[:, pred_vars] = dataset.loc[:, pred_vars].where(
            dataset.loc[:, pred_vars] >= 0, 0)

        # Remove rows that still contain zeros (nodata) across any predictor
        dat_i = dataset[(dataset.loc[:, pred_vars] != 0).all(axis=1)]

        # Create NDVI by season
        if self.model_type == "spring_summer":
            dat_i.loc[:, "NDVI_summer"] = self.ndvi(dat_i, "summer")
            dat_i.loc[:, "NDVI_spring"] = self.ndvi(dat_i, "spring")
        elif self.model_type == "summer_long":
            dat_i.loc[:, "NDVI_summerlong"] = self.ndvi(dat_i, "summerlong")

        # Drop obvious outliers (e.g., cloud/water contamination)
        self.matrix = self.drop_outliers(dat_i)

        # Create PCA features by group
        if self.model_type == "spring_summer":
            spring_pcas = self.pca_reduction("spring")
            summer_pcas = self.pca_reduction("summer")
            global_pcas = self.pca_reduction(
                "global",
                ["shadow", "dem", "slope"]
            )
        elif self.model_type == "summer_long":
            summerlong_pcas = self.pca_reduction("summerlong")
            global_pcas = self.pca_reduction(
                "global",
                ["shadow", "dem", "slope"]
            )

        # Merge PCA features back into the working matrix
        if self.model_type == "spring_summer":
            pcas_df = pd.concat(
                [spring_pcas, summer_pcas, global_pcas, self.matrix],
                axis=1
            )
        elif self.model_type == "summer_long":
            pcas_df = pd.concat(
                [summerlong_pcas, global_pcas, self.matrix],
                axis=1
            )
        self.matrix = pcas_df

    @staticmethod
    def bg_undersampling(df):
        """
        Reduce the dominant class (bare ground) to mitigate class imbalance.

        Notes
        -----
        - Finds the most frequent class (bare soil) in 'code_v1_reclass'
        and limits it to 800 samples using RandomUnderSampler.
        - Returns a clean DataFrame where X and y are concatenated again with
        aligned indices (required by downstream operations).
        """
        X = df.drop(columns=['code_v1_reclass'])
        y = df['code_v1_reclass']

        # Identify the majority class (the one to be reduced)
        max_clss = y.value_counts().idxmax()

        # Undersample only the majority class to 800 samples
        # rus = RandomUnderSampler(
        #     sampling_strategy={max_clss: 800},
        #     random_state=42
        # )
        # Undersample only the majority class to 100 samples
        rus = RandomUnderSampler(
            sampling_strategy={max_clss: 100},
            random_state=42
        )

        X_resampled, y_resampled = rus.fit_resample(X, y)

        # Rebuild a single DataFrame (indices are reset to keep them aligned)
        dataset = pd.concat(
            [X_resampled.reset_index(drop=True),
            y_resampled.reset_index(drop=True)],
            axis=1
        )
        return dataset

    def drop_outliers(self, df):
        """
        Remove obvious outliers likely caused by water, snow, clouds, or 
        sensor issues.
        """
        target = df.copy()

        if self.model_type == "summer_long":
            # Exclude negative NDVI cases that are unlikely for vegetation
            # (e.g., water/snow/clouds)
            target.query("NDVI_summerlong > 0", inplace=True)
            # Discard coastal outlayers
            target.query("coastal_summerlong < 4000", inplace=True)

        elif self.model_type == "spring_summer":
            # Exclude negative NDVI cases that are unlikely for vegetation
            # (e.g., water/snow/clouds)
            target.query(
                "(NDVI_summer > 0) and (NDVI_spring > 0)", inplace=True)
            # Remove extreme SWIR1 cases
            # (often cloud contamination or L7 striping)
            target.query("swir1_spring < 9000", inplace=True)
        
        return target

    def ndvi(self, df, suffix):
        """
        Compute NDVI for the given seasonal suffix ('spring' or 'summer').
        NDVI = (NIR - RED) / (NIR + RED)
        """
        numerator = df[f"nir_{suffix}"] - df[f"red_{suffix}"]
        denominator = df[f"nir_{suffix}"] + df[f"red_{suffix}"]
        return numerator / denominator

    def pca_reduction(self, suffix, columns=None):
        """
        Standardize features and compute the first 3 PCs for a group.

        Parameters
        ----------
        suffix : str
            If columns is None, selects columns that end with this suffix.
            Use 'global' and pass explicit 'columns' to use global covariates.
        columns : list[str] | None
            Specific columns to use; if None, auto-select by suffix.
        """
        if columns is None:
            columns_bol = self.matrix.columns.str.endswith(suffix)
            columns = self.matrix.columns[columns_bol]

        # Standardize (mean=0, var=1) then apply PCA
        X = StandardScaler().fit_transform(self.matrix[columns])

        # Fit PCA and transform
        pca = PCA(n_components=3, random_state=42)
        pca_np = pca.fit_transform(X)

        # Name components consistently per group
        pca_cols = [f"PC{i}_{suffix}" for i in range(1, pca.n_components_ + 1)]

        return pd.DataFrame(pca_np, columns=pca_cols, index=self.matrix.index)

    def define_predictor_groups_summer_long(self):
        """
        Define predictor set dictionaries used to build experiments.

        Valid for summer_long model type.

        Conventions
        -----------
        - Keys end with a version code ('_1', '_2', '_3').
        - 'N': numeric predictors; 'C': categorical predictors.
        - PCA groups list the PC columns to use.
        """
        pred_dict = {}

        # Hand-crafted sets
        pred_dict["manual_ndvi_1"] = {
            "N": ["NDVI_summerlong", "shadow", "dem"],
            "C": ["acibasi"]
        }
        pred_dict["manual_1"] = {
            "N": ["nir_summerlong", "swir1_summerlong", "shadow", "dem"],
            "C": ["acibasi"]
        }
        pred_dict["manual_pca_1"] = {
            "N": ["PC2_summerlong", "PC1_summerlong", "PC1_global"]
        }

        # Reuse same definition for versions 2 and 3
        pred_dict["manual_ndvi_2"] = pred_dict["manual_ndvi_1"]
        pred_dict["manual_2"] = pred_dict["manual_1"]
        pred_dict["manual_pca_2"] = pred_dict["manual_pca_1"]

        pred_dict["manual_ndvi_3"] = pred_dict["manual_ndvi_1"]
        pred_dict["manual_3"] = pred_dict["manual_1"]
        pred_dict["manual_pca_3"] = pred_dict["manual_pca_1"]

        # VIF-curated PCA sets
        pred_dict["vif_pca_1"] = {
            "N": [
                "PC1_summerlong", "PC2_summerlong", "PC3_summerlong",
                "PC1_global", "PC2_global", "PC3_global"
            ]
        }
        pred_dict["vif_pca_2"] = pred_dict["vif_pca_1"]
        pred_dict["vif_pca_3"] = pred_dict["vif_pca_1"]

        # VIF-selected raw + categorical variables
        pred_dict["vif_1"] = {
            "N": [
                "nir_summerlong", "coastal_summerlong", "dem",
                "shadow", "slope"
            ],
            "C": ["acibasi"]
        }
        pred_dict["vif_2"] = pred_dict["vif_1"]
        pred_dict["vif_3"] = {
            "N": ["nir_summerlong", "dem", "shadow", "slope"],
            "C": ["acibasi"]
        }

        # VIF global + PCA combined
        pred_dict["vif_global_1"] = {
            "N": [
                "coastal_summerlong", "dem", "shadow", "slope",
                "PC2_summerlong", "PC3_summerlong"
            ],
            "C": ["acibasi"],
        }
        pred_dict["vif_global_2"] = pred_dict["vif_global_1"]
        pred_dict["vif_global_2"] = pred_dict["vif_global_1"]

        self.predictor_sets = pred_dict

    def define_predictor_groups_spring_summer(self):
        """
        Define predictor set dictionaries used to build experiments.

        Valid for spring_summer model type only.

        Conventions
        -----------
        - Keys end with a version code ('_1', '_2', '_3').
        - 'N': numeric predictors; 'C': categorical predictors.
        - PCA groups list the PC columns to use.
        """
        pred_dict = {}

        # Hand-crafted sets
        pred_dict["manual_ndvi_1"] = {
            "N": ["NDVI_summer", "shadow", "dem"],
            "C": ["acibasi"]
        }
        
        pred_dict["manual_1"] = {
            "N": ["nir_summer", "swir1_spring", "shadow", "dem"],
            "C": ["acibasi"]
        }

        pred_dict["manual_pca_1"] = {
            "N": ["PC2_spring", "PC1_summer", "PC1_global"]
        }

        # Reuse same definition for versions 2 and 3
        pred_dict["manual_ndvi_2"] = pred_dict["manual_ndvi_1"]
        pred_dict["manual_2"] = pred_dict["manual_1"]
        pred_dict["manual_pca_2"] = pred_dict["manual_pca_1"]

        pred_dict["manual_ndvi_3"] = pred_dict["manual_ndvi_1"]
        pred_dict["manual_3"] = pred_dict["manual_1"]
        pred_dict["manual_pca_3"] = pred_dict["manual_pca_1"]

        # VIF-curated PCA sets
        pred_dict["vif_pca_1"] = {
            "N": [
                "PC1_spring", "PC2_spring", "PC3_spring",
                "PC2_summer", "PC3_summer",
                "PC1_global", "PC2_global", "PC3_global"
            ]
        }
        pred_dict["vif_pca_2"] = pred_dict["vif_pca_1"]
        pred_dict["vif_pca_3"] = pred_dict["vif_pca_1"]

        # VIF-selected raw + categorical variables
        pred_dict["vif_1"] = {
            "N": ["nir_summer", "coastal_spring", "dem", "shadow", "slope"], 
            "C": ["acibasi"]
        }
        pred_dict["vif_2"] = {
            "N": [
                "coastal_summer", "nir_summer", "nir_spring", 
                "coastal_spring", "dem", "shadow", "slope"],
            "C": ["acibasi"],
        }
        pred_dict["vif_3"] = pred_dict["vif_2"]

        # VIF global + PCA combined
        pred_dict["vif_global_3"] = {
            "N": [
                "coastal_spring", "dem", "shadow", "slope", "PC2_spring", 
                "PC3_spring", "PC2_summer", "PC3_summer"],
            "C": ["acibasi"],
        }
        pred_dict["vif_global_1"] = pred_dict["vif_global_3"]
        pred_dict["vif_global_2"] = {
            "N": [
                "coastal_summer", "coastal_spring", "dem", "shadow", "slope",
                "PC2_spring", "PC3_spring", "PC2_summer", "PC3_summer"
            ],
            "C": ["acibasi"],
        }

        self.predictor_sets = pred_dict

    def get_predictor_groups(self, version):
        """Return predictor set keys that belong to the requested version."""
        if not isinstance(version, str):
            version = str(version)
        return [k for k in self.predictor_sets.keys() if k.endswith(version)]

    def split(self, pred_id: str, label_col: str = "code_v1_reclass"):
        """
        Build train/test splits for the chosen predictor set.

        Notes
        -----
        - Stratified split preserves class proportions.
        - 70/30 train/test split with fixed seed for reproducibility.
        """
        pred_vars = self.predictor_sets[pred_id]

        # Flatten the predictor lists (both numeric and categorical groups)
        target_cols = [
            x for v in pred_vars.values()
            for x in v if isinstance(v, list)
        ]

        X = self.matrix[target_cols]
        y = self.matrix[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=327, test_size=0.3
        )
        return X_train, X_test, y_train, y_test


class Pipeline:
    """
    Construct an imbalanced-learn pipeline with optional under/oversampling.

    Designed to:
    - Handle both numeric and categorical predictors (SMOTENC if needed).
    - Allow different imbalance strategies (TomekLinks, random undersampling,
      SMOTE/ADASYN).
    - Integrate seamlessly with GridSearchCV.
    """

    def __init__(
        self,
        y: Union[pd.DataFrame, pd.Series],
        under_strategy: str = None,
        over_strategy: str = None,
        X: Union[pd.DataFrame, None] = None,
        categorical_predictors: list = None
    ):
        """
        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target labels (needed by some samplers, e.g., TomekLinks).
        under_strategy : {'random', 'tomeklinks', 'none', None}
            Undersampling method to apply.
        over_strategy : {'smote', 'adasyn', 'none', None}
            Oversampling method to apply.
        X : pd.DataFrame, optional
            Training features (required when using SMOTENC to locate
            categorical columns).
        categorical_predictors : list[str], optional
            Column names corresponding to categorical predictors (for SMOTENC).
        """
        self.y = y

        # Choose undersampling strategy
        if under_strategy == "tomeklinks":
            self.under = self.tomek()
        elif under_strategy == "random":
            self.under = self.random()
        else:
            self.under = self.none()
            # warnings.warn(...)

        # Choose oversampling strategy
        if over_strategy == "adasyn":
            self.over = self.adasyn()
        elif over_strategy == "smote":
            if categorical_predictors is None:
                # No categorical predictors -> SMOTE
                self.over = self.smote()
            else:
                # Map categorical column names to index positions
                # (SMOTENC requires indices)
                col_idxs = [
                    X.columns.get_loc(col)
                    for col in categorical_predictors
                ]
                self.over = self.smotenc(col_idxs)
        else:
            self.over = self.none()
            # warnings.warn(...)

        self.under_name = under_strategy
        self.over_name = over_strategy

    def add_model(self, clf):
        """
        Build the imbalanced-learn pipeline.
        (scaling -> over -> under -> model)

        Notes
        -----
        - Standardization is applied before resampling here so distances
          used by k-NN–based methods (SMOTE/ADASYN) are meaningful.
        """
        self.imb_pipe = imbPipe(
            steps=[
                ("scaler", StandardScaler()),
                ("over", self.over),
                ("under", self.under),
                ("clf", clf),
            ]
        )

    def grid_search(self, param_grid: dict):
        """
        Configure GridSearchCV for the pipeline.

        Notes
        -----
        - Uses StratifiedKFold(n_splits=2) to keep label proportions per fold.
        - n_jobs=1 avoids heavy memory pressure with nested parallelism +
          resampling inside folds.
        """
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=self.imb_pipe,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            error_score="raise",
            cv=cv,
            n_jobs=1,  # Intentional. Safer with resampling in the loop
            # verbose=3,
        )
        return grid_search, cv

    def none(self):
        """Identity transformer: no (under/over) sampling applied."""
        return FunctionTransformer(func=lambda x: x, validate=False)

    def random(self):
        """
        Random undersampling to limit large classes as defined in
        undersampling().
        """
        return RandomUnderSampler(
            sampling_strategy=self.undersampling, 
            random_state=42
        )

    def tomek(self):
        """
        TomekLinks undersampling.

        Important
        ---------
        - Only Tomek links are removed; if a majority class has no Tomek links,
          it will remain unchanged.
        - We target only classes exceeding 250 samples (see undersampling()).
        """
        # For cleaning methods, sampling_strategy expects the list of
        # classes to clean.
        cls_dict = self.undersampling(self.y)
        y_res_values = [cls for cls, val in cls_dict.items() if val == 250]
        return TomekLinks(sampling_strategy=y_res_values)

    def adasyn(self):
        """
        ADASYN oversampling.

        Caveats
        -------
        - Can fail when no neighbors belong to the majority class (division by 
          zero).
          For small classes, SMOTE is often more stable.
        """
        return ADASYN(
            sampling_strategy=self.augmentation_perc,
            n_neighbors=2,
            random_state=42
        )

    def smote(self):
        """SMOTE oversampling for fully numeric feature sets."""
        return SMOTE(
            sampling_strategy=self.augmentation,
            k_neighbors=2,
            random_state=42
        )

    def smotenc(self, cf):
        """SMOTENC oversampling for mixed numeric/categorical feature sets."""
        return SMOTENC(
            sampling_strategy=self.augmentation,
            k_neighbors=2,
            categorical_features=cf,
            random_state=42,
        )

    def undersampling(self, y):
        """
        Create an undersampling map for imbalanced-learn.

        Classes with more than 250 samples are reduced to 250;
        others are left as-is.
        """
        sampling_dict = {}
        cls_counter = Counter(y)
        for cls, count in cls_counter.items():
            if count > 250:
                sampling_dict[cls] = 250
        return sampling_dict

    def augmentation(self, y):
        """
        Create an oversampling map for SMOTE/SMOTENC.

        Classes with fewer than 150 samples are increased up to 150;
        larger classes keep their original size.
        """
        sampling_dict = {}
        cls_counter = Counter(y)
        for cls, count in cls_counter.items():
            sampling_dict[cls] = 150 if count < 150 else count
        return sampling_dict

    def augmentation_perc(self, y):
        """
        Create an oversampling map for ADASYN (percentage-based).

        Classes with fewer than 150 samples are increased by +30% (rounded);
        larger classes keep their original size.
        """
        sampling_dict = {}
        cls_counter = Counter(y)
        for cls, count in cls_counter.items():
            if count < 150:
                to_increase = int(round(count * 0.3, 0))
                sampling_dict[cls] = count + to_increase
            else:
                sampling_dict[cls] = count
        return sampling_dict


class Model:
    """
    Wrap model definitions, parameter grids, and evaluation helpers.
    """

    models = {
        # Random Forest
        "rf": {
            "clf": RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ),
            # Parameter grid for GridSearchCV
            "param_grid": {
                "clf__n_estimators": [500, 600],
                "clf__min_samples_leaf": [5, 10, 15, 20],
                "clf__min_impurity_decrease": [0.0, 0.05, 0.1],
            },
        },
        # Support Vector Machine (SVC)
        "svm": {
            # SVC supports class_weight and is robust; kernels tuned via grid
            "clf": SVC(random_state=42, class_weight="balanced"),
            "param_grid": {
                "clf__C": [5, 15, 25],
                "clf__kernel": ["rbf", "sigmoid"],
                # "clf__kernel": ["poly", "rbf", "sigmoid"],
                # "clf__degree": [1, 2, 3],
                "clf__gamma": ["scale", "auto"],
            },
        },
    }

    def __init__(self, model_key):
        """Select a concrete model by key ('rf' or 'svm')."""
        self.key = model_key

    def get_clf(self):
        return self.models[self.key]["clf"]

    def get_grid(self):
        return self.models[self.key]["param_grid"]

    def add_params(self, params):
        """Update the model with best parameters (post-GridSearchCV)."""
        updated_clf = self.models[self.key]["clf"].set_params(**params)
        self.models[self.key]["clf"] = updated_clf

    def omission_error(self, cm):
        """
        Compute omission error per class (producer's perspective).

        Omission error (class i) = 1 - Producer's Accuracy
        or
        Omission error (class i) = misclassified as NOT i / total true i
        """
        ncls = cm.shape[0]
        omerrs = []
        for cls in range(ncls):
            cls_truth = cm[cls, ]
            cls_omerrs = cls_truth[np.arange(len(cls_truth)) != cls]
            omerrs.append(cls_omerrs.sum() / cls_truth.sum())
        return omerrs

    def commission_error(self, cm):
        """
        Compute commission error per class (user's perspective).

        Commission error (class j) = 1 - User's Accuracy
        or
        Commission error (class j) = misclassified AS j / total predicted j
        """
        ncls = cm.shape[1]
        coerrs = []
        for cls in range(ncls):
            cls_truth = cm[:, cls]
            cls_coerrs = cls_truth[np.arange(len(cls_truth)) != cls]
            coerrs.append(cls_coerrs.sum() / cls_truth.sum())
        return coerrs

    def compute_metrics(self, grid_search, X_test, y_test, pipe_name, pred_id):
        """
        Evaluate a fitted GridSearchCV model and package metrics.

        Returns
        -------
        (cm_dict, metrics_df)
          - cm_dict: {'cm', 'labels', 'model', 'pipe', 'pred_id'}
          - metrics_df: tidy DataFrame with overall, per-class producer/user 
            accuracies
        """
        stats_cols = [
            'model', 'metric', 'label_code', 'data', 'pipe_name', 'pred_id']

        # Predictions on held-out test set
        y_predict = grid_search.predict(X_test)

        # Overall scores
        accuracy = accuracy_score(y_test, y_predict)
        balanced_accuracy = balanced_accuracy_score(y_test, y_predict)
        kappa = cohen_kappa_score(y_test, y_predict)

        # Restrict labels to those actually predicted (keeps CM aligned)
        y_labels = np.unique(y_predict)
        cm = confusion_matrix(y_test, y_predict, labels=y_labels)

        # Class-wise errors
        om_err = self.omission_error(cm)
        co_err = self.commission_error(cm)

        # Producer’s and User’s accuracies
        prod_acc = [1 - oe for oe in om_err]
        user_acc = [1 - ce for ce in co_err]

        # Collect global metrics
        metrics = [
            [self.key, 'overall_accuracy', None, accuracy, pipe_name, pred_id],
            [
                self.key, 'balanced_accuracy', None,
                balanced_accuracy, pipe_name, pred_id
            ],
            [self.key, 'kappa', None, kappa, pipe_name, pred_id],
        ]

        # Per-class metrics
        for i, label in enumerate(y_labels):
            metrics.append([
                self.key, 'producer_acc', label, prod_acc[i],
                pipe_name, pred_id
            ])
        for i, label in enumerate(y_labels):
            metrics.append([
                self.key, 'user_acc', label, user_acc[i], pipe_name, pred_id])

        cm_dict = {
            'cm': cm,
            'labels': y_labels,
            'model': self.key,
            'pipe': pipe_name,
            'pred_id': pred_id
        }
        return cm_dict, pd.DataFrame(metrics, columns=stats_cols)


class Reduction:
    """
    Compute PCA on a chosen predictor subset for visualization/diagnostics.
    """

    def __init__(self, df: pd.DataFrame, predictors: list, label_col: str):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with predictors and labels (must be free of NaNs).
        predictors : list[str]
            Feature columns to include in PCA.
        label_col : str
            Name of the label column to retain alongside PCs.
        """
        # Standardize before PCA
        X = StandardScaler().fit_transform(df[predictors])

        # Fit PCA with 5 components (adjust as needed)
        pca = PCA(n_components=5, random_state=42)
        pca.fit(X)

        # Transform features
        transformation_np = pca.transform(X)

        # Component names
        pca_cols = [f"PCA{i}" for i in range(1, pca.n_components_ + 1)]

        # Assemble transformed DataFrame + labels
        transformation_df = pd.DataFrame(
            transformation_np, columns=pca_cols, index=df.index)
        df_pca = pd.concat([df[label_col], transformation_df], axis=1)

        # Store artifacts
        self.df = df_pca
        self.pca_cols = pca_cols
        self.pca = pca
        self.label_code = label_col

        # Variable loadings (feature contributions per component)
        self.loads = pd.DataFrame(
            pca.components_.T, columns=pca_cols, index=predictors)

    def explained_variance(self):
        """
        Bar plot of the explained variance ratio for each principal component.
        """
        plt.bar(
            self.pca_cols,
            self.pca.explained_variance_ratio_,
            color='gold'
        )
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(self.pca_cols)

    def plot_loadings(self):
        """
        Horizontal bar plots of feature loadings for each principal component."""
        fig, axs = plt.subplots(
            self.loads.shape[1], figsize=(5, 15), sharex=True)

        for i, pca_col in enumerate(self.pca_cols):
            self.loads.iloc[:, i].plot.barh(ax=axs[i])
            axs[i].set_title(pca_col)


def loop_training(dataset: Dataset, pred_id, cm_list, stats_list, grid_list):
    """
    Train models across combinations of (over/under) sampling strategies.

    For each combination:
      - Build a pipeline
        (SMOTENC if categorical predictors exist and SMOTE is selected).
      - Run GridSearchCV on the chosen model.
      - Evaluate on the test set and collect metrics and best params.
    """
    X_train, X_test, y_train, y_test = dataset.split(
        pred_id,
        "code_v1_reclass"
    )
    pred_vars = dataset.predictor_sets[pred_id]

    # Oversampling (data augmentation) strategies
    for da in ["smote", "none"]:
        # Undersampling strategies
        for du in ["tomeklinks", "random", "none"]:

            pipe_name = f"{da}_{du}"
            print(f"  {pipe_name}")

            # Use SMOTENC when categorical predictors are present with SMOTE
            if ("C" in pred_vars) and (da == "smote"):
                pipe = Pipeline(
                    y_train,
                    du,
                    da,
                    X_train,
                    categorical_predictors=pred_vars["C"]
                )
            else:
                pipe = Pipeline(y_train, du, da)

            for model_key in ["rf", "svm"]:
                print(f"    {model_key.upper()}")

                model = Model(model_key)
                pipe.add_model(model.get_clf())

                gridcv, cv = pipe.grid_search(model.get_grid())
                gridcv.fit(X_train, y_train)

                cm_dict, stats = model.compute_metrics(
                    gridcv, X_test, y_test, pipe_name, pred_id
                )

                # Strip 'clf__' prefix from best params for readability
                grid_best = {
                    k.removeprefix("clf__"): v
                    for k, v in gridcv.best_params_.items()
                }

                # Persist grid search summary
                grid_dict = {
                    'pred_id': pred_id,
                    'model': model_key,
                    'pipe': pipe_name,
                    'best_params': grid_best,
                    'best_score': gridcv.best_score_
                }

                cm_list.append(cm_dict)
                stats_list.append(stats)
                grid_list.append(grid_dict)

    return (cm_list, stats_list, grid_list)

def search_grid(
        lst: Union[List[Dict], pd.DataFrame],
        **kwargs: Any
    ) -> Union[Dict, pd.DataFrame]:
    """
    Search for a matching dictionary inside a list of dicts
    or filter a DataFrame based on unique identifier keys.

    Parameters
    ----------
    lst : list[dict] | pd.DataFrame
        Collection to search. If a list of dictionaries is provided,
        the function returns the unique dictionary matching the
        specified identifiers. If a DataFrame is provided, the
        function returns a filtered DataFrame.
    **kwargs
        Identifier key-value pairs used for matching or filtering.
        Expected keys include:

        - pred_id : hashable
        - model : hashable
        - pipe : hashable
        - dataset : hashable

        All provided key-value pairs are combined using logical AND.

    Returns
    -------
    dict | pd.DataFrame
        - If `lst` is a list: returns the unique matching dictionary.
        - If `lst` is a DataFrame: returns the filtered DataFrame.

    Raises
    ------
    ValueError
        If no unique match is found in the list case.
    """
    # In order to match a unique grid, the **kargs dict must contain these keys
    unique_id_keys = ["pred_id", "model", "pipe", "dataset"]

    # Check that arguments passed contains the required keys
    hits = (k in unique_id_keys for k in kwargs.keys())
    if sum(hits) != len(unique_id_keys):
        raise ValueError(
            f"The passed keys {list(kwargs.keys())}\
             do not contain all the required keys: {unique_id_keys}")

    pred_id, model, pipe, dataset = [kwargs[k] for k in unique_id_keys]

    if isinstance(lst, list):
        # Use a generator to avoid building an intermediate list
        matches = (
            d for d in lst
            if d.get("pred_id") == pred_id
            and d.get("model") == model
            and d.get("pipe") == pipe
            and d.get("dataset") == dataset
        )
        result = list(matches)

        if len(result) == 1:
            return result[0]
        raise ValueError("No unique dict found with the requested parameters.")

    elif isinstance(lst, pd.DataFrame):
        query = (
            f"(pred_id == '{pred_id}') "
            f"& (model == '{model}') "
            f"& (pipe == '{pipe}') "
            f"& (dataset == '{dataset}')"
        )
        return lst.query(query).copy()

    else:
        raise TypeError(
            "lst must be either a list of dicts or a pandas DataFrame.")
    
def set_version(dataset_version: int, lst: List[Dict]) -> List[Dict]:
    """
    Ensure each dict in `lst` has a 'dataset' key set to `dataset_version`.
    If the list is empty, it is returned unchanged.
    """
    if not lst:
        return lst

    # If any item lacks 'dataset', set it for all to keep the list consistent
    if not all("dataset" in d for d in lst):
        lst = [dict(d, dataset = dataset_version) for d in lst]
        # Remove version number of each predictor set id
        return [d | {"pred_id": d["pred_id"][:-2]} for d in lst]
    return lst

def extract_version(text: str) -> int:
    """
    Extract the version number.
    """
    m = re.search(r"_v(\d+)_", text)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not extract trailing integer from: {text!r}")


def load_stats(
    folder_name: str,
    logs_folder: Path,
    all_stats: bool = False
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read grid search stats, confusion matrices, and find the best model
    for a given result's prefix.

    Parameters
    ----------
    folder_name : str
        Prefix of the result's folder(s) inside logs_folder
        (e.g., 'train_dataset_v3_TIMESTAMP').
    logs_folder : Path
        Main folder where logs are stored.
    all_stats : bool
        Whether to return the entire CSV with all the model stats (from the 
        multiple training steps — pipe_id, pred_iv, etc. — with the best CV 
        model). This table is returned if a specific model is requested.

    Returns
    -------
    gridcv_best_stats : pd.DataFrame
        DataFrame loaded from 'best_gridcv_stats.csv' with a 'dataset'
        column ensured. Contains the stats computed in the CV step.
    cm_stats : list[dict]
        Confusion matrices stats list with 'dataset' injected if missing.
    best_model : dict
        Dict describing the statistics of the best model from the CV step.
        One model per pipeline.

    Raises
    ------
    FileNotFoundError
        If no matching results folder is found.
    ValueError
        If dataset version cannot be inferred from folder name.
    """
    candidates = list(logs_folder.glob(folder_name + "*"))
    if not candidates:
        raise FileNotFoundError(f"No folders found matching: {logs_folder / (folder_name + '*')}")

    # Choose the most recent matching folder by modification time
    folder_path = max(candidates, key=lambda p: p.stat().st_mtime)
    # Infer dataset version from the provided folder_name
    dataset_version = extract_version(folder_path.name)

    # Load pickled stats
    with open(folder_path / "gridcv_stats.pkl", "rb") as f:
        model_grid_stats: list[dict[str, Any]] = pickle.load(f)
        model_grid_stats = set_version(dataset_version, model_grid_stats)

    with open(folder_path / "confusion_matrices.pkl", "rb") as f:
        cm_stats: list[dict[str, Any]] = pickle.load(f)
        cm_stats = set_version(dataset_version, cm_stats)

    # Load best gridcv stats CSV and ensure 'dataset' column exists
    gridcv_best_stats = pd.read_csv(folder_path / "best_gridcv_stats.csv")
    if "dataset" not in gridcv_best_stats.columns:
        gridcv_best_stats["dataset"] = dataset_version
    # Remove dataset version from predictor ids
    gridcv_best_stats["pred_id"] = gridcv_best_stats["pred_id"].str[:-2]

    # Robustly select the best model
    # (support 'best_score' or sklearn's 'best_score_')
    def _score(d: dict[str, Any]) -> float:
        return float(d.get("best_score", d.get("best_score_", float("-inf"))))

    best_model = max(model_grid_stats, key=_score)

    # For SVM, ensure probability=True to allow SHAP computation
    # on probabilities
    if best_model.get("model") == "svm":
        best_model.setdefault("best_params", {})
        best_model["best_params"]["probability"] = True

    # Add the folder where the model is retrieved
    best_model["folder_name"] = folder_path.name

    if all_stats:
        return model_grid_stats
    else:
        return gridcv_best_stats, cm_stats, best_model

def retrieve_model(
    logs_dir,
    log_folder_name,
    labels_path,
    label_codes_path,
    model_type: str,
    **kwargs
):
    """
    Load the model used for the classification task.

    Parameters
    ----------
    logs_dir : str
        Folder where the logs are saved.
    folder_name : str
        Prefix of the result's folder(s) inside logs_folder
        (e.g., 'train_dataset_v3_TIMESTAMP').
    labels_path : str
        Path pointing the dataset used to train the model.
    label_code_paths : str
        The path to the dataset containing the labels for the reference data.
    model_type : str
        One of the model types available: 'summer_long' or 'spring_summer'

    **kwargs
        Identifier key-value pairs used for matching or filtering.
        Expected keys include:

        - pred_id : hashable
        - model : hashable
        - pipe : hashable
        - dataset : hashable

    """
    model_stats = load_stats(log_folder_name, Path(logs_dir), True)

    # Target model best hiperparams
    model_params = search_grid(
        model_stats,
        **kwargs
    )
    dataset_v = model_params["dataset"]
    predictor_id = f'{model_params["pred_id"]}_{dataset_v}'

    dataset = Dataset(labels_path, label_codes_path, dataset_v, model_type)
    X_train, X_test, y_train, y_test = dataset.split(
        predictor_id,
        "code_v1_reclass"
    )
    pred_vars = dataset.predictor_sets[predictor_id]
    # Construct the preprocessing pipeline
    da, du = model_params["pipe"].split("_")
    # Use SMOTENC when categorical predictors are present with SMOTE
    if ("C" in pred_vars) and (da == "smote"):
        pipe = Pipeline(
            y_train,
            du,
            da,
            X_train,
            categorical_predictors=pred_vars["C"]
        )

    else:
        pipe = Pipeline(y_train, du, da)
    
    pred_vars_list = sum(pred_vars.values(), [])

    model = Model(model_params["model"])
    model.add_params(model_params["best_params"])
    pipe.add_model(model.get_clf())
    pipe.imb_pipe.fit(X_train, y_train)
    return (model, pipe, pred_vars_list, model_params, X_train, X_test)
