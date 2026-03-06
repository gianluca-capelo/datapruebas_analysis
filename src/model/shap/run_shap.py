import ast
import os

import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MAX_SELECTED_FEATURES, CLASSIFICATION_RESULTS_DIR, REGRESSION_RESULTS_DIR
from src.model.run_models import retrieve_dataset, get_models


def _fresh_estimator(model_name, global_seed, is_classification):
    models = get_models(random_state=global_seed, is_classification=is_classification)
    for m in models:
        if m.__class__.__name__ == model_name:
            return m.__class__(**m.get_params())
    raise ValueError(f"Estimator '{model_name}' not found in model zoo.")


def _build_pipeline(model, is_classification, feature_selection, X_train_shape, select_score_func):
    step_name = 'classifier' if is_classification else 'regressor'
    select_step = (
        ('select', SelectKBest(score_func=select_score_func, k=min(MAX_SELECTED_FEATURES, X_train_shape[1])))
        if feature_selection else ('select_noop', 'passthrough')
    )
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        select_step,
        ('scaler', StandardScaler()),
        (step_name, model),
    ])
    return pipe, step_name


def parse_hparams(s):
    import math
    d = eval(s, {"__builtins__": {}}, {"None": None, "True": True, "False": False, "nan": math.nan})
    if not isinstance(d, dict):
        raise ValueError("Hyperparameters string is not a valid dictionary.")
    return d


def shap_after_nested_cv(
        dataset_name: str,
        is_classification: bool,
        feature_selection: bool,
        global_seed: int,
        model_name_to_explain: str,
        folds_csv_path: str
):
    """
    Refit per outer LOO fold with stored hyperparameters,
    compute SHAP for that fold's test sample(s), and return:
      shap_values_df: DataFrame [samples x features]
      mean_abs_shap: Series (mean |SHAP| per feature)
    """
    X, y, feature_names, target_name = retrieve_dataset(dataset_name)
    feature_names = np.array(feature_names)

    folds_df = load_folds_info(folds_csv_path, model_name_to_explain)

    loo = LeaveOneOut()
    fold_splits = list(loo.split(X, y))

    if len(fold_splits) != len(folds_df):
        raise RuntimeError("Current LOO fold count and folds.csv rows differ. "
                           "Persist and reuse train/test indices to guarantee identity.")

    select_score_func = f_classif if is_classification else f_regression

    explanations = []
    for fold_id, ((train_idx, test_idx), fold_row) in enumerate(zip(fold_splits, folds_df.itertuples(index=False))):
        if fold_id != fold_row.fold:
            raise RuntimeError(f"Fold order mismatch at fold {fold_id} vs {fold_row.fold}.")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        estimator = _fresh_estimator(model_name_to_explain, global_seed, is_classification)
        pipeline, estimator_step_name = _build_pipeline(estimator, is_classification, feature_selection, X_train.shape,
                                                        select_score_func)
        # Apply per-fold estimator params
        final_est = pipeline.named_steps[estimator_step_name]
        final_est.set_params(**fold_row.hyperparameters)
        pipeline.named_steps[estimator_step_name] = final_est

        # Fit on training portion of this fold
        pipeline.fit(X_train, y_train)

        expl = compute_shap_for_pipeline(X_test, X_train, estimator_step_name, feature_names, feature_selection,
                                         pipeline, seed=global_seed, is_classification=is_classification)

        explanations.append(expl)

    return explanations


def _callable_for_shap(estimator, is_classification):
    if is_classification:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba
        else:
            raise ValueError("At the moment, only classifiers with predict_proba are supported for SHAP.")
    else:
        return estimator.predict


def compute_shap_for_pipeline(X_test, X_train, estimator_step_name, feature_names, feature_selection, pipeline,
                              seed: int, is_classification=True):
    if seed is None:
        raise ValueError("seed must be provided for reproducible SHAP results.")
    # SHAP computation
    preprocess = pipeline[:-1]
    estimator = pipeline.named_steps[estimator_step_name]
    X_train_transformed = preprocess.transform(X_train)
    X_test_transformed = preprocess.transform(X_test)
    # Map names after SelectKBest
    if feature_selection:
        support_idx = pipeline.named_steps['select'].get_support(indices=True)
        shap_feature_names = np.asarray(feature_names)[support_idx]
    else:
        shap_feature_names = np.asarray(feature_names)

    f_callable = _callable_for_shap(estimator, is_classification)
    explainer = shap.Explainer(f_callable, X_train_transformed, feature_names=shap_feature_names, seed=seed)

    kind = f"{explainer.__module__}.{explainer.__class__.__name__}"
    link = getattr(explainer.link, "__class__", type(explainer.link)).__name__
    print(f"[SHAP] backend: {kind} | link: {link}")
    # Explain the test sample(s)
    return explainer(X_test_transformed)  # Explanation


def load_folds_info(folds_csv_path, model_name_to_explain):
    folds_df = pd.read_csv(folds_csv_path)
    folds_df = folds_df[folds_df['model'] == model_name_to_explain].copy()
    if folds_df.empty:
        raise ValueError(f"No rows for model '{model_name_to_explain}' in {folds_csv_path}")
    # Parse hyperparameters (stringified dict → dict)
    if folds_df['hyperparameters'].dtype == object:
        folds_df['hyperparameters'] = folds_df['hyperparameters'].apply(
            lambda s: parse_hparams(s)
        )
    folds_df = folds_df.sort_values('fold').reset_index(drop=True)
    return folds_df


from src.config import MODEL_OUTER_SEED


def _explanations_to_dataframe(explanations: list) -> pd.DataFrame:
    """Convert a list of shap.Explanation objects (one per LOO fold) to a DataFrame.

    Returns a DataFrame with columns: fold, base_value, and one column per feature.
    Features not selected in a fold appear as NaN.
    """
    rows = []
    for fold_id, expl in enumerate(explanations):
        vals = expl.values.flatten()
        names = expl.feature_names
        row = {"fold": fold_id, "base_value": float(np.atleast_1d(expl.base_values).flat[0])}
        row.update(dict(zip(names, vals)))
        rows.append(row)
    return pd.DataFrame(rows)


def run_shap(task: str, dataset_name: str, timestamp: str, model_name_to_explain: str):
    """
    Run SHAP analysis on a trained model.

    Args:
        task: 'classification' or 'regression'
        dataset_name: Name of the dataset (e.g., 'tmt_ssrt')
        timestamp: Timestamp folder of the results
        model_name_to_explain: Name of the model to explain (e.g., 'Ridge')

    Returns:
        tuple: (shap_explanations, target_name)
    """
    if task != 'classification' and task != 'regression':
        raise ValueError("task must be 'classification' or 'regression'")

    is_classification = task == 'classification'

    # Get target_name from dataset
    _, _, _, target_name = retrieve_dataset(dataset_name)

    results_dir = CLASSIFICATION_RESULTS_DIR if is_classification else REGRESSION_RESULTS_DIR

    dataset_dir = os.path.join(
        results_dir,
        timestamp,
        target_name,
        dataset_name,
    )

    folds_path = os.path.join(dataset_dir, "folds.csv")

    shap_explanations = shap_after_nested_cv(
        dataset_name=dataset_name,
        is_classification=is_classification,
        feature_selection=True,
        global_seed=MODEL_OUTER_SEED,
        model_name_to_explain=model_name_to_explain,
        folds_csv_path=folds_path,
    )

    # Save raw SHAP values to CSV
    shap_df = _explanations_to_dataframe(shap_explanations)
    shap_csv_path = os.path.join(dataset_dir, f"shap_values_{model_name_to_explain}.csv")
    shap_df.to_csv(shap_csv_path, index=False)
    print(f"[SHAP] Raw values saved to: {shap_csv_path}")

    return shap_explanations, target_name
