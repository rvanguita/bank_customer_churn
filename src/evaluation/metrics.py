import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from typing import Tuple, Union

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef,
    cohen_kappa_score, brier_score_loss
)
from sklearn.preprocessing import label_binarize
import mlflow
from mlflow.models.signature import infer_signature



from sklearn.metrics import (
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)

class ClassifierMetricsEvaluator:
    def __init__(self, model):
        self.model = model

    def _validate_inputs(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X, np.asarray(y).ravel()


    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 1
        is_multiclass = n_classes > 2

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        }

        if is_multiclass:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            brier = np.mean([
                brier_score_loss(y_true_bin[:, i], y_proba[:, i])
                for i in range(n_classes)
            ])
        else:
            y_proba_bin = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            brier = brier_score_loss(y_true, y_proba_bin)

        results["brier_score"] = float(brier)
        results["log_loss"] = float(log_loss(y_true, y_proba))

        try:
            if is_multiclass:
                results["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            else:
                results["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba))
        except Exception:
            results["roc_auc"] = float("nan")

        return results
    
    
    def evaluate_fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], predict_proba: bool = True):
        X, y = self._validate_inputs(X, y)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        n_classes = len(np.unique(y))
        if predict_proba and hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = np.zeros((len(y), n_classes), dtype=float)

        metrics = self._compute_metrics(y, y_pred, y_proba)
        final_metrics = {
            k: round(float(np.nanmean(v)), 6)
            for k, v in metrics.items()
            if not np.isnan(np.nanmean(v))
        }
        return final_metrics, X, y, y_pred, y_proba


    def cross_validate(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], n_splits: int = 5, predict_proba: bool = True
    ):
        X, y = self._validate_inputs(X, y)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        

        n_classes = len(np.unique(y))
        aggregate_metrics = {
            k: [] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
                            "matthews_corrcoef", "cohen_kappa", "brier_score", "log_loss"]
        }

        X_tests, y_trues, y_preds, y_probas = [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, y), total=n_splits, desc="Cross-validation")):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if predict_proba and hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_test)
            else:
                y_proba = np.zeros((len(X_test), n_classes), dtype=float)

            X_tests.append(X_test.to_numpy())
            y_trues.extend(y_test)
            y_preds.extend(y_pred)
            y_probas.append(y_proba)

            fold_metrics = self._compute_metrics(y_test, y_pred, y_proba)
            for metric, value in fold_metrics.items():
                aggregate_metrics[metric].append(value)

        # final_metrics = pd.DataFrame([{k: round(np.mean(v), 2) for k, v in aggregate_metrics.items()}])
        final_metrics = {k.replace(" ", "_").lower(): round(float(np.mean(v)), 2) for k, v in aggregate_metrics.items()}

        return (
            final_metrics,
            np.vstack(X_tests),
            np.array(y_trues),
            np.array(y_preds),
            np.vstack(y_probas)
        )
        
    
    def cross_val_predict_summary(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], cv: int = 5
    ):
        X, y = self._validate_inputs(X, y)
        y_pred = cross_val_predict(self.model, X, y, cv=cv, method="predict")

        n_classes = len(np.unique(y))
        if hasattr(self.model, "predict_proba"):
            y_proba = cross_val_predict(self.model, X, y, cv=cv, method="predict_proba")
        else:
            y_proba = np.zeros((len(y), n_classes), dtype=float)

        metrics = self._compute_metrics(y, y_pred, y_proba)
        return pd.DataFrame([metrics]), X, y, y_pred, y_proba
    

    def cross_validate_with_mlflow(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        predict_proba: bool = True,
        mlflow=None,
        experiment_name: str = "default"
    ):
        X, y = self._validate_inputs(X, y)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        n_classes = len(np.unique(y))
        aggregate_metrics = {
            k: [] for k in [
                "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
                "matthews_corrcoef", "cohen_kappa", "brier_score", "log_loss"
            ]
        }

        X_tests, y_trues, y_preds, y_probas = [], [], [], []

        mlflow.set_experiment(experiment_name)
        model_name = str(self.model).split('(')[0]
        
        
        with mlflow.start_run(run_name=model_name):
            for fold_idx, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, y), total=n_splits, desc="Cross-validation")):
                X_train, X_test = _subset(X, train_idx), _subset(X, test_idx)
                y_train, y_test = _subset(y, train_idx), _subset(y, test_idx)

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                if predict_proba and hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(X_test)
                else:
                    y_proba = np.zeros((len(X_test), n_classes), dtype=float)

                X_tests.append(X_test.to_numpy())
                y_trues.extend(y_test)
                y_preds.extend(y_pred)
                y_probas.append(y_proba)

            fold_metrics = self._compute_metrics(y_test, y_pred, y_proba)
            for metric, value in fold_metrics.items():
                aggregate_metrics[metric].append(value)

            final_metrics = {
                k: round(float(np.nanmean(v)), 6)
                for k, v in aggregate_metrics.items()
                if not np.isnan(np.nanmean(v))
            }

            mlflow.log_metrics(final_metrics)

            X_example = pd.DataFrame(X_tests[0], columns=X.columns)
            signature = infer_signature(X_example, self.model.predict(X_example))
            
            mlflow.set_tag("estimator", model_name)
            
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path=model_name,
                signature=signature,
                input_example=X_example
            )


        return (
            final_metrics,
            np.vstack(X_tests),
            np.array(y_trues),
            np.array(y_preds),
            np.vstack(y_probas)
        )


def _subset(data, idx):
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]