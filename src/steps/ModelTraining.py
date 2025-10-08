import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.steps.MLFlowLogging import LogMetrics


def TrainModel(
    model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
) -> BaseEstimator:
    model.fit(X_train, y_train)
    return model


def EvaluateModel(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    LogMetrics(
        [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score],
        y_test,
        y_pred,
        metricPrefix="testing_",
    )
