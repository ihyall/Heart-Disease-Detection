import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

from config import (
    EvaluationMetrics,
    GridSearchCVValue,
    GridSearchNJobs,
    GridSearchScoringMethod,
)
from src.steps.MLFlowLogging import LogFigure, LogMetrics
from src.steps.Visualization import PlotFeatureImportances


def TrainModel(
    model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
) -> BaseEstimator:
    model.fit(X_train, y_train)
    return model


def AdvancedTrainModel(
    model: BaseEstimator, params: dict, X_train: pd.DataFrame, y_train: pd.Series
):
    paramGrid = params.pop("param_grid", {})
    grid = GridSearchCV(
        estimator=model,
        scoring=GridSearchScoringMethod,
        cv=GridSearchCVValue,
        param_grid=paramGrid,
        n_jobs=GridSearchNJobs,
        **params,
    )
    grid.fit(X_train, y_train)
    return grid


def EvaluateModel(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, prefix: str
):
    y_pred = model.predict(X_test)
    LogMetrics(EvaluationMetrics, y_test, y_pred, metricPrefix=prefix)
    LogFigure(
        PlotFeatureImportances(CalculateFeatureImportances(model, X_test, y_test)),
        "evaluation",
        "feature_importances",
    )


def CalculateFeatureImportances(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> pd.Series:
    result = permutation_importance(model, X, y, n_repeats=10, n_jobs=2)
    return pd.Series(result.importances_mean, index=X.columns)
