import mlflow
import pandas as pd
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import from_pandas
from sklearn.base import BaseEstimator

from config import DatasetTargetName
from src.steps.ModelTraining import EvaluateModel, TrainModel


def MakeDataset(df: pd.DataFrame, name: str, source: str):
    dataset = from_pandas(
        df=df,
        source=HTTPDatasetSource(url=source),
        targets=DatasetTargetName,
        name=name,
    )
    return dataset


def MakeDefaultModelRun(
    parentRunID: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    mlflow.sklearn.autolog()
    with mlflow.start_run(parent_run_id=parentRunID, nested=True):
        model = TrainModel(model=model, X_train=X_train, y_train=y_train)
        EvaluateModel(model=model, X_test=X_test, y_test=y_test)


def RunMultipleModelCycles(
    parentRunID: str,
    models: list[BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    for model in models:
        MakeDefaultModelRun(
            parentRunID=parentRunID,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
