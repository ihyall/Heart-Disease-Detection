import mlflow
import pandas as pd
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import from_pandas

from config import DatasetTargetName
from src.steps.ModelTraining import AdvancedTrainModel, EvaluateModel
from src.steps.Preprocessing import ScaleNumericalValues


def MakeDataset(df: pd.DataFrame, name: str, source: str):
    dataset = from_pandas(
        df=df,
        source=HTTPDatasetSource(url=source),
        targets=DatasetTargetName,
        name=name,
    )
    return dataset


def MakeModelRun(
    parentRunID: str,
    model: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    with mlflow.start_run(parent_run_id=parentRunID, nested=True):
        model = AdvancedTrainModel(
            model=model["instance"],
            params=model.setdefault("gridSearchParams", {}),
            X_train=X_train,
            y_train=y_train,
        )

        EvaluateModel(model=model, X_test=X_test, y_test=y_test, prefix="training")
        EvaluateModel(model=model, X_test=X_test, y_test=y_test, prefix="testing")
        mlflow.log_param(
            "model",
            str(model.best_estimator_.__class__).split(".")[-1].removesuffix("'>"),
        )

        match model.__module__.split(".")[0]:
            case "catboost":
                mlflow.catboost.log_model(
                    cb_model=model,
                    params=model.get_params(),
                    name=str(model.best_estimator_.__class__)
                    .split(".")[-1]
                    .removesuffix("'>"),
                    input_example=X_test,
                )
            case "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    params=model.get_params(),
                    name=str(model.best_estimator_.__class__)
                    .split(".")[-1]
                    .removesuffix("'>"),
                    input_example=X_test,
                )


def RunMultipleModelCycles(
    parentRunID: str,
    models: list[dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    mlflow.autolog(log_models=False)
    for model in models:
        if model.setdefault("scaleValues", False):
            X_train = ScaleNumericalValues(X_train, X_train.columns)
            X_test = ScaleNumericalValues(X_test, X_test.columns)

        MakeModelRun(
            parentRunID=parentRunID,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
