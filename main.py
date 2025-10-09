import mlflow

from config import (
    DatasetFilename,
    DatasetName,
    DatasetSourceURL,
    DefaultVisualizations,
    ExperimentName,
    Models,
    PostProcessingVisualizations,
)
from src.steps.DataOperations import LoadData
from src.steps.MLFlowBase import MakeDataset, RunMultipleModelCycles
from src.steps.MLFlowLogging import LogFigures
from src.steps.Preprocessing import (
    FixTargetImbalance,
    MergeTargetWithOthers,
    SeparateTargetFromOthers,
    SplitIntoTrainAndTestSamples,
)

# Тренировочный код для продумывания проекта
mlflow.set_tracking_uri("http://localhost:5000")

if mlflow.get_experiment_by_name(ExperimentName) is None:
    mlflow.create_experiment(name=ExperimentName)
mlflow.set_experiment(ExperimentName)

with mlflow.start_run() as run:
    df = LoadData(DatasetFilename)

    dataset = MakeDataset(df=df, name=DatasetName, source=DatasetSourceURL)

    mlflow.log_input(dataset=dataset, context="Raw")
    LogFigures(DefaultVisualizations, "preprocessing", df=dataset.df, nCols=2)

    X, y = FixTargetImbalance(*SeparateTargetFromOthers(dataset.df))

    processedDataset = MakeDataset(
        df=MergeTargetWithOthers(X, y),
        name=f"{DatasetName} (processed)",
        source=DatasetSourceURL,
    )
    mlflow.log_input(dataset=processedDataset, context="Processed")
    LogFigures(
        PostProcessingVisualizations, "postprocessing", df=processedDataset.df, nCols=2
    )

    X_train, X_test, y_train, y_test = SplitIntoTrainAndTestSamples(X, y)

    RunMultipleModelCycles(
        parentRunID=run.info.run_id,
        models=Models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
