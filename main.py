import mlflow

from config import (
    DatasetFilename,
    DatasetName,
    DatasetSourceURL,
    ExperimentName,
    Models,
    DefaultVisualizations,
    PostProcessingVisualizations,
)
from src.steps.DataOperations import LoadData
from src.steps.MLFlowBase import (
    MakeDataset,
    MakeDefaultModelRun,
    RunMultipleModelCycles,
)
from src.steps.MLFlowLogging import LogFigures
from src.steps.Preprocessing import (
    SplitIntoTrainAndTestSamples,
    ScaleNumericalValues,
    FixTargetImbalance,
    SeparateTargetFromOthers,
    MergeTargetWithOthers,
)

# Тренировочный код для продумывания проекта
mlflow.set_tracking_uri("http://localhost:5000")

if mlflow.get_experiment_by_name(ExperimentName) is None:
    mlflow.create_experiment(name=ExperimentName)
mlflow.set_experiment(ExperimentName)

with mlflow.start_run() as run:
    df = LoadData(DatasetFilename)

    dataset = MakeDataset(df=df.sample(5000), name=DatasetName, source=DatasetSourceURL)

    mlflow.log_input(dataset=dataset, context="raw")
    LogFigures(DefaultVisualizations, "preprocessing", df=dataset.df, nCols=2)

    # preprocessing if anything will be needed
    X, y = FixTargetImbalance(*SeparateTargetFromOthers(dataset.df))
    # TODO MergeTargetWithOthers makes all accuracy 1, need to be fixed
    processedDataset = MakeDataset(
        df=MergeTargetWithOthers(X, y),
        name=f"{DatasetName} (processed)",
        source=DatasetSourceURL,
    )

    mlflow.log_input(dataset=processedDataset, context="processed")
    LogFigures(
        PostProcessingVisualizations,
        "postprocessing",
        df=MergeTargetWithOthers(X, y),
        nCols=2,
    )

    X_train, X_test, y_train, y_test = SplitIntoTrainAndTestSamples(X, y)

    X_train = ScaleNumericalValues(X_train, ["Age", "BMI"])
    X_test = ScaleNumericalValues(X_test, ["Age", "BMI"])

    RunMultipleModelCycles(
        parentRunID=run.info.run_id,
        models=Models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
