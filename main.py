import mlflow
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from config import DatasetFilename, DatasetName, DatasetSourceURL, ExperimentName
from src.steps.DataOperations import LoadData
from src.steps.MLFlowBase import MakeDataset, MakeDefaultModelRun
from src.steps.MLFlowLogging import LogFigures
from src.steps.Preprocessing import SplitIntoTrainAndTestSamples
from src.steps.Visualization import defaultVisualizations

# Тренировочный код для продумывания проекта
mlflow.set_tracking_uri("http://localhost:5000")

if mlflow.get_experiment_by_name(ExperimentName) is None:
    mlflow.create_experiment(name=ExperimentName)
mlflow.set_experiment(ExperimentName)

with mlflow.start_run() as run:
    df = LoadData(DatasetFilename)

    dataset = MakeDataset(
        df=df.sample(10000), name=DatasetName, source=DatasetSourceURL
    )

    mlflow.log_input(dataset=dataset, context="raw")
    LogFigures(defaultVisualizations, "preprocessing", df=dataset.df, nCols=2)

    # preprocessing if anything will be needed
    # mlflow.log_input(dataset=dataset, context="preprocessed")

    X_train, X_test, y_train, y_test = SplitIntoTrainAndTestSamples(dataset.df)

    MakeDefaultModelRun(
        parentRunID=run.info.run_id,
        model=LogisticRegression,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        max_iter=400,
    )

    MakeDefaultModelRun(
        parentRunID=run.info.run_id,
        model=LogisticRegressionCV,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        max_iter=400,
    )
