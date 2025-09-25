from typing import Callable

import mlflow
from matplotlib.figure import Figure

from config import DatasetTargetName


def LogMetric(metricName: str, value: float):
    mlflow.log_metric(key=metricName, value=value)


def LogMetrics(metricFuncs: list[Callable], *args, metricPrefix: str = ""):
    for metric in metricFuncs:
        LogMetric(metricName=f"{metricPrefix}_{metric.__name__}", value=metric(*args))


def LogFigure(figure: Figure, dirName: str, plotName: str):
    mlflow.log_figure(figure=figure, artifact_file=f"{dirName}/{plotName}.png")
    # print(figure, f"{dirName}/{plotName}.png")


def LogFigures(visFuncs: list[Callable], dirName: str, **kwargs):
    for visFunc in visFuncs:
        LogFigure(
            visFunc(**kwargs),
            dirName=dirName,
            plotName=visFunc.__name__.removeprefix("Plot"),
        )


if __name__ == "__main__":
    import sys

    import pandas as pd
    from sklearn.metrics import accuracy_score

    sys.path.append("../")
    from src.steps.Visualization import (
        PlotKDE,
        PlotMissing,
        PlotOutliers,
        PlotTargetBalance,
    )

    LogMetrics([accuracy_score], [1, 2, 3], [1, 2, 2])

    df = pd.read_csv("data/heart_disease_health_indicators_BRFSS2015.csv").sample(10000)

    # LogFigure(PlotKde(df=df, columns=GetNumericalColumns(df)))
    LogFigures(
        [PlotKDE, PlotMissing, PlotOutliers, PlotTargetBalance],
        dirName="preprocessing",
        df=df,
        targetName=DatasetTargetName,
    )
