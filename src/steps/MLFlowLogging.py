from typing import Callable

import mlflow
from matplotlib.figure import Figure


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
