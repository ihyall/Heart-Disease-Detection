import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from config import DatasetTargetName, VisualizationsPath
from src.steps.DataOperations import GetNumericalColumns


def PlotKDE(df: pd.DataFrame, nCols: int = 2, **kwargs) -> Figure:
    columns = GetNumericalColumns(df)
    nPlots = len(columns)
    nRows = (nPlots // nCols) + (nPlots % nCols)

    fig = plt.figure(figsize=(5 * nCols, 3 * (len(columns) // nCols)))

    for i, col in enumerate(columns):
        plt.subplot(nRows, nCols, i + 1)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Распределение {col}")
        plt.xlabel("")

    plt.tight_layout()
    return fig


def PlotMissing(df: pd.DataFrame, **kwargs) -> Figure:
    fig = plt.figure(figsize=(df.shape[1] // 2.5, df.shape[1] // 2.5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Пропуски в наборе данных")
    plt.tight_layout()
    return fig


def PlotOutliers(df: pd.DataFrame, nCols: int = 2, **kwargs) -> Figure:
    columns = GetNumericalColumns(df)
    nPlots = len(columns)
    nRows = (nPlots // nCols) + (nPlots % nCols)

    fig = plt.figure(figsize=(5 * nCols, 3 * (nPlots // nCols)))
    for i, col in enumerate(columns):
        plt.subplot(nRows, nCols, i + 1)
        sns.boxplot(data=df, x=col, hue=DatasetTargetName)
        plt.title(f"Выбросы {col}")
        plt.xlabel("")

    plt.tight_layout()
    return fig


def PlotTargetBalance(df: pd.DataFrame, **kwargs) -> Figure:
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=DatasetTargetName)
    plt.title(f"Сбалансированность {DatasetTargetName}")
    return fig


defaultVisualizations = [PlotKDE, PlotMissing, PlotOutliers, PlotTargetBalance]


def MakeDefaultVisualizations(df: pd.DataFrame, **kwargs):
    for visFunc in defaultVisualizations:
        visFunc(df, **kwargs)
        plt.savefig(VisualizationsPath + visFunc.__name__.removeprefix("Plot"))


if __name__ == "__main__":
    import sys

    sys.path.append("../../")
    from config import VisualizationsPath

    df = pd.read_csv("../../data/heart_disease_health_indicators_BRFSS2015.csv").sample(
        100000
    )
    # plotKde(df, GetNumericalColumns(df))
    # plotOutliers(df, GetNumericalColumns(df))
    # plotTargetBalance(df, DatasetTargetName)
    plt.savefig("../../" + VisualizationsPath + "test.png")
    plt.show()
