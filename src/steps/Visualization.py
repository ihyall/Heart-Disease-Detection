import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.steps.DataOperations import GetNumericalColumns, GetOutliersIndexes

# from .DataOperations import GetNumericalColumns, GetOutliersIndexes
from matplotlib.figure import Figure


def plotKde(df: pd.DataFrame, columns: list[str], nCols: int = 2) -> Figure:
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


def plotMissing(df: pd.DataFrame) -> Figure:
    fig = plt.figure(figsize=(df.shape[1] // 2.5, df.shape[1] // 2.5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Пропуски в наборе данных")
    plt.tight_layout()
    return fig


def plotOutliers(df: pd.DataFrame, columns: list[str], nCols: int = 2) -> Figure:
    nPlots = len(columns)
    nRows = (nPlots // nCols) + (nPlots % nCols)

    # indexes = GetOutliersIndexes(df, columns, 1.5)

    fig = plt.figure(figsize=(5 * nCols, 3 * (nPlots // nCols)))
    for i, col in enumerate(columns):
        # if col in indexes:
        plt.subplot(nRows, nCols, i + 1)
        # sns.kdeplot(data=df, x=col, fill=True)
        # plt.boxplot(x=df[col], vert=False)
        sns.boxplot(data=df, x=col, hue="HeartDiseaseorAttack")
        plt.title(f"Выбросы {col}")
        plt.xlabel("")

    plt.tight_layout()
    return fig


def plotTargetBalance(df: pd.DataFrame, targetName: str) -> Figure:
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=targetName)
    plt.title(f"Сбалансированность {targetName}")
    return fig


if __name__ == "__main__":
    import sys

    sys.path.append("../../")
    from config import VisualizationPath

    df = pd.read_csv("../../data/heart_disease_health_indicators_BRFSS2015.csv").sample(
        100000
    )
    # plotKde(df, GetNumericalColumns(df))
    plotOutliers(df, GetNumericalColumns(df))
    # plotTargetBalance(df, "HeartDiseaseorAttack")
    plt.savefig("../../" + VisualizationPath + "test.png")
    plt.show()
