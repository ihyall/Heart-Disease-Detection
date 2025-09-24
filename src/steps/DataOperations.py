import numpy as np
import pandas as pd

from config import DataPath

RootPath: str = ""


def LoadData(fileName: str) -> pd.DataFrame:
    return pd.read_csv(RootPath + DataPath + fileName)


def GetNumericalColumns(df: pd.DataFrame) -> list[str]:
    # columns = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
    # for col in list(columns):
    #     if pd.unique(df[col]).shape[0] < 10:
    #         columns.remove(col)
    # return columns
    return df.select_dtypes(include=["number"]).columns.to_list()


def GetCategoricalColumns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()


def GetNMissing(df: pd.DataFrame) -> str:
    missing = df.isnull().sum()
    result = missing[missing > 0]
    if result.size > 0:
        return missing[missing > 0].to_string()
    return "No missing values"


def GetOutliersIndexes(
    df: pd.DataFrame, columns: list[str], k: float = 1.5
) -> dict[str, list[int]]:
    subDF: pd.DataFrame = df.loc[:, columns]
    Q1 = subDF.quantile(0.25)
    Q3 = subDF.quantile(0.75)
    lowerBound = Q1 - k * (Q3 - Q1)
    upperBound = Q3 + k * (Q3 - Q1)
    bottomOutliers = subDF <= lowerBound
    upperOutliers = subDF >= upperBound

    outliersMask = bottomOutliers + upperOutliers

    result = dict()
    for column in subDF:
        outlierIndexes = subDF[column].index[outliersMask[column]].to_list()
        if len(outlierIndexes) > 0:
            result[column] = outlierIndexes
    return result


def RemoveRowsWithOutliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    indexesDict: dict[str, list[int]] = GetOutliersIndexes(df, GetNumericalColumns(df))
    indexes = set()
    for indexList in indexesDict.values():
        indexes.update(indexList)

    return df.drop(index=list(indexes)).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    sys.path.append("../../")
    RootPath = "../../"
    # df = pd.DataFrame({"a": [np.nan, 1, np.nan], "b": [1, np.nan, 3]})
    df = pd.DataFrame(
        {"a": [-1000, 1, 2, 1, 100, 100], "b": [-200, 1, 100, 3, 200, 300]}
    )
    print(GetOutliersIndexes(df, GetNumericalColumns(df)))
    print(RemoveRowsWithOutliers(df, GetNumericalColumns(df)))
