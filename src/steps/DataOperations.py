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
    """
    Returns a string of a DataFrame with null value counters

    If there are none, returns `"No missing values"`
    """
    missing = df.isnull().sum()
    result = missing[missing > 0]
    if result.size > 0:
        return missing[missing > 0].to_string()
    return "No missing values"


def GetOutliersIndices(
    df: pd.DataFrame, columns: list[str], k: float = 1.5
) -> dict[str, list[int]]:
    """
    Returns a dictionary, where:
    - keys represent passed DataFrame columns
    - values are the indices of rows with missing values in corresponding column
    """
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
        outlierIndices = subDF[column].index[outliersMask[column]].to_list()
        if len(outlierIndices) > 0:
            result[column] = outlierIndices
    return result


def RemoveRowsWithOutliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    indicesDict: dict[str, list[int]] = GetOutliersIndices(df, GetNumericalColumns(df))
    indices = set()
    for indexList in indicesDict.values():
        indices.update(indexList)

    return df.drop(index=list(indices)).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    sys.path.append("../../")
    RootPath = "../../"
    # df = pd.DataFrame({"a": [np.nan, 1, np.nan], "b": [1, np.nan, 3]})
    df = pd.DataFrame(
        {"a": [-1000, 1, 2, 1, 100, 100], "b": [-200, 1, 100, 3, 200, 300]}
    )
    print(GetOutliersIndices(df, GetNumericalColumns(df)))
    print(RemoveRowsWithOutliers(df, GetNumericalColumns(df)))
