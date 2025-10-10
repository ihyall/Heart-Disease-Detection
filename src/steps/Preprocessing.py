import io

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DatasetTargetName


def GetDFIndo(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def DescribeDF(df: pd.DataFrame) -> str:
    return df.describe().transpose().to_string()


def SplitIntoTrainAndTestSamples(
    X: pd.DataFrame, y: pd.Series, testSize: int = 0.2, randomState: int | None = None
) -> list[pd.DataFrame | pd.Series]:
    return train_test_split(X, y, test_size=testSize, random_state=randomState)


def DropDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(ignore_index=True)


def SeparateTargetFromOthers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[DatasetTargetName]), df[DatasetTargetName]


def MergeTargetWithOthers(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return X.copy().assign(**{DatasetTargetName: y})


def ScaleNumericalValues(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    numColumns = X[columns]
    numColumns = scaler.fit_transform(numColumns)
    X[columns] = numColumns
    return X


def FixTargetImbalance(
    X: pd.DataFrame, y: pd.Series, randomState: int | None = None
) -> pd.DataFrame:
    smote = SMOTE(random_state=randomState)
    return smote.fit_resample(X, y)
