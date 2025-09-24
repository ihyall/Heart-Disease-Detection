import io

import pandas as pd
from sklearn.model_selection import train_test_split


def GetDFIndo(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def DescribeDF(df: pd.DataFrame) -> str:
    return df.describe().transpose().to_string()


def SplitIntoTrainAndTestSamples(
    df: pd.DataFrame, targetName: str, randomState: int | None = None
) -> list[pd.DataFrame | pd.Series]:
    return train_test_split(
        df.drop(columns=[targetName]),
        df[targetName],
        test_size=0.2,
        random_state=randomState,
    )


def DropDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(ignore_index=True)


if __name__ == "__main__":
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    print(DescribeDF(df))
