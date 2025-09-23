import io

import pandas as pd


def GetDFIndo(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def DescribeDF(df: pd.DataFrame) -> str:
    return df.describe().transpose().to_string()


df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

print(DescribeDF(df))
