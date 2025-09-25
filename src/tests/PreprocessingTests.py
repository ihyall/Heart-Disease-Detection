import sys

import pandas as pd

sys.path.append("../")

import src.steps.Preprocessing as pre


def test_GetDFInfo():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", 12],
            "d": [True, True, False],
        }
    )
    info = pre.GetDFIndo(df=df)
    assert (
        info
        == "<class 'pandas.core.frame.DataFrame'>\n\
RangeIndex: 3 entries, 0 to 2\n\
Data columns (total 4 columns):\n\
 #   Column  Non-Null Count  Dtype  \n\
---  ------  --------------  -----  \n\
 0   a       3 non-null      int64  \n\
 1   b       3 non-null      float64\n\
 2   c       3 non-null      object \n\
 3   d       3 non-null      bool   \n\
dtypes: bool(1), float64(1), int64(1), object(1)\n\
memory usage: 207.0+ bytes\n"
    )


def test_DescribeDF():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", 12],
            "d": [True, True, False],
        }
    )
    info = pre.DescribeDF(df=df)
    assert (
        info
        == "   count  mean      std  min  25%  50%   75%  max\n\
a    3.0   2.0  1.00000  1.0  1.5  2.0  2.50  3.0\n\
b    3.0  -1.4  4.60326 -6.7 -2.9  0.9  1.25  1.6"
    )


def test_SplitIntoTrainAndTestSamples():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", "123123"],
            "d": [True, True, False],
        }
    )
    target = "d"
    samples = pre.SplitIntoTrainAndTestSamples(df, target, randomState=0)
    print(*samples, sep="\n")
    validation: list[pd.DataFrame | pd.Series] = [
        pd.DataFrame(
            {
                "a": {1: 2, 0: 1},
                "b": {1: 0.9, 0: 1.6},
                "c": {1: "asudjyfasdk", 0: "asdfasdf"},
            }
        ),
        pd.DataFrame({"a": {2: 3}, "b": {2: -6.7}, "c": {2: "123123"}}),
        pd.Series({1: True, 0: True}),
        pd.Series({2: False}),
    ]
    for i, sample in enumerate(samples):
        assert sample.equals(validation[i])


def test_DropDuplicates():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", "123123"],
            "d": [True, True, False],
        }
    )
    assert pre.DropDuplicates(df).equals(df)

    df1 = pd.DataFrame(
        {
            "a": [1, 1, 2, 3],
            "b": [1.6, 1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asdfasdf", "asudjyfasdk", "123123"],
            "d": [True, True, True, False],
        }
    )
    assert pre.DropDuplicates(df1).equals(df)


if __name__ == "__main__":
    test_GetDFInfo()
    test_DescribeDF()
    test_SplitIntoTrainAndTestSamples()
    test_DropDuplicates()
