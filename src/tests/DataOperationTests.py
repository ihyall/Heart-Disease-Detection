import sys

import numpy as np
import pandas as pd

sys.path.append("../")

import src.steps.DataOperations as do


def test_LoadData():
    data = do.LoadData("heart_disease_health_indicators_BRFSS2015.csv")
    assert data.shape == (253680, 22)


def test_GetNumericalColumns():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", 12],
            "d": [True, True, False],
        }
    )
    numColumns = do.GetNumericalColumns(df=df)
    assert numColumns == ["a", "b"]

    df = pd.DataFrame(
        {
            "a": list("123"),
            "b": list("982"),
            "c": ["asdfasdf", "asudjyfasdk", 12],
            "d": [True, True, False],
        }
    )
    numColumns = do.GetNumericalColumns(df=df)
    assert numColumns == []


def test_GetCategoricalColumns():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", 12],
            "d": [True, True, False],
        }
    )
    catColumns = do.GetCategoricalColumns(df=df)
    assert catColumns == ["c", "d"]

    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1.6, 0.9, -6.7], "c": [15, 10974, 12], "d": [12, 312, 3]}
    )
    catColumns = do.GetCategoricalColumns(df=df)
    assert catColumns == []


def test_GetNMissing():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, np.nan, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", 12],
            "d": [np.nan, True, np.nan],
        }
    )
    missing = do.GetNMissing(df=df)
    assert missing == "b    1\nd    2"

    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1.6, 0.9, -6.7], "c": [15, 10974, 12], "d": [12, 312, 3]}
    )
    missing = do.GetNMissing(df=df)
    assert missing == "No missing values"


if __name__ == "__main__":
    do.RootPath += "../../"
    test_LoadData()
    test_GetNumericalColumns()
    test_GetCategoricalColumns()
    test_GetNMissing()
