import sys

import pandas as pd
import numpy as np

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
    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", "123123"],
        }
    )
    y = pd.Series([True, True, False])
    samples = pre.SplitIntoTrainAndTestSamples(X, y, randomState=0)
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


# TODO make tests for new preprocessing steps
def test_SeparateTargetFromOthers():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.6, 0.9, -6.7],
            "c": ["asdfasdf", "asudjyfasdk", "123123"],
            "HeartDiseaseorAttack": [True, True, False],
        }
    )
    df, target = pre.SeparateTargetFromOthers(df)
    assert df.equals(
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.6, 0.9, -6.7],
                "c": ["asdfasdf", "asudjyfasdk", "123123"],
            }
        )
    )
    assert target.equals(pd.Series([True, True, False]))


def test_MergeTargetWithOthers():
    assert pre.MergeTargetWithOthers(
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.6, 0.9, -6.7],
                "c": ["asdfasdf", "asudjyfasdk", "123123"],
            }
        ),
        pd.Series([True, True, False]),
    ).equals(
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.6, 0.9, -6.7],
                "c": ["asdfasdf", "asudjyfasdk", "123123"],
                "HeartDiseaseorAttack": [True, True, False],
            }
        )
    )


def test_ScaleNumericalValues():  # TODO
    a = [1, 2, 3]
    b = [1.6, 0.9, -6.7]
    assert pre.ScaleNumericalValues(
        X=pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.6, 0.9, -6.7],
                "c": ["asdfasdf", "asudjyfasdk", "123123"],
            }
        ),
        columns=["a", "b"],
    ).equals(
        pd.DataFrame(
            {
                "a": (a - np.mean(a)) / np.std(a),
                "b": (b - np.mean(b)) / np.std(b),
                "c": ["asdfasdf", "asudjyfasdk", "123123"],
            }
        )
    )


def test_FixTargetImbalance():  # TODO
    a = [
        0.000000,
        0.142857,
        0.285714,
        0.428571,
        0.571429,
        0.714286,
        0.857143,
        1.000000,
        1.142857,
        1.285714,
        1.428571,
        1.571429,
        1.714286,
        1.857143,
        2.000000,
        2.142857,
        2.285714,
        2.428571,
        2.571429,
        2.714286,
        2.857143,
        3.000000,
        3.142857,
        3.285714,
        3.428571,
        3.571429,
        3.714286,
        3.857143,
        4.000000,
        4.142857,
        4.285714,
        4.428571,
        4.571429,
        4.714286,
        4.857143,
        5.000000,
        5.142857,
        5.285714,
        5.428571,
        5.571429,
        5.714286,
        5.857143,
        6.000000,
        6.142857,
        6.285714,
        6.428571,
        6.571429,
        6.714286,
        6.857143,
        7.000000,
        3.347397,
        2.636892,
        3.179803,
        3.637496,
    ]
    b = [
        1.000000,
        1.096182,
        1.201614,
        1.317187,
        1.443876,
        1.582751,
        1.734982,
        1.901855,
        2.084779,
        2.285296,
        2.505100,
        2.746044,
        3.010163,
        3.299685,
        3.617054,
        3.964948,
        4.346303,
        4.764337,
        5.222579,
        5.724895,
        6.275524,
        6.879114,
        7.540758,
        8.266040,
        9.061081,
        9.932590,
        10.887922,
        11.935140,
        13.083080,
        14.341432,
        15.720813,
        17.232866,
        18.890350,
        20.707254,
        22.698910,
        24.882127,
        27.275330,
        29.898714,
        32.774419,
        35.926715,
        39.382203,
        43.170046,
        47.322209,
        51.873734,
        56.863031,
        62.332208,
        68.327418,
        74.899257,
        82.103186,
        90.000000,
        8.665380,
        5.475612,
        8.011798,
        10.524805,
    ]
    c = [
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
    ]
    # assert False, print(
    #     list(
    #         map(
    #             lambda x: float(x),
    #             pre.FixTargetImbalance(
    #                 pd.DataFrame({"a": a, "b": b}), pd.Series(c), randomState=0
    #             )
    #         )
    #     )
    # )
    X, y = pre.FixTargetImbalance(
        pd.DataFrame({"a": a, "b": b}), pd.Series(c), randomState=0
    )
    assert X.equals(
        pd.DataFrame(
            {
                "a": [
                    0.0,
                    0.142857,
                    0.285714,
                    0.428571,
                    0.571429,
                    0.714286,
                    0.857143,
                    1.0,
                    1.142857,
                    1.285714,
                    1.428571,
                    1.571429,
                    1.714286,
                    1.857143,
                    2.0,
                    2.142857,
                    2.285714,
                    2.428571,
                    2.571429,
                    2.714286,
                    2.857143,
                    3.0,
                    3.142857,
                    3.285714,
                    3.428571,
                    3.571429,
                    3.714286,
                    3.857143,
                    4.0,
                    4.142857,
                    4.285714,
                    4.428571,
                    4.571429,
                    4.714286,
                    4.857143,
                    5.0,
                    5.142857,
                    5.285714,
                    5.428571,
                    5.571429,
                    5.714286,
                    5.857143,
                    6.0,
                    6.142857,
                    6.285714,
                    6.428571,
                    6.571429,
                    6.714286,
                    6.857143,
                    7.0,
                    3.347397,
                    2.636892,
                    3.179803,
                    3.637496,
                    3.8879766395635333,
                    4.13130364158747,
                    4.215203007901178,
                    4.682106331922956,
                    4.577128656278937,
                    2.659304585149254,
                    5.6300136615831695,
                    1.615452080493701,
                    3.9059411696102626,
                    6.3364031072435,
                    0.25719937336004833,
                    4.999503712590954,
                    5.301522920365178,
                    2.8259310993008566,
                    4.261331721522453,
                    6.811031630052234,
                    2.907800012737287,
                    2.4536284718118297,
                    6.064647282497823,
                    6.266924199563644,
                ],
                "b": [
                    1.0,
                    1.096182,
                    1.201614,
                    1.317187,
                    1.443876,
                    1.582751,
                    1.734982,
                    1.901855,
                    2.084779,
                    2.285296,
                    2.5051,
                    2.746044,
                    3.010163,
                    3.299685,
                    3.617054,
                    3.964948,
                    4.346303,
                    4.764337,
                    5.222579,
                    5.724895,
                    6.275524,
                    6.879114,
                    7.540758,
                    8.26604,
                    9.061081,
                    9.93259,
                    10.887922,
                    11.93514,
                    13.08308,
                    14.341432,
                    15.720813,
                    17.232866,
                    18.89035,
                    20.707254,
                    22.69891,
                    24.882127,
                    27.27533,
                    29.898714,
                    32.774419,
                    35.926715,
                    39.382203,
                    43.170046,
                    47.322209,
                    51.873734,
                    56.863031,
                    62.332208,
                    68.327418,
                    74.899257,
                    82.103186,
                    90.0,
                    8.66538,
                    5.475612,
                    8.011798,
                    10.524805,
                    12.441966493875618,
                    14.252292280192398,
                    15.897683416129347,
                    21.379970199412888,
                    19.66283927250835,
                    5.625925442731244,
                    41.969542572661325,
                    3.0128731580915122,
                    12.391639364598715,
                    58.99625527499233,
                    1.2098363849003642,
                    25.421213202686708,
                    30.23224982335619,
                    6.818386037494758,
                    15.54039963179456,
                    81.23342888291295,
                    6.674850776136093,
                    4.922094591266482,
                    52.745941053490746,
                    56.35637686466738,
                ],
            }
        )
    )
    assert y.equals(
        pd.Series(
            [
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )
    )
