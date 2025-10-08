# Main folder paths
# VisualizationsPath = "visualizations/"
DataPath = "data/"

# MLFlow experiment name
ExperimentName = "Heart Disease Detection"

# Dataset info
DatasetFilename = "heart_disease_health_indicators_BRFSS2015.csv"
DatasetSourceURL = (
    "https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset"
)
DatasetName = "Heart Disease Health Indicators Dataset"
DatasetTargetName = "HeartDiseaseorAttack"
DatasetNumericalColumns = ["BMI"]  # TODO


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier

Models = [
    LogisticRegression(max_iter=400),
    LogisticRegressionCV(max_iter=400),
    DecisionTreeClassifier(),
]

from src.steps.Visualization import (  # noqa: E402
    PlotKDE,
    PlotMissing,
    PlotOutliers,
    PlotTargetBalance,
)

DefaultVisualizations = [PlotKDE, PlotMissing, PlotOutliers, PlotTargetBalance]
PostProcessingVisualizations = [PlotKDE, PlotOutliers, PlotTargetBalance]
