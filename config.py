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


from catboost import CatBoostClassifier  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

Models = [
    {
        "instance": GradientBoostingClassifier(),
        "gridSearchParams": {"param_grid": {"learning_rate": [0.1, 0.05, 0.01]}},
    },
    {"instance": RandomForestClassifier(n_estimators=100)},
    {
        "instance": CatBoostClassifier(verbose=0),
        "gridSearchParams": {"param_grid": {"learning_rate": [0.1, 0.05, 0.01]}},
    },
    {"instance": LogisticRegression(max_iter=1000), "scaleValues": True},
    {"instance": LinearSVC(max_iter=1000), "scaleValues": True},
]

# Scoring names: https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
GridSearchScoringMethod = "accuracy"
GridSearchCVValue = 5
GridSearchNJobs = 8


from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

EvaluationMetrics = [
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
]


from src.steps.Visualization import (  # noqa: E402
    PlotKDE,
    PlotMissing,
    PlotOutliers,
    PlotTargetBalance,
    PlotFeatureCorrelation
)

DefaultVisualizations = [PlotKDE, PlotMissing, PlotOutliers, PlotTargetBalance, PlotFeatureCorrelation]
PostProcessingVisualizations = [PlotKDE, PlotOutliers, PlotTargetBalance, PlotFeatureCorrelation]
