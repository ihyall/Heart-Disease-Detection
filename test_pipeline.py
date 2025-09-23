import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import from_pandas
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

from config import VisualizationPath

# Тренировочный код для продумывания проекта

df = pd.read_csv("./data/heart_disease_health_indicators_BRFSS2015.csv")

datasetSourceURL = (
    "https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset"
)
datasetSource = HTTPDatasetSource(url=datasetSourceURL)
dataset = from_pandas(
    df=df,
    source=datasetSource,
    targets="HeartDiseaseorAttack",
    name="Heart Disease Health Indicators Dataset",
)

X_train, X_test, y_train, y_test = train_test_split(
    dataset.df.drop(columns=["HeartDiseaseorAttack"]),
    dataset.df["HeartDiseaseorAttack"],
    test_size=0.2,
    random_state=42,
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run():
    # mlflow.sklearn.autolog()
    mlflow.log_input(dataset=dataset)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    # Сохранение графика confusion matrix как артефакта
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(VisualizationPath + "confusion_matrix.png")
    mlflow.log_artifact(VisualizationPath + "confusion_matrix.png")

    # Логирование модели
    mlflow.sklearn.log_model(model, "model")

mlflow.end_run()
