import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import uuid
import mlflow
from mlflow.models import infer_signature

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

X = pd.read_csv("data/x.csv", index_col=[0])
y = pd.read_csv("data/y.csv", index_col=[0])

(X_train, X_test, y_train, y_test) = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    train_size=100,
)


def training_vanilla():
    """Very basic training, no cross validation or any other technique: train and print the score."""
    params = {
        "criterion": "entropy",
        "min_samples_split": 4,
        "min_impurity_decrease": 0.02,
    }

    decission_tree_model = DecisionTreeClassifier(**params)
    decission_tree_model = decission_tree_model.fit(X_train, y_train)
    prediction = decission_tree_model.predict(X_test)

    print(decission_tree_model.score(X_test, y_test))

    # Define parameters
    params = {
        "n_estimators": 10,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    # Instantiate the RandomForestClassifier model with parameters
    random_forest_model = RandomForestClassifier(**params)

    # Train the model
    random_forest_model.fit(X_train, y_train)

    # Predict
    predictions = random_forest_model.predict(X_test)

    # Evaluate your model
    score_train = random_forest_model.score(
        X_train, y_train.values.ravel()
    )  # .values will give the values in a numpy array (shape: (n,1))
    # .ravel will convert that array shape to (n, ) (i.e. flatten it)
    print(score_train)


# -----------------------------------------------------------------------------
def register_model_mlflow(
    run_name,
    params,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    tagging_info,
):
    """We are going to use this function to log different elements of our training into MLFlow."""
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Instantiate the Ridge model with parameters
        model = model(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        score = model.score(X_train, y_train.values.ravel())

        # Create a table for log
        X_test_pred = X_test.copy()
        X_test_pred["ground_truth"] = y_test
        X_test_pred["predictions"] = predictions
        mlflow.log_table(data=X_test_pred, artifact_file="val.json")

        # In this case we only are going to log the score, but we can log more info if necessary
        metric_eval = {
            "score": score,
        }

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the desired metric
        mlflow.log_metrics(metric_eval)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", tagging_info)

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="PythonBCN_artifact",
            signature=signature,
            input_example=X_train,
            registered_model_name=run_name,
        )

        return model, metric_eval


def training_mlflow():
    """Just train the models using the previous function to log the info."""
    mlflow.set_tracking_uri(uri="http://localhost:8080")
    mlflow.set_experiment("Python Barcelona 2024 Workshop")

    # Lets register the Decission Tree
    model, metric_eval = register_model_mlflow(
        run_name=f"decission_tree_classifier_{uuid.uuid4()}",
        params={
            "criterion": "entropy",
            "min_samples_split": 4,
            "min_impurity_decrease": 0.02,
        },
        model=DecisionTreeClassifier,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tagging_info="DT classifier",
    )

    # Lets register the Random Forest
    model, metric_eval = register_model_mlflow(
        run_name=f"random_forest_classifier_{uuid.uuid4()}",
        params={
            "n_estimators": 14,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        model=RandomForestClassifier,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tagging_info="RF classifier",
    )


def main():
    # training_vanilla()
    training_mlflow()


if __name__ == "__main__":
    main()
