# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib
# from pathlib import Path

# # 1. Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# print("Dataset loaded successfully!")
# print("Features shape:", X.shape)
# print("Target shape:", y.shape)

# # 2. Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# print("\nTraining samples:", len(X_train))
# print("Testing samples:", len(X_test))

# # 3. Initialize and train the Random Forest model
# model = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42
# )

# model.fit(X_train, y_train)

# print("\nModel trained successfully!")

# # 4. Predict on the test dataset
# y_pred = model.predict(X_test)

# # 5. Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)

# print(f"\nModel Accuracy: {accuracy:.2%}")

# model_dir=Path("models")
# model_dir.mkdir(parents=True, exist_ok=True)
# model_path = model_dir / "iris.joblib"
# joblib.dump(model, model_path)
# print(f"\nModel saved to {model_path}")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
# Client API lets us manage registered models and their aliases
from mlflow import MlflowClient


#Configure MLFlow Tracking
mlflow.set_tracking_uri("http://localhost:7001") 

# Group related training runs under one experiment
mlflow.set_experiment("Iris Classification")

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Initialize and train the Random Forest model

n_estimators = 100
random_state = 42

model_parameters = {"model-type":"RandomForestClassifier",
                    "n_estimators":n_estimators,
                    "random_state":random_state,
                    "test-size":0.2
                }

# Start a tracked training run
with mlflow.start_run() as run:
    # Record the training configuration before model fitting
    mlflow.log_params(model_parameters)

    model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state
    )
    model.fit(X_train, y_train)

# print("\nModel trained successfully!")

# 4. Predict on the test dataset
    y_pred = model.predict(X_test)

# 5. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")

    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\nModel Precision: {precision:.2%}")

    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\nModel Recall: {recall:.2%}")

    # Model Quality Gate
    # Define the min accepatable test accuracy for now
    MIN_TEST_ACCURACY = 0.9
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError (
            f"Model rejected: test accuracy: {accuracy: .2f%}"
            f"is below the required {MIN_TEST_ACCURACY}"
        )
    print("Model Quality Gate Passed...")

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    model_dir=Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "iris.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Model registration
    registered_model_name = "iris-classifier"
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        registered_model_name=registered_model_name,
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    # Assign the champion alias to the newly registered model
    client = MlflowClient()
    # Identify current training run
    # We use this run id to find the model version registered by this particular run 
    current_run_id = mlflow.active_run().info.run_id
    # Retrieve all versions registered under this model name
    registered_versions = client.search_model_versions(
        "name='iris-classifier'"
    )
    # Find the version created by the current training run
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=current_model_version.version
    )

    print("Registered Model: iris-classifier")

