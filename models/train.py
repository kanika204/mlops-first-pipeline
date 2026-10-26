from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

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
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# 4. Predict on the test dataset
y_pred = model.predict(X_test)

# 5. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")

model_dir=Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "iris.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

