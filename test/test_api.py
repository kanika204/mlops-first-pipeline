# Import TestClient so we can send simulated HTTP requests directly to FastAPI
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

# Create a reusable test client for all test functions
client = TestClient(app)

def test_health_endpoint():
    """
    Verify that the application is running and can respond to a health-check request
    """
    response = client.get("/health")
    # A successful HTTP response should return status code 200
    assert response.status_code == 200
    # Now check response body
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction():
    """
    Verify that valid flower measurements produce a prediction
    """
    # Define a valid flower measurement
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Convert the JSON response into a Python dictionary
    result = response.json()
    # Check the response contract: both fields must exist
    assert "prediction" in result
    assert "class_name" in result

    # The predicted class must be one of the three Iris classes
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    # The predicted integer must be between 0 and 2
    assert result["prediction"] in [0, 1, 2]

def test_invalid_iris_input():
    """
    Verify that invalid flower measurements are handled correctly
    """
    # Define an invalid flower measurement
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)
    # FastAPI normally returns 422 for validation error
    assert response.status_code == 422