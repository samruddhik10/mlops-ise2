from app import app


def test_home():
    response = app.test_client().get("/")
    assert response.status_code == 200


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_prediction():
    response = app.test_client().post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 200
    assert "predicted_species" in response.json


def test_invalid_input():
    response = app.test_client().post(
        "/predict",
        json={}
    )
    assert response.status_code == 400