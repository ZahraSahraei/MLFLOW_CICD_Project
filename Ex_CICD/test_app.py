import pytest 
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_predict(client):
    response = client.post('/predict', json={"features":[0.02729,0,7.07,0.0,0.469,7.185,61.1,4.9671,2,242,17.8,392.83,4.03]})
    assert response.status_code == 200 
    assert "prediction" in response.get_json()
