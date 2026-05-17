from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_response_time():
    import time
    start = time.time()
    response = client.get("/health")
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 1.0


def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "model_type" in body
    assert "translation_backend" in body
