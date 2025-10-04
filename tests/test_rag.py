import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "LangChain FastAPI Demo - See /docs for endpoints"}

# Note: For full tests, mock services and add auth headers
