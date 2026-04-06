"""
Unit tests for the Flask inference API.
Run with:  pytest test_api.py -v
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    """Create a test Flask client with the model mocked out."""
    dummy_mask = np.zeros((64, 64), dtype=np.uint8)
    dummy_mask[10:30, 10:30] = 1   # small house patch

    with patch("app.load_model_if_needed", return_value=True), \
         patch("app.predict_mask", return_value=(dummy_mask, int(dummy_mask.sum()), 9.77)), \
         patch("app.fetch_image", return_value=Image.new("RGB", (64, 64))):
        import app as application
        application.app.config["TESTING"] = True
        with application.app.test_client() as c:
            yield c


# ── Health endpoint ───────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        data = json.loads(client.get("/health").data)
        assert data["status"] == "ok"

    def test_contains_device_field(self, client):
        data = json.loads(client.get("/health").data)
        assert "device" in data


# ── Predict endpoint ──────────────────────────────────────────────────────────
class TestPredict:
    VALID_PAYLOAD = {"image_url": "https://example.com/aerial.jpg"}

    def test_valid_request_returns_200(self, client):
        resp = client.post("/predict",
                           data=json.dumps(self.VALID_PAYLOAD),
                           content_type="application/json")
        assert resp.status_code == 200

    def test_response_contains_required_keys(self, client):
        resp = client.post("/predict",
                           data=json.dumps(self.VALID_PAYLOAD),
                           content_type="application/json")
        data = json.loads(resp.data)
        for key in ("input", "house_pixels", "coverage_percent", "mask_png_base64", "image_size"):
            assert key in data, f"Missing key: {key}"

    def test_coverage_is_numeric(self, client):
        resp = client.post("/predict",
                           data=json.dumps(self.VALID_PAYLOAD),
                           content_type="application/json")
        data = json.loads(resp.data)
        assert isinstance(data["coverage_percent"], (int, float))

    def test_mask_base64_is_string(self, client):
        resp = client.post("/predict",
                           data=json.dumps(self.VALID_PAYLOAD),
                           content_type="application/json")
        data = json.loads(resp.data)
        assert isinstance(data["mask_png_base64"], str)
        assert len(data["mask_png_base64"]) > 0

    def test_missing_image_url_returns_400(self, client):
        resp = client.post("/predict",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        resp = client.post("/predict",
                           data="",
                           content_type="application/json")
        assert resp.status_code == 400

    def test_input_echoed_in_response(self, client):
        resp = client.post("/predict",
                           data=json.dumps(self.VALID_PAYLOAD),
                           content_type="application/json")
        data = json.loads(resp.data)
        assert data["input"] == self.VALID_PAYLOAD["image_url"]
