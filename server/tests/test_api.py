"""HTTP tests proving the explicitly mocked server surface."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "mode": "schema_scaffold",
        "inference": False,
    }


def test_create_response_minimal():
    resp = client.post("/v1/responses", json={"model": "gpt-4o"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["model"] == "gpt-4o"
    assert data["id"].startswith("resp_")
    assert data["status"] == "completed"
    assert data["output"][0]["content"][0]["text"].startswith(
        "CKE schema scaffold:"
    )
    assert "output_items" not in data


def test_create_response_with_input():
    resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "Hello, world!", "temperature": 0.5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "gpt-4o"
    assert data["id"].startswith("resp_")


def test_get_response():
    create_resp = client.post("/v1/responses", json={"model": "gpt-4o"})
    resp_id = create_resp.json()["id"]

    get_resp = client.get(f"/v1/responses/{resp_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == resp_id


def test_get_response_not_found():
    resp = client.get("/v1/responses/nonexistent")
    assert resp.status_code == 404


def test_cancel_response():
    # Mock store creates responses as completed; flip to in_progress to test successful cancel.
    from server.routes.responses import _response_store
    from server.schemas.common import ResponseStatus

    create_resp = client.post("/v1/responses", json={"model": "gpt-4o"})
    resp_id = create_resp.json()["id"]
    _response_store[resp_id].status = ResponseStatus.in_progress

    cancel_resp = client.post(f"/v1/responses/{resp_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


def test_cancel_completed_is_rejected():
    create_resp = client.post("/v1/responses", json={"model": "gpt-4o"})
    resp_id = create_resp.json()["id"]
    # Stored status is completed → cancel should be 409
    cancel_resp = client.post(f"/v1/responses/{resp_id}/cancel")
    assert cancel_resp.status_code == 409


def test_cancel_response_not_found():
    resp = client.post("/v1/responses/nonexistent/cancel")
    assert resp.status_code == 404


def test_stream_response():
    stream_resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "stream": True},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]

    text = stream_resp.text
    assert "response.created" in text
    assert "response.in_progress" in text
    assert "response.output_text.delta" in text
    assert "response.completed" in text


def test_nonstandard_stream_endpoint_is_not_exposed():
    create_resp = client.post("/v1/responses", json={"model": "gpt-4o"})
    resp_id = create_resp.json()["id"]
    assert client.post(f"/v1/responses/{resp_id}/stream").status_code == 404


def test_create_conversation():
    resp = client.post("/v1/conversations", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "conversation"
    assert data["id"].startswith("conv_")


def test_get_conversation():
    create_resp = client.post("/v1/conversations", json={})
    conv_id = create_resp.json()["id"]

    get_resp = client.get(f"/v1/conversations/{conv_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == conv_id


def test_get_conversation_not_found():
    resp = client.get("/v1/conversations/nonexistent")
    assert resp.status_code == 404


def test_add_conversation_items():
    create_resp = client.post("/v1/conversations", json={})
    conv_id = create_resp.json()["id"]

    add_resp = client.post(
        f"/v1/conversations/{conv_id}/items",
        json={"items": ["item1", "item2"]},
    )
    assert add_resp.status_code == 200
    assert add_resp.json()["items"] == ["item1", "item2"]


def test_add_items_to_nonexistent_conversation():
    resp = client.post(
        "/v1/conversations/nonexistent/items",
        json={"items": ["item1"]},
    )
    assert resp.status_code == 404


def test_create_response_with_tools():
    resp = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                }
            ],
        },
    )
    assert resp.status_code == 200


def test_create_response_with_web_search_tool():
    resp = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4o",
            "tools": [{"type": "web_search"}],
        },
    )
    assert resp.status_code == 200


def test_create_response_with_file_search_tool():
    resp = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_abc123"],
                }
            ],
        },
    )
    assert resp.status_code == 200


def test_conversation_with_items():
    resp = client.post(
        "/v1/conversations",
        json={
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
