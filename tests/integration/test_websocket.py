"""
Tests for API-002: WebSocket Real-time Updates
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add the project root to the path so we can import from ipe
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipe.api.main import app  # noqa: E402

client = TestClient(app)


def test_websocket_health_endpoint():
    """Test the WebSocket health check endpoint."""
    response = client.get("/ws/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_connections" in data
    assert "timestamp" in data


def test_websocket_connection():
    """Test WebSocket connection establishment."""
    with client.websocket_connect("/ws/test-client") as websocket:
        # Should receive connection confirmation
        data = websocket.receive_json()
        assert data["type"] == "connection_established"
        assert "client_id" in data
        assert "reconnection_token" in data


def test_websocket_ping_pong():
    """Test WebSocket ping/pong functionality."""
    with client.websocket_connect("/ws/ping-test") as websocket:
        # Receive connection confirmation
        websocket.receive_json()

        # Send ping
        websocket.send_json({"type": "ping"})

        # Should receive pong
        response = websocket.receive_json()
        assert response["type"] == "pong"


def test_simulation_subscription():
    """Test simulation subscription mechanism."""
    with client.websocket_connect("/ws/sub-test") as websocket:
        # Receive connection confirmation
        websocket.receive_json()

        # Subscribe to a simulation
        websocket.send_json(
            {"type": "subscribe_simulation", "simulation_id": "test-sim-123"}
        )

        # Should receive subscription confirmation
        response = websocket.receive_json()
        assert response["type"] == "subscription_confirmed"
        assert response["simulation_id"] == "test-sim-123"


def test_simulation_unsubscription():
    """Test simulation unsubscription mechanism."""
    with client.websocket_connect("/ws/unsub-test") as websocket:
        # Receive connection confirmation
        websocket.receive_json()

        # Subscribe first
        websocket.send_json(
            {"type": "subscribe_simulation", "simulation_id": "test-sim-456"}
        )
        websocket.receive_json()  # subscription confirmation

        # Then unsubscribe
        websocket.send_json(
            {"type": "unsubscribe_simulation", "simulation_id": "test-sim-456"}
        )

        # Should receive unsubscription confirmation
        response = websocket.receive_json()
        assert response["type"] == "unsubscription_confirmed"
        assert response["simulation_id"] == "test-sim-456"


def test_unknown_message_type():
    """Test handling of unknown message types."""
    with client.websocket_connect("/ws/error-test") as websocket:
        # Receive connection confirmation
        websocket.receive_json()

        # Send unknown message type
        websocket.send_json({"type": "unknown_message_type", "data": "test"})

        # Should receive error response
        response = websocket.receive_json()
        assert response["type"] == "error"
        assert "unknown_message_type" in response["message"]


# Note: More comprehensive tests would require running the WebSocket server
# and client in separate processes to test reconnection and message queuing
