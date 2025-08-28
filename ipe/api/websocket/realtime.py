"""
API-002: WebSocket Real-time Updates

WebSocket connection management, real-time simulation updates,
client reconnection handling, and message queuing for reliability.
"""

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.routing import APIRouter
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
import uuid
from datetime import datetime
from queue import Queue
import threading

logger = logging.getLogger(__name__)

# WebSocket router
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.simulation_subscribers: Dict[str, List[str]] = {}  # sim_id -> [client_ids]
        self.message_queues: Dict[str, Queue] = {}  # client_id -> message queue
        self.reconnection_tokens: Dict[str, str] = {}  # token -> client_id

    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept WebSocket connection and assign client ID."""
        await websocket.accept()

        if not client_id:
            client_id = str(uuid.uuid4())

        self.active_connections[client_id] = websocket
        self.message_queues[client_id] = Queue()

        # Generate reconnection token
        reconnection_token = str(uuid.uuid4())
        self.reconnection_tokens[reconnection_token] = client_id

        logger.info(f"WebSocket client {client_id} connected")

        # Send connection confirmation
        await self.send_personal_message(
            {
                "type": "connection_established",
                "client_id": client_id,
                "reconnection_token": reconnection_token,
                "timestamp": datetime.now().isoformat(),
            },
            client_id,
        )

        return client_id

    async def disconnect(self, client_id: str):
        """Handle client disconnection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        # Remove from simulation subscriptions
        for sim_id in list(self.simulation_subscribers.keys()):
            if client_id in self.simulation_subscribers[sim_id]:
                self.simulation_subscribers[sim_id].remove(client_id)
                if not self.simulation_subscribers[sim_id]:
                    del self.simulation_subscribers[sim_id]

        # Clean up message queue
        if client_id in self.message_queues:
            del self.message_queues[client_id]

        logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                await self.disconnect(client_id)
        else:
            # Queue message for when client reconnects
            if client_id in self.message_queues:
                self.message_queues[client_id].put(message)

    async def broadcast_to_simulation(
        self, simulation_id: str, message: Dict[str, Any]
    ):
        """Broadcast message to all clients subscribed to a simulation."""
        if simulation_id in self.simulation_subscribers:
            for client_id in self.simulation_subscribers[simulation_id].copy():
                await self.send_personal_message(message, client_id)

    def subscribe_to_simulation(self, client_id: str, simulation_id: str):
        """Subscribe client to simulation updates."""
        if simulation_id not in self.simulation_subscribers:
            self.simulation_subscribers[simulation_id] = []

        if client_id not in self.simulation_subscribers[simulation_id]:
            self.simulation_subscribers[simulation_id].append(client_id)
            logger.info(f"Client {client_id} subscribed to simulation {simulation_id}")

    def unsubscribe_from_simulation(self, client_id: str, simulation_id: str):
        """Unsubscribe client from simulation updates."""
        if simulation_id in self.simulation_subscribers:
            if client_id in self.simulation_subscribers[simulation_id]:
                self.simulation_subscribers[simulation_id].remove(client_id)
                logger.info(
                    f"Client {client_id} unsubscribed from simulation {simulation_id}"
                )

    async def send_queued_messages(self, client_id: str):
        """Send any queued messages to recently reconnected client."""
        if client_id in self.message_queues:
            while not self.message_queues[client_id].empty():
                message = self.message_queues[client_id].get()
                await self.send_personal_message(message, client_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """WebSocket endpoint for real-time updates."""
    client_id = await manager.connect(websocket, client_id)

    try:
        # Send any queued messages
        await manager.send_queued_messages(client_id)

        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            await handle_client_message(client_id, message)

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.disconnect(client_id)


async def handle_client_message(client_id: str, message: Dict[str, Any]):
    """Handle incoming messages from clients."""
    message_type = message.get("type")

    if message_type == "subscribe_simulation":
        simulation_id = message.get("simulation_id")
        if simulation_id:
            manager.subscribe_to_simulation(client_id, simulation_id)
            await manager.send_personal_message(
                {
                    "type": "subscription_confirmed",
                    "simulation_id": simulation_id,
                    "timestamp": datetime.now().isoformat(),
                },
                client_id,
            )

    elif message_type == "unsubscribe_simulation":
        simulation_id = message.get("simulation_id")
        if simulation_id:
            manager.unsubscribe_from_simulation(client_id, simulation_id)
            await manager.send_personal_message(
                {
                    "type": "unsubscription_confirmed",
                    "simulation_id": simulation_id,
                    "timestamp": datetime.now().isoformat(),
                },
                client_id,
            )

    elif message_type == "ping":
        await manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.now().isoformat()}, client_id
        )

    else:
        await manager.send_personal_message(
            {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat(),
            },
            client_id,
        )


# Simulation update functions (to be called by simulation engine)
async def send_simulation_update(simulation_id: str, update_data: Dict[str, Any]):
    """Send real-time update for a simulation."""
    message = {
        "type": "simulation_update",
        "simulation_id": simulation_id,
        "data": update_data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast_to_simulation(simulation_id, message)


async def send_simulation_status_change(
    simulation_id: str, status: str, details: Dict[str, Any] = None
):
    """Send simulation status change notification."""
    message = {
        "type": "simulation_status_change",
        "simulation_id": simulation_id,
        "status": status,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast_to_simulation(simulation_id, message)


async def send_state_space_update(simulation_id: str, new_points: List[Dict[str, Any]]):
    """Send state space points update."""
    message = {
        "type": "state_space_update",
        "simulation_id": simulation_id,
        "new_points": new_points,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast_to_simulation(simulation_id, message)


# Health check endpoint
@router.get("/ws/health")
async def websocket_health():
    """WebSocket service health check."""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "total_subscriptions": sum(
            len(subs) for subs in manager.simulation_subscribers.values()
        ),
        "timestamp": datetime.now().isoformat(),
    }
