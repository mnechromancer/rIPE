/**
 * API-002: WebSocket Real-time Updates Client
 * 
 * TypeScript WebSocket client with reconnection handling,
 * simulation subscription management, and message queuing.
 */

interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

interface SimulationUpdate {
  type: 'simulation_update';
  simulation_id: string;
  data: any;
  timestamp: string;
}

interface StatusChange {
  type: 'simulation_status_change';
  simulation_id: string;
  status: string;
  details: any;
  timestamp: string;
}

interface StateSpaceUpdate {
  type: 'state_space_update';
  simulation_id: string;
  new_points: any[];
  timestamp: string;
}

type MessageHandler = (message: WebSocketMessage) => void;

export class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private clientId: string | null = null;
  private reconnectionToken: string | null = null;
  private messageHandlers: Map<string, MessageHandler[]> = new Map();
  private subscriptions: Set<string> = new Set();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;
  private pingInterval: number = 30000; // 30 seconds
  private pingTimer: number | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private connectionState: 'disconnected' | 'connecting' | 'connected' = 'disconnected';

  constructor(baseUrl: string = 'ws://localhost:8000') {
    this.url = `${baseUrl}/ws`;
  }

  /**
   * Connect to the WebSocket server
   */
  async connect(): Promise<string> {
    if (this.connectionState === 'connected') {
      return this.clientId!;
    }

    this.connectionState = 'connecting';
    
    return new Promise((resolve, reject) => {
      const wsUrl = this.clientId ? `${this.url}/${this.clientId}` : `${this.url}/new`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.connectionState = 'connected';
        this.reconnectAttempts = 0;
        this.startPinging();
        this.processMessageQueue();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);

          // Handle connection establishment
          if (message.type === 'connection_established') {
            this.clientId = message.client_id;
            this.reconnectionToken = message.reconnection_token;
            resolve(this.clientId);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket closed', event);
        this.connectionState = 'disconnected';
        this.stopPinging();
        
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
    this.stopPinging();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.connectionState = 'disconnected';
    this.subscriptions.clear();
  }

  /**
   * Subscribe to simulation updates
   */
  subscribeToSimulation(simulationId: string): void {
    this.subscriptions.add(simulationId);
    
    this.sendMessage({
      type: 'subscribe_simulation',
      simulation_id: simulationId
    });
  }

  /**
   * Unsubscribe from simulation updates
   */
  unsubscribeFromSimulation(simulationId: string): void {
    this.subscriptions.delete(simulationId);
    
    this.sendMessage({
      type: 'unsubscribe_simulation',
      simulation_id: simulationId
    });
  }

  /**
   * Add message handler for specific message types
   */
  onMessage(messageType: string, handler: MessageHandler): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType)!.push(handler);
  }

  /**
   * Remove message handler
   */
  offMessage(messageType: string, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(messageType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * Send a message to the server
   */
  private sendMessage(message: WebSocketMessage): void {
    if (this.connectionState === 'connected' && this.ws) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending message:', error);
        this.messageQueue.push(message);
      }
    } else {
      // Queue message for when connection is restored
      this.messageQueue.push(message);
    }
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(message: WebSocketMessage): void {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error('Error in message handler:', error);
        }
      });
    }

    // Handle specific message types
    switch (message.type) {
      case 'pong':
        // Heartbeat response
        break;
      case 'subscription_confirmed':
        console.log(`Subscribed to simulation ${message.simulation_id}`);
        break;
      case 'unsubscription_confirmed':
        console.log(`Unsubscribed from simulation ${message.simulation_id}`);
        break;
      case 'error':
        console.error('WebSocket error:', message.message);
        break;
    }
  }

  /**
   * Start periodic ping to keep connection alive
   */
  private startPinging(): void {
    this.pingTimer = window.setInterval(() => {
      this.sendMessage({ type: 'ping' });
    }, this.pingInterval);
  }

  /**
   * Stop periodic ping
   */
  private stopPinging(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(async () => {
      try {
        await this.connect();
        // Resubscribe to previous subscriptions
        this.subscriptions.forEach(simId => {
          this.subscribeToSimulation(simId);
        });
      } catch (error) {
        console.error('Reconnection failed:', error);
      }
    }, delay);
  }

  /**
   * Process queued messages after reconnection
   */
  private processMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
    }
  }

  /**
   * Get current connection state
   */
  getConnectionState(): 'disconnected' | 'connecting' | 'connected' {
    return this.connectionState;
  }

  /**
   * Get client ID
   */
  getClientId(): string | null {
    return this.clientId;
  }

  /**
   * Check if subscribed to a simulation
   */
  isSubscribedTo(simulationId: string): boolean {
    return this.subscriptions.has(simulationId);
  }
}

// Create and export singleton instance
export const webSocketService = new WebSocketService();

// Export typed message interfaces for use by components
export type {
  WebSocketMessage,
  SimulationUpdate,
  StatusChange,
  StateSpaceUpdate,
  MessageHandler
};
