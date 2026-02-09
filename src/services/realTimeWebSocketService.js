const WebSocket = require('ws');
const logger = require('../utils/logger');
const kafkaConsumer = require('./kafkaConsumerService');
const jwt = require('jsonwebtoken');

class RealTimeWebSocketService {
  constructor() {
    this.wss = null;
    this.clients = new Map(); // userId -> Set of WebSocket connections
    this.subscriptions = new Map(); // connectionId -> Set of topics
  }

  initialize(server) {
    this.wss = new WebSocket.Server({ 
      server,
      path: '/ws',
      verifyClient: this.verifyClient.bind(this)
    });

    this.wss.on('connection', this.handleConnection.bind(this));

    // Start consuming Kafka topics for broadcasting
    this.startKafkaConsumers();

    logger.info('WebSocket service initialized');
  }

  verifyClient(info, callback) {
    try {
      const token = new URL(info.req.url, 'http://localhost').searchParams.get('token');
      
      if (!token) {
        callback(false, 401, 'Unauthorized');
        return;
      }

      const JWT_SECRET = process.env.JWT_SECRET || process.env.JWT_PUBLIC_KEY?.replace(/\\n/g, '\n');
      const decoded = jwt.verify(token, JWT_SECRET);
      
      info.req.userId = decoded.sub || decoded.userId;
      callback(true);
    } catch (error) {
      logger.warn('WebSocket authentication failed', { error: error.message });
      callback(false, 401, 'Unauthorized');
    }
  }

  handleConnection(ws, req) {
    const userId = req.userId;
    const connectionId = this.generateConnectionId();

    // Store connection
    if (!this.clients.has(userId)) {
      this.clients.set(userId, new Set());
    }
    this.clients.get(userId).add(ws);

    // Initialize subscriptions
    this.subscriptions.set(connectionId, new Set());

    ws.connectionId = connectionId;
    ws.userId = userId;
    ws.isAlive = true;

    // Send welcome message
    this.sendToClient(ws, {
      type: 'CONNECTED',
      message: 'WebSocket connection established',
      connectionId,
      timestamp: new Date().toISOString()
    });

    // Handle messages from client
    ws.on('message', (message) => this.handleMessage(ws, message));

    // Handle pong (heartbeat)
    ws.on('pong', () => {
      ws.isAlive = true;
    });

    // Handle close
    ws.on('close', () => this.handleDisconnect(ws));

    // Handle errors
    ws.on('error', (error) => {
      logger.error('WebSocket error', { 
        userId, 
        connectionId, 
        error: error.message 
      });
    });

    logger.info('WebSocket client connected', { userId, connectionId });
  }

  handleMessage(ws, message) {
    try {
      const data = JSON.parse(message.toString());

      switch (data.type) {
        case 'SUBSCRIBE':
          this.handleSubscribe(ws, data.topics);
          break;
        case 'UNSUBSCRIBE':
          this.handleUnsubscribe(ws, data.topics);
          break;
        case 'PING':
          this.sendToClient(ws, { type: 'PONG', timestamp: Date.now() });
          break;
        default:
          logger.warn('Unknown message type', { type: data.type });
      }
    } catch (error) {
      logger.error('Failed to handle WebSocket message', { 
        error: error.message 
      });
    }
  }

  handleSubscribe(ws, topics) {
    const subscriptions = this.subscriptions.get(ws.connectionId);
    
    topics.forEach(topic => {
      subscriptions.add(topic);
    });

    this.sendToClient(ws, {
      type: 'SUBSCRIBED',
      topics: Array.from(subscriptions),
      timestamp: new Date().toISOString()
    });

    logger.debug('Client subscribed to topics', { 
      userId: ws.userId, 
      topics 
    });
  }

  handleUnsubscribe(ws, topics) {
    const subscriptions = this.subscriptions.get(ws.connectionId);
    
    topics.forEach(topic => {
      subscriptions.delete(topic);
    });

    this.sendToClient(ws, {
      type: 'UNSUBSCRIBED',
      topics,
      timestamp: new Date().toISOString()
    });
  }

  handleDisconnect(ws) {
    const userId = ws.userId;
    const connectionId = ws.connectionId;

    // Remove from clients
    if (this.clients.has(userId)) {
      this.clients.get(userId).delete(ws);
      if (this.clients.get(userId).size === 0) {
        this.clients.delete(userId);
      }
    }

    // Remove subscriptions
    this.subscriptions.delete(connectionId);

    logger.info('WebSocket client disconnected', { userId, connectionId });
  }

  async startKafkaConsumers() {
    // Consumer for market data
    await kafkaConsumer.createConsumer(
      'websocket-market-data',
      'market-data',
      (message) => this.broadcastMarketData(message.value)
    );

    // Consumer for ML predictions
    await kafkaConsumer.createConsumer(
      'websocket-ml-predictions',
      'ml-predictions',
      (message) => this.broadcastMLPrediction(message.value)
    );

    // Consumer for risk alerts
    await kafkaConsumer.createConsumer(
      'websocket-risk-alerts',
      'risk-alerts',
      (message) => this.broadcastRiskAlert(message.value)
    );

    logger.info('WebSocket Kafka consumers started');
  }

  broadcastMarketData(data) {
    this.broadcast('MARKET_UPDATE', data, 'market-data');
  }

  broadcastMLPrediction(data) {
    // Send to specific user
    if (data.userId) {
      this.sendToUser(data.userId, {
        type: 'ML_PREDICTION',
        data,
        timestamp: new Date().toISOString()
      });
    }
  }

  broadcastRiskAlert(data) {
    // Send to specific user with high priority
    if (data.userId) {
      this.sendToUser(data.userId, {
        type: 'RISK_ALERT',
        data,
        priority: data.severity,
        timestamp: new Date().toISOString()
      });
    }
  }

  broadcast(type, data, topic = null) {
    const message = {
      type,
      data,
      timestamp: new Date().toISOString()
    };

    let sentCount = 0;

    this.wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        // Check if client is subscribed to this topic
        if (!topic || this.subscriptions.get(client.connectionId)?.has(topic)) {
          this.sendToClient(client, message);
          sentCount++;
        }
      }
    });

    logger.debug('Broadcast message sent', { type, topic, sentCount });
  }

  sendToUser(userId, message) {
    const userConnections = this.clients.get(userId);
    
    if (userConnections) {
      userConnections.forEach(ws => {
        if (ws.readyState === WebSocket.OPEN) {
          this.sendToClient(ws, message);
        }
      });
    }
  }

  sendToClient(ws, message) {
    try {
      ws.send(JSON.stringify(message));
    } catch (error) {
      logger.error('Failed to send message to client', { 
        error: error.message 
      });
    }
  }

  startHeartbeat() {
    setInterval(() => {
      this.wss.clients.forEach(ws => {
        if (!ws.isAlive) {
          logger.warn('Terminating inactive WebSocket connection', { 
            userId: ws.userId 
          });
          return ws.terminate();
        }

        ws.isAlive = false;
        ws.ping();
      });
    }, 30000); // 30 seconds
  }

  generateConnectionId() {
    return `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getMetrics() {
    return {
      total_connections: this.wss ? this.wss.clients.size : 0,
      unique_users: this.clients.size,
      total_subscriptions: Array.from(this.subscriptions.values())
        .reduce((sum, subs) => sum + subs.size, 0)
    };
  }

  async shutdown() {
    if (this.wss) {
      this.wss.clients.forEach(client => {
        client.close(1000, 'Server shutting down');
      });
      this.wss.close();
    }

    await kafkaConsumer.stopAll();
    logger.info('WebSocket service shut down');
  }
}

module.exports = new RealTimeWebSocketService();
