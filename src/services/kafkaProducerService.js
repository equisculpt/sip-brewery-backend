const { Kafka } = require('kafkajs');
const logger = require('../utils/logger');

class KafkaProducerService {
  constructor() {
    this.kafka = new Kafka({
      clientId: 'sip-brewery-backend',
      brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
      retry: {
        initialRetryTime: 100,
        retries: 8
      }
    });

    this.producer = this.kafka.producer({
      allowAutoTopicCreation: true,
      transactionTimeout: 30000
    });

    this.isConnected = false;
    this.topics = {
      MARKET_DATA: 'market-data',
      USER_EVENTS: 'user-events',
      FUND_UPDATES: 'fund-updates',
      PORTFOLIO_CHANGES: 'portfolio-changes',
      ML_PREDICTIONS: 'ml-predictions',
      RISK_ALERTS: 'risk-alerts'
    };
  }

  async connect() {
    try {
      await this.producer.connect();
      this.isConnected = true;
      logger.info('Kafka producer connected successfully');
    } catch (error) {
      logger.error('Failed to connect Kafka producer', { error: error.message });
      throw error;
    }
  }

  async disconnect() {
    try {
      await this.producer.disconnect();
      this.isConnected = false;
      logger.info('Kafka producer disconnected');
    } catch (error) {
      logger.error('Failed to disconnect Kafka producer', { error: error.message });
    }
  }

  async sendMessage(topic, message, key = null) {
    if (!this.isConnected) {
      await this.connect();
    }

    try {
      const result = await this.producer.send({
        topic,
        messages: [
          {
            key: key,
            value: JSON.stringify(message),
            timestamp: Date.now().toString()
          }
        ]
      });

      logger.debug('Message sent to Kafka', { topic, partition: result[0].partition });
      return result;
    } catch (error) {
      logger.error('Failed to send message to Kafka', { 
        topic, 
        error: error.message 
      });
      throw error;
    }
  }

  async sendBatch(topic, messages) {
    if (!this.isConnected) {
      await this.connect();
    }

    try {
      const kafkaMessages = messages.map(msg => ({
        key: msg.key || null,
        value: JSON.stringify(msg.value),
        timestamp: Date.now().toString()
      }));

      const result = await this.producer.send({
        topic,
        messages: kafkaMessages
      });

      logger.info('Batch messages sent to Kafka', { 
        topic, 
        count: messages.length 
      });
      return result;
    } catch (error) {
      logger.error('Failed to send batch to Kafka', { 
        topic, 
        error: error.message 
      });
      throw error;
    }
  }

  // Domain-specific methods

  async publishMarketData(data) {
    return this.sendMessage(this.topics.MARKET_DATA, {
      type: 'MARKET_UPDATE',
      timestamp: new Date().toISOString(),
      data
    }, data.symbol);
  }

  async publishUserEvent(userId, eventType, eventData) {
    return this.sendMessage(this.topics.USER_EVENTS, {
      userId,
      eventType,
      eventData,
      timestamp: new Date().toISOString()
    }, userId);
  }

  async publishFundUpdate(fundId, updateType, updateData) {
    return this.sendMessage(this.topics.FUND_UPDATES, {
      fundId,
      updateType,
      updateData,
      timestamp: new Date().toISOString()
    }, fundId);
  }

  async publishPortfolioChange(userId, changeType, changeData) {
    return this.sendMessage(this.topics.PORTFOLIO_CHANGES, {
      userId,
      changeType,
      changeData,
      timestamp: new Date().toISOString()
    }, userId);
  }

  async publishMLPrediction(modelType, userId, prediction) {
    return this.sendMessage(this.topics.ML_PREDICTIONS, {
      modelType,
      userId,
      prediction,
      timestamp: new Date().toISOString()
    }, userId);
  }

  async publishRiskAlert(userId, alertType, alertData) {
    return this.sendMessage(this.topics.RISK_ALERTS, {
      userId,
      alertType,
      alertData,
      severity: alertData.severity || 'MEDIUM',
      timestamp: new Date().toISOString()
    }, userId);
  }

  async sendTransaction(messages) {
    if (!this.isConnected) {
      await this.connect();
    }

    const transaction = await this.producer.transaction();
    
    try {
      await transaction.send({
        topic: this.topics.USER_EVENTS,
        messages: messages.map(msg => ({
          value: JSON.stringify(msg)
        }))
      });

      await transaction.commit();
      logger.info('Transaction committed successfully');
    } catch (error) {
      await transaction.abort();
      logger.error('Transaction aborted', { error: error.message });
      throw error;
    }
  }

  getMetrics() {
    return {
      isConnected: this.isConnected,
      topics: Object.keys(this.topics).length
    };
  }
}

module.exports = new KafkaProducerService();
