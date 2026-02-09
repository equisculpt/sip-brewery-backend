const { Kafka } = require('kafkajs');
const logger = require('../utils/logger');

class KafkaConsumerService {
  constructor() {
    this.kafka = new Kafka({
      clientId: 'sip-brewery-backend',
      brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
      retry: {
        initialRetryTime: 100,
        retries: 8
      }
    });

    this.consumers = new Map();
    this.messageHandlers = new Map();
  }

  async createConsumer(groupId, topics, handler) {
    const consumer = this.kafka.consumer({ 
      groupId,
      sessionTimeout: 30000,
      heartbeatInterval: 3000
    });

    try {
      await consumer.connect();
      await consumer.subscribe({ 
        topics: Array.isArray(topics) ? topics : [topics],
        fromBeginning: false 
      });

      this.consumers.set(groupId, consumer);
      this.messageHandlers.set(groupId, handler);

      await consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
          try {
            const value = JSON.parse(message.value.toString());
            const key = message.key ? message.key.toString() : null;

            await handler({
              topic,
              partition,
              offset: message.offset,
              key,
              value,
              timestamp: message.timestamp
            });

            logger.debug('Message processed', { topic, partition, offset: message.offset });
          } catch (error) {
            logger.error('Error processing message', {
              topic,
              partition,
              offset: message.offset,
              error: error.message
            });
          }
        }
      });

      logger.info('Kafka consumer started', { groupId, topics });
      return consumer;
    } catch (error) {
      logger.error('Failed to create Kafka consumer', { 
        groupId, 
        error: error.message 
      });
      throw error;
    }
  }

  async stopConsumer(groupId) {
    const consumer = this.consumers.get(groupId);
    if (consumer) {
      try {
        await consumer.disconnect();
        this.consumers.delete(groupId);
        this.messageHandlers.delete(groupId);
        logger.info('Kafka consumer stopped', { groupId });
      } catch (error) {
        logger.error('Failed to stop Kafka consumer', { 
          groupId, 
          error: error.message 
        });
      }
    }
  }

  async stopAll() {
    const promises = Array.from(this.consumers.keys()).map(groupId => 
      this.stopConsumer(groupId)
    );
    await Promise.all(promises);
    logger.info('All Kafka consumers stopped');
  }

  getMetrics() {
    return {
      activeConsumers: this.consumers.size,
      consumerGroups: Array.from(this.consumers.keys())
    };
  }
}

module.exports = new KafkaConsumerService();
