const EventEmitter = require('events');
const logger = require('../../utils/logger');

/**
 * CQRS Command Handler
 * Handles write operations (Commands)
 */

class CommandBus extends EventEmitter {
  constructor() {
    super();
    this.handlers = new Map();
  }

  register(commandName, handler) {
    this.handlers.set(commandName, handler);
    logger.info(`Registered command handler: ${commandName}`);
  }

  async execute(command) {
    const handler = this.handlers.get(command.type);
    
    if (!handler) {
      throw new Error(`No handler registered for command: ${command.type}`);
    }

    logger.info(`Executing command: ${command.type}`, { commandId: command.id });

    try {
      const result = await handler.handle(command);
      
      // Emit event for event sourcing
      this.emit('command.executed', {
        command,
        result,
        timestamp: new Date()
      });

      return result;
    } catch (error) {
      logger.error(`Command execution failed: ${command.type}`, { error: error.message });
      
      this.emit('command.failed', {
        command,
        error,
        timestamp: new Date()
      });

      throw error;
    }
  }
}

// Command Handlers

class CreatePortfolioCommandHandler {
  constructor(portfolioRepository, eventStore) {
    this.portfolioRepository = portfolioRepository;
    this.eventStore = eventStore;
  }

  async handle(command) {
    const { userId, initialInvestment, riskProfile } = command.payload;

    // Create portfolio aggregate
    const portfolio = {
      id: command.id,
      userId,
      totalValue: initialInvestment,
      totalInvested: initialInvestment,
      riskProfile,
      holdings: [],
      createdAt: new Date(),
      version: 1
    };

    // Save to write database
    await this.portfolioRepository.create(portfolio);

    // Store event
    await this.eventStore.append({
      aggregateId: portfolio.id,
      aggregateType: 'Portfolio',
      eventType: 'PortfolioCreated',
      data: portfolio,
      version: 1,
      timestamp: new Date()
    });

    return { portfolioId: portfolio.id };
  }
}

class PlaceOrderCommandHandler {
  constructor(orderRepository, eventStore, kafkaProducer) {
    this.orderRepository = orderRepository;
    this.eventStore = eventStore;
    this.kafkaProducer = kafkaProducer;
  }

  async handle(command) {
    const { userId, fundId, amount, orderType } = command.payload;

    // Create order aggregate
    const order = {
      id: command.id,
      userId,
      fundId,
      amount,
      orderType,
      status: 'PENDING',
      createdAt: new Date(),
      version: 1
    };

    // Save to write database
    await this.orderRepository.create(order);

    // Store event
    await this.eventStore.append({
      aggregateId: order.id,
      aggregateType: 'Order',
      eventType: 'OrderPlaced',
      data: order,
      version: 1,
      timestamp: new Date()
    });

    // Publish event to Kafka for async processing
    await this.kafkaProducer.publishUserEvent(userId, 'ORDER_PLACED', order);

    return { orderId: order.id };
  }
}

class RebalancePortfolioCommandHandler {
  constructor(portfolioRepository, eventStore) {
    this.portfolioRepository = portfolioRepository;
    this.eventStore = eventStore;
  }

  async handle(command) {
    const { portfolioId, newWeights } = command.payload;

    // Load portfolio aggregate
    const portfolio = await this.portfolioRepository.findById(portfolioId);

    if (!portfolio) {
      throw new Error(`Portfolio not found: ${portfolioId}`);
    }

    // Update portfolio
    portfolio.holdings = newWeights;
    portfolio.version += 1;
    portfolio.lastRebalanced = new Date();

    // Save changes
    await this.portfolioRepository.update(portfolio);

    // Store event
    await this.eventStore.append({
      aggregateId: portfolio.id,
      aggregateType: 'Portfolio',
      eventType: 'PortfolioRebalanced',
      data: { newWeights },
      version: portfolio.version,
      timestamp: new Date()
    });

    return { success: true };
  }
}

// Event Store
class EventStore {
  constructor(mongoClient) {
    this.collection = mongoClient.db('sip_brewery').collection('events');
  }

  async append(event) {
    await this.collection.insertOne({
      ...event,
      _id: `${event.aggregateId}_${event.version}`,
      storedAt: new Date()
    });

    logger.info(`Event stored: ${event.eventType}`, { aggregateId: event.aggregateId });
  }

  async getEvents(aggregateId, fromVersion = 0) {
    const events = await this.collection
      .find({
        aggregateId,
        version: { $gt: fromVersion }
      })
      .sort({ version: 1 })
      .toArray();

    return events;
  }

  async getAllEvents(aggregateType, fromTimestamp) {
    const query = { aggregateType };
    
    if (fromTimestamp) {
      query.timestamp = { $gte: fromTimestamp };
    }

    return await this.collection
      .find(query)
      .sort({ timestamp: 1 })
      .toArray();
  }
}

// Repository (Write Model)
class PortfolioWriteRepository {
  constructor(mongoClient) {
    this.collection = mongoClient.db('sip_brewery_write').collection('portfolios');
  }

  async create(portfolio) {
    await this.collection.insertOne(portfolio);
  }

  async update(portfolio) {
    await this.collection.updateOne(
      { id: portfolio.id, version: portfolio.version - 1 },
      { $set: portfolio }
    );
  }

  async findById(id) {
    return await this.collection.findOne({ id });
  }
}

module.exports = {
  CommandBus,
  CreatePortfolioCommandHandler,
  PlaceOrderCommandHandler,
  RebalancePortfolioCommandHandler,
  EventStore,
  PortfolioWriteRepository
};
