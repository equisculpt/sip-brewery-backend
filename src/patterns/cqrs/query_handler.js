const logger = require('../../utils/logger');

/**
 * CQRS Query Handler
 * Handles read operations (Queries)
 * Uses denormalized read models for fast queries
 */

class QueryBus {
  constructor() {
    this.handlers = new Map();
  }

  register(queryName, handler) {
    this.handlers.set(queryName, handler);
    logger.info(`Registered query handler: ${queryName}`);
  }

  async execute(query) {
    const handler = this.handlers.get(query.type);
    
    if (!handler) {
      throw new Error(`No handler registered for query: ${query.type}`);
    }

    logger.debug(`Executing query: ${query.type}`);

    return await handler.handle(query);
  }
}

// Query Handlers

class GetPortfolioQueryHandler {
  constructor(portfolioReadRepository, cache) {
    this.repository = portfolioReadRepository;
    this.cache = cache;
  }

  async handle(query) {
    const { portfolioId } = query.payload;

    // Check cache first
    const cacheKey = `portfolio:${portfolioId}`;
    const cached = await this.cache.get(cacheKey);
    
    if (cached) {
      logger.debug(`Cache hit for portfolio: ${portfolioId}`);
      return JSON.parse(cached);
    }

    // Query read model
    const portfolio = await this.repository.findById(portfolioId);

    if (!portfolio) {
      throw new Error(`Portfolio not found: ${portfolioId}`);
    }

    // Cache result
    await this.cache.setex(cacheKey, 300, JSON.stringify(portfolio));

    return portfolio;
  }
}

class GetUserPortfoliosQueryHandler {
  constructor(portfolioReadRepository) {
    this.repository = portfolioReadRepository;
  }

  async handle(query) {
    const { userId, includePerformance = true } = query.payload;

    const portfolios = await this.repository.findByUserId(userId);

    if (includePerformance) {
      // Enrich with performance data
      for (const portfolio of portfolios) {
        portfolio.performance = await this.repository.getPerformance(portfolio.id);
      }
    }

    return portfolios;
  }
}

class GetTopPerformingFundsQueryHandler {
  constructor(fundReadRepository) {
    this.repository = fundReadRepository;
  }

  async handle(query) {
    const { timeframe = '1Y', limit = 10, category } = query.payload;

    return await this.repository.getTopPerformers({
      timeframe,
      limit,
      category
    });
  }
}

class SearchFundsQueryHandler {
  constructor(fundReadRepository, vectorDatabase) {
    this.repository = fundReadRepository;
    this.vectorDatabase = vectorDatabase;
  }

  async handle(query) {
    const { searchTerm, filters = {}, limit = 20 } = query.payload;

    // Use vector database for semantic search
    if (searchTerm) {
      const semanticResults = await this.vectorDatabase.searchFundsByQuery(
        searchTerm,
        limit,
        filters
      );
      
      // Enrich with full fund data
      const fundIds = semanticResults.map(r => r.fundId);
      const funds = await this.repository.findByIds(fundIds);
      
      return funds;
    }

    // Regular filtered search
    return await this.repository.search(filters, limit);
  }
}

// Read Model Repositories (Optimized for queries)

class PortfolioReadRepository {
  constructor(mongoClient, cache) {
    this.collection = mongoClient.db('sip_brewery_read').collection('portfolios');
    this.cache = cache;
  }

  async findById(id) {
    return await this.collection.findOne({ id });
  }

  async findByUserId(userId) {
    return await this.collection
      .find({ userId })
      .sort({ createdAt: -1 })
      .toArray();
  }

  async getPerformance(portfolioId) {
    // Query pre-calculated performance metrics
    const performance = await this.collection.findOne(
      { id: portfolioId },
      { projection: { performance: 1 } }
    );

    return performance?.performance || {};
  }

  async updateReadModel(event) {
    // Update read model based on event
    switch (event.eventType) {
      case 'PortfolioCreated':
        await this.collection.insertOne(event.data);
        break;
      
      case 'PortfolioRebalanced':
        await this.collection.updateOne(
          { id: event.aggregateId },
          { $set: { holdings: event.data.newWeights, lastRebalanced: event.timestamp } }
        );
        break;
      
      case 'PortfolioValueUpdated':
        await this.collection.updateOne(
          { id: event.aggregateId },
          { $set: { totalValue: event.data.newValue, performance: event.data.performance } }
        );
        break;
    }

    // Invalidate cache
    await this.cache.del(`portfolio:${event.aggregateId}`);
  }
}

class FundReadRepository {
  constructor(mongoClient) {
    this.collection = mongoClient.db('sip_brewery_read').collection('funds');
  }

  async findByIds(ids) {
    return await this.collection
      .find({ id: { $in: ids } })
      .toArray();
  }

  async getTopPerformers({ timeframe, limit, category }) {
    const query = category ? { category } : {};
    
    return await this.collection
      .find(query)
      .sort({ [`returns.${timeframe}`]: -1 })
      .limit(limit)
      .toArray();
  }

  async search(filters, limit) {
    const query = {};
    
    if (filters.category) query.category = filters.category;
    if (filters.riskLevel) query.riskLevel = filters.riskLevel;
    if (filters.minReturn) query['returns.1Y'] = { $gte: filters.minReturn };

    return await this.collection
      .find(query)
      .limit(limit)
      .toArray();
  }
}

// Event Projector (Updates read models from events)
class EventProjector {
  constructor(eventStore, readRepositories) {
    this.eventStore = eventStore;
    this.readRepositories = readRepositories;
    this.lastProcessedTimestamp = null;
  }

  async start() {
    logger.info('Starting event projector');

    // Process events continuously
    setInterval(async () => {
      await this.processNewEvents();
    }, 1000); // Check every second
  }

  async processNewEvents() {
    try {
      const events = await this.eventStore.getAllEvents(
        null,
        this.lastProcessedTimestamp
      );

      for (const event of events) {
        await this.projectEvent(event);
        this.lastProcessedTimestamp = event.timestamp;
      }

      if (events.length > 0) {
        logger.debug(`Projected ${events.length} events`);
      }
    } catch (error) {
      logger.error('Event projection failed', { error: error.message });
    }
  }

  async projectEvent(event) {
    // Route event to appropriate read repository
    switch (event.aggregateType) {
      case 'Portfolio':
        await this.readRepositories.portfolio.updateReadModel(event);
        break;
      
      case 'Order':
        // Update order read model
        break;
      
      case 'Fund':
        // Update fund read model
        break;
    }
  }

  async rebuild() {
    logger.info('Rebuilding read models from event store');

    // Clear read models
    // Replay all events
    const allEvents = await this.eventStore.getAllEvents();

    for (const event of allEvents) {
      await this.projectEvent(event);
    }

    logger.info(`Rebuilt read models from ${allEvents.length} events`);
  }
}

module.exports = {
  QueryBus,
  GetPortfolioQueryHandler,
  GetUserPortfoliosQueryHandler,
  GetTopPerformingFundsQueryHandler,
  SearchFundsQueryHandler,
  PortfolioReadRepository,
  FundReadRepository,
  EventProjector
};
