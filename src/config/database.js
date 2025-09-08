/**
 * 🗄️ OPTIMIZED DATABASE CONFIGURATION
 * 
 * High-performance MongoDB configuration for 100,000+ users
 */

const mongoose = require('mongoose');
const logger = require('../utils/logger');

// Connection pool optimization
const connectionOptions = {
  // Connection pool settings
  maxPoolSize: 50, // Maximum number of connections
  minPoolSize: 5,  // Minimum number of connections
  maxIdleTimeMS: 30000, // Close connections after 30 seconds of inactivity
  
  // Performance settings
  serverSelectionTimeoutMS: 5000, // How long to try selecting a server
  socketTimeoutMS: 45000, // How long a send or receive on a socket can take
  bufferMaxEntries: 0, // Disable mongoose buffering
  bufferCommands: false, // Disable mongoose buffering
  
  // Replica set settings
  readPreference: 'secondaryPreferred', // Read from secondary when possible
  readConcern: { level: 'majority' },
  writeConcern: { w: 'majority', j: true },
  
  // Connection management
  heartbeatFrequencyMS: 10000, // How often to check server status
  retryWrites: true,
  retryReads: true,
  
  // Compression
  compressors: ['zlib'],
  zlibCompressionLevel: 6
};

// Database indexes for performance
const setupIndexes = async () => {
  try {
    const db = mongoose.connection.db;
    
    // User collection indexes
    await db.collection('users').createIndex({ email: 1 }, { unique: true });
    await db.collection('users').createIndex({ 'profile.pan': 1 }, { sparse: true });
    await db.collection('users').createIndex({ createdAt: 1 });
    await db.collection('users').createIndex({ 'subscription.type': 1 });
    
    // Portfolio collection indexes
    await db.collection('portfolios').createIndex({ userId: 1 });
    await db.collection('portfolios').createIndex({ 'holdings.fundCode': 1 });
    await db.collection('portfolios').createIndex({ lastUpdated: -1 });
    
    // Transaction collection indexes
    await db.collection('transactions').createIndex({ userId: 1, createdAt: -1 });
    await db.collection('transactions').createIndex({ 'fund.code': 1 });
    await db.collection('transactions').createIndex({ type: 1, status: 1 });
    
    // Fund data indexes
    await db.collection('funds').createIndex({ code: 1 }, { unique: true });
    await db.collection('funds').createIndex({ category: 1 });
    await db.collection('funds').createIndex({ 'performance.returns.1Y': -1 });
    
    // Market data indexes
    await db.collection('marketdata').createIndex({ symbol: 1, date: -1 });
    await db.collection('marketdata').createIndex({ date: -1 });
    
    // Corporate actions indexes
    await db.collection('corporateactions').createIndex({ symbol: 1, date: -1 });
    await db.collection('corporateactions').createIndex({ type: 1, priority: 1 });
    
    logger.info('✅ Database indexes created successfully');
  } catch (error) {
    logger.error('❌ Failed to create database indexes:', error);
  }
};

// Connection with retry logic
const connectWithRetry = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI, connectionOptions);
    logger.info('✅ Database connected successfully');
    
    // Setup indexes after connection
    await setupIndexes();
    
    // Setup connection event handlers
    mongoose.connection.on('error', (error) => {
      logger.error('Database connection error:', error);
    });
    
    mongoose.connection.on('disconnected', () => {
      logger.warn('Database disconnected. Attempting to reconnect...');
      setTimeout(connectWithRetry, 5000);
    });
    
    mongoose.connection.on('reconnected', () => {
      logger.info('Database reconnected successfully');
    });
    
  } catch (error) {
    logger.error('Database connection failed:', error);
    setTimeout(connectWithRetry, 5000);
  }
};

// Query optimization helpers
const optimizedQueries = {
  // Paginated queries with proper indexing
  paginatedFind: (model, filter = {}, options = {}) => {
    const { page = 1, limit = 20, sort = { createdAt: -1 } } = options;
    const skip = (page - 1) * limit;
    
    return model
      .find(filter)
      .sort(sort)
      .skip(skip)
      .limit(limit)
      .lean(); // Return plain objects for better performance
  },
  
  // Aggregation with proper indexing
  optimizedAggregate: (model, pipeline) => {
    return model.aggregate(pipeline).allowDiskUse(true);
  },
  
  // Bulk operations for better performance
  bulkWrite: (model, operations) => {
    return model.bulkWrite(operations, { ordered: false });
  }
};

module.exports = {
  connectWithRetry,
  setupIndexes,
  optimizedQueries,
  connectionOptions
};