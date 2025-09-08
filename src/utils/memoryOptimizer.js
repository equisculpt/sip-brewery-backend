/**
 * 🧠 MEMORY OPTIMIZATION UTILITIES
 * 
 * Advanced memory management for high-performance applications
 */

const logger = require('./logger');

class MemoryOptimizer {
  constructor() {
    this.memoryThreshold = 0.85; // 85% memory usage threshold
    this.gcInterval = 60000; // Force GC every minute
    this.monitoringInterval = 10000; // Monitor every 10 seconds
    
    this.startMemoryMonitoring();
  }

  // Start memory monitoring
  startMemoryMonitoring() {
    setInterval(() => {
      this.checkMemoryUsage();
    }, this.monitoringInterval);
    
    // Force garbage collection periodically
    if (global.gc) {
      setInterval(() => {
        global.gc();
        logger.debug('Forced garbage collection completed');
      }, this.gcInterval);
    }
  }

  // Check memory usage and trigger cleanup if needed
  checkMemoryUsage() {
    const usage = process.memoryUsage();
    const totalMemory = usage.heapTotal;
    const usedMemory = usage.heapUsed;
    const memoryUsagePercent = usedMemory / totalMemory;

    if (memoryUsagePercent > this.memoryThreshold) {
      logger.warn('High memory usage detected:', {
        usedMemory: Math.round(usedMemory / 1024 / 1024) + 'MB',
        totalMemory: Math.round(totalMemory / 1024 / 1024) + 'MB',
        percentage: Math.round(memoryUsagePercent * 100) + '%'
      });
      
      this.performMemoryCleanup();
    }
  }

  // Perform memory cleanup
  performMemoryCleanup() {
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
    
    // Clear require cache for non-essential modules
    this.clearRequireCache();
    
    logger.info('Memory cleanup performed');
  }

  // Clear require cache for non-essential modules
  clearRequireCache() {
    const essentialModules = [
      'express',
      'mongoose',
      'redis',
      'jsonwebtoken',
      'bcrypt',
      'helmet',
      'cors'
    ];
    
    Object.keys(require.cache).forEach(key => {
      const isEssential = essentialModules.some(module => key.includes(module));
      if (!isEssential && key.includes('node_modules')) {
        delete require.cache[key];
      }
    });
  }

  // Memory-efficient object processing
  processLargeDataset(data, batchSize = 1000) {
    const results = [];
    
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      const processedBatch = this.processBatch(batch);
      results.push(...processedBatch);
      
      // Allow event loop to process other tasks
      if (i % (batchSize * 10) === 0) {
        setImmediate(() => {});
      }
    }
    
    return results;
  }

  // Process batch with memory optimization
  processBatch(batch) {
    return batch.map(item => {
      // Process item and return only necessary data
      const processed = this.processItem(item);
      
      // Explicitly nullify references to help GC
      item = null;
      
      return processed;
    });
  }

  // Process individual item
  processItem(item) {
    // Implement your processing logic here
    return item;
  }

  // Get memory statistics
  getMemoryStats() {
    const usage = process.memoryUsage();
    
    return {
      heapUsed: Math.round(usage.heapUsed / 1024 / 1024) + 'MB',
      heapTotal: Math.round(usage.heapTotal / 1024 / 1024) + 'MB',
      external: Math.round(usage.external / 1024 / 1024) + 'MB',
      rss: Math.round(usage.rss / 1024 / 1024) + 'MB',
      arrayBuffers: Math.round(usage.arrayBuffers / 1024 / 1024) + 'MB'
    };
  }

  // Memory-efficient stream processing
  createMemoryEfficientStream(transform) {
    const { Transform } = require('stream');
    
    return new Transform({
      objectMode: true,
      highWaterMark: 16, // Limit buffer size
      transform(chunk, encoding, callback) {
        try {
          const result = transform(chunk);
          callback(null, result);
        } catch (error) {
          callback(error);
        }
      }
    });
  }
}

// Singleton instance
const memoryOptimizer = new MemoryOptimizer();

module.exports = {
  MemoryOptimizer,
  memoryOptimizer
};