const logger = require('../utils/logger');

let client;
let elasticsearchAvailable = false;

try {
  const { Client } = require('@elastic/elasticsearch');
  client = new Client({ node: process.env.ELASTICSEARCH_URL || 'http://localhost:9200' });
  elasticsearchAvailable = true;
  logger.info('✅ Elasticsearch client initialized');
} catch (error) {
  logger.warn('⚠️ Elasticsearch not installed. Search functionality will use in-memory fallback. Install with: npm install @elastic/elasticsearch');
  
  // Mock Elasticsearch client
  client = {
    indices: {
      exists: async () => ({ body: true }),
      create: async () => ({ acknowledged: true })
    },
    index: async (params) => {
      logger.debug('Mock: Indexing document', params);
      return { result: 'created', _id: Date.now().toString() };
    },
    search: async (params) => {
      logger.debug('Mock: Searching', params);
      return { hits: { hits: [], total: { value: 0 } } };
    },
    delete: async (params) => {
      logger.debug('Mock: Deleting document', params);
      return { result: 'deleted' };
    },
    update: async (params) => {
      logger.debug('Mock: Updating document', params);
      return { result: 'updated' };
    }
  };
}

async function ensureIndex(index) {
  if (!elasticsearchAvailable) {
    logger.debug(`Mock: Index ${index} ensured`);
    return;
  }
  
  try {
    const exists = await client.indices.exists({ index });
    if (!exists.body) {
      await client.indices.create({ index });
      logger.info(`Elasticsearch index created: ${index}`);
    }
  } catch (error) {
    logger.error(`Error ensuring index ${index}:`, error);
  }
}

module.exports = { client, ensureIndex, elasticsearchAvailable };
