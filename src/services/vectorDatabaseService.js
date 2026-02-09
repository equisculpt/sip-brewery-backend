const { PineconeClient } = require('@pinecone-database/pinecone');
const { Configuration, OpenAIApi } = require('openai');
const logger = require('../utils/logger');

class VectorDatabaseService {
  constructor() {
    this.pinecone = new PineconeClient();
    this.isInitialized = false;
    this.indexes = {
      FUND_EMBEDDINGS: 'fund-embeddings',
      USER_PROFILES: 'user-profiles',
      PORTFOLIO_SIMILARITY: 'portfolio-similarity'
    };

    // Initialize OpenAI for embeddings (alternative: use local model)
    const configuration = new Configuration({
      apiKey: process.env.OPENAI_API_KEY
    });
    this.openai = new OpenAIApi(configuration);
  }

  async initialize() {
    try {
      await this.pinecone.init({
        environment: process.env.PINECONE_ENVIRONMENT || 'us-west1-gcp',
        apiKey: process.env.PINECONE_API_KEY
      });

      this.isInitialized = true;
      logger.info('Pinecone Vector Database initialized successfully');

      // Ensure indexes exist
      await this.ensureIndexes();
    } catch (error) {
      logger.error('Failed to initialize Pinecone', { error: error.message });
      throw error;
    }
  }

  async ensureIndexes() {
    try {
      const existingIndexes = await this.pinecone.listIndexes();

      // Create fund embeddings index if not exists
      if (!existingIndexes.includes(this.indexes.FUND_EMBEDDINGS)) {
        await this.pinecone.createIndex({
          createRequest: {
            name: this.indexes.FUND_EMBEDDINGS,
            dimension: 1536, // OpenAI ada-002 embedding size
            metric: 'cosine',
            pods: 1,
            replicas: 1,
            pod_type: 'p1.x1'
          }
        });
        logger.info('Created fund-embeddings index');
      }

      // Create user profiles index if not exists
      if (!existingIndexes.includes(this.indexes.USER_PROFILES)) {
        await this.pinecone.createIndex({
          createRequest: {
            name: this.indexes.USER_PROFILES,
            dimension: 512,
            metric: 'cosine',
            pods: 1,
            replicas: 1,
            pod_type: 'p1.x1'
          }
        });
        logger.info('Created user-profiles index');
      }
    } catch (error) {
      logger.error('Failed to ensure indexes', { error: error.message });
    }
  }

  async generateEmbedding(text) {
    try {
      const response = await this.openai.createEmbedding({
        model: 'text-embedding-ada-002',
        input: text
      });

      return response.data.data[0].embedding;
    } catch (error) {
      logger.error('Failed to generate embedding', { error: error.message });
      throw error;
    }
  }

  // Fund operations

  async indexFund(fundId, fundDescription, metadata = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const embedding = await this.generateEmbedding(fundDescription);
      const index = this.pinecone.Index(this.indexes.FUND_EMBEDDINGS);

      await index.upsert({
        upsertRequest: {
          vectors: [
            {
              id: fundId,
              values: embedding,
              metadata: {
                ...metadata,
                indexed_at: new Date().toISOString()
              }
            }
          ]
        }
      });

      logger.info('Fund indexed in vector database', { fundId });
    } catch (error) {
      logger.error('Failed to index fund', { fundId, error: error.message });
      throw error;
    }
  }

  async findSimilarFunds(fundId, topK = 10) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const index = this.pinecone.Index(this.indexes.FUND_EMBEDDINGS);

      // Fetch the fund's embedding
      const fetchResponse = await index.fetch({
        ids: [fundId]
      });

      if (!fetchResponse.vectors[fundId]) {
        throw new Error(`Fund ${fundId} not found in vector database`);
      }

      const embedding = fetchResponse.vectors[fundId].values;

      // Query for similar funds
      const queryResponse = await index.query({
        queryRequest: {
          vector: embedding,
          topK: topK + 1, // +1 to exclude the query fund itself
          includeMetadata: true
        }
      });

      // Filter out the query fund and return results
      const results = queryResponse.matches
        .filter(match => match.id !== fundId)
        .slice(0, topK)
        .map(match => ({
          fundId: match.id,
          similarity: match.score,
          metadata: match.metadata
        }));

      return results;
    } catch (error) {
      logger.error('Failed to find similar funds', { fundId, error: error.message });
      throw error;
    }
  }

  async searchFundsByQuery(queryText, topK = 10, filter = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const embedding = await this.generateEmbedding(queryText);
      const index = this.pinecone.Index(this.indexes.FUND_EMBEDDINGS);

      const queryResponse = await index.query({
        queryRequest: {
          vector: embedding,
          topK,
          includeMetadata: true,
          filter
        }
      });

      return queryResponse.matches.map(match => ({
        fundId: match.id,
        similarity: match.score,
        metadata: match.metadata
      }));
    } catch (error) {
      logger.error('Failed to search funds', { error: error.message });
      throw error;
    }
  }

  // User profile operations

  async indexUserProfile(userId, profileVector, metadata = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const index = this.pinecone.Index(this.indexes.USER_PROFILES);

      await index.upsert({
        upsertRequest: {
          vectors: [
            {
              id: userId,
              values: profileVector,
              metadata: {
                ...metadata,
                indexed_at: new Date().toISOString()
              }
            }
          ]
        }
      });

      logger.info('User profile indexed', { userId });
    } catch (error) {
      logger.error('Failed to index user profile', { userId, error: error.message });
      throw error;
    }
  }

  async findSimilarUsers(userId, topK = 10) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const index = this.pinecone.Index(this.indexes.USER_PROFILES);

      const fetchResponse = await index.fetch({
        ids: [userId]
      });

      if (!fetchResponse.vectors[userId]) {
        throw new Error(`User ${userId} not found in vector database`);
      }

      const embedding = fetchResponse.vectors[userId].values;

      const queryResponse = await index.query({
        queryRequest: {
          vector: embedding,
          topK: topK + 1,
          includeMetadata: true
        }
      });

      return queryResponse.matches
        .filter(match => match.id !== userId)
        .slice(0, topK)
        .map(match => ({
          userId: match.id,
          similarity: match.score,
          metadata: match.metadata
        }));
    } catch (error) {
      logger.error('Failed to find similar users', { userId, error: error.message });
      throw error;
    }
  }

  // Batch operations

  async batchIndexFunds(funds) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const index = this.pinecone.Index(this.indexes.FUND_EMBEDDINGS);
      const vectors = [];

      for (const fund of funds) {
        const embedding = await this.generateEmbedding(fund.description);
        vectors.push({
          id: fund.id,
          values: embedding,
          metadata: fund.metadata || {}
        });
      }

      // Upsert in batches of 100
      const batchSize = 100;
      for (let i = 0; i < vectors.length; i += batchSize) {
        const batch = vectors.slice(i, i + batchSize);
        await index.upsert({
          upsertRequest: {
            vectors: batch
          }
        });
      }

      logger.info('Batch indexed funds', { count: funds.length });
    } catch (error) {
      logger.error('Failed to batch index funds', { error: error.message });
      throw error;
    }
  }

  async deleteFund(fundId) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const index = this.pinecone.Index(this.indexes.FUND_EMBEDDINGS);
      await index.delete1({
        ids: [fundId]
      });

      logger.info('Fund deleted from vector database', { fundId });
    } catch (error) {
      logger.error('Failed to delete fund', { fundId, error: error.message });
      throw error;
    }
  }

  async getIndexStats(indexName) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const index = this.pinecone.Index(indexName);
      const stats = await index.describeIndexStats();
      return stats;
    } catch (error) {
      logger.error('Failed to get index stats', { indexName, error: error.message });
      throw error;
    }
  }

  getMetrics() {
    return {
      isInitialized: this.isInitialized,
      indexes: Object.keys(this.indexes).length
    };
  }
}

module.exports = new VectorDatabaseService();
