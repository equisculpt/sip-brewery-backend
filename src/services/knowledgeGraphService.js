const neo4j = require('neo4j-driver');
const logger = require('../utils/logger');

class KnowledgeGraphService {
  constructor() {
    this.driver = null;
    this.isConnected = false;
  }

  async connect() {
    try {
      this.driver = neo4j.driver(
        process.env.NEO4J_URI || 'bolt://localhost:7687',
        neo4j.auth.basic(
          process.env.NEO4J_USER || 'neo4j',
          process.env.NEO4J_PASSWORD || 'password'
        ),
        {
          maxConnectionPoolSize: 50,
          connectionAcquisitionTimeout: 60000
        }
      );

      await this.driver.verifyConnectivity();
      this.isConnected = true;
      logger.info('Neo4j Knowledge Graph connected successfully');

      await this.createConstraints();
      await this.createIndexes();
    } catch (error) {
      logger.error('Failed to connect to Neo4j', { error: error.message });
      throw error;
    }
  }

  async disconnect() {
    if (this.driver) {
      await this.driver.close();
      this.isConnected = false;
      logger.info('Neo4j Knowledge Graph disconnected');
    }
  }

  async createConstraints() {
    const session = this.driver.session();
    try {
      await session.run('CREATE CONSTRAINT fund_id IF NOT EXISTS FOR (f:Fund) REQUIRE f.id IS UNIQUE');
      await session.run('CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE');
      await session.run('CREATE CONSTRAINT stock_id IF NOT EXISTS FOR (s:Stock) REQUIRE s.id IS UNIQUE');
      await session.run('CREATE CONSTRAINT sector_id IF NOT EXISTS FOR (sec:Sector) REQUIRE sec.id IS UNIQUE');
      logger.info('Neo4j constraints created');
    } finally {
      await session.close();
    }
  }

  async createIndexes() {
    const session = this.driver.session();
    try {
      await session.run('CREATE INDEX fund_category IF NOT EXISTS FOR (f:Fund) ON (f.category)');
      await session.run('CREATE INDEX user_risk IF NOT EXISTS FOR (u:User) ON (u.risk_profile)');
      await session.run('CREATE INDEX stock_sector IF NOT EXISTS FOR (s:Stock) ON (s.sector)');
      logger.info('Neo4j indexes created');
    } finally {
      await session.close();
    }
  }

  // Fund operations

  async createFund(fundData) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `CREATE (f:Fund {
          id: $id,
          name: $name,
          category: $category,
          aum: $aum,
          expense_ratio: $expense_ratio,
          nav: $nav,
          created_at: datetime()
        })
        RETURN f`,
        fundData
      );

      logger.info('Fund created in knowledge graph', { fundId: fundData.id });
      return result.records[0].get('f').properties;
    } finally {
      await session.close();
    }
  }

  async createHolding(fundId, stockId, weight) {
    const session = this.driver.session();
    try {
      await session.run(
        `MATCH (f:Fund {id: $fundId})
         MATCH (s:Stock {id: $stockId})
         MERGE (f)-[h:HOLDS {weight: $weight, updated_at: datetime()}]->(s)
         RETURN h`,
        { fundId, stockId, weight }
      );

      logger.debug('Holding relationship created', { fundId, stockId, weight });
    } finally {
      await session.close();
    }
  }

  async createUserInvestment(userId, fundId, amount) {
    const session = this.driver.session();
    try {
      await session.run(
        `MATCH (u:User {id: $userId})
         MATCH (f:Fund {id: $fundId})
         MERGE (u)-[inv:INVESTED_IN {
           amount: $amount,
           date: datetime(),
           updated_at: datetime()
         }]->(f)
         RETURN inv`,
        { userId, fundId, amount }
      );

      logger.info('User investment created', { userId, fundId, amount });
    } finally {
      await session.close();
    }
  }

  // Query operations

  async findSimilarFunds(fundId, limit = 10) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (f1:Fund {id: $fundId})-[:HOLDS]->(s:Stock)<-[:HOLDS]-(f2:Fund)
         WHERE f1 <> f2
         WITH f2, COUNT(s) AS common_stocks
         ORDER BY common_stocks DESC
         LIMIT $limit
         RETURN f2.id AS fundId, f2.name AS fundName, common_stocks`,
        { fundId, limit: neo4j.int(limit) }
      );

      return result.records.map(record => ({
        fundId: record.get('fundId'),
        fundName: record.get('fundName'),
        commonStocks: record.get('common_stocks').toNumber()
      }));
    } finally {
      await session.close();
    }
  }

  async findFundsByRiskProfile(riskProfile, limit = 20) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (f:Fund)
         WHERE f.risk_level = $riskProfile
         RETURN f
         LIMIT $limit`,
        { riskProfile, limit: neo4j.int(limit) }
      );

      return result.records.map(record => record.get('f').properties);
    } finally {
      await session.close();
    }
  }

  async getUserPortfolioGraph(userId) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (u:User {id: $userId})-[inv:INVESTED_IN]->(f:Fund)-[:HOLDS]->(s:Stock)
         RETURN u, inv, f, s
         LIMIT 100`,
        { userId }
      );

      const nodes = new Map();
      const relationships = [];

      result.records.forEach(record => {
        const user = record.get('u');
        const investment = record.get('inv');
        const fund = record.get('f');
        const stock = record.get('s');

        if (!nodes.has(user.identity.toString())) {
          nodes.set(user.identity.toString(), { ...user.properties, type: 'User' });
        }
        if (!nodes.has(fund.identity.toString())) {
          nodes.set(fund.identity.toString(), { ...fund.properties, type: 'Fund' });
        }
        if (!nodes.has(stock.identity.toString())) {
          nodes.set(stock.identity.toString(), { ...stock.properties, type: 'Stock' });
        }

        relationships.push({
          from: user.identity.toString(),
          to: fund.identity.toString(),
          type: 'INVESTED_IN',
          properties: investment.properties
        });
      });

      return {
        nodes: Array.from(nodes.values()),
        relationships
      };
    } finally {
      await session.close();
    }
  }

  async detectConcentrationRisk(userId, threshold = 0.35) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (u:User {id: $userId})-[inv:INVESTED_IN]->(f:Fund)-[:BELONGS_TO]->(sec:Sector)
         WITH u, sec, SUM(inv.amount) AS sector_exposure, u.total_portfolio_value AS total
         WHERE sector_exposure > $threshold * total
         RETURN sec.name AS sector, sector_exposure, total`,
        { userId, threshold }
      );

      return result.records.map(record => ({
        sector: record.get('sector'),
        exposure: record.get('sector_exposure').toNumber(),
        total: record.get('total').toNumber(),
        percentage: (record.get('sector_exposure').toNumber() / record.get('total').toNumber()) * 100
      }));
    } finally {
      await session.close();
    }
  }

  async findCorrelatedFunds(fundId, minCorrelation = 0.7, limit = 10) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (f1:Fund {id: $fundId})-[c:CORRELATED_WITH]-(f2:Fund)
         WHERE c.correlation >= $minCorrelation
         RETURN f2.id AS fundId, f2.name AS fundName, c.correlation AS correlation
         ORDER BY c.correlation DESC
         LIMIT $limit`,
        { fundId, minCorrelation, limit: neo4j.int(limit) }
      );

      return result.records.map(record => ({
        fundId: record.get('fundId'),
        fundName: record.get('fundName'),
        correlation: record.get('correlation')
      }));
    } finally {
      await session.close();
    }
  }

  async getGraphStatistics() {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (n)
         RETURN labels(n) AS label, COUNT(n) AS count`
      );

      const stats = {};
      result.records.forEach(record => {
        const label = record.get('label')[0];
        const count = record.get('count').toNumber();
        stats[label] = count;
      });

      return stats;
    } finally {
      await session.close();
    }
  }

  async executeCypherQuery(query, params = {}) {
    const session = this.driver.session();
    try {
      const result = await session.run(query, params);
      return result.records.map(record => record.toObject());
    } finally {
      await session.close();
    }
  }

  getMetrics() {
    return {
      isConnected: this.isConnected,
      driver: this.driver ? 'initialized' : 'not initialized'
    };
  }
}

module.exports = new KnowledgeGraphService();
