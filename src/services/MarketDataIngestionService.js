const MarketData = require('../models/MarketData');

class MarketDataIngestionService {
  constructor({ sources = [] } = {}) {
    this.sources = sources;
  }

  async ingestLatestSnapshot() {
    return {
      status: 'pending',
      message: 'No ingestion sources configured'
    };
  }

  async persistSnapshot(snapshot, metadata = {}) {
    if (!snapshot || !snapshot.date) {
      throw new Error('Market data snapshot requires a date');
    }

    return MarketData.create({
      ...snapshot,
      metadata: {
        ...(snapshot.metadata || {}),
        ...metadata
      }
    });
  }
}

module.exports = MarketDataIngestionService;
