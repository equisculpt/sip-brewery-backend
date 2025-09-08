const { Client } = require('@elastic/elasticsearch');
const client = new Client({ node: process.env.ELASTICSEARCH_URL || 'http://localhost:9200' });
async function ensureIndex(index) {
  const exists = await client.indices.exists({ index });
  if (!exists.body) await client.indices.create({ index });
}
module.exports = { client, ensureIndex };
