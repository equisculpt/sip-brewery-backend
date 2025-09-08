const { client, ensureIndex } = require('./elasticsearch');
async function indexData(index, docs) {
  await ensureIndex(index);
  const body = docs.flatMap(doc => [{ index: { _index: index } }, doc]);
  await client.bulk({ refresh: true, body });
}
module.exports = indexData;
