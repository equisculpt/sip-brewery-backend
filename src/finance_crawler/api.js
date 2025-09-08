const express = require('express');
const { client } = require('./elasticsearch');
const app = express();

app.get('/search', async (req, res) => {
  const q = req.query.q || '';
  const index = req.query.index || 'chittorgarh';
  const { body } = await client.search({
    index,
    body: { query: { match: { name: q } } }
  });
  res.json(body.hits.hits.map(hit => hit._source));
});

app.listen(3000, () => console.log('Finance Search API running on port 3000'));
