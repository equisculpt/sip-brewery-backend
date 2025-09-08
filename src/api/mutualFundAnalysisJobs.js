const express = require('express');
const router = express.Router();
const { v4: uuidv4 } = require('uuid');

// In-memory job store and queue (swap with Redis or DB for production)
const jobs = {};
const jobQueue = [];

// Progress steps for premium UX
const steps = [
  { progress: 10, message: 'Fetching latest market data…' },
  { progress: 30, message: 'Running AI prediction models…' },
  { progress: 60, message: 'Synthesizing insights…' },
  { progress: 90, message: 'Finalizing analysis…' },
  { progress: 100, message: 'Analysis complete!' },
];

// POST /api/analyze-mutual-fund
router.post('/analyze-mutual-fund', (req, res) => {
  const { fundId, userParams } = req.body;
  const jobId = uuidv4();
  jobs[jobId] = {
    status: 'pending',
    progress: 0,
    message: 'Job queued',
    result: null,
    error: null,
    createdAt: Date.now(),
    fundId,
    userParams
  };
  jobQueue.push(jobId);
  res.json({ jobId });
});

// GET /api/job-status/:jobId
router.get('/job-status/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json({
    status: job.status,
    progress: job.progress,
    message: job.message,
    result: job.result,
    error: job.error
  });
});

// Worker to process jobs from the queue (calls real Python ASI backend)
const axios = require('axios');
setInterval(async () => {
  if (jobQueue.length === 0) return;
  const jobId = jobQueue.shift();
  const job = jobs[jobId];
  if (!job) return;
  job.status = 'running';
  for (const step of steps) {
    job.progress = step.progress;
    job.message = step.message;
    // Simulate work for each step (2–3s per step for demo UX)
    await new Promise(res => setTimeout(res, 2000 + Math.random() * 1000));
  }
  // Call Python ASI backend for real analysis and chart data
  try {
    const asiRes = await axios.post('http://localhost:8001/analyze', {
      fundId: job.fundId,
      userParams: job.userParams
    }, { timeout: 20000 });
    job.result = {
      fundId: job.fundId,
      insights: asiRes.data.insights || [],
      chartData: asiRes.data.chartData || [],
      generatedAt: new Date().toISOString()
    };
    job.status = 'completed';
    job.message = 'Analysis complete!';
    job.progress = 100;
  } catch (err) {
    job.status = 'failed';
    job.error = 'ASI backend error: ' + (err.response?.data?.error || err.message);
  }
}, 2000);

module.exports = router;
