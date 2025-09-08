// Prediction Monitor: Stores predictions, tracks outcomes, and enables feedback-driven improvement
const fs = require('fs');
const path = require('path');
const PREDICTION_LOG = path.join(__dirname, 'prediction_log.jsonl');

function storePrediction(prediction) {
  fs.appendFileSync(PREDICTION_LOG, JSON.stringify(prediction) + '\n');
}

function getPredictions(filter = {}) {
  if (!fs.existsSync(PREDICTION_LOG)) return [];
  return fs.readFileSync(PREDICTION_LOG, 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
    .filter(pred => {
      for (const key in filter) {
        if (pred[key] !== filter[key]) return false;
      }
      return true;
    });
}

function updatePredictionOutcome(id, actual, feedback) {
  const preds = getPredictions();
  const idx = preds.findIndex(p => p.id === id);
  if (idx !== -1) {
    preds[idx].actual = actual;
    preds[idx].feedback = feedback;
    preds[idx].accuracy = (typeof actual === 'number' && typeof preds[idx].predicted === 'number')
      ? 1 - Math.abs(actual - preds[idx].predicted) / Math.max(Math.abs(actual), 1)
      : null;
    fs.writeFileSync(PREDICTION_LOG, preds.map(p => JSON.stringify(p)).join('\n') + '\n');
  }
}

function rollingAccuracy(window = 100) {
  const preds = getPredictions().slice(-window);
  const valid = preds.filter(p => typeof p.accuracy === 'number');
  if (valid.length === 0) return null;
  return valid.reduce((sum, p) => sum + p.accuracy, 0) / valid.length;
}

module.exports = { storePrediction, getPredictions, updatePredictionOutcome, rollingAccuracy };
