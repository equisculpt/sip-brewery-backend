/**
 * TensorFlow Loader with GPU/CPU/Mock Fallback
 * Handles TensorFlow loading gracefully across the application
 */

const logger = require('./logger');

let tf = null;
let tensorflowMode = 'none';

// Try to load TensorFlow in order of preference: GPU -> CPU -> Mock
try {
  tf = require('@tensorflow/tfjs-node-gpu');
  tensorflowMode = 'gpu';
  logger.info('✅ TensorFlow GPU loaded successfully');
} catch (gpuError) {
  logger.warn('⚠️ TensorFlow GPU not available, trying CPU mode...');
  
  try {
    tf = require('@tensorflow/tfjs-node');
    tensorflowMode = 'cpu';
    logger.info('✅ TensorFlow CPU loaded successfully');
  } catch (cpuError) {
    logger.warn('⚠️ TensorFlow not available, using mock mode for AI operations');
    tensorflowMode = 'mock';
    
    // Create mock TensorFlow API
    tf = {
      tensor: () => ({ dispose: () => {}, print: () => {} }),
      tensor1d: () => ({ dispose: () => {}, print: () => {} }),
      tensor2d: () => ({ dispose: () => {}, print: () => {} }),
      tensor3d: () => ({ dispose: () => {}, print: () => {} }),
      tensor4d: () => ({ dispose: () => {}, print: () => {} }),
      zeros: () => ({ dispose: () => {}, print: () => {} }),
      ones: () => ({ dispose: () => {}, print: () => {} }),
      randomNormal: () => ({ dispose: () => {}, print: () => {} }),
      sequential: () => ({
        add: () => {},
        compile: () => {},
        fit: async () => ({ history: { loss: [0.5], acc: [0.8] } }),
        predict: () => ({ dispose: () => {}, print: () => {}, data: async () => [0.5] }),
        evaluate: () => [{ dispose: () => {} }, { dispose: () => {} }],
        save: async () => {},
        summary: () => {}
      }),
      loadLayersModel: async () => ({
        predict: () => ({ dispose: () => {}, data: async () => [0.5] }),
        evaluate: () => [{ dispose: () => {} }],
        save: async () => {}
      }),
      layers: {
        dense: () => ({}),
        conv2d: () => ({}),
        maxPooling2d: () => ({}),
        flatten: () => ({}),
        dropout: () => ({}),
        lstm: () => ({}),
        gru: () => ({}),
        embedding: () => ({}),
        batchNormalization: () => ({})
      },
      train: {
        adam: () => ({}),
        sgd: () => ({}),
        rmsprop: () => ({})
      },
      losses: {
        meanSquaredError: () => 0,
        categoricalCrossentropy: () => 0
      },
      metrics: {
        categoricalAccuracy: () => 0
      },
      tidy: (fn) => fn(),
      dispose: () => {},
      memory: () => ({ numTensors: 0, numBytes: 0 }),
      version: { 'tfjs-core': '0.0.0-mock' }
    };
  }
}

module.exports = tf;
module.exports.tensorflowMode = tensorflowMode;
module.exports.isGPU = tensorflowMode === 'gpu';
module.exports.isCPU = tensorflowMode === 'cpu';
module.exports.isMock = tensorflowMode === 'mock';
