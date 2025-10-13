/**
 * Fix TensorFlow imports across all files
 * Replace @tensorflow/tfjs-node-gpu with tensorflowLoader
 */

const fs = require('fs');
const path = require('path');

const filesToFix = [
  'src/ai/AdvancedMLModels.js',
  'src/ai/BacktestingFramework.js',
  'src/ai/ContinuousLearningEngine.js',
  'src/ai/MutualFundAnalyzer.js',
  'src/ai/PerformanceMetrics.js',
  'src/asi/ASIIntelligenceEnhancer.js',
  'src/asi/AdvancedMutualFundPredictor.js',
  'src/asi/AutonomousLearningSystem.js',
  'src/asi/BehavioralFinanceEngine.js',
  'src/asi/DocumentIntelligenceAnalyzer.js',
  'src/asi/EnhancedPortfolioAnalyzer.js',
  'src/asi/FactorInvestingEngine.js',
  'src/asi/GPUQuantumEngine.js',
  'src/asi/ModernPortfolioTheory.js',
  'src/asi/MultiModalDataProcessor.js',
  'src/asi/QuantumInspiredOptimizer.js',
  'src/asi/RealTimeAdaptiveLearner.js',
  'src/asi/ReinforcementLearningEngine.js',
  'src/asi/models/AlphaModel.js',
  'src/asi/models/DeepForecastingModel.js'
];

let filesFixed = 0;
let errors = 0;

filesToFix.forEach(file => {
  try {
    const filePath = path.join(__dirname, file);
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Replace TensorFlow GPU import
    const originalContent = content;
    content = content.replace(
      /const tf = require\('@tensorflow\/tfjs-node-gpu'\);/g,
      "const tf = require('../utils/tensorflowLoader');"
    );
    
    // Fix relative path for nested directories
    if (file.includes('/models/')) {
      content = content.replace(
        "const tf = require('../utils/tensorflowLoader');",
        "const tf = require('../../utils/tensorflowLoader');"
      );
    }
    
    if (content !== originalContent) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`✅ Fixed: ${file}`);
      filesFixed++;
    } else {
      console.log(`⏭️ Skipped (no change): ${file}`);
    }
  } catch (error) {
    console.error(`❌ Error fixing ${file}:`, error.message);
    errors++;
  }
});

console.log(`\n✨ Complete! Fixed ${filesFixed} files. Errors: ${errors}`);
