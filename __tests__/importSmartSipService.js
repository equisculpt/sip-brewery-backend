console.log('Importing smartSipService.js...');
try {
  require('../src/services/smartSipService');
  console.log('Import successful!');
} catch (err) {
  console.error('Import failed:', err);
  process.exit(1);
}
