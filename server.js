/**
 * DEPRECATED: server.js
 * 
 * This file is deprecated and maintained only for backward compatibility.
 * The main application entry point is now src/app.js
 * 
 * Please use:
 * - npm start (uses src/app.js)
 * - npm run dev (uses src/app.js with nodemon)
 * 
 * This file will be removed in a future version.
 */

console.warn('⚠️  WARNING: server.js is deprecated!');
console.warn('⚠️  Please use "npm start" or "npm run dev" instead.');
console.warn('⚠️  Main entry point is now src/app.js');
console.warn('⚠️  This file will be removed in a future version.\n');

// Delegate to the main application
require('./src/app.js'); 