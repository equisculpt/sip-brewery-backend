const fs = require('fs');
const path = require('path');

// Patterns to remove completely
const completeRemovals = [
  "const { MongoMemoryServer } = require('mongodb-memory-server');",
  "let mongoServer;",
  "mongoServer = await MongoMemoryServer.create();",
  "const mongoUri = mongoServer.getUri();",
  "await mongoose.connect(mongoUri);",
  "await mongoose.connect(mongoUri, {",
  "useNewUrlParser: true,",
  "useUnifiedTopology: true,",
  "});",
  "await mongoServer.stop();",
  "await mongoose.disconnect();"
];

// Patterns to replace
const replacements = [
  {
    pattern: /beforeAll\(async \(\) => \{[^}]*await mongoose\.connect[^}]*\}/gs,
    replacement: 'beforeAll(async () => {\n    // Using global setup - connection already established\n  });'
  },
  {
    pattern: /afterAll\(async \(\) => \{[^}]*await mongoose\.disconnect[^}]*\}/gs,
    replacement: 'afterAll(async () => {\n    // Using global setup - cleanup handled globally\n  });'
  },
  {
    pattern: /beforeEach\(async \(\) => \{[^}]*const collections = mongoose\.connection\.collections[^}]*\}/gs,
    replacement: 'beforeEach(async () => {\n    // Using global setup - database cleaned automatically\n  });'
  }
];

// Fix Transaction test data
const transactionFixes = [
  {
    pattern: /type: ['"]purchase['"]/g,
    replacement: "type: 'SIP'"
  },
  {
    pattern: /type: ['"]sale['"]/g,
    replacement: "type: 'REDEMPTION'"
  },
  {
    pattern: /type: ['"]dividend['"]/g,
    replacement: "type: 'DIVIDEND_PAYOUT'"
  },
  {
    pattern: /status: ['"]completed['"]/g,
    replacement: "status: 'SUCCESS'"
  },
  {
    pattern: /orderType: ['"]buy['"]/g,
    replacement: "orderType: 'BUY'"
  },
  {
    pattern: /orderType: ['"]sell['"]/g,
    replacement: "orderType: 'SELL'"
  }
];

function fixTestFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;

    // Remove complete patterns
    completeRemovals.forEach(pattern => {
      if (content.includes(pattern)) {
        content = content.replace(new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), '');
        modified = true;
      }
    });

    // Apply replacements
    replacements.forEach(({ pattern, replacement }) => {
      if (pattern.test(content)) {
        content = content.replace(pattern, replacement);
        modified = true;
      }
    });

    // Fix Transaction test data
    transactionFixes.forEach(({ pattern, replacement }) => {
      if (pattern.test(content)) {
        content = content.replace(pattern, replacement);
        modified = true;
      }
    });

    // Clean up extra whitespace and newlines
    content = content.replace(/\n\s*\n\s*\n/g, '\n\n');
    content = content.replace(/\s+$/gm, ''); // Remove trailing whitespace
    content = content.replace(/\n+$/, '\n'); // Ensure single newline at end

    if (modified) {
      fs.writeFileSync(filePath, content);
      console.log(`Fixed: ${filePath}`);
    }
  } catch (error) {
    console.error(`Error fixing ${filePath}:`, error.message);
  }
}

function findTestFiles(dir) {
  const files = [];
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      files.push(...findTestFiles(fullPath));
    } else if (item.endsWith('.test.js') || item.endsWith('.spec.js')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

// Find and fix all test files
const testFiles = findTestFiles('__tests__');
console.log(`Found ${testFiles.length} test files to process`);

testFiles.forEach(fixTestFile);
console.log('All test file fixes completed!'); 