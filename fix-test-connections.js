const fs = require('fs');
const path = require('path');

// Patterns to remove from test files
const patternsToRemove = [
  {
    start: "const { MongoMemoryServer } = require('mongodb-memory-server');",
    end: "await mongoose.connect(mongoUri);",
    replacement: "// Using global setup - connection already established"
  },
  {
    start: "let mongoServer;",
    end: "await mongoose.connect(mongoUri);",
    replacement: "// Using global setup - connection already established"
  },
  {
    start: "beforeAll(async () => {",
    end: "await mongoose.connect(mongoUri);",
    replacement: "beforeAll(async () => {\n    // Using global setup - connection already established"
  }
];

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

    // Fix beforeAll blocks
    const beforeAllRegex = /beforeAll\(async \(\) => \{[^}]*await mongoose\.connect[^}]*\}/gs;
    if (beforeAllRegex.test(content)) {
      content = content.replace(beforeAllRegex, 'beforeAll(async () => {\n    // Using global setup - connection already established\n  });');
      modified = true;
    }

    // Fix afterAll blocks that only disconnect
    const afterAllRegex = /afterAll\(async \(\) => \{[^}]*await mongoose\.disconnect[^}]*\}/gs;
    if (afterAllRegex.test(content)) {
      content = content.replace(afterAllRegex, 'afterAll(async () => {\n    // Using global setup - cleanup handled globally\n  });');
      modified = true;
    }

    // Clean up extra whitespace
    content = content.replace(/\n\s*\n\s*\n/g, '\n\n');

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
console.log('Test file connection fixes completed!'); 