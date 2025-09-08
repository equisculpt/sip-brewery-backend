module.exports = {
  "testEnvironment": "node",
  "roots": [
    "<rootDir>/tests"
  ],
  "testMatch": [
    "**/tests/**/*.test.js",
    "**/tests/**/*.spec.js"
  ],
  "collectCoverageFrom": [
    "src/**/*.js",
    "!src/**/*.test.js",
    "!src/test/**"
  ],
  "coverageDirectory": "coverage",
  "coverageReporters": [
    "text",
    "lcov",
    "html"
  ],
  "setupFilesAfterEnv": [
    "<rootDir>/tests/setup.js"
  ],
  "testTimeout": 30000
};