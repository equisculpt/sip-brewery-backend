/**
 * Simple Logger Utility
 * Provides consistent logging across the application
 */

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

class Logger {
  constructor() {
    this.logLevel = process.env.LOG_LEVEL || 'info';
  }

  formatMessage(level, message, ...args) {
    const timestamp = new Date().toISOString();
    const formattedArgs = args.length > 0 ? ' ' + args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
    ).join(' ') : '';
    
    return `[${timestamp}] [${level.toUpperCase()}] ${message}${formattedArgs}`;
  }

  info(message, ...args) {
    console.log(colors.blue + this.formatMessage('info', message, ...args) + colors.reset);
  }

  warn(message, ...args) {
    console.warn(colors.yellow + this.formatMessage('warn', message, ...args) + colors.reset);
  }

  error(message, ...args) {
    console.error(colors.red + this.formatMessage('error', message, ...args) + colors.reset);
  }

  debug(message, ...args) {
    if (this.logLevel === 'debug') {
      console.log(colors.dim + this.formatMessage('debug', message, ...args) + colors.reset);
    }
  }

  success(message, ...args) {
    console.log(colors.green + this.formatMessage('success', message, ...args) + colors.reset);
  }

  log(message, ...args) {
    console.log(this.formatMessage('log', message, ...args));
  }
}

// Export singleton instance
const logger = new Logger();

module.exports = logger;
