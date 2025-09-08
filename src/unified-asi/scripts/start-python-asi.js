/**
 * 🚀 PYTHON ASI STARTUP SCRIPT
 * 
 * Automated startup script for Python ASI services
 * Ensures Python ASI bridge is running before Node.js services
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified Finance ASI
 */

const { spawn } = require('child_process');
const path = require('path');
const axios = require('axios');
const logger = require('../../utils/logger');

class PythonASIStarter {
  constructor() {
    this.pythonProcess = null;
    this.isRunning = false;
    this.healthCheckInterval = null;
    this.maxRetries = 3;
    this.retryDelay = 5000; // 5 seconds
  }

  /**
   * Start Python ASI services
   */
  async start() {
    try {
      logger.info('🐍 Starting Python ASI services...');

      // Check if Python is available
      await this.checkPythonAvailability();

      // Start Python ASI bridge
      await this.startPythonBridge();

      // Wait for service to be ready
      await this.waitForService();

      // Setup health monitoring
      this.setupHealthMonitoring();

      logger.info('✅ Python ASI services started successfully');
      return true;

    } catch (error) {
      logger.error('❌ Failed to start Python ASI services:', error);
      throw error;
    }
  }

  /**
   * Check if Python is available
   */
  async checkPythonAvailability() {
    return new Promise((resolve, reject) => {
      const python = spawn('python', ['--version']);
      
      python.on('close', (code) => {
        if (code === 0) {
          logger.info('✅ Python is available');
          resolve(true);
        } else {
          // Try python3
          const python3 = spawn('python3', ['--version']);
          python3.on('close', (code3) => {
            if (code3 === 0) {
              logger.info('✅ Python3 is available');
              resolve(true);
            } else {
              reject(new Error('Python is not available. Please install Python 3.8+'));
            }
          });
        }
      });

      python.on('error', () => {
        // Try python3 if python fails
        const python3 = spawn('python3', ['--version']);
        python3.on('close', (code) => {
          if (code === 0) {
            logger.info('✅ Python3 is available');
            resolve(true);
          } else {
            reject(new Error('Python is not available. Please install Python 3.8+'));
          }
        });
        python3.on('error', () => {
          reject(new Error('Python is not available. Please install Python 3.8+'));
        });
      });
    });
  }

  /**
   * Start Python ASI bridge
   */
  async startPythonBridge() {
    const pythonBridgePath = path.join(__dirname, '../python-asi/python_bridge.py');
    
    logger.info(`🚀 Starting Python ASI bridge: ${pythonBridgePath}`);

    // Try python first, then python3
    let pythonCommand = 'python';
    try {
      await this.checkPythonAvailability();
    } catch (error) {
      pythonCommand = 'python3';
    }

    this.pythonProcess = spawn(pythonCommand, [pythonBridgePath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONPATH: path.join(__dirname, '../python-asi'),
        PYTHONUNBUFFERED: '1'
      }
    });

    // Handle Python process output
    this.pythonProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        logger.info(`🐍 Python ASI: ${output}`);
      }
    });

    this.pythonProcess.stderr.on('data', (data) => {
      const error = data.toString().trim();
      if (error && !error.includes('WARNING')) {
        logger.error(`🐍 Python ASI Error: ${error}`);
      }
    });

    this.pythonProcess.on('close', (code) => {
      logger.warn(`🐍 Python ASI process exited with code ${code}`);
      this.isRunning = false;
      
      // Auto-restart if unexpected exit
      if (code !== 0 && this.isRunning) {
        logger.info('🔄 Attempting to restart Python ASI...');
        setTimeout(() => this.start(), this.retryDelay);
      }
    });

    this.pythonProcess.on('error', (error) => {
      logger.error('🐍 Python ASI process error:', error);
      this.isRunning = false;
    });

    this.isRunning = true;
    logger.info('🐍 Python ASI process started');
  }

  /**
   * Wait for Python service to be ready
   */
  async waitForService(maxWaitTime = 30000) {
    const startTime = Date.now();
    const checkInterval = 1000; // 1 second

    logger.info('⏳ Waiting for Python ASI service to be ready...');

    while (Date.now() - startTime < maxWaitTime) {
      try {
        const response = await axios.get('http://localhost:8001/health', {
          timeout: 2000
        });

        if (response.status === 200) {
          logger.info('✅ Python ASI service is ready');
          return true;
        }
      } catch (error) {
        // Service not ready yet, continue waiting
      }

      await new Promise(resolve => setTimeout(resolve, checkInterval));
    }

    throw new Error('Python ASI service failed to start within timeout period');
  }

  /**
   * Setup health monitoring
   */
  setupHealthMonitoring() {
    this.healthCheckInterval = setInterval(async () => {
      try {
        const response = await axios.get('http://localhost:8001/health', {
          timeout: 5000
        });

        if (response.status !== 200) {
          logger.warn('⚠️ Python ASI health check failed');
        }
      } catch (error) {
        logger.error('❌ Python ASI health check error:', error.message);
        
        // Attempt restart if service is down
        if (this.isRunning && this.pythonProcess) {
          logger.info('🔄 Attempting to restart Python ASI...');
          await this.restart();
        }
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Stop Python ASI services
   */
  async stop() {
    try {
      logger.info('🛑 Stopping Python ASI services...');

      // Clear health check interval
      if (this.healthCheckInterval) {
        clearInterval(this.healthCheckInterval);
        this.healthCheckInterval = null;
      }

      // Stop Python process
      if (this.pythonProcess) {
        this.pythonProcess.kill('SIGTERM');
        
        // Wait for graceful shutdown
        await new Promise((resolve) => {
          const timeout = setTimeout(() => {
            if (this.pythonProcess) {
              this.pythonProcess.kill('SIGKILL');
            }
            resolve();
          }, 5000);

          this.pythonProcess.on('close', () => {
            clearTimeout(timeout);
            resolve();
          });
        });

        this.pythonProcess = null;
      }

      this.isRunning = false;
      logger.info('✅ Python ASI services stopped');

    } catch (error) {
      logger.error('❌ Error stopping Python ASI services:', error);
      throw error;
    }
  }

  /**
   * Restart Python ASI services
   */
  async restart() {
    await this.stop();
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
    await this.start();
  }

  /**
   * Get service status
   */
  async getStatus() {
    try {
      if (!this.isRunning) {
        return {
          status: 'STOPPED',
          process: 'Not running',
          health: 'N/A'
        };
      }

      const response = await axios.get('http://localhost:8001/health', {
        timeout: 3000
      });

      return {
        status: 'RUNNING',
        process: 'Active',
        health: response.data,
        pid: this.pythonProcess?.pid
      };

    } catch (error) {
      return {
        status: 'ERROR',
        process: this.isRunning ? 'Running but unhealthy' : 'Not running',
        health: 'Service unreachable',
        error: error.message
      };
    }
  }

  /**
   * Check if service is healthy
   */
  async isHealthy() {
    try {
      const response = await axios.get('http://localhost:8001/health', {
        timeout: 3000
      });
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }
}

// Create singleton instance
const pythonASIStarter = new PythonASIStarter();

module.exports = {
  PythonASIStarter,
  pythonASIStarter
};
