/**
 * 🚀 PM2 ECOSYSTEM CONFIGURATION
 * Production-ready process management for SIP Brewery Backend
 */

module.exports = {
  apps: [
    {
      name: 'sipbrewery-backend',
      script: './src/app.js',
      instances: process.env.CLUSTER_WORKERS || 'max',
      exec_mode: 'cluster',
      max_memory_restart: '1G',
      node_args: '--max-old-space-size=1024',
      
      // Environment configurations
      env: {
        NODE_ENV: 'development',
        PORT: 3000
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
        ENABLE_CLUSTERING: true,
        LOG_LEVEL: 'info'
      },
      
      // Logging configuration
      log_file: './logs/combined.log',
      out_file: './logs/out.log',
      error_file: './logs/error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      
      // Auto-restart configuration
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '10s',
      
      // Health monitoring
      health_check_grace_period: 3000,
      health_check_fatal_exceptions: true,
      
      // Performance monitoring
      pmx: true,
      monitoring: true
    },
    
    // Python ASI Bridge Service
    {
      name: 'sipbrewery-python-asi',
      script: 'python3',
      args: './src/unified-asi/python-asi/python_bridge.py',
      interpreter: 'none',
      instances: 1,
      exec_mode: 'fork',
      
      env: {
        PYTHONPATH: './src/unified-asi/python-asi',
        PYTHONUNBUFFERED: '1',
        ASI_PORT: 8001
      },
      env_production: {
        PYTHONPATH: './src/unified-asi/python-asi',
        PYTHONUNBUFFERED: '1',
        ASI_PORT: 8001,
        LOG_LEVEL: 'info'
      },
      
      // Logging
      log_file: './logs/python-asi.log',
      out_file: './logs/python-asi-out.log',
      error_file: './logs/python-asi-error.log',
      
      // Auto-restart
      autorestart: true,
      max_restarts: 5,
      min_uptime: '10s'
    },

    // Unified ASI System
    {
      name: 'unified-asi-system',
      script: './src/unified-asi/index.js',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      max_restarts: 100,
      restart_delay: 5000
    },
    {
      name: 'asi-web-research-agent',
      script: 'src/asi/WebResearchAgent.js',
      watch: true,
      autorestart: true,
      max_restarts: 100,
      restart_delay: 5000
    },
    {
      name: 'sipbrewery-backend',
      script: 'src/app.js',
      watch: true,
      autorestart: true,
      max_restarts: 100,
      restart_delay: 5000
    }
  ]
};
