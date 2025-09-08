const cron = require('node-cron');
const leaderboardService = require('./leaderboardService');
const logger = require('../utils/logger');

class LeaderboardCronService {
  constructor() {
    this.jobs = new Map();
  }

  /**
   * Initialize all leaderboard cron jobs
   */
  init() {
    logger.info('Initializing leaderboard cron jobs...');
    
    // Daily XIRR update (8:00 AM IST)
    this.startXIRRUpdateJob();
    
    // Daily leaderboard generation (8:30 AM IST)
    this.startLeaderboardGenerationJob();
    
    // Weekly leaderboard cleanup (Sunday 9:00 AM IST)
    this.startLeaderboardCleanupJob();
    
    logger.info('Leaderboard cron jobs initialized successfully');
  }

  /**
   * Start daily XIRR update job
   */
  startXIRRUpdateJob() {
    const job = cron.schedule('0 8 * * *', async () => {
      try {
        logger.info('Running daily XIRR update job...');
        
        const updatedCount = await leaderboardService.updateAllPortfolioXIRR();
        
        logger.info(`Daily XIRR update completed. Updated ${updatedCount} portfolios.`);
        
      } catch (error) {
        logger.error('Error in daily XIRR update job:', error);
      }
    }, {
      timezone: 'Asia/Kolkata'
    });

    this.jobs.set('xirrUpdate', job);
    logger.info('XIRR update cron job scheduled');
  }

  /**
   * Start daily leaderboard generation job
   */
  startLeaderboardGenerationJob() {
    const job = cron.schedule('30 8 * * *', async () => {
      try {
        logger.info('Running daily leaderboard generation job...');
        
        const results = await leaderboardService.generateAllLeaderboards();
        
        logger.info('Daily leaderboard generation completed:', results);
        
      } catch (error) {
        logger.error('Error in daily leaderboard generation job:', error);
      }
    }, {
      timezone: 'Asia/Kolkata'
    });

    this.jobs.set('leaderboardGeneration', job);
    logger.info('Leaderboard generation cron job scheduled');
  }

  /**
   * Start weekly leaderboard cleanup job
   */
  startLeaderboardCleanupJob() {
    const job = cron.schedule('0 9 * * 0', async () => {
      try {
        logger.info('Running weekly leaderboard cleanup job...');
        
        // Clean up old leaderboard history entries
        const Leaderboard = require('../models/Leaderboard');
        const UserPortfolio = require('../models/UserPortfolio');
        
        // Keep only last 30 days of leaderboard data
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
        
        const deletedLeaderboards = await Leaderboard.deleteMany({
          generatedAt: { $lt: thirtyDaysAgo },
          isActive: false
        });
        
        // Clean up old leaderboard history from portfolios
        const portfolios = await UserPortfolio.find({});
        let cleanedPortfolios = 0;
        
        for (const portfolio of portfolios) {
          const originalLength = portfolio.leaderboardHistory.length;
          portfolio.leaderboardHistory = portfolio.leaderboardHistory.filter(entry => 
            new Date(entry.date) >= thirtyDaysAgo
          );
          
          if (portfolio.leaderboardHistory.length !== originalLength) {
            await portfolio.save();
            cleanedPortfolios++;
          }
        }
        
        logger.info(`Weekly cleanup completed. Deleted ${deletedLeaderboards.deletedCount} old leaderboards, cleaned ${cleanedPortfolios} portfolios.`);
        
      } catch (error) {
        logger.error('Error in weekly leaderboard cleanup job:', error);
      }
    }, {
      timezone: 'Asia/Kolkata'
    });

    this.jobs.set('leaderboardCleanup', job);
    logger.info('Leaderboard cleanup cron job scheduled');
  }

  /**
   * Start a test job for development
   */
  startTestJob() {
    const job = cron.schedule('*/10 * * * *', async () => {
      try {
        logger.info('Running leaderboard test job...');
        
        // Test XIRR calculation
        const testPortfolio = {
          transactions: [
            {
              date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
              type: 'SIP',
              amount: 10000
            },
            {
              date: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000), // 15 days ago
              type: 'SIP',
              amount: 10000
            }
          ],
          totalCurrentValue: 21000
        };
        
        const xirr = await leaderboardService.calculatePortfolioXIRR(testPortfolio, '1M');
        logger.info('Test XIRR calculation:', xirr);
        
      } catch (error) {
        logger.error('Error in leaderboard test job:', error);
      }
    });

    this.jobs.set('test', job);
    logger.info('Leaderboard test cron job scheduled (runs every 10 minutes)');
  }

  /**
   * Stop all cron jobs
   */
  stopAll() {
    logger.info('Stopping all leaderboard cron jobs...');
    
    for (const [name, job] of this.jobs) {
      job.stop();
      logger.info(`Stopped leaderboard cron job: ${name}`);
    }
    
    this.jobs.clear();
    logger.info('All leaderboard cron jobs stopped');
  }

  /**
   * Stop a specific cron job
   */
  stopJob(jobName) {
    const job = this.jobs.get(jobName);
    if (job) {
      job.stop();
      this.jobs.delete(jobName);
      logger.info(`Stopped leaderboard cron job: ${jobName}`);
    } else {
      logger.warn(`Leaderboard cron job not found: ${jobName}`);
    }
  }

  /**
   * Get status of all cron jobs
   */
  getJobStatus() {
    const status = {};
    
    for (const [name, job] of this.jobs) {
      status[name] = {
        running: job.running,
        lastRun: job.lastDate,
        nextRun: job.nextDate
      };
    }
    
    return status;
  }

  /**
   * Manually trigger XIRR update
   */
  async triggerXIRRUpdate() {
    try {
      logger.info('Manually triggering XIRR update...');
      
      const updatedCount = await leaderboardService.updateAllPortfolioXIRR();
      
      logger.info('Manual XIRR update completed');
      return { success: true, updatedCount, message: `XIRR updated for ${updatedCount} portfolios` };
      
    } catch (error) {
      logger.error('Error in manual XIRR update:', error);
      throw error;
    }
  }

  /**
   * Manually trigger leaderboard generation
   */
  async triggerLeaderboardGeneration() {
    try {
      logger.info('Manually triggering leaderboard generation...');
      
      const results = await leaderboardService.generateAllLeaderboards();
      
      logger.info('Manual leaderboard generation completed');
      return { success: true, results, message: 'Leaderboards generated successfully' };
      
    } catch (error) {
      logger.error('Error in manual leaderboard generation:', error);
      throw error;
    }
  }
}

module.exports = new LeaderboardCronService(); 