const cron = require('node-cron');
const smartSipService = require('./smartSipService');
const marketScoreService = require('./marketScoreService');
const logger = require('../utils/logger');

class CronService {
  constructor() {
    this.jobs = new Map();
  }

  /**
   * Initialize all cron jobs
   */
  init() {
    logger.info('Initializing cron jobs...');
    
    // 6:00 AM daily job: check SIPs due today and consult AGI
    this.startMorningSIPCheckJob();

    // 8:00 AM daily job: initiate orders and pause/resume for SIP-only
    this.startMorningSIPActionJob();

    // Daily market analysis update (9:00 AM IST)
    this.startMarketAnalysisJob();
    
    // Daily SIP execution check (9:30 AM IST)
    this.startSIPExecutionJob();
    
    // Weekly SIP analytics update (Sunday 10:00 AM IST)
    this.startSIPAnalyticsJob();
    
    logger.info('Cron jobs initialized successfully');
  }

  /**
   * Start 6:00 AM daily job: check SIPs due today and consult AGI
   */
  startMorningSIPCheckJob() {
    const job = cron.schedule('0 6 * * *', async () => {
      try {
        logger.info('6:00 AM: Checking SIPs due today and consulting AGI...');
        // For SIP-only: find SIPs due 7 days from now, consult AGI for pause/resume/start
        const sipOnlySIPs = await smartSipService.getSIPsDueInDays(7, { isSipOnly: true });
        await smartSipService.consultAGIForSIPOnly(sipOnlySIPs);
        // For normal funds: find SIPs due today, consult AGI for lumpsum
        const normalSIPs = await smartSipService.getSIPsDueInDays(0, { isSipOnly: false });
        await smartSipService.consultAGIForNormal(normalSIPs);
        logger.info('6:00 AM SIP check and AGI consultation complete.');
      } catch (error) {
        logger.error('Error in 6:00 AM SIP check job:', error);
      }
    }, { timezone: 'Asia/Kolkata' });
    this.jobs.set('morningSIPCheck', job);
    logger.info('6:00 AM SIP check cron job scheduled');
  }

  /**
   * Start 8:00 AM daily job: initiate orders and pause/resume for SIP-only
   */
  startMorningSIPActionJob() {
    const job = cron.schedule('0 8 * * *', async () => {
      try {
        logger.info('8:00 AM: Initiating SIP orders and pause/resume actions...');
        // For SIP-only: send pause/resume/start instructions for SIPs due in 7 days
        const sipOnlySIPs = await smartSipService.getSIPsDueInDays(7, { isSipOnly: true });
        await smartSipService.initiatePauseResumeForSIPOnly(sipOnlySIPs);
        // For normal funds: place lumpsum orders for SIPs due today
        const normalSIPs = await smartSipService.getSIPsDueInDays(0, { isSipOnly: false });
        await smartSipService.placeLumpsumOrdersForNormal(normalSIPs);
        logger.info('8:00 AM SIP action job complete.');
      } catch (error) {
        logger.error('Error in 8:00 AM SIP action job:', error);
      }
    }, { timezone: 'Asia/Kolkata' });
    this.jobs.set('morningSIPAction', job);
    logger.info('8:00 AM SIP action cron job scheduled');
  }

  /**
   * Start daily market analysis job
   */
  startMarketAnalysisJob() {
    const job = cron.schedule('0 9 * * *', async () => {
      try {
        logger.info('Running daily market analysis job...');
        
        // Update market analysis for all active SIPs
        const activeSIPs = await smartSipService.getAllActiveSIPs();
        
        for (const sip of activeSIPs) {
          try {
            // Get fresh market analysis
            const marketAnalysis = await marketScoreService.calculateMarketScore();
            
            // Update SIP with latest analysis
            sip.marketAnalysis = {
              lastUpdated: new Date(),
              currentScore: marketAnalysis.score,
              currentReason: marketAnalysis.reason,
              indicators: marketAnalysis.indicators
            };
            
            await sip.save();
            logger.info(`Updated market analysis for SIP ${sip._id}`);
            
          } catch (error) {
            logger.error(`Error updating market analysis for SIP ${sip._id}:`, error);
          }
        }
        
        logger.info('Daily market analysis job completed');
        
      } catch (error) {
        logger.error('Error in daily market analysis job:', error);
      }
    }, {
      timezone: 'Asia/Kolkata'
    });

    this.jobs.set('marketAnalysis', job);
    logger.info('Market analysis cron job scheduled');
  }

  /**
   * Start daily SIP execution job
   */
  startSIPExecutionJob() {
    const job = cron.schedule('30 9 * * *', async () => {
      try {
        logger.info('Running daily SIP execution job...');
        
        const today = new Date();
        const activeSIPs = await smartSipService.getAllActiveSIPs();
        
        let executedCount = 0;
        let skippedCount = 0;
        
        for (const sip of activeSIPs) {
          try {
            // Check if it's time for this SIP
            const nextSIPDate = new Date(sip.nextSIPDate);
            
            if (today >= nextSIPDate) {
              // Execute SIP
              const result = await smartSipService.executeSIP(sip.userId);
              
              if (result.success) {
                executedCount++;
                logger.info(`SIP executed for user ${sip.userId}: ₹${result.amount}`);
                
                // TODO: Send notification to user
                // await notificationService.sendSIPExecutionNotification(sip.userId, result);
                
              } else {
                skippedCount++;
                logger.info(`SIP skipped for user ${sip.userId}: ${result.message}`);
              }
            } else {
              skippedCount++;
            }
            
          } catch (error) {
            logger.error(`Error executing SIP for user ${sip.userId}:`, error);
          }
        }
        
        logger.info(`SIP execution job completed: ${executedCount} executed, ${skippedCount} skipped`);
        
      } catch (error) {
        logger.error('Error in daily SIP execution job:', error);
      }
    }, {
      timezone: 'Asia/Kolkata'
    });

    this.jobs.set('sipExecution', job);
    logger.info('SIP execution cron job scheduled');
  }

  /**
   * Start weekly SIP analytics job
   */
  startSIPAnalyticsJob() {
    const job = cron.schedule('0 10 * * 0', async () => {
      try {
        logger.info('Running weekly SIP analytics job...');
        
        const activeSIPs = await smartSipService.getAllActiveSIPs();
        
        for (const sip of activeSIPs) {
          try {
            // Get analytics for the week
            const analytics = await smartSipService.getSIPAnalytics(sip.userId);
            
            if (analytics && analytics.totalSIPs > 0) {
              // TODO: Generate weekly report and send to user
              // await reportService.generateWeeklyReport(sip.userId, analytics);
              
              logger.info(`Weekly analytics generated for user ${sip.userId}`);
            }
            
          } catch (error) {
            logger.error(`Error generating analytics for user ${sip.userId}:`, error);
          }
        }
        
        logger.info('Weekly SIP analytics job completed');
        
      } catch (error) {
        logger.error('Error in weekly SIP analytics job:', error);
      }
    }, {
      timezone: 'Asia/Kolkata'
    });

    this.jobs.set('sipAnalytics', job);
    logger.info('SIP analytics cron job scheduled');
  }

  /**
   * Start a custom job for testing
   */
  startTestJob() {
    const job = cron.schedule('*/5 * * * *', async () => {
      try {
        logger.info('Running test job...');
        
        // Test market analysis
        const marketAnalysis = await marketScoreService.calculateMarketScore();
        logger.info('Market analysis test:', marketAnalysis);
        
        // Test SIP recommendation
        const activeSIPs = await smartSipService.getAllActiveSIPs();
        if (activeSIPs.length > 0) {
          const testUserId = activeSIPs[0].userId;
          const recommendation = await smartSipService.getSIPRecommendation(testUserId);
          logger.info('SIP recommendation test:', recommendation);
        }
        
      } catch (error) {
        logger.error('Error in test job:', error);
      }
    });

    this.jobs.set('test', job);
    logger.info('Test cron job scheduled (runs every 5 minutes)');
  }

  /**
   * Stop all cron jobs
   */
  stopAll() {
    logger.info('Stopping all cron jobs...');
    
    for (const [name, job] of this.jobs) {
      job.stop();
      logger.info(`Stopped cron job: ${name}`);
    }
    
    this.jobs.clear();
    logger.info('All cron jobs stopped');
  }

  /**
   * Stop a specific cron job
   */
  stopJob(jobName) {
    const job = this.jobs.get(jobName);
    if (job) {
      job.stop();
      this.jobs.delete(jobName);
      logger.info(`Stopped cron job: ${jobName}`);
    } else {
      logger.warn(`Cron job not found: ${jobName}`);
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
   * Manually trigger market analysis update
   */
  async triggerMarketAnalysis() {
    try {
      logger.info('Manually triggering market analysis...');
      
      const activeSIPs = await smartSipService.getAllActiveSIPs();
      
      for (const sip of activeSIPs) {
        try {
          const marketAnalysis = await marketScoreService.calculateMarketScore();
          
          sip.marketAnalysis = {
            lastUpdated: new Date(),
            currentScore: marketAnalysis.score,
            currentReason: marketAnalysis.reason,
            indicators: marketAnalysis.indicators
          };
          
          await sip.save();
          
        } catch (error) {
          logger.error(`Error updating market analysis for SIP ${sip._id}:`, error);
        }
      }
      
      logger.info('Manual market analysis update completed');
      return { success: true, message: 'Market analysis updated for all active SIPs' };
      
    } catch (error) {
      logger.error('Error in manual market analysis:', error);
      throw error;
    }
  }

  /**
   * Manually trigger SIP execution for all eligible users
   */
  async triggerSIPExecution() {
    try {
      logger.info('Manually triggering SIP execution...');
      
      const activeSIPs = await smartSipService.getAllActiveSIPs();
      const results = [];
      
      for (const sip of activeSIPs) {
        try {
          const result = await smartSipService.executeSIP(sip.userId);
          results.push({
            userId: sip.userId,
            success: result.success,
            message: result.message,
            amount: result.amount
          });
          
        } catch (error) {
          results.push({
            userId: sip.userId,
            success: false,
            message: error.message
          });
        }
      }
      
      logger.info('Manual SIP execution completed');
      return { success: true, results };
      
    } catch (error) {
      logger.error('Error in manual SIP execution:', error);
      throw error;
    }
  }
}

module.exports = new CronService(); 