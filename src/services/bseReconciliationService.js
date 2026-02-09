const logger = require('../utils/logger');
const { MfOrder } = require('../models');
const mfApiClient = require('../utils/mfApiClient');
const cron = require('node-cron');

class BSEReconciliationService {
  constructor() {
    this.isRunning = false;
    this.cronJob = null;
    this.reconciliationInterval = '0 */2 * * *'; // Every 2 hours
    this.batchSize = 50;
  }

  async initialize() {
    logger.info('Initializing BSE Reconciliation Service');
    this.startCronJob();
  }

  startCronJob() {
    if (this.cronJob) {
      logger.warn('BSE Reconciliation cron job already running');
      return;
    }

    this.cronJob = cron.schedule(this.reconciliationInterval, async () => {
      await this.reconcileOrders();
    });

    logger.info('BSE Reconciliation cron job started', { 
      interval: this.reconciliationInterval 
    });
  }

  stopCronJob() {
    if (this.cronJob) {
      this.cronJob.stop();
      this.cronJob = null;
      logger.info('BSE Reconciliation cron job stopped');
    }
  }

  async reconcileOrders() {
    if (this.isRunning) {
      logger.warn('BSE Reconciliation already running, skipping this cycle');
      return;
    }

    this.isRunning = true;
    const startTime = Date.now();

    try {
      logger.info('Starting BSE order reconciliation');

      const ordersToReconcile = await MfOrder.findOrdersForReconciliation();
      
      if (ordersToReconcile.length === 0) {
        logger.info('No orders to reconcile');
        return;
      }

      logger.info(`Found ${ordersToReconcile.length} orders to reconcile`);

      let reconciledCount = 0;
      let failedCount = 0;
      const errors = [];

      for (let i = 0; i < ordersToReconcile.length; i += this.batchSize) {
        const batch = ordersToReconcile.slice(i, i + this.batchSize);
        
        const results = await Promise.allSettled(
          batch.map(order => this.reconcileOrder(order))
        );

        results.forEach((result, index) => {
          if (result.status === 'fulfilled' && result.value.success) {
            reconciledCount++;
          } else {
            failedCount++;
            errors.push({
              orderId: batch[index]._id,
              bseOrderId: batch[index].bseOrderId,
              error: result.reason || result.value?.error
            });
          }
        });

        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      const duration = Date.now() - startTime;

      logger.info('BSE order reconciliation completed', {
        total: ordersToReconcile.length,
        reconciled: reconciledCount,
        failed: failedCount,
        duration: `${duration}ms`
      });

      if (errors.length > 0) {
        logger.error('BSE reconciliation errors', { errors: errors.slice(0, 10) });
      }

      return {
        success: true,
        stats: {
          total: ordersToReconcile.length,
          reconciled: reconciledCount,
          failed: failedCount,
          duration
        }
      };

    } catch (error) {
      logger.error('BSE order reconciliation failed', { 
        error: error.message,
        stack: error.stack 
      });
      return {
        success: false,
        error: error.message
      };
    } finally {
      this.isRunning = false;
    }
  }

  async reconcileOrder(order) {
    try {
      logger.debug('Reconciling order', { 
        orderId: order._id,
        bseOrderId: order.bseOrderId 
      });

      const statusResponse = await mfApiClient.getOrderStatus(order.bseOrderId);

      if (!statusResponse.success) {
        throw new Error(statusResponse.error || 'Failed to get order status from BSE');
      }

      const bseStatus = statusResponse.data;

      await this.updateOrderFromBSEStatus(order, bseStatus);

      logger.debug('Order reconciled successfully', { 
        orderId: order._id,
        bseOrderId: order.bseOrderId,
        status: bseStatus.status 
      });

      return { success: true };

    } catch (error) {
      logger.error('Failed to reconcile order', {
        orderId: order._id,
        bseOrderId: order.bseOrderId,
        error: error.message
      });

      if (order.retryCount < 5) {
        await order.incrementRetry();
      } else {
        logger.warn('Order exceeded max retry count', {
          orderId: order._id,
          bseOrderId: order.bseOrderId,
          retryCount: order.retryCount
        });
      }

      return { success: false, error: error.message };
    }
  }

  async updateOrderFromBSEStatus(order, bseStatus) {
    const statusMapping = {
      'SUCCESS': 'COMPLETED',
      'COMPLETED': 'COMPLETED',
      'ACCEPTED': 'ACCEPTED',
      'PENDING': 'SUBMITTED',
      'SUBMITTED': 'SUBMITTED',
      'REJECTED': 'REJECTED',
      'FAILED': 'FAILED',
      'CANCELLED': 'CANCELLED'
    };

    const mappedStatus = statusMapping[bseStatus.status] || order.status;

    order.status = mappedStatus;
    order.bseStatus = bseStatus.status;
    order.bseResponse = bseStatus;

    if (bseStatus.allottedUnits) {
      order.allottedUnits = parseFloat(bseStatus.allottedUnits);
    }

    if (bseStatus.nav) {
      order.nav = parseFloat(bseStatus.nav);
    }

    if (bseStatus.allottedAmount) {
      order.allottedAmount = parseFloat(bseStatus.allottedAmount);
    }

    if (bseStatus.settlementDate) {
      order.settlementDate = new Date(bseStatus.settlementDate);
    }

    if (bseStatus.errorMessage) {
      order.errorMessage = bseStatus.errorMessage;
    }

    if (mappedStatus === 'COMPLETED' || mappedStatus === 'REJECTED' || mappedStatus === 'FAILED') {
      order.reconciledAt = new Date();
    }

    await order.save();

    logger.info('Order updated from BSE status', {
      orderId: order._id,
      bseOrderId: order.bseOrderId,
      oldStatus: order.status,
      newStatus: mappedStatus,
      allottedUnits: order.allottedUnits,
      nav: order.nav
    });
  }

  async manualReconciliation(orderId) {
    try {
      const order = await MfOrder.findById(orderId);

      if (!order) {
        return {
          success: false,
          error: 'Order not found'
        };
      }

      if (!order.bseOrderId) {
        return {
          success: false,
          error: 'Order does not have BSE order ID'
        };
      }

      const result = await this.reconcileOrder(order);

      return result;

    } catch (error) {
      logger.error('Manual reconciliation failed', {
        orderId,
        error: error.message
      });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getReconciliationStats(days = 7) {
    try {
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);

      const stats = await MfOrder.aggregate([
        {
          $match: {
            createdAt: { $gte: startDate }
          }
        },
        {
          $group: {
            _id: '$status',
            count: { $sum: 1 }
          }
        }
      ]);

      const pendingReconciliation = await MfOrder.countDocuments({
        status: { $in: ['SUBMITTED', 'ACCEPTED'] },
        bseOrderId: { $exists: true, $ne: null },
        reconciledAt: null,
        createdAt: { $gte: startDate }
      });

      return {
        success: true,
        data: {
          statusBreakdown: stats,
          pendingReconciliation,
          period: `Last ${days} days`
        }
      };

    } catch (error) {
      logger.error('Failed to get reconciliation stats', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }
}

const bseReconciliationService = new BSEReconciliationService();

module.exports = bseReconciliationService;
