const logger = require('../utils/logger');
const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');
const { User, UserPortfolio, Holding, Transaction, ComplianceLog, AuditLog } = require('../models');

class ComplianceEngine {
  constructor() {
    this.regulatoryLimits = {
      MAX_SINGLE_FUND_ALLOCATION: 0.25, // 25% in single fund
      MAX_SECTOR_ALLOCATION: 0.35, // 35% in single sector
      MAX_ELSS_ALLOCATION: 0.15, // 15% in ELSS
      MIN_DIVERSIFICATION: 0.05, // Minimum 5% in each category
      MAX_ANNUAL_CHURN: 0.5, // Maximum 50% annual portfolio churn
      MAX_DAILY_TRANSACTIONS: 10, // Maximum 10 transactions per day
      MAX_MONTHLY_INVESTMENT: 1000000, // Maximum 10L per month
      MIN_HOLDING_PERIOD: 90 // Minimum 90 days holding period
    };

    this.reportTypes = {
      SEBI_MONTHLY: 'sebi_monthly',
      AMFI_QUARTERLY: 'amfi_quarterly',
      COMPLIANCE_AUDIT: 'compliance_audit',
      USER_BEHAVIOR: 'user_behavior',
      RISK_ASSESSMENT: 'risk_assessment',
      TAX_COMPLIANCE: 'tax_compliance'
    };

    this.violationTypes = {
      OVER_ALLOCATION: 'over_allocation',
      UNDER_DIVERSIFICATION: 'under_diversification',
      EXCESSIVE_CHURN: 'excessive_churn',
      FREQUENT_TRADING: 'frequent_trading',
      LARGE_INVESTMENTS: 'large_investments',
      SHORT_HOLDING: 'short_holding',
      TAX_EVASION: 'tax_evasion',
      KYC_VIOLATION: 'kyc_violation'
    };
  }

  /**
   * Generate SEBI/AMFI reports (dummy now)
   */
  async generateSEBIReport(userId, period = 'monthly') {
    try {
      logger.info('Generating SEBI report', { userId, period });

      const user = await User.findById(userId);
      const portfolio = await UserPortfolio.findOne({ userId });
      const holdings = await Holding.find({ userId, isActive: true });
      const transactions = await Transaction.find({ userId });

      if (!user || !portfolio) {
        throw new Error('User or portfolio not found');
      }

      const reportData = {
        userInfo: {
          userId: user._id,
          name: user.name,
          email: user.email,
          phone: user.phone,
          kycStatus: user.kycStatus || 'PENDING'
        },
        portfolioSummary: {
          totalValue: portfolio.totalValue || 0,
          totalInvested: portfolio.totalInvested || 0,
          holdingsCount: holdings.length,
          assetAllocation: this.calculateAssetAllocation(holdings)
        },
        complianceMetrics: await this.calculateComplianceMetrics(userId, holdings, transactions),
        violations: await this.checkSEBIViolations(userId, holdings, transactions),
        recommendations: await this.generateComplianceRecommendations(userId),
        reportPeriod: period,
        generatedDate: new Date().toISOString()
      };

      // Generate PDF report
      const pdfPath = await this.generatePDFReport(reportData, 'SEBI');

      // Store report
      await this.storeComplianceReport(userId, 'SEBI', reportData, pdfPath);

      return {
        success: true,
        data: {
          report: reportData,
          pdfPath,
          message: 'SEBI report generated successfully'
        }
      };
    } catch (error) {
      logger.error('SEBI report generation failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to generate SEBI report',
        error: error.message
      };
    }
  }

  /**
   * Generate AMFI quarterly report
   */
  async generateAMFIReport(userId, quarter = 'Q1') {
    try {
      logger.info('Generating AMFI report', { userId, quarter });

      const user = await User.findById(userId);
      const portfolio = await UserPortfolio.findOne({ userId });
      const holdings = await Holding.find({ userId, isActive: true });
      const transactions = await Transaction.find({ userId });

      if (!user || !portfolio) {
        throw new Error('User or portfolio not found');
      }

      const reportData = {
        userInfo: {
          userId: user._id,
          name: user.name,
          email: user.email,
          phone: user.phone
        },
        portfolioAnalysis: {
          totalValue: portfolio.totalValue || 0,
          totalInvested: portfolio.totalInvested || 0,
          returns: await this.calculatePortfolioReturns(holdings),
          riskMetrics: await this.calculateRiskMetrics(holdings),
          fundDistribution: this.calculateFundDistribution(holdings)
        },
        transactionAnalysis: await this.analyzeTransactions(transactions, quarter),
        complianceStatus: await this.checkAMFICompliance(userId, holdings, transactions),
        recommendations: await this.generateAMFIRecommendations(userId),
        quarter,
        generatedDate: new Date().toISOString()
      };

      // Generate PDF report
      const pdfPath = await this.generatePDFReport(reportData, 'AMFI');

      // Store report
      await this.storeComplianceReport(userId, 'AMFI', reportData, pdfPath);

      return {
        success: true,
        data: {
          report: reportData,
          pdfPath,
          message: 'AMFI report generated successfully'
        }
      };
    } catch (error) {
      logger.error('AMFI report generation failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to generate AMFI report',
        error: error.message
      };
    }
  }

  /**
   * Auto-flag user behavior violating regulatory limits
   */
  async checkRegulatoryViolations(userId) {
    try {
      logger.info('Checking regulatory violations', { userId });

      const user = await User.findById(userId);
      const holdings = await Holding.find({ userId, isActive: true });
      const transactions = await Transaction.find({ userId });

      if (!user) {
        throw new Error('User not found');
      }

      const violations = [];

      // Check single fund allocation
      const fundAllocationViolations = await this.checkFundAllocationViolations(holdings);
      violations.push(...fundAllocationViolations);

      // Check sector allocation
      const sectorAllocationViolations = await this.checkSectorAllocationViolations(holdings);
      violations.push(...sectorAllocationViolations);

      // Check diversification
      const diversificationViolations = await this.checkDiversificationViolations(holdings);
      violations.push(...diversificationViolations);

      // Check portfolio churn
      const churnViolations = await this.checkChurnViolations(transactions);
      violations.push(...churnViolations);

      // Check trading frequency
      const tradingViolations = await this.checkTradingFrequencyViolations(transactions);
      violations.push(...tradingViolations);

      // Check investment limits
      const investmentViolations = await this.checkInvestmentLimitViolations(transactions);
      violations.push(...investmentViolations);

      // Check holding period
      const holdingViolations = await this.checkHoldingPeriodViolations(holdings);
      violations.push(...holdingViolations);

      // Store violations
      await this.storeViolations(userId, violations);

      // Generate alerts for high-priority violations
      const alerts = this.generateViolationAlerts(violations);

      return {
        success: true,
        data: {
          violations,
          alerts,
          totalViolations: violations.length,
          highPriorityViolations: violations.filter(v => v.priority === 'HIGH').length
        }
      };
    } catch (error) {
      logger.error('Regulatory violations check failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to check regulatory violations',
        error: error.message
      };
    }
  }

  /**
   * Auto-generate reports in PDF format for admin
   */
  async generateAdminReports(adminId, reportType = 'comprehensive') {
    try {
      logger.info('Generating admin reports', { adminId, reportType });

      const reports = [];

      if (reportType === 'comprehensive' || reportType === 'compliance') {
        const complianceReport = await this.generateComprehensiveComplianceReport();
        reports.push(complianceReport);
      }

      if (reportType === 'comprehensive' || reportType === 'user_behavior') {
        const userBehaviorReport = await this.generateUserBehaviorReport();
        reports.push(userBehaviorReport);
      }

      if (reportType === 'comprehensive' || reportType === 'risk_assessment') {
        const riskReport = await this.generateRiskAssessmentReport();
        reports.push(riskReport);
      }

      if (reportType === 'comprehensive' || reportType === 'tax_compliance') {
        const taxReport = await this.generateTaxComplianceReport();
        reports.push(taxReport);
      }

      // Generate combined PDF
      const combinedPdfPath = await this.generateCombinedPDFReport(reports, adminId);

      return {
        success: true,
        data: {
          reports,
          combinedPdfPath,
          totalReports: reports.length
        }
      };
    } catch (error) {
      logger.error('Admin reports generation failed', { error: error.message, adminId });
      return {
        success: false,
        message: 'Failed to generate admin reports',
        error: error.message
      };
    }
  }

  /**
   * Add compliance metrics in Admin dashboard
   */
  async getComplianceMetrics() {
    try {
      logger.info('Getting compliance metrics');

      const metrics = {
        overall: await this.calculateOverallComplianceMetrics(),
        byCategory: await this.calculateComplianceMetricsByCategory(),
        violations: await this.getViolationMetrics(),
        trends: await this.getComplianceTrends(),
        alerts: await this.getActiveComplianceAlerts(),
        lastUpdated: new Date().toISOString()
      };

      return {
        success: true,
        data: metrics
      };
    } catch (error) {
      logger.error('Compliance metrics retrieval failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to get compliance metrics',
        error: error.message
      };
    }
  }

  /**
   * Monitor real-time compliance
   */
  async monitorRealTimeCompliance() {
    try {
      logger.info('Starting real-time compliance monitoring');

      const monitoringResults = {
        activeViolations: await this.getActiveViolations(),
        pendingReviews: await this.getPendingReviews(),
        complianceScore: await this.calculateOverallComplianceScore(),
        riskLevel: await this.assessOverallRiskLevel(),
        recommendations: await this.generateSystemWideRecommendations(),
        timestamp: new Date().toISOString()
      };

      // Store monitoring results
      await this.storeMonitoringResults(monitoringResults);

      return {
        success: true,
        data: monitoringResults
      };
    } catch (error) {
      logger.error('Real-time compliance monitoring failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to monitor real-time compliance',
        error: error.message
      };
    }
  }

  // Helper methods for SEBI report generation
  async calculateComplianceMetrics(userId, holdings, transactions) {
    try {
      const metrics = {
        diversificationScore: this.calculateDiversificationScore(holdings),
        allocationCompliance: this.calculateAllocationCompliance(holdings),
        tradingCompliance: this.calculateTradingCompliance(transactions),
        kycCompliance: await this.checkKYCCompliance(userId),
        taxCompliance: await this.checkTaxCompliance(userId, transactions),
        overallScore: 0
      };

      // Calculate overall compliance score
      metrics.overallScore = (
        metrics.diversificationScore * 0.3 +
        metrics.allocationCompliance * 0.25 +
        metrics.tradingCompliance * 0.2 +
        metrics.kycCompliance * 0.15 +
        metrics.taxCompliance * 0.1
      );

      return metrics;
    } catch (error) {
      logger.error('Compliance metrics calculation failed', { error: error.message });
      return {
        diversificationScore: 0,
        allocationCompliance: 0,
        tradingCompliance: 0,
        kycCompliance: 0,
        taxCompliance: 0,
        overallScore: 0
      };
    }
  }

  async checkSEBIViolations(userId, holdings, transactions) {
    try {
      const violations = [];

      // Check for SEBI-specific violations
      const fundAllocationViolations = await this.checkFundAllocationViolations(holdings);
      violations.push(...fundAllocationViolations);

      const tradingViolations = await this.checkTradingFrequencyViolations(transactions);
      violations.push(...tradingViolations);

      return violations;
    } catch (error) {
      logger.error('SEBI violations check failed', { error: error.message });
      return [];
    }
  }

  async generateComplianceRecommendations(userId) {
    try {
      const recommendations = [];

      // Get user's compliance status
      const complianceStatus = await this.getUserComplianceStatus(userId);

      if (complianceStatus.diversificationScore < 0.7) {
        recommendations.push({
          type: 'DIVERSIFICATION',
          priority: 'HIGH',
          message: 'Increase portfolio diversification',
          action: 'Add funds from different categories'
        });
      }

      if (complianceStatus.allocationScore < 0.8) {
        recommendations.push({
          type: 'ALLOCATION',
          priority: 'MEDIUM',
          message: 'Review fund allocations',
          action: 'Rebalance portfolio'
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('Compliance recommendations generation failed', { error: error.message });
      return [];
    }
  }

  // Helper methods for AMFI report generation
  async calculatePortfolioReturns(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      
      return {
        absolute: totalValue - totalInvested,
        percentage: totalInvested > 0 ? ((totalValue - totalInvested) / totalInvested) * 100 : 0,
        timeWeighted: await this.calculateTimeWeightedReturn(holdings)
      };
    } catch (error) {
      logger.error('Portfolio returns calculation failed', { error: error.message });
      return { absolute: 0, percentage: 0, timeWeighted: 0 };
    }
  }

  async calculateRiskMetrics(holdings) {
    try {
      return {
        volatility: this.calculatePortfolioVolatility(holdings),
        sharpeRatio: this.calculateSharpeRatio(holdings),
        maxDrawdown: this.calculateMaxDrawdown(holdings),
        beta: this.calculatePortfolioBeta(holdings)
      };
    } catch (error) {
      logger.error('Risk metrics calculation failed', { error: error.message });
      return { volatility: 0, sharpeRatio: 0, maxDrawdown: 0, beta: 0 };
    }
  }

  calculateFundDistribution(holdings) {
    try {
      const distribution = {};
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);

      holdings.forEach(holding => {
        const category = holding.fundCategory || 'others';
        distribution[category] = (distribution[category] || 0) + (holding.currentValue / totalValue);
      });

      return distribution;
    } catch (error) {
      logger.error('Fund distribution calculation failed', { error: error.message });
      return {};
    }
  }

  async analyzeTransactions(transactions, quarter) {
    try {
      const quarterTransactions = transactions.filter(t => {
        const transactionDate = new Date(t.date);
        const quarterStart = this.getQuarterStart(quarter);
        const quarterEnd = this.getQuarterEnd(quarter);
        return transactionDate >= quarterStart && transactionDate <= quarterEnd;
      });

      return {
        totalTransactions: quarterTransactions.length,
        buyTransactions: quarterTransactions.filter(t => t.type === 'BUY').length,
        sellTransactions: quarterTransactions.filter(t => t.type === 'SELL').length,
        totalVolume: quarterTransactions.reduce((sum, t) => sum + t.amount, 0),
        averageTransactionSize: quarterTransactions.length > 0 ? 
          quarterTransactions.reduce((sum, t) => sum + t.amount, 0) / quarterTransactions.length : 0
      };
    } catch (error) {
      logger.error('Transaction analysis failed', { error: error.message });
      return {
        totalTransactions: 0,
        buyTransactions: 0,
        sellTransactions: 0,
        totalVolume: 0,
        averageTransactionSize: 0
      };
    }
  }

  async checkAMFICompliance(userId, holdings, transactions) {
    try {
      const compliance = {
        fundAllocation: this.checkFundAllocationCompliance(holdings),
        tradingPatterns: this.checkTradingPatternCompliance(transactions),
        riskManagement: this.checkRiskManagementCompliance(holdings),
        documentation: await this.checkDocumentationCompliance(userId)
      };

      return compliance;
    } catch (error) {
      logger.error('AMFI compliance check failed', { error: error.message });
      return {
        fundAllocation: false,
        tradingPatterns: false,
        riskManagement: false,
        documentation: false
      };
    }
  }

  async generateAMFIRecommendations(userId) {
    try {
      const recommendations = [];

      // Get AMFI-specific recommendations
      const amfiStatus = await this.getAMFIComplianceStatus(userId);

      if (!amfiStatus.fundAllocation) {
        recommendations.push({
          type: 'FUND_ALLOCATION',
          priority: 'HIGH',
          message: 'Review fund allocation for AMFI compliance',
          action: 'Rebalance portfolio'
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('AMFI recommendations generation failed', { error: error.message });
      return [];
    }
  }

  // Helper methods for violation checking
  async checkFundAllocationViolations(holdings) {
    try {
      const violations = [];
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);

      holdings.forEach(holding => {
        const allocation = holding.currentValue / totalValue;
        
        if (allocation > this.regulatoryLimits.MAX_SINGLE_FUND_ALLOCATION) {
          violations.push({
            type: this.violationTypes.OVER_ALLOCATION,
            fundName: holding.fundName,
            currentAllocation: allocation,
            maxAllowed: this.regulatoryLimits.MAX_SINGLE_FUND_ALLOCATION,
            priority: 'HIGH',
            message: `Fund allocation exceeds ${this.regulatoryLimits.MAX_SINGLE_FUND_ALLOCATION * 100}% limit`,
            recommendation: 'Consider reducing allocation or diversifying'
          });
        }
      });

      return violations;
    } catch (error) {
      logger.error('Fund allocation violations check failed', { error: error.message });
      return [];
    }
  }

  async checkSectorAllocationViolations(holdings) {
    try {
      const violations = [];
      const sectorAllocation = {};
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);

      // Calculate sector allocation
      holdings.forEach(holding => {
        const sector = holding.fundCategory || 'others';
        sectorAllocation[sector] = (sectorAllocation[sector] || 0) + (holding.currentValue / totalValue);
      });

      // Check for violations
      Object.entries(sectorAllocation).forEach(([sector, allocation]) => {
        if (allocation > this.regulatoryLimits.MAX_SECTOR_ALLOCATION) {
          violations.push({
            type: this.violationTypes.OVER_ALLOCATION,
            sector: sector,
            currentAllocation: allocation,
            maxAllowed: this.regulatoryLimits.MAX_SECTOR_ALLOCATION,
            priority: 'MEDIUM',
            message: `Sector allocation exceeds ${this.regulatoryLimits.MAX_SECTOR_ALLOCATION * 100}% limit`,
            recommendation: 'Diversify across sectors'
          });
        }
      });

      return violations;
    } catch (error) {
      logger.error('Sector allocation violations check failed', { error: error.message });
      return [];
    }
  }

  async checkDiversificationViolations(holdings) {
    try {
      const violations = [];
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);

      // Check minimum diversification
      holdings.forEach(holding => {
        const allocation = holding.currentValue / totalValue;
        
        if (allocation < this.regulatoryLimits.MIN_DIVERSIFICATION) {
          violations.push({
            type: this.violationTypes.UNDER_DIVERSIFICATION,
            fundName: holding.fundName,
            currentAllocation: allocation,
            minRequired: this.regulatoryLimits.MIN_DIVERSIFICATION,
            priority: 'LOW',
            message: `Fund allocation below ${this.regulatoryLimits.MIN_DIVERSIFICATION * 100}% minimum`,
            recommendation: 'Consider increasing allocation or switching'
          });
        }
      });

      return violations;
    } catch (error) {
      logger.error('Diversification violations check failed', { error: error.message });
      return [];
    }
  }

  async checkChurnViolations(transactions) {
    try {
      const violations = [];
      const now = new Date();
      const oneYearAgo = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());

      // Get transactions from last year
      const yearlyTransactions = transactions.filter(t => new Date(t.date) >= oneYearAgo);
      const totalVolume = yearlyTransactions.reduce((sum, t) => sum + t.amount, 0);

      // Calculate portfolio value
      const portfolioValue = 1000000; // Mock value
      const churnRate = totalVolume / portfolioValue;

      if (churnRate > this.regulatoryLimits.MAX_ANNUAL_CHURN) {
        violations.push({
          type: this.violationTypes.EXCESSIVE_CHURN,
          currentChurn: churnRate,
          maxAllowed: this.regulatoryLimits.MAX_ANNUAL_CHURN,
          priority: 'HIGH',
          message: `Annual portfolio churn rate exceeds ${this.regulatoryLimits.MAX_ANNUAL_CHURN * 100}%`,
          recommendation: 'Reduce trading frequency'
        });
      }

      return violations;
    } catch (error) {
      logger.error('Churn violations check failed', { error: error.message });
      return [];
    }
  }

  async checkTradingFrequencyViolations(transactions) {
    try {
      const violations = [];
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

      // Get today's transactions
      const todayTransactions = transactions.filter(t => {
        const transactionDate = new Date(t.date);
        return transactionDate >= today;
      });

      if (todayTransactions.length > this.regulatoryLimits.MAX_DAILY_TRANSACTIONS) {
        violations.push({
          type: this.violationTypes.FREQUENT_TRADING,
          currentTransactions: todayTransactions.length,
          maxAllowed: this.regulatoryLimits.MAX_DAILY_TRANSACTIONS,
          priority: 'MEDIUM',
          message: `Daily transaction count exceeds ${this.regulatoryLimits.MAX_DAILY_TRANSACTIONS}`,
          recommendation: 'Limit daily trading activity'
        });
      }

      return violations;
    } catch (error) {
      logger.error('Trading frequency violations check failed', { error: error.message });
      return [];
    }
  }

  async checkInvestmentLimitViolations(transactions) {
    try {
      const violations = [];
      const now = new Date();
      const thisMonth = new Date(now.getFullYear(), now.getMonth(), 1);

      // Get this month's transactions
      const monthlyTransactions = transactions.filter(t => new Date(t.date) >= thisMonth);
      const monthlyInvestment = monthlyTransactions
        .filter(t => t.type === 'BUY')
        .reduce((sum, t) => sum + t.amount, 0);

      if (monthlyInvestment > this.regulatoryLimits.MAX_MONTHLY_INVESTMENT) {
        violations.push({
          type: this.violationTypes.LARGE_INVESTMENTS,
          currentInvestment: monthlyInvestment,
          maxAllowed: this.regulatoryLimits.MAX_MONTHLY_INVESTMENT,
          priority: 'HIGH',
          message: `Monthly investment exceeds ₹${this.regulatoryLimits.MAX_MONTHLY_INVESTMENT.toLocaleString()}`,
          recommendation: 'Review investment limits'
        });
      }

      return violations;
    } catch (error) {
      logger.error('Investment limit violations check failed', { error: error.message });
      return [];
    }
  }

  async checkHoldingPeriodViolations(holdings) {
    try {
      const violations = [];
      const now = new Date();

      holdings.forEach(holding => {
        const holdingPeriod = (now - new Date(holding.purchaseDate)) / (1000 * 60 * 60 * 24);
        
        if (holdingPeriod < this.regulatoryLimits.MIN_HOLDING_PERIOD) {
          violations.push({
            type: this.violationTypes.SHORT_HOLDING,
            fundName: holding.fundName,
            holdingPeriod: Math.floor(holdingPeriod),
            minRequired: this.regulatoryLimits.MIN_HOLDING_PERIOD,
            priority: 'LOW',
            message: `Holding period below ${this.regulatoryLimits.MIN_HOLDING_PERIOD} days`,
            recommendation: 'Hold for longer period'
          });
        }
      });

      return violations;
    } catch (error) {
      logger.error('Holding period violations check failed', { error: error.message });
      return [];
    }
  }

  // Helper methods for PDF generation
  async generatePDFReport(reportData, reportType) {
    try {
      const doc = new PDFDocument();
      const fileName = `${reportType}_report_${Date.now()}.pdf`;
      const filePath = path.join(__dirname, '../reports', fileName);

      // Ensure reports directory exists
      const reportsDir = path.dirname(filePath);
      if (!fs.existsSync(reportsDir)) {
        fs.mkdirSync(reportsDir, { recursive: true });
      }

      const stream = fs.createWriteStream(filePath);
      doc.pipe(stream);

      // Add content to PDF
      doc.fontSize(20).text(`${reportType} Compliance Report`, { align: 'center' });
      doc.moveDown();
      doc.fontSize(12).text(`Generated on: ${new Date().toLocaleString()}`);
      doc.moveDown();

      // Add user information
      doc.fontSize(16).text('User Information');
      doc.fontSize(10).text(`Name: ${reportData.userInfo.name}`);
      doc.text(`Email: ${reportData.userInfo.email}`);
      doc.text(`Phone: ${reportData.userInfo.phone}`);
      doc.moveDown();

      // Add portfolio summary
      doc.fontSize(16).text('Portfolio Summary');
      doc.fontSize(10).text(`Total Value: ₹${reportData.portfolioSummary.totalValue.toLocaleString()}`);
      doc.text(`Total Invested: ₹${reportData.portfolioSummary.totalInvested.toLocaleString()}`);
      doc.text(`Holdings Count: ${reportData.portfolioSummary.holdingsCount}`);
      doc.moveDown();

      // Add compliance metrics
      if (reportData.complianceMetrics) {
        doc.fontSize(16).text('Compliance Metrics');
        doc.fontSize(10).text(`Overall Score: ${(reportData.complianceMetrics.overallScore * 100).toFixed(1)}%`);
        doc.text(`Diversification Score: ${(reportData.complianceMetrics.diversificationScore * 100).toFixed(1)}%`);
        doc.moveDown();
      }

      // Add violations
      if (reportData.violations && reportData.violations.length > 0) {
        doc.fontSize(16).text('Violations');
        reportData.violations.forEach(violation => {
          doc.fontSize(10).text(`• ${violation.message}`, { color: 'red' });
        });
        doc.moveDown();
      }

      // Add recommendations
      if (reportData.recommendations && reportData.recommendations.length > 0) {
        doc.fontSize(16).text('Recommendations');
        reportData.recommendations.forEach(rec => {
          doc.fontSize(10).text(`• ${rec.message}`);
        });
      }

      doc.end();

      return new Promise((resolve, reject) => {
        stream.on('finish', () => resolve(filePath));
        stream.on('error', reject);
      });
    } catch (error) {
      logger.error('PDF report generation failed', { error: error.message });
      throw error;
    }
  }

  // Utility methods
  calculateAssetAllocation(holdings) {
    try {
      const allocation = {};
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);

      holdings.forEach(holding => {
        const category = holding.fundCategory || 'others';
        allocation[category] = (allocation[category] || 0) + (holding.currentValue / totalValue);
      });

      return allocation;
    } catch (error) {
      logger.error('Asset allocation calculation failed', { error: error.message });
      return {};
    }
  }

  calculateDiversificationScore(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const categories = new Set(holdings.map(h => h.fundCategory));
      const maxAllocation = Math.max(...holdings.map(h => h.currentValue / totalValue));
      
      const diversificationScore = (categories.size / 5) * (1 - maxAllocation);
      return Math.min(1, Math.max(0, diversificationScore));
    } catch (error) {
      logger.error('Diversification score calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateAllocationCompliance(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      let complianceScore = 1;

      holdings.forEach(holding => {
        const allocation = holding.currentValue / totalValue;
        if (allocation > this.regulatoryLimits.MAX_SINGLE_FUND_ALLOCATION) {
          complianceScore -= 0.1;
        }
      });

      return Math.max(0, complianceScore);
    } catch (error) {
      logger.error('Allocation compliance calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateTradingCompliance(transactions) {
    try {
      // Mock trading compliance calculation
      return 0.9;
    } catch (error) {
      logger.error('Trading compliance calculation failed', { error: error.message });
      return 0;
    }
  }

  async checkKYCCompliance(userId) {
    try {
      const user = await User.findById(userId);
      return user?.kycStatus === 'COMPLETED' ? 1 : 0;
    } catch (error) {
      logger.error('KYC compliance check failed', { error: error.message });
      return 0;
    }
  }

  async checkTaxCompliance(userId, transactions) {
    try {
      // Mock tax compliance check
      return 0.95;
    } catch (error) {
      logger.error('Tax compliance check failed', { error: error.message });
      return 0;
    }
  }

  // Storage methods
  async storeComplianceReport(userId, reportType, reportData, pdfPath) {
    try {
      // In real implementation, store in database
      logger.info('Compliance report stored', { userId, reportType, pdfPath });
    } catch (error) {
      logger.error('Compliance report storage failed', { error: error.message });
    }
  }

  async storeViolations(userId, violations) {
    try {
      // In real implementation, store in database
      logger.info('Violations stored', { userId, violationsCount: violations.length });
    } catch (error) {
      logger.error('Violations storage failed', { error: error.message });
    }
  }

  generateViolationAlerts(violations) {
    try {
      return violations
        .filter(v => v.priority === 'HIGH')
        .map(v => ({
          type: 'COMPLIANCE_VIOLATION',
          severity: 'HIGH',
          message: v.message,
          action: v.recommendation
        }));
    } catch (error) {
      logger.error('Violation alerts generation failed', { error: error.message });
      return [];
    }
  }

  // Additional helper methods for comprehensive functionality
  async calculateOverallComplianceMetrics() {
    try {
      // Mock overall compliance metrics
      return {
        totalUsers: 1000,
        compliantUsers: 850,
        complianceRate: 0.85,
        averageScore: 0.78
      };
    } catch (error) {
      logger.error('Overall compliance metrics calculation failed', { error: error.message });
      return { totalUsers: 0, compliantUsers: 0, complianceRate: 0, averageScore: 0 };
    }
  }

  async calculateComplianceMetricsByCategory() {
    try {
      return {
        diversification: { compliant: 800, total: 1000, rate: 0.8 },
        allocation: { compliant: 900, total: 1000, rate: 0.9 },
        trading: { compliant: 950, total: 1000, rate: 0.95 },
        kyc: { compliant: 980, total: 1000, rate: 0.98 }
      };
    } catch (error) {
      logger.error('Compliance metrics by category calculation failed', { error: error.message });
      return {};
    }
  }

  async getViolationMetrics() {
    try {
      return {
        totalViolations: 150,
        highPriority: 25,
        mediumPriority: 75,
        lowPriority: 50,
        resolvedViolations: 100
      };
    } catch (error) {
      logger.error('Violation metrics retrieval failed', { error: error.message });
      return { totalViolations: 0, highPriority: 0, mediumPriority: 0, lowPriority: 0, resolvedViolations: 0 };
    }
  }

  async getComplianceTrends() {
    try {
      return {
        monthly: [0.75, 0.78, 0.80, 0.82, 0.85, 0.87],
        quarterly: [0.70, 0.75, 0.80, 0.85],
        yearly: [0.65, 0.75, 0.85]
      };
    } catch (error) {
      logger.error('Compliance trends retrieval failed', { error: error.message });
      return { monthly: [], quarterly: [], yearly: [] };
    }
  }

  async getActiveComplianceAlerts() {
    try {
      return [
        {
          id: '1',
          type: 'HIGH_RISK_VIOLATION',
          message: 'Multiple users with excessive fund allocation',
          count: 15,
          priority: 'HIGH'
        }
      ];
    } catch (error) {
      logger.error('Active compliance alerts retrieval failed', { error: error.message });
      return [];
    }
  }
}

module.exports = new ComplianceEngine(); 