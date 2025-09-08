const logger = require('../utils/logger');
const fs = require('fs').promises;
const path = require('path');

class ComplianceService {
  constructor() {
    this.sebiGuidelines = new Map();
    this.complianceChecks = new Map();
    this.testResults = [];
    this.riskLevels = {
      LOW: 'low',
      MEDIUM: 'medium',
      HIGH: 'high',
      CRITICAL: 'critical'
    };
  }

  /**
   * Initialize compliance service
   */
  async initialize() {
    try {
      await this.loadSEBIGuidelines();
      await this.setupComplianceChecks();
      logger.info('Compliance Service initialized successfully');
      return true;
    } catch (error) {
      logger.error('Error initializing compliance service:', error);
      return false;
    }
  }

  /**
   * Load SEBI guidelines
   */
  async loadSEBIGuidelines() {
    this.sebiGuidelines.set('expense_ratio', {
      equity: { max: 2.5, description: 'Maximum 2.5% for equity funds' },
      debt: { max: 2.25, description: 'Maximum 2.25% for debt funds' },
      index: { max: 1.5, description: 'Maximum 1.5% for index funds' },
      etf: { max: 1.0, description: 'Maximum 1% for ETFs' }
    });

    this.sebiGuidelines.set('kyc_requirements', {
      mandatory: true,
      documents: ['PAN', 'Aadhaar', 'Address Proof', 'Bank Account'],
      threshold: 50000,
      description: 'KYC mandatory for investments above ₹50,000'
    });

    this.sebiGuidelines.set('disclosure_requirements', {
      nav_disclosure: 'Daily for open-ended funds',
      portfolio_disclosure: 'Monthly',
      performance_disclosure: 'Regular intervals',
      risk_disclosure: 'Mandatory in all communications'
    });

    this.sebiGuidelines.set('investment_limits', {
      single_stock: 10, // Maximum 10% in single stock
      sector_limit: 25, // Maximum 25% in single sector
      debt_rating: 'AA+ and above',
      description: 'Investment limits for diversification and risk management'
    });

    logger.info('SEBI guidelines loaded successfully');
  }

  /**
   * Setup compliance checks
   */
  async setupComplianceChecks() {
    // Fund compliance checks
    this.complianceChecks.set('fund_expense_ratio', {
      name: 'Fund Expense Ratio Check',
      description: 'Verify fund expense ratio is within SEBI limits',
      check: (fundData) => this.checkExpenseRatio(fundData),
      risk: this.riskLevels.HIGH
    });

    this.complianceChecks.set('fund_investment_limits', {
      name: 'Fund Investment Limits Check',
      description: 'Verify fund follows SEBI investment limits',
      check: (fundData) => this.checkInvestmentLimits(fundData),
      risk: this.riskLevels.CRITICAL
    });

    this.complianceChecks.set('fund_disclosure', {
      name: 'Fund Disclosure Check',
      description: 'Verify fund meets SEBI disclosure requirements',
      check: (fundData) => this.checkDisclosureRequirements(fundData),
      risk: this.riskLevels.MEDIUM
    });

    // User compliance checks
    this.complianceChecks.set('user_kyc', {
      name: 'User KYC Check',
      description: 'Verify user KYC is complete',
      check: (userData) => this.checkUserKYC(userData),
      risk: this.riskLevels.CRITICAL
    });

    this.complianceChecks.set('user_investment_limits', {
      name: 'User Investment Limits Check',
      description: 'Verify user investments are within limits',
      check: (userData) => this.checkUserInvestmentLimits(userData),
      risk: this.riskLevels.HIGH
    });

    // AI response compliance checks
    this.complianceChecks.set('ai_response_compliance', {
      name: 'AI Response Compliance Check',
      description: 'Verify AI responses comply with SEBI guidelines',
      check: (response) => this.checkAIResponseCompliance(response),
      risk: this.riskLevels.HIGH
    });

    logger.info('Compliance checks setup completed');
  }

  /**
   * Check fund expense ratio compliance
   */
  checkExpenseRatio(fundData) {
    const { category, expenseRatio } = fundData;
    const guidelines = this.sebiGuidelines.get('expense_ratio');
    
    if (!guidelines[category]) {
      return {
        compliant: false,
        issue: `Unknown fund category: ${category}`,
        recommendation: 'Verify fund category classification'
      };
    }

    const maxAllowed = guidelines[category].max;
    const isCompliant = expenseRatio <= maxAllowed;

    return {
      compliant: isCompliant,
      current: expenseRatio,
      limit: maxAllowed,
      issue: isCompliant ? null : `Expense ratio ${expenseRatio}% exceeds SEBI limit of ${maxAllowed}%`,
      recommendation: isCompliant ? 'Compliant' : 'Reduce expense ratio to meet SEBI guidelines'
    };
  }

  /**
   * Check fund investment limits compliance
   */
  checkInvestmentLimits(fundData) {
    const { holdings } = fundData;
    const guidelines = this.sebiGuidelines.get('investment_limits');
    const issues = [];

    // Check single stock limit
    for (const holding of holdings) {
      if (holding.allocation > guidelines.single_stock) {
        issues.push(`Stock ${holding.name}: ${holding.allocation}% exceeds ${guidelines.single_stock}% limit`);
      }
    }

    // Check sector concentration
    const sectorAllocation = {};
    for (const holding of holdings) {
      const sector = holding.sector || 'Unknown';
      sectorAllocation[sector] = (sectorAllocation[sector] || 0) + holding.allocation;
    }

    for (const [sector, allocation] of Object.entries(sectorAllocation)) {
      if (allocation > guidelines.sector_limit) {
        issues.push(`Sector ${sector}: ${allocation}% exceeds ${guidelines.sector_limit}% limit`);
      }
    }

    return {
      compliant: issues.length === 0,
      issues: issues,
      recommendation: issues.length === 0 ? 'Compliant' : 'Rebalance portfolio to meet SEBI limits'
    };
  }

  /**
   * Check fund disclosure requirements
   */
  checkDisclosureRequirements(fundData) {
    const { disclosures } = fundData;
    const guidelines = this.sebiGuidelines.get('disclosure_requirements');
    const required = Object.keys(guidelines);
    const missing = [];

    for (const requirement of required) {
      if (!disclosures || !disclosures[requirement]) {
        missing.push(requirement);
      }
    }

    return {
      compliant: missing.length === 0,
      missing: missing,
      recommendation: missing.length === 0 ? 'Compliant' : `Add missing disclosures: ${missing.join(', ')}`
    };
  }

  /**
   * Check user KYC compliance
   */
  checkUserKYC(userData) {
    const { kyc, totalInvestment } = userData;
    const guidelines = this.sebiGuidelines.get('kyc_requirements');

    if (totalInvestment > guidelines.threshold && !kyc.complete) {
      return {
        compliant: false,
        issue: `KYC required for investments above ₹${guidelines.threshold}`,
        recommendation: 'Complete KYC process immediately'
      };
    }

    if (kyc.complete && kyc.documents.length < guidelines.documents.length) {
      return {
        compliant: false,
        issue: 'Incomplete KYC documents',
        recommendation: `Provide all required documents: ${guidelines.documents.join(', ')}`
      };
    }

    return {
      compliant: true,
      recommendation: 'KYC compliant'
    };
  }

  /**
   * Check user investment limits
   */
  checkUserInvestmentLimits(userData) {
    const { investments, riskProfile } = userData;
    const issues = [];

    // Check if investments align with risk profile
    const highRiskAllocation = investments.filter(inv => inv.category === 'equity').reduce((sum, inv) => sum + inv.allocation, 0);
    
    if (riskProfile === 'conservative' && highRiskAllocation > 30) {
      issues.push('High equity allocation for conservative risk profile');
    }

    if (riskProfile === 'moderate' && highRiskAllocation > 70) {
      issues.push('Very high equity allocation for moderate risk profile');
    }

    return {
      compliant: issues.length === 0,
      issues: issues,
      recommendation: issues.length === 0 ? 'Compliant' : 'Consider rebalancing based on risk profile'
    };
  }

  /**
   * Check AI response compliance
   */
  checkAIResponseCompliance(response) {
    const issues = [];
    const responseText = response.toLowerCase();

    // Check for investment advice
    const adviceKeywords = ['buy this', 'sell this', 'invest in', 'guaranteed returns', 'best fund'];
    for (const keyword of adviceKeywords) {
      if (responseText.includes(keyword)) {
        issues.push(`Contains investment advice: "${keyword}"`);
      }
    }

    // Check for performance guarantees
    if (responseText.includes('guarantee') || responseText.includes('assured returns')) {
      issues.push('Contains performance guarantees');
    }

    // Check for international/crypto content
    if (responseText.includes('crypto') || responseText.includes('bitcoin') || responseText.includes('international')) {
      issues.push('Contains non-Indian mutual fund content');
    }

    // Check for proper disclaimers
    if (!responseText.includes('consult your advisor') && !responseText.includes('educational purpose')) {
      issues.push('Missing appropriate disclaimers');
    }

    return {
      compliant: issues.length === 0,
      issues: issues,
      recommendation: issues.length === 0 ? 'Compliant' : 'Review and modify response to comply with SEBI guidelines'
    };
  }

  /**
   * Run comprehensive compliance audit
   */
  async runComplianceAudit(data) {
    const auditResults = {
      timestamp: new Date().toISOString(),
      overallCompliant: true,
      checks: [],
      summary: {
        total: 0,
        compliant: 0,
        nonCompliant: 0,
        criticalIssues: 0
      }
    };

    for (const [checkId, check] of this.complianceChecks.entries()) {
      try {
        const result = await check.check(data);
        const checkResult = {
          id: checkId,
          name: check.name,
          description: check.description,
          risk: check.risk,
          result: result,
          compliant: result.compliant
        };

        auditResults.checks.push(checkResult);
        auditResults.summary.total++;

        if (result.compliant) {
          auditResults.summary.compliant++;
        } else {
          auditResults.summary.nonCompliant++;
          auditResults.overallCompliant = false;

          if (check.risk === this.riskLevels.CRITICAL) {
            auditResults.summary.criticalIssues++;
          }
        }
      } catch (error) {
        logger.error(`Error running compliance check ${checkId}:`, error);
        auditResults.checks.push({
          id: checkId,
          name: check.name,
          error: error.message,
          compliant: false
        });
        auditResults.summary.nonCompliant++;
        auditResults.overallCompliant = false;
      }
    }

    this.testResults.push(auditResults);
    logger.info(`Compliance audit completed: ${auditResults.summary.compliant}/${auditResults.summary.total} checks passed`);
    
    return auditResults;
  }

  /**
   * Generate compliance report
   */
  async generateComplianceReport() {
    const report = {
      generatedAt: new Date().toISOString(),
      summary: {
        totalAudits: this.testResults.length,
        overallCompliance: this.calculateOverallCompliance(),
        criticalIssues: this.countCriticalIssues(),
        recentIssues: this.getRecentIssues()
      },
      detailedResults: this.testResults.slice(-10), // Last 10 audits
      recommendations: this.generateRecommendations()
    };

    return report;
  }

  /**
   * Calculate overall compliance percentage
   */
  calculateOverallCompliance() {
    if (this.testResults.length === 0) return 0;

    let totalChecks = 0;
    let compliantChecks = 0;

    for (const result of this.testResults) {
      totalChecks += result.summary.total;
      compliantChecks += result.summary.compliant;
    }

    return totalChecks > 0 ? (compliantChecks / totalChecks) * 100 : 0;
  }

  /**
   * Count critical issues
   */
  countCriticalIssues() {
    return this.testResults.reduce((count, result) => count + result.summary.criticalIssues, 0);
  }

  /**
   * Get recent compliance issues
   */
  getRecentIssues() {
    const recentIssues = [];
    const recentResults = this.testResults.slice(-5); // Last 5 audits

    for (const result of recentResults) {
      for (const check of result.checks) {
        if (!check.compliant && check.result && check.result.issues) {
          recentIssues.push({
            timestamp: result.timestamp,
            check: check.name,
            issues: check.result.issues
          });
        }
      }
    }

    return recentIssues;
  }

  /**
   * Generate compliance recommendations
   */
  generateRecommendations() {
    const recommendations = [];

    // Analyze common issues
    const issueFrequency = {};
    for (const result of this.testResults) {
      for (const check of result.checks) {
        if (!check.compliant && check.result && check.result.issues) {
          for (const issue of check.result.issues) {
            issueFrequency[issue] = (issueFrequency[issue] || 0) + 1;
          }
        }
      }
    }

    // Generate recommendations based on frequency
    for (const [issue, frequency] of Object.entries(issueFrequency)) {
      if (frequency > 2) { // Issue appears more than twice
        recommendations.push({
          issue: issue,
          frequency: frequency,
          priority: frequency > 5 ? 'HIGH' : 'MEDIUM',
          action: this.getRecommendationAction(issue)
        });
      }
    }

    return recommendations;
  }

  /**
   * Get recommendation action for specific issue
   */
  getRecommendationAction(issue) {
    const actionMap = {
      'expense ratio': 'Review and optimize fund expense ratios',
      'investment limits': 'Rebalance portfolio to meet SEBI limits',
      'KYC': 'Complete KYC process for all users',
      'disclosure': 'Ensure all required disclosures are made',
      'investment advice': 'Review AI responses to avoid investment advice',
      'disclaimers': 'Add appropriate disclaimers to all communications'
    };

    for (const [key, action] of Object.entries(actionMap)) {
      if (issue.toLowerCase().includes(key)) {
        return action;
      }
    }

    return 'Review and address the specific compliance issue';
  }

  /**
   * Save compliance report
   */
  async saveComplianceReport(report) {
    try {
      const reportsPath = path.join(__dirname, '../../reports');
      await fs.mkdir(reportsPath, { recursive: true });
      
      const filename = `compliance-report-${new Date().toISOString().split('T')[0]}.json`;
      const filepath = path.join(reportsPath, filename);
      
      await fs.writeFile(filepath, JSON.stringify(report, null, 2));
      logger.info(`Compliance report saved: ${filename}`);
      
      return filepath;
    } catch (error) {
      logger.error('Error saving compliance report:', error);
      throw error;
    }
  }
}

module.exports = ComplianceService; 