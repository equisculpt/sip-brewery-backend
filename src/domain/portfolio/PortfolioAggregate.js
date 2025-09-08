/**
 * 🏛️ DOMAIN-DRIVEN DESIGN: PORTFOLIO AGGREGATE
 * 
 * Portfolio aggregate root implementing DDD patterns
 * Encapsulates business logic and maintains consistency
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const { v4: uuidv4 } = require('uuid');
const { DomainEvent } = require('../base/DomainEvent');
const { AggregateRoot } = require('../base/AggregateRoot');
const { Money } = require('../valueObjects/Money');
const { AssetAllocation } = require('../valueObjects/AssetAllocation');

/**
 * Portfolio Domain Events
 */
class PortfolioCreated extends DomainEvent {
  constructor(portfolioId, userId, initialAllocation) {
    super('portfolio.created', {
      portfolioId,
      userId,
      initialAllocation
    });
  }
}

class InvestmentAdded extends DomainEvent {
  constructor(portfolioId, investment) {
    super('portfolio.investment.added', {
      portfolioId,
      investment
    });
  }
}

class PortfolioRebalanced extends DomainEvent {
  constructor(portfolioId, oldAllocation, newAllocation, reason) {
    super('portfolio.rebalanced', {
      portfolioId,
      oldAllocation,
      newAllocation,
      reason
    });
  }
}

class RiskLevelChanged extends DomainEvent {
  constructor(portfolioId, oldRiskLevel, newRiskLevel, reason) {
    super('portfolio.risk.changed', {
      portfolioId,
      oldRiskLevel,
      newRiskLevel,
      reason
    });
  }
}

/**
 * Portfolio Business Rules
 */
class PortfolioBusinessRules {
  static validateMinimumInvestment(amount) {
    const minimumAmount = new Money(500, 'INR'); // ₹500 minimum
    if (amount.isLessThan(minimumAmount)) {
      throw new Error(`Minimum investment amount is ${minimumAmount.toString()}`);
    }
  }

  static validateAllocationLimits(allocation) {
    // Maximum 80% in equity for retail investors
    if (allocation.equityPercentage > 80) {
      throw new Error('Equity allocation cannot exceed 80% for retail investors');
    }

    // Minimum 10% in debt for risk management
    if (allocation.debtPercentage < 10) {
      throw new Error('Minimum 10% allocation required in debt instruments');
    }

    // Total allocation must equal 100%
    if (allocation.getTotalPercentage() !== 100) {
      throw new Error('Total allocation must equal 100%');
    }
  }

  static validateRebalancingThreshold(currentAllocation, targetAllocation) {
    const threshold = 5; // 5% threshold
    const equityDrift = Math.abs(currentAllocation.equityPercentage - targetAllocation.equityPercentage);
    
    if (equityDrift < threshold) {
      throw new Error(`Rebalancing threshold not met. Current drift: ${equityDrift}%`);
    }
  }

  static validateRiskCapacity(userAge, riskLevel) {
    const maxEquityByAge = Math.max(100 - userAge, 20); // Rule of thumb: 100 - age
    
    if (riskLevel === 'AGGRESSIVE' && userAge > 50) {
      throw new Error('Aggressive risk profile not recommended for investors over 50');
    }

    if (riskLevel === 'CONSERVATIVE' && userAge < 30) {
      console.warn('Conservative risk profile may limit growth potential for young investors');
    }
  }
}

/**
 * Portfolio Aggregate Root
 */
class PortfolioAggregate extends AggregateRoot {
  constructor(id, userId) {
    super(id);
    this.userId = userId;
    this.investments = new Map();
    this.totalValue = new Money(0, 'INR');
    this.currentAllocation = null;
    this.targetAllocation = null;
    this.riskProfile = 'MODERATE';
    this.goals = [];
    this.lastRebalancedAt = null;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }

  /**
   * Create new portfolio
   */
  static create(userId, initialAllocation, riskProfile = 'MODERATE') {
    const portfolioId = uuidv4();
    const portfolio = new PortfolioAggregate(portfolioId, userId);
    
    // Validate initial allocation
    PortfolioBusinessRules.validateAllocationLimits(initialAllocation);
    
    portfolio.targetAllocation = initialAllocation;
    portfolio.currentAllocation = new AssetAllocation(0, 0, 0, 0); // Empty initially
    portfolio.riskProfile = riskProfile;
    
    // Raise domain event
    portfolio.addDomainEvent(new PortfolioCreated(
      portfolioId,
      userId,
      initialAllocation
    ));
    
    return portfolio;
  }

  /**
   * Add investment to portfolio
   */
  addInvestment(fundCode, amount, investmentType = 'SIP') {
    // Validate minimum investment
    PortfolioBusinessRules.validateMinimumInvestment(amount);
    
    const investment = {
      id: uuidv4(),
      fundCode,
      amount,
      investmentType,
      investedAt: new Date(),
      units: null, // Will be calculated based on NAV
      currentValue: amount // Initially same as invested amount
    };
    
    // Add to investments
    this.investments.set(investment.id, investment);
    
    // Update total value
    this.totalValue = this.totalValue.add(amount);
    
    // Recalculate current allocation
    this.recalculateCurrentAllocation();
    
    // Update timestamp
    this.updatedAt = new Date();
    
    // Raise domain event
    this.addDomainEvent(new InvestmentAdded(this.id, investment));
    
    return investment.id;
  }

  /**
   * Rebalance portfolio
   */
  rebalance(newTargetAllocation, reason = 'Manual rebalancing') {
    if (!this.currentAllocation) {
      throw new Error('Cannot rebalance empty portfolio');
    }
    
    // Validate new allocation
    PortfolioBusinessRules.validateAllocationLimits(newTargetAllocation);
    
    // Check if rebalancing is needed
    PortfolioBusinessRules.validateRebalancingThreshold(
      this.currentAllocation,
      newTargetAllocation
    );
    
    const oldAllocation = this.targetAllocation;
    this.targetAllocation = newTargetAllocation;
    this.lastRebalancedAt = new Date();
    this.updatedAt = new Date();
    
    // Raise domain event
    this.addDomainEvent(new PortfolioRebalanced(
      this.id,
      oldAllocation,
      newTargetAllocation,
      reason
    ));
    
    return this.generateRebalancingInstructions(oldAllocation, newTargetAllocation);
  }

  /**
   * Change risk profile
   */
  changeRiskProfile(newRiskLevel, userAge, reason = 'User preference change') {
    // Validate risk capacity
    PortfolioBusinessRules.validateRiskCapacity(userAge, newRiskLevel);
    
    const oldRiskLevel = this.riskProfile;
    this.riskProfile = newRiskLevel;
    this.updatedAt = new Date();
    
    // Suggest new allocation based on risk profile
    const suggestedAllocation = this.getSuggestedAllocationForRisk(newRiskLevel);
    
    // Raise domain event
    this.addDomainEvent(new RiskLevelChanged(
      this.id,
      oldRiskLevel,
      newRiskLevel,
      reason
    ));
    
    return {
      newRiskProfile: newRiskLevel,
      suggestedAllocation,
      requiresRebalancing: this.requiresRebalancing(suggestedAllocation)
    };
  }

  /**
   * Calculate portfolio performance
   */
  calculatePerformance(marketData) {
    let totalCurrentValue = new Money(0, 'INR');
    let totalInvestedValue = new Money(0, 'INR');
    
    for (const investment of this.investments.values()) {
      const currentNav = marketData[investment.fundCode]?.nav;
      if (currentNav && investment.units) {
        const currentValue = new Money(investment.units * currentNav, 'INR');
        totalCurrentValue = totalCurrentValue.add(currentValue);
        
        // Update investment current value
        investment.currentValue = currentValue;
      }
      
      totalInvestedValue = totalInvestedValue.add(investment.amount);
    }
    
    this.totalValue = totalCurrentValue;
    
    const absoluteReturn = totalCurrentValue.subtract(totalInvestedValue);
    const percentageReturn = totalInvestedValue.amount > 0 
      ? (absoluteReturn.amount / totalInvestedValue.amount) * 100 
      : 0;
    
    return {
      totalInvested: totalInvestedValue,
      totalCurrent: totalCurrentValue,
      absoluteReturn,
      percentageReturn,
      updatedAt: new Date()
    };
  }

  /**
   * Get portfolio health score
   */
  getHealthScore() {
    let score = 100;
    const issues = [];
    
    // Check diversification
    if (this.investments.size < 3) {
      score -= 20;
      issues.push('Insufficient diversification (less than 3 funds)');
    }
    
    // Check allocation drift
    if (this.currentAllocation && this.targetAllocation) {
      const drift = this.calculateAllocationDrift();
      if (drift > 10) {
        score -= 15;
        issues.push(`High allocation drift: ${drift}%`);
      }
    }
    
    // Check rebalancing frequency
    if (this.lastRebalancedAt) {
      const daysSinceRebalancing = (Date.now() - this.lastRebalancedAt.getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceRebalancing > 365) {
        score -= 10;
        issues.push('Portfolio not rebalanced in over a year');
      }
    }
    
    return {
      score: Math.max(score, 0),
      grade: this.getGradeFromScore(score),
      issues,
      recommendations: this.getRecommendations(issues)
    };
  }

  /**
   * Private helper methods
   */
  recalculateCurrentAllocation() {
    if (this.totalValue.amount === 0) {
      this.currentAllocation = new AssetAllocation(0, 0, 0, 0);
      return;
    }
    
    // This would typically fetch fund categories and calculate allocation
    // For now, using simplified logic
    let equityValue = 0;
    let debtValue = 0;
    let hybridValue = 0;
    let otherValue = 0;
    
    for (const investment of this.investments.values()) {
      const value = investment.currentValue.amount;
      // Simplified categorization based on fund code
      if (investment.fundCode.includes('EQUITY')) {
        equityValue += value;
      } else if (investment.fundCode.includes('DEBT')) {
        debtValue += value;
      } else if (investment.fundCode.includes('HYBRID')) {
        hybridValue += value;
      } else {
        otherValue += value;
      }
    }
    
    const total = this.totalValue.amount;
    this.currentAllocation = new AssetAllocation(
      (equityValue / total) * 100,
      (debtValue / total) * 100,
      (hybridValue / total) * 100,
      (otherValue / total) * 100
    );
  }

  calculateAllocationDrift() {
    if (!this.currentAllocation || !this.targetAllocation) {
      return 0;
    }
    
    const equityDrift = Math.abs(
      this.currentAllocation.equityPercentage - this.targetAllocation.equityPercentage
    );
    const debtDrift = Math.abs(
      this.currentAllocation.debtPercentage - this.targetAllocation.debtPercentage
    );
    
    return Math.max(equityDrift, debtDrift);
  }

  getSuggestedAllocationForRisk(riskLevel) {
    const allocations = {
      'CONSERVATIVE': new AssetAllocation(30, 60, 10, 0),
      'MODERATE': new AssetAllocation(50, 40, 10, 0),
      'AGGRESSIVE': new AssetAllocation(70, 20, 10, 0)
    };
    
    return allocations[riskLevel] || allocations['MODERATE'];
  }

  requiresRebalancing(targetAllocation) {
    if (!this.currentAllocation) return false;
    
    const drift = Math.abs(
      this.currentAllocation.equityPercentage - targetAllocation.equityPercentage
    );
    
    return drift > 5; // 5% threshold
  }

  generateRebalancingInstructions(oldAllocation, newAllocation) {
    const instructions = [];
    
    const equityChange = newAllocation.equityPercentage - oldAllocation.equityPercentage;
    const debtChange = newAllocation.debtPercentage - oldAllocation.debtPercentage;
    
    if (equityChange > 0) {
      instructions.push({
        action: 'INCREASE',
        category: 'EQUITY',
        percentage: equityChange,
        suggestedAmount: this.totalValue.multiply(equityChange / 100)
      });
    } else if (equityChange < 0) {
      instructions.push({
        action: 'DECREASE',
        category: 'EQUITY',
        percentage: Math.abs(equityChange),
        suggestedAmount: this.totalValue.multiply(Math.abs(equityChange) / 100)
      });
    }
    
    if (debtChange > 0) {
      instructions.push({
        action: 'INCREASE',
        category: 'DEBT',
        percentage: debtChange,
        suggestedAmount: this.totalValue.multiply(debtChange / 100)
      });
    } else if (debtChange < 0) {
      instructions.push({
        action: 'DECREASE',
        category: 'DEBT',
        percentage: Math.abs(debtChange),
        suggestedAmount: this.totalValue.multiply(Math.abs(debtChange) / 100)
      });
    }
    
    return instructions;
  }

  getGradeFromScore(score) {
    if (score >= 90) return 'A+';
    if (score >= 80) return 'A';
    if (score >= 70) return 'B+';
    if (score >= 60) return 'B';
    if (score >= 50) return 'C';
    return 'D';
  }

  getRecommendations(issues) {
    const recommendations = [];
    
    issues.forEach(issue => {
      if (issue.includes('diversification')) {
        recommendations.push('Consider adding more funds from different categories');
      }
      if (issue.includes('drift')) {
        recommendations.push('Rebalance your portfolio to maintain target allocation');
      }
      if (issue.includes('rebalanced')) {
        recommendations.push('Review and rebalance your portfolio annually');
      }
    });
    
    return recommendations;
  }

  /**
   * Serialization for persistence
   */
  toSnapshot() {
    return {
      id: this.id,
      userId: this.userId,
      investments: Array.from(this.investments.entries()),
      totalValue: this.totalValue.toJSON(),
      currentAllocation: this.currentAllocation?.toJSON(),
      targetAllocation: this.targetAllocation?.toJSON(),
      riskProfile: this.riskProfile,
      goals: this.goals,
      lastRebalancedAt: this.lastRebalancedAt,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      version: this.version
    };
  }

  /**
   * Deserialization from persistence
   */
  static fromSnapshot(snapshot) {
    const portfolio = new PortfolioAggregate(snapshot.id, snapshot.userId);
    
    portfolio.investments = new Map(snapshot.investments);
    portfolio.totalValue = Money.fromJSON(snapshot.totalValue);
    portfolio.currentAllocation = snapshot.currentAllocation 
      ? AssetAllocation.fromJSON(snapshot.currentAllocation) 
      : null;
    portfolio.targetAllocation = snapshot.targetAllocation 
      ? AssetAllocation.fromJSON(snapshot.targetAllocation) 
      : null;
    portfolio.riskProfile = snapshot.riskProfile;
    portfolio.goals = snapshot.goals;
    portfolio.lastRebalancedAt = snapshot.lastRebalancedAt 
      ? new Date(snapshot.lastRebalancedAt) 
      : null;
    portfolio.createdAt = new Date(snapshot.createdAt);
    portfolio.updatedAt = new Date(snapshot.updatedAt);
    portfolio.version = snapshot.version;
    
    return portfolio;
  }
}

module.exports = { 
  PortfolioAggregate, 
  PortfolioBusinessRules,
  PortfolioCreated,
  InvestmentAdded,
  PortfolioRebalanced,
  RiskLevelChanged
};
