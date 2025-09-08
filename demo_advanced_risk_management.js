/**
 * 🎯 ADVANCED RISK MANAGEMENT DEMO
 * 
 * Demonstration of enterprise-grade risk management capabilities
 * - Value-at-Risk calculations using multiple methodologies
 * - Comprehensive stress testing scenarios
 * - Factor-based risk attribution
 * - Real-time risk monitoring
 * - Regulatory capital requirements
 */

const { EnterpriseIntegrationService } = require('./src/services/EnterpriseIntegrationService');
const logger = require('./src/utils/logger');

class AdvancedRiskManagementDemo {
  constructor() {
    this.integrationService = null;
  }

  async initialize() {
    try {
      console.log('🚀 Initializing Advanced Risk Management Demo...\n');
      
      // Initialize enterprise integration service
      this.integrationService = new EnterpriseIntegrationService();
      await this.integrationService.initializeServices();
      
      console.log('✅ Enterprise services initialized successfully\n');
      
    } catch (error) {
      console.error('❌ Initialization failed:', error);
      throw error;
    }
  }

  async runDemo() {
    try {
      await this.initialize();
      
      console.log('🎯 ADVANCED RISK MANAGEMENT DEMONSTRATION\n');
      console.log('=' .repeat(60));
      
      // Demo portfolio
      const demoPortfolio = {
        id: 'DEMO_PORTFOLIO_001',
        name: 'Institutional Equity Fund',
        totalValue: 50000000, // 5 Crores
        currency: 'INR',
        holdings: [
          { symbol: 'RELIANCE', quantity: 5000, avgPrice: 2400, currentPrice: 2500, sector: 'Energy', weight: 0.25 },
          { symbol: 'TCS', quantity: 2000, avgPrice: 3200, currentPrice: 3500, sector: 'IT', weight: 0.14 },
          { symbol: 'INFY', quantity: 3000, avgPrice: 1600, currentPrice: 1800, sector: 'IT', weight: 0.108 },
          { symbol: 'HDFCBANK', quantity: 1500, avgPrice: 1400, currentPrice: 1500, sector: 'Banking', weight: 0.045 },
          { symbol: 'ICICIBANK', quantity: 2500, avgPrice: 800, currentPrice: 900, sector: 'Banking', weight: 0.045 },
          { symbol: 'BHARTIARTL', quantity: 4000, avgPrice: 700, currentPrice: 750, sector: 'Telecom', weight: 0.06 },
          { symbol: 'ITC', quantity: 6000, avgPrice: 350, currentPrice: 400, sector: 'FMCG', weight: 0.048 },
          { symbol: 'HINDUNILVR', quantity: 800, avgPrice: 2200, currentPrice: 2400, sector: 'FMCG', weight: 0.0384 },
          { symbol: 'MARUTI', quantity: 500, avgPrice: 8000, currentPrice: 8500, sector: 'Auto', weight: 0.085 },
          { symbol: 'ASIANPAINT', quantity: 600, avgPrice: 2800, currentPrice: 3000, sector: 'Paints', weight: 0.036 }
        ]
      };

      // 1. Comprehensive Portfolio Analysis
      console.log('1️⃣  COMPREHENSIVE PORTFOLIO ANALYSIS');
      console.log('-'.repeat(40));
      
      const analysisResults = await this.integrationService.executeComprehensivePortfolioAnalysis(
        demoPortfolio.id, 
        'full'
      );
      
      this.displayAnalysisResults(analysisResults);
      
      // 2. Real-time Risk Monitoring
      console.log('\n2️⃣  REAL-TIME RISK MONITORING');
      console.log('-'.repeat(40));
      
      const monitoringResults = await this.integrationService.executeRealTimeRiskMonitoring(demoPortfolio.id);
      this.displayMonitoringResults(monitoringResults);
      
      // 3. System Status and Health
      console.log('\n3️⃣  SYSTEM STATUS & HEALTH');
      console.log('-'.repeat(40));
      
      const systemStatus = this.integrationService.getIntegratedSystemStatus();
      this.displaySystemStatus(systemStatus);
      
      // 4. Advanced Risk Scenarios
      console.log('\n4️⃣  ADVANCED RISK SCENARIOS');
      console.log('-'.repeat(40));
      
      await this.demonstrateRiskScenarios(demoPortfolio);
      
      console.log('\n🎉 DEMO COMPLETED SUCCESSFULLY!');
      console.log('=' .repeat(60));
      
    } catch (error) {
      console.error('❌ Demo failed:', error);
    }
  }

  displayAnalysisResults(results) {
    console.log(`📊 Analysis ID: ${results.analysis_id}`);
    console.log(`📅 Timestamp: ${results.timestamp}`);
    console.log(`💼 Portfolio: ${results.portfolio_id}\n`);
    
    // VaR Analysis
    if (results.components.var_analysis) {
      const var95 = results.components.var_analysis.composite_var;
      console.log(`📈 Value-at-Risk (95%): ₹${(var95 * 50000000).toLocaleString('en-IN')} (${(var95 * 100).toFixed(2)}%)`);
      
      console.log('   Methodologies:');
      Object.entries(results.components.var_analysis.methodologies).forEach(([method, result]) => {
        console.log(`   • ${method}: ${(result.value * 100).toFixed(3)}%`);
      });
    }
    
    // Stress Testing
    if (results.components.stress_testing) {
      console.log('\n🧪 Stress Testing Results:');
      Object.entries(results.components.stress_testing.scenarios).forEach(([scenario, result]) => {
        console.log(`   • ${result.scenario_name}: ${result.loss_percentage.toFixed(2)}% loss`);
      });
    }
    
    // Risk Attribution
    if (results.components.risk_attribution) {
      console.log('\n📊 Risk Attribution:');
      console.log(`   Total Risk (Volatility): ${(results.components.risk_attribution.total_volatility * 100).toFixed(2)}%`);
      
      Object.entries(results.components.risk_attribution.factor_contributions).forEach(([factor, contrib]) => {
        console.log(`   • ${factor}: ${contrib.percentage.toFixed(1)}%`);
      });
    }
    
    // Regulatory Capital
    if (results.components.regulatory_capital) {
      console.log('\n🏛️ Regulatory Capital (SEBI):');
      console.log(`   Total Required: ₹${results.components.regulatory_capital.total_capital_required.toLocaleString('en-IN')}`);
    }
    
    // ASI Insights
    if (results.components.asi_insights) {
      console.log('\n🤖 ASI-Powered Insights:');
      console.log(`   Confidence: ${((results.components.asi_insights.confidence || 0.8) * 100).toFixed(1)}%`);
      console.log('   Key Observations: Advanced AI analysis completed');
    }
  }

  displayMonitoringResults(results) {
    console.log(`🔍 Monitoring Portfolio: ${results.portfolio_id}`);
    console.log(`📡 Stream ID: ${results.stream_id}`);
    console.log(`⚡ Monitoring ID: ${results.monitoring_id}`);
    console.log(`✅ Status: ${results.status.toUpperCase()}`);
    console.log(`🕐 Started: ${results.started_at}`);
    
    console.log('\n📊 Real-time monitoring features:');
    console.log('   • Continuous VaR monitoring (1-minute intervals)');
    console.log('   • Stress level monitoring (5-minute intervals)');
    console.log('   • Concentration risk monitoring (3-minute intervals)');
    console.log('   • Liquidity monitoring (2-minute intervals)');
    console.log('   • Automated alert system for threshold breaches');
  }

  displaySystemStatus(status) {
    console.log(`🏥 Overall Status: ${status.overall_status.toUpperCase()}`);
    console.log(`🕐 Last Health Check: ${new Date(status.integration_metrics.lastHealthCheck).toLocaleString()}`);
    
    console.log('\n🔧 Service Status:');
    Object.entries(status.services).forEach(([service, serviceStatus]) => {
      const statusIcon = serviceStatus.status === 'healthy' || serviceStatus.status === 'operational' ? '✅' : '❌';
      console.log(`   ${statusIcon} ${service}: ${serviceStatus.status}`);
    });
    
    console.log('\n⚡ Active Workflows:');
    status.active_workflows.forEach(workflow => {
      const activeIcon = workflow.active ? '🟢' : '🔴';
      console.log(`   ${activeIcon} ${workflow.name}`);
    });
    
    console.log('\n📈 Integration Metrics:');
    console.log(`   • Total Requests: ${status.integration_metrics.totalRequests}`);
    console.log(`   • Success Rate: ${status.integration_metrics.totalRequests > 0 ? 
      ((status.integration_metrics.successfulRequests / status.integration_metrics.totalRequests) * 100).toFixed(1) : 0}%`);
  }

  async demonstrateRiskScenarios(portfolio) {
    const riskService = this.integrationService.services.get('risk_management');
    
    console.log('🎭 Simulating Market Crisis Scenarios:\n');
    
    // Scenario 1: Market Crash
    console.log('📉 Scenario 1: Market Crash (-30% equity markets)');
    const crashScenario = {
      name: 'Market Crash Simulation',
      shocks: {
        equity: -0.30,
        volatility: 2.0,
        liquidity: -0.40
      },
      duration: 90
    };
    
    const crashResult = await riskService.runStressScenario(portfolio, crashScenario);
    console.log(`   Impact: ₹${Math.abs(crashResult.total_loss).toLocaleString('en-IN')} loss (${Math.abs(crashResult.loss_percentage).toFixed(1)}%)`);
    
    // Scenario 2: Interest Rate Shock
    console.log('\n📈 Scenario 2: Interest Rate Shock (+300 bps)');
    const rateScenario = {
      name: 'Interest Rate Shock',
      shocks: {
        interest_rates: 0.03,
        banking_sector: -0.15,
        bond_prices: -0.12
      },
      duration: 126
    };
    
    const rateResult = await riskService.runStressScenario(portfolio, rateScenario);
    console.log(`   Impact: ₹${Math.abs(rateResult.total_loss).toLocaleString('en-IN')} loss (${Math.abs(rateResult.loss_percentage).toFixed(1)}%)`);
    
    // Scenario 3: Currency Crisis
    console.log('\n💱 Scenario 3: Currency Crisis (INR devaluation)');
    const currencyScenario = {
      name: 'Currency Crisis',
      shocks: {
        inr_usd: -0.20,
        foreign_outflows: -0.35,
        inflation: 0.04
      },
      duration: 60
    };
    
    const currencyResult = await riskService.runStressScenario(portfolio, currencyScenario);
    console.log(`   Impact: ₹${Math.abs(currencyResult.total_loss).toLocaleString('en-IN')} loss (${Math.abs(currencyResult.loss_percentage).toFixed(1)}%)`);
    
    console.log('\n🛡️ Risk Mitigation Recommendations:');
    console.log('   • Increase diversification across sectors');
    console.log('   • Implement dynamic hedging strategies');
    console.log('   • Maintain adequate liquidity buffers');
    console.log('   • Regular stress testing and scenario analysis');
    console.log('   • Continuous monitoring of risk metrics');
  }

  async cleanup() {
    console.log('\n🧹 Cleaning up demo resources...');
    // Add cleanup logic if needed
    console.log('✅ Cleanup completed');
  }
}

// Run the demo
async function runAdvancedRiskManagementDemo() {
  const demo = new AdvancedRiskManagementDemo();
  
  try {
    await demo.runDemo();
  } catch (error) {
    console.error('❌ Demo execution failed:', error);
  } finally {
    await demo.cleanup();
    process.exit(0);
  }
}

// Execute if run directly
if (require.main === module) {
  runAdvancedRiskManagementDemo();
}

module.exports = { AdvancedRiskManagementDemo };
