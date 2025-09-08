/**
 * 🔍 ON-DEMAND SOCIAL MEDIA ANALYSIS SCRIPT
 * 
 * Utility script for performing on-demand analysis of social media data
 * and management philosophy with detailed reporting
 * 
 * @author Financial Intelligence Team
 * @version 1.0.0 - Social Media Analysis Tool
 */

const fs = require('fs').promises;
const path = require('path');
const logger = require('../src/utils/logger');
const { FreeSocialMediaIntegration } = require('../src/finance_crawler/free-social-media-integration');

class SocialMediaAnalyzer {
  constructor() {
    this.socialMediaSystem = null;
    this.analysisResults = new Map();
    this.reportPath = './reports/social-media-analysis';
  }

  async initialize() {
    try {
      console.log('🔍 Initializing Social Media Analyzer...\n');
      
      // Create reports directory
      await this.createReportsDirectory();
      
      // Initialize social media system
      this.socialMediaSystem = new FreeSocialMediaIntegration({
        enableRealTimeTracking: false, // Disable for analysis mode
        enablePhilosophyAnalysis: true,
        enableSentimentAnalysis: true,
        enableTrendAnalysis: true,
        enableASIIntegration: false // Disable for standalone analysis
      });
      
      await this.socialMediaSystem.initialize();
      
      console.log('✅ Social Media Analyzer initialized successfully!\n');
      
    } catch (error) {
      console.error('❌ Analyzer initialization failed:', error);
      throw error;
    }
  }

  async createReportsDirectory() {
    const dirs = [
      this.reportPath,
      path.join(this.reportPath, 'company-analysis'),
      path.join(this.reportPath, 'sentiment-reports'),
      path.join(this.reportPath, 'philosophy-reports'),
      path.join(this.reportPath, 'trend-analysis'),
      path.join(this.reportPath, 'comprehensive-reports')
    ];
    
    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') throw error;
      }
    }
  }

  async runComprehensiveAnalysis(options = {}) {
    try {
      console.log('🚀 Starting Comprehensive Social Media Analysis...\n');
      
      const analysisConfig = {
        includeAllCompanies: true,
        generateDetailedReports: true,
        analyzeTrends: true,
        generateInsights: true,
        exportData: true,
        ...options
      };
      
      const results = {
        timestamp: new Date().toISOString(),
        analysisConfig,
        companiesAnalyzed: 0,
        totalDataPoints: 0,
        insights: [],
        reports: []
      };
      
      // Get all available management data
      const allData = await this.socialMediaSystem.getAllManagementData();
      
      console.log('📊 Analysis Overview:');
      console.log(`   Management Insights: ${Object.keys(allData.insights).length}`);
      console.log(`   Sentiment Trends: ${Object.keys(allData.sentimentTrends).length}`);
      console.log(`   Philosophy Profiles: ${Object.keys(allData.philosophyProfiles).length}`);
      console.log('');
      
      // Analyze each company
      for (const [company, insights] of Object.entries(allData.insights)) {
        console.log(`🔍 Analyzing ${company}...`);
        
        const companyAnalysis = await this.analyzeCompany(company, {
          insights: insights,
          sentimentTrends: allData.sentimentTrends[company],
          philosophyProfile: allData.philosophyProfiles[company]
        });
        
        this.analysisResults.set(company, companyAnalysis);
        results.companiesAnalyzed++;
        results.totalDataPoints += companyAnalysis.dataPoints;
        
        // Generate company report
        if (analysisConfig.generateDetailedReports) {
          await this.generateCompanyReport(company, companyAnalysis);
          results.reports.push(`company-analysis/${company.replace(/[^a-zA-Z0-9]/g, '_')}_analysis.json`);
        }
      }
      
      // Generate trend analysis
      if (analysisConfig.analyzeTrends) {
        console.log('\n📈 Analyzing trends across all companies...');
        const trendAnalysis = await this.analyzeTrends(allData);
        results.trendAnalysis = trendAnalysis;
        
        await this.generateTrendReport(trendAnalysis);
        results.reports.push('trend-analysis/trend_analysis.json');
      }
      
      // Generate comprehensive insights
      if (analysisConfig.generateInsights) {
        console.log('\n💡 Generating comprehensive insights...');
        const comprehensiveInsights = await this.generateComprehensiveInsights();
        results.insights = comprehensiveInsights;
        
        await this.generateInsightsReport(comprehensiveInsights);
        results.reports.push('comprehensive-reports/insights.json');
      }
      
      // Generate summary report
      await this.generateSummaryReport(results);
      results.reports.push('comprehensive-reports/summary.json');
      
      console.log('\n✅ Comprehensive Analysis Completed!');
      console.log(`📊 Companies Analyzed: ${results.companiesAnalyzed}`);
      console.log(`📈 Total Data Points: ${results.totalDataPoints}`);
      console.log(`📄 Reports Generated: ${results.reports.length}`);
      console.log(`💡 Insights Generated: ${results.insights.length}`);
      console.log(`📁 Reports saved to: ${this.reportPath}`);
      
      return results;
      
    } catch (error) {
      console.error('❌ Comprehensive analysis failed:', error);
      throw error;
    }
  }

  async analyzeCompany(company, data) {
    const analysis = {
      company,
      analyzedAt: new Date().toISOString(),
      dataPoints: 0,
      
      // Analysis components
      managementInsights: null,
      sentimentAnalysis: null,
      philosophyAnalysis: null,
      communicationPatterns: null,
      
      // Derived metrics
      overallScore: 0,
      riskLevel: 'unknown',
      recommendationLevel: 'neutral',
      
      // Insights and recommendations
      keyInsights: [],
      recommendations: [],
      alerts: []
    };
    
    // Analyze management insights
    if (data.insights) {
      analysis.managementInsights = this.analyzeManagementInsights(data.insights);
      analysis.dataPoints += data.insights.dataPoints || 0;
    }
    
    // Analyze sentiment trends
    if (data.sentimentTrends) {
      analysis.sentimentAnalysis = this.analyzeSentimentTrends(data.sentimentTrends);
    }
    
    // Analyze philosophy profile
    if (data.philosophyProfile) {
      analysis.philosophyAnalysis = this.analyzePhilosophyProfile(data.philosophyProfile);
    }
    
    // Generate communication patterns analysis
    analysis.communicationPatterns = this.analyzeCommunicationPatterns(data);
    
    // Calculate overall score
    analysis.overallScore = this.calculateOverallScore(analysis);
    
    // Determine risk level
    analysis.riskLevel = this.determineRiskLevel(analysis);
    
    // Generate recommendation level
    analysis.recommendationLevel = this.generateRecommendationLevel(analysis);
    
    // Generate insights and recommendations
    analysis.keyInsights = this.generateKeyInsights(analysis);
    analysis.recommendations = this.generateRecommendations(analysis);
    analysis.alerts = this.generateAlerts(analysis);
    
    return analysis;
  }

  analyzeManagementInsights(insights) {
    const analysis = {
      totalInsights: insights.insights?.length || 0,
      insightTypes: {},
      confidenceDistribution: {},
      impactDistribution: {},
      averageConfidence: 0
    };
    
    if (insights.insights && insights.insights.length > 0) {
      // Analyze insight types
      for (const insight of insights.insights) {
        analysis.insightTypes[insight.type] = (analysis.insightTypes[insight.type] || 0) + 1;
        analysis.confidenceDistribution[insight.confidence] = (analysis.confidenceDistribution[insight.confidence] || 0) + 1;
        analysis.impactDistribution[insight.impact] = (analysis.impactDistribution[insight.impact] || 0) + 1;
      }
      
      // Calculate average confidence
      const totalConfidence = insights.insights.reduce((sum, insight) => sum + (insight.confidence || 0), 0);
      analysis.averageConfidence = totalConfidence / insights.insights.length;
    }
    
    return analysis;
  }

  analyzeSentimentTrends(sentimentTrends) {
    const analysis = {
      currentTrend: sentimentTrends.currentTrend,
      confidence: sentimentTrends.confidence,
      historyLength: sentimentTrends.history?.length || 0,
      sentimentDistribution: {},
      trendDirection: 'stable'
    };
    
    if (sentimentTrends.history && sentimentTrends.history.length > 0) {
      // Analyze sentiment distribution
      for (const entry of sentimentTrends.history) {
        analysis.sentimentDistribution[entry.sentiment] = (analysis.sentimentDistribution[entry.sentiment] || 0) + 1;
      }
      
      // Determine trend direction
      if (sentimentTrends.history.length >= 3) {
        const recent = sentimentTrends.history.slice(-3);
        const positiveCount = recent.filter(h => h.sentiment === 'positive').length;
        const negativeCount = recent.filter(h => h.sentiment === 'negative').length;
        
        if (positiveCount > negativeCount) {
          analysis.trendDirection = 'improving';
        } else if (negativeCount > positiveCount) {
          analysis.trendDirection = 'declining';
        }
      }
    }
    
    return analysis;
  }

  analyzePhilosophyProfile(philosophyProfile) {
    const analysis = {
      primaryPhilosophy: null,
      philosophyScore: 0,
      consistencyScore: 0,
      confidenceLevel: 'unknown',
      leadershipStyle: null,
      riskApproach: null
    };
    
    if (philosophyProfile.analysis) {
      const profile = philosophyProfile.analysis;
      
      analysis.primaryPhilosophy = profile.investmentPhilosophy?.primaryPhilosophy;
      analysis.philosophyScore = profile.philosophyScore || 0;
      analysis.consistencyScore = profile.consistencyScore || 0;
      analysis.confidenceLevel = profile.confidenceLevel || 'unknown';
      analysis.leadershipStyle = profile.leadershipStyle?.primaryStyle;
      analysis.riskApproach = profile.riskManagementApproach?.primaryApproach;
    }
    
    return analysis;
  }

  analyzeCommunicationPatterns(data) {
    const patterns = {
      frequency: 'unknown',
      consistency: 'unknown',
      tone: 'unknown',
      topics: []
    };
    
    // Analyze based on available data
    if (data.philosophyProfile?.analysis?.communicationPatterns) {
      const commPatterns = data.philosophyProfile.analysis.communicationPatterns;
      patterns.frequency = commPatterns.frequency?.level || 'unknown';
      patterns.tone = commPatterns.tone?.dominantTone || 'unknown';
    }
    
    return patterns;
  }

  calculateOverallScore(analysis) {
    let score = 50; // Base score
    
    // Philosophy analysis contribution (30%)
    if (analysis.philosophyAnalysis) {
      score += (analysis.philosophyAnalysis.philosophyScore * 0.3);
    }
    
    // Sentiment analysis contribution (25%)
    if (analysis.sentimentAnalysis) {
      if (analysis.sentimentAnalysis.trendDirection === 'improving') score += 15;
      else if (analysis.sentimentAnalysis.trendDirection === 'declining') score -= 15;
      
      score += (analysis.sentimentAnalysis.confidence * 0.1);
    }
    
    // Management insights contribution (25%)
    if (analysis.managementInsights) {
      score += (analysis.managementInsights.averageConfidence * 0.25);
    }
    
    // Communication patterns contribution (20%)
    if (analysis.communicationPatterns) {
      if (analysis.communicationPatterns.frequency === 'high') score += 10;
      if (analysis.communicationPatterns.consistency === 'high') score += 10;
    }
    
    return Math.min(Math.max(Math.round(score), 0), 100);
  }

  determineRiskLevel(analysis) {
    const score = analysis.overallScore;
    
    if (score >= 80) return 'low';
    if (score >= 60) return 'medium';
    if (score >= 40) return 'medium-high';
    return 'high';
  }

  generateRecommendationLevel(analysis) {
    const score = analysis.overallScore;
    
    if (score >= 85) return 'strong_buy';
    if (score >= 70) return 'buy';
    if (score >= 55) return 'hold';
    if (score >= 40) return 'weak_hold';
    return 'avoid';
  }

  generateKeyInsights(analysis) {
    const insights = [];
    
    // Philosophy insights
    if (analysis.philosophyAnalysis?.primaryPhilosophy) {
      insights.push({
        type: 'investment_philosophy',
        message: `Clear ${analysis.philosophyAnalysis.primaryPhilosophy} investment approach`,
        confidence: analysis.philosophyAnalysis.philosophyScore
      });
    }
    
    // Sentiment insights
    if (analysis.sentimentAnalysis?.trendDirection === 'improving') {
      insights.push({
        type: 'sentiment_trend',
        message: 'Improving sentiment trend detected',
        confidence: analysis.sentimentAnalysis.confidence
      });
    }
    
    // Communication insights
    if (analysis.communicationPatterns?.frequency === 'high') {
      insights.push({
        type: 'communication_frequency',
        message: 'High management communication frequency',
        confidence: 85
      });
    }
    
    return insights;
  }

  generateRecommendations(analysis) {
    const recommendations = [];
    
    if (analysis.overallScore >= 70) {
      recommendations.push({
        type: 'investment',
        message: 'Strong management communication and philosophy clarity',
        priority: 'high'
      });
    }
    
    if (analysis.philosophyAnalysis?.consistencyScore >= 80) {
      recommendations.push({
        type: 'consistency',
        message: 'Highly consistent strategy communication',
        priority: 'medium'
      });
    }
    
    return recommendations;
  }

  generateAlerts(analysis) {
    const alerts = [];
    
    if (analysis.sentimentAnalysis?.trendDirection === 'declining') {
      alerts.push({
        type: 'sentiment_decline',
        message: 'Declining sentiment trend requires attention',
        severity: 'medium'
      });
    }
    
    if (analysis.overallScore < 40) {
      alerts.push({
        type: 'low_score',
        message: 'Low overall management analysis score',
        severity: 'high'
      });
    }
    
    return alerts;
  }

  async analyzeTrends(allData) {
    const trendAnalysis = {
      analyzedAt: new Date().toISOString(),
      
      // Philosophy trends
      philosophyDistribution: {},
      leadershipStyleDistribution: {},
      riskApproachDistribution: {},
      
      // Sentiment trends
      overallSentimentTrend: 'neutral',
      sentimentDistribution: {},
      
      // Performance trends
      topPerformers: [],
      consistencyLeaders: [],
      
      // Market insights
      marketOutlookDistribution: {},
      communicationPatternTrends: {}
    };
    
    // Analyze philosophy distribution
    for (const [company, profile] of Object.entries(allData.philosophyProfiles)) {
      if (profile.analysis?.investmentPhilosophy?.primaryPhilosophy) {
        const philosophy = profile.analysis.investmentPhilosophy.primaryPhilosophy;
        trendAnalysis.philosophyDistribution[philosophy] = (trendAnalysis.philosophyDistribution[philosophy] || 0) + 1;
      }
      
      if (profile.analysis?.leadershipStyle?.primaryStyle) {
        const style = profile.analysis.leadershipStyle.primaryStyle;
        trendAnalysis.leadershipStyleDistribution[style] = (trendAnalysis.leadershipStyleDistribution[style] || 0) + 1;
      }
    }
    
    // Analyze sentiment trends
    for (const [company, trends] of Object.entries(allData.sentimentTrends)) {
      if (trends.currentTrend) {
        trendAnalysis.sentimentDistribution[trends.currentTrend] = (trendAnalysis.sentimentDistribution[trends.currentTrend] || 0) + 1;
      }
    }
    
    // Identify top performers
    const performanceScores = [];
    for (const [company, result] of this.analysisResults) {
      performanceScores.push({
        company,
        score: result.overallScore,
        philosophyScore: result.philosophyAnalysis?.philosophyScore || 0
      });
    }
    
    trendAnalysis.topPerformers = performanceScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    
    trendAnalysis.consistencyLeaders = performanceScores
      .sort((a, b) => b.philosophyScore - a.philosophyScore)
      .slice(0, 5);
    
    return trendAnalysis;
  }

  async generateComprehensiveInsights() {
    const insights = [];
    
    // Analyze all results
    const allResults = Array.from(this.analysisResults.values());
    
    // Overall performance insights
    const averageScore = allResults.reduce((sum, result) => sum + result.overallScore, 0) / allResults.length;
    insights.push({
      type: 'overall_performance',
      message: `Average management analysis score: ${averageScore.toFixed(1)}`,
      category: 'performance'
    });
    
    // Philosophy insights
    const philosophyTypes = {};
    allResults.forEach(result => {
      if (result.philosophyAnalysis?.primaryPhilosophy) {
        const philosophy = result.philosophyAnalysis.primaryPhilosophy;
        philosophyTypes[philosophy] = (philosophyTypes[philosophy] || 0) + 1;
      }
    });
    
    const dominantPhilosophy = Object.entries(philosophyTypes)
      .sort(([,a], [,b]) => b - a)[0];
    
    if (dominantPhilosophy) {
      insights.push({
        type: 'dominant_philosophy',
        message: `Most common investment philosophy: ${dominantPhilosophy[0]} (${dominantPhilosophy[1]} companies)`,
        category: 'philosophy'
      });
    }
    
    // Risk insights
    const riskLevels = {};
    allResults.forEach(result => {
      riskLevels[result.riskLevel] = (riskLevels[result.riskLevel] || 0) + 1;
    });
    
    insights.push({
      type: 'risk_distribution',
      message: `Risk distribution: ${Object.entries(riskLevels).map(([level, count]) => `${level}: ${count}`).join(', ')}`,
      category: 'risk'
    });
    
    return insights;
  }

  async generateCompanyReport(company, analysis) {
    const reportPath = path.join(this.reportPath, 'company-analysis', `${company.replace(/[^a-zA-Z0-9]/g, '_')}_analysis.json`);
    
    const report = {
      company,
      generatedAt: new Date().toISOString(),
      analysis,
      summary: {
        overallScore: analysis.overallScore,
        riskLevel: analysis.riskLevel,
        recommendationLevel: analysis.recommendationLevel,
        keyInsights: analysis.keyInsights.length,
        recommendations: analysis.recommendations.length,
        alerts: analysis.alerts.length
      }
    };
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
  }

  async generateTrendReport(trendAnalysis) {
    const reportPath = path.join(this.reportPath, 'trend-analysis', 'trend_analysis.json');
    
    const report = {
      generatedAt: new Date().toISOString(),
      trendAnalysis,
      summary: {
        totalCompanies: Object.values(trendAnalysis.philosophyDistribution).reduce((sum, count) => sum + count, 0),
        dominantPhilosophy: Object.entries(trendAnalysis.philosophyDistribution).sort(([,a], [,b]) => b - a)[0]?.[0],
        topPerformer: trendAnalysis.topPerformers[0]?.company
      }
    };
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
  }

  async generateInsightsReport(insights) {
    const reportPath = path.join(this.reportPath, 'comprehensive-reports', 'insights.json');
    
    const report = {
      generatedAt: new Date().toISOString(),
      totalInsights: insights.length,
      insights,
      categories: {
        performance: insights.filter(i => i.category === 'performance').length,
        philosophy: insights.filter(i => i.category === 'philosophy').length,
        risk: insights.filter(i => i.category === 'risk').length
      }
    };
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
  }

  async generateSummaryReport(results) {
    const reportPath = path.join(this.reportPath, 'comprehensive-reports', 'summary.json');
    
    const summary = {
      generatedAt: new Date().toISOString(),
      analysisResults: results,
      executionSummary: {
        companiesAnalyzed: results.companiesAnalyzed,
        totalDataPoints: results.totalDataPoints,
        reportsGenerated: results.reports.length,
        insightsGenerated: results.insights.length,
        averageScore: this.calculateAverageScore(),
        topPerformer: this.getTopPerformer(),
        recommendationDistribution: this.getRecommendationDistribution()
      }
    };
    
    await fs.writeFile(reportPath, JSON.stringify(summary, null, 2));
    
    console.log(`\n📄 Summary report saved to: ${reportPath}`);
  }

  calculateAverageScore() {
    const scores = Array.from(this.analysisResults.values()).map(r => r.overallScore);
    return scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
  }

  getTopPerformer() {
    const results = Array.from(this.analysisResults.entries());
    if (results.length === 0) return null;
    
    return results.sort(([,a], [,b]) => b.overallScore - a.overallScore)[0][0];
  }

  getRecommendationDistribution() {
    const distribution = {};
    for (const [, result] of this.analysisResults) {
      distribution[result.recommendationLevel] = (distribution[result.recommendationLevel] || 0) + 1;
    }
    return distribution;
  }
}

// CLI interface
async function main() {
  try {
    const analyzer = new SocialMediaAnalyzer();
    await analyzer.initialize();
    
    // Parse command line arguments
    const args = process.argv.slice(2);
    const options = {};
    
    // Simple argument parsing
    if (args.includes('--no-reports')) options.generateDetailedReports = false;
    if (args.includes('--no-trends')) options.analyzeTrends = false;
    if (args.includes('--no-insights')) options.generateInsights = false;
    
    // Run comprehensive analysis
    await analyzer.runComprehensiveAnalysis(options);
    
  } catch (error) {
    console.error('❌ Analysis failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { SocialMediaAnalyzer };
