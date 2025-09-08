const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');

class ComprehensiveReportService {
    constructor() {
        this.reportTypes = {
            CLIENT_STATEMENT: 'client_statement',
            ASI_DIAGNOSTIC: 'asi_diagnostic',
            PORTFOLIO_ALLOCATION: 'portfolio_allocation',
            PERFORMANCE_BENCHMARK: 'performance_benchmark',
            FY_PNL: 'fy_pnl',
            ELSS_INVESTMENT: 'elss_investment',
            TOP_PERFORMER: 'top_performer',
            ASSET_ALLOCATION_TRENDS: 'asset_allocation_trends',
            SIP_FLOW_RETENTION: 'sip_flow_retention',
            CAMPAIGN_PERFORMANCE: 'campaign_performance',
            COMPLIANCE_AUDIT: 'compliance_audit',
            COMMISSION_BROKERAGE: 'commission_brokerage',
            CUSTOM_BUILDER: 'custom_builder'
        };
    }

    async generateReport(reportType, userId, options = {}) {
        const userData = await this.getUserData(userId);
        const reportData = await this.getReportData(reportType, userId, options);
        
        switch (reportType) {
            case this.reportTypes.CLIENT_STATEMENT:
                return this.generateClientStatement(userData, reportData, options);
            case this.reportTypes.ASI_DIAGNOSTIC:
                return this.generateASIDiagnostic(userData, reportData, options);
            case this.reportTypes.PORTFOLIO_ALLOCATION:
                return this.generatePortfolioAllocation(userData, reportData, options);
            case this.reportTypes.PERFORMANCE_BENCHMARK:
                return this.generatePerformanceBenchmark(userData, reportData, options);
            case this.reportTypes.FY_PNL:
                return this.generateFYPnL(userData, reportData, options);
            case this.reportTypes.ELSS_INVESTMENT:
                return this.generateELSSReport(userData, reportData, options);
            case this.reportTypes.TOP_PERFORMER:
                return this.generateTopPerformerReport(userData, reportData, options);
            case this.reportTypes.ASSET_ALLOCATION_TRENDS:
                return this.generateAssetAllocationTrends(userData, reportData, options);
            case this.reportTypes.SIP_FLOW_RETENTION:
                return this.generateSIPFlowRetention(userData, reportData, options);
            case this.reportTypes.CAMPAIGN_PERFORMANCE:
                return this.generateCampaignPerformance(userData, reportData, options);
            case this.reportTypes.COMPLIANCE_AUDIT:
                return this.generateComplianceAudit(userData, reportData, options);
            case this.reportTypes.COMMISSION_BROKERAGE:
                return this.generateCommissionBrokerage(userData, reportData, options);
            case this.reportTypes.CUSTOM_BUILDER:
                return this.generateCustomReport(userData, reportData, options);
            default:
                throw new Error(`Unknown report type: ${reportType}`);
        }
    }

    async generateClientStatement(userData, reportData, options) {
        const doc = new PDFDocument({ margin: 50 });
        const filename = `Client_Statement_${userData.userId}_${Date.now()}.pdf`;
        const filepath = path.join(__dirname, '../../reports', filename);
        
        doc.pipe(fs.createWriteStream(filepath));

        // Header with SIP Brewery branding
        this.addHeader(doc, 'Client Investment Statement');
        
        // Client Details Section
        doc.fontSize(14).text('Client Details', 50, 120);
        doc.fontSize(10)
           .text(`Name: ${userData.name}`, 50, 140)
           .text(`PAN: ${userData.pan}`, 50, 155)
           .text(`Folio No: ${userData.folioNumber}`, 50, 170)
           .text(`Report Period: ${options.startDate} to ${options.endDate}`, 50, 185);

        // Portfolio Summary
        doc.fontSize(14).text('Portfolio Summary', 50, 220);
        const summary = reportData.portfolioSummary;
        doc.fontSize(10)
           .text(`Total Invested: ₹${summary.totalInvested.toLocaleString()}`, 50, 240)
           .text(`Current Value: ₹${summary.currentValue.toLocaleString()}`, 50, 255)
           .text(`Absolute Return: ${summary.absoluteReturn}%`, 50, 270)
           .text(`XIRR: ${summary.xirr}%`, 50, 285);

        // Holdings Table
        doc.fontSize(14).text('Current Holdings', 50, 320);
        this.addHoldingsTable(doc, reportData.holdings, 340);

        // SIP History
        doc.fontSize(14).text('SIP History', 50, 480);
        this.addSIPHistory(doc, reportData.sipHistory, 500);

        // AI Insights
        doc.fontSize(14).text('🧠 AI Insights', 50, 600);
        doc.fontSize(10)
           .text(reportData.aiInsights.summary, 50, 620, { width: 500 })
           .text(`🔥 ${reportData.aiInsights.topPerformer}`, 50, 650)
           .text(`⚠️ ${reportData.aiInsights.riskAlert}`, 50, 665);

        // Action Items
        doc.fontSize(12).text('Recommended Actions:', 50, 700);
        reportData.actionItems.forEach((action, index) => {
            doc.fontSize(10).text(`${index + 1}. ${action}`, 60, 720 + (index * 15));
        });

        // Footer with compliance
        this.addFooter(doc);
        
        doc.end();
        return { filename, filepath };
    }

    async generateASIDiagnostic(userData, reportData, options) {
        const doc = new PDFDocument({ margin: 50 });
        const filename = `ASI_Diagnostic_${userData.userId}_${Date.now()}.pdf`;
        const filepath = path.join(__dirname, '../../reports', filename);
        
        doc.pipe(fs.createWriteStream(filepath));

        this.addHeader(doc, 'ASI Portfolio Diagnostic Report');

        // Overall ASI Score
        doc.fontSize(16).text('Overall ASI Score', 50, 120);
        doc.fontSize(24).fillColor('#00FF87').text(`${reportData.overallScore}/100`, 50, 145);
        doc.fillColor('black');

        // Subscores
        doc.fontSize(14).text('Performance Breakdown', 50, 190);
        const subscores = reportData.subscores;
        let yPos = 210;
        
        Object.entries(subscores).forEach(([metric, score]) => {
            doc.fontSize(10)
               .text(`${metric}: ${score}/100`, 50, yPos)
               .rect(200, yPos - 2, (score * 2), 10)
               .fillAndStroke('#00FF87', '#00FF87');
            yPos += 25;
        });

        // Fund-wise ASI Comparison
        doc.fontSize(14).text('Fund-wise ASI Analysis', 50, 350);
        this.addFundASITable(doc, reportData.fundAnalysis, 370);

        // AI Recommendations
        doc.fontSize(14).text('🧠 ASI Recommendations', 50, 500);
        reportData.recommendations.forEach((rec, index) => {
            doc.fontSize(10).text(`${rec.emoji} ${rec.text}`, 50, 520 + (index * 20));
        });

        // Risk Alerts
        doc.fontSize(14).text('⚠️ Risk Alerts', 50, 600);
        reportData.riskAlerts.forEach((alert, index) => {
            doc.fontSize(10).fillColor('red').text(`• ${alert}`, 50, 620 + (index * 15));
        });

        this.addFooter(doc);
        doc.end();
        return { filename, filepath };
    }

    async generatePortfolioAllocation(userData, reportData, options) {
        const doc = new PDFDocument({ margin: 50 });
        const filename = `Portfolio_Allocation_${userData.userId}_${Date.now()}.pdf`;
        const filepath = path.join(__dirname, '../../reports', filename);
        
        doc.pipe(fs.createWriteStream(filepath));

        this.addHeader(doc, 'Portfolio Allocation & Overlap Report');

        // Current Allocation
        doc.fontSize(14).text('Asset Class Distribution', 50, 120);
        const allocation = reportData.assetAllocation;
        let yPos = 140;
        
        Object.entries(allocation).forEach(([asset, percentage]) => {
            doc.fontSize(10)
               .text(`${asset}: ${percentage}%`, 50, yPos)
               .rect(200, yPos - 2, (percentage * 3), 12)
               .fillAndStroke('#4AE3F7', '#4AE3F7');
            yPos += 20;
        });

        // Sectoral Exposure
        doc.fontSize(14).text('Sectoral Exposure', 50, 280);
        this.addSectorTable(doc, reportData.sectorExposure, 300);

        // Overlap Analysis
        doc.fontSize(14).text('Fund Overlap Analysis', 50, 420);
        doc.fontSize(10).text('Funds with >10% overlap:', 50, 440);
        
        reportData.overlapAnalysis.forEach((overlap, index) => {
            doc.fontSize(9)
               .text(`${overlap.fund1} ↔ ${overlap.fund2}: ${overlap.percentage}%`, 50, 460 + (index * 15));
        });

        // AI Suggestions
        doc.fontSize(14).text('🧠 Diversification Insights', 50, 550);
        reportData.diversificationInsights.forEach((insight, index) => {
            doc.fontSize(10).text(`${insight.emoji} ${insight.text}`, 50, 570 + (index * 20));
        });

        this.addFooter(doc);
        doc.end();
        return { filename, filepath };
    }

    async generatePerformanceBenchmark(userData, reportData, options) {
        const doc = new PDFDocument({ margin: 50 });
        const filename = `Performance_Benchmark_${userData.userId}_${Date.now()}.pdf`;
        const filepath = path.join(__dirname, '../../reports', filename);
        
        doc.pipe(fs.createWriteStream(filepath));

        this.addHeader(doc, 'Performance vs Benchmark Report');

        // Performance Metrics Table
        doc.fontSize(14).text('Fund Performance Analysis', 50, 120);
        this.addPerformanceTable(doc, reportData.performanceData, 140);

        // Risk Metrics
        doc.fontSize(14).text('Risk-Adjusted Returns', 50, 350);
        const riskMetrics = reportData.riskMetrics;
        let yPos = 370;
        
        Object.entries(riskMetrics).forEach(([fund, metrics]) => {
            doc.fontSize(12).text(fund, 50, yPos);
            doc.fontSize(10)
               .text(`Alpha: ${metrics.alpha}%`, 60, yPos + 15)
               .text(`Beta: ${metrics.beta}`, 60, yPos + 30)
               .text(`Sharpe Ratio: ${metrics.sharpeRatio}`, 60, yPos + 45)
               .text(`Std Deviation: ${metrics.stdDev}%`, 60, yPos + 60);
            yPos += 85;
        });

        // Underperforming Funds Alert
        if (reportData.underperformingFunds.length > 0) {
            doc.fontSize(14).text('⚠️ Underperforming Funds', 50, 600);
            reportData.underperformingFunds.forEach((fund, index) => {
                doc.fontSize(10)
                   .fillColor('red')
                   .text(`• ${fund.name} (${fund.underperformance}% below benchmark)`, 50, 620 + (index * 15));
            });
        }

        this.addFooter(doc);
        doc.end();
        return { filename, filepath };
    }

    async generateFYPnL(userData, reportData, options) {
        const doc = new PDFDocument({ margin: 50 });
        const filename = `FY_PnL_${userData.userId}_${Date.now()}.pdf`;
        const filepath = path.join(__dirname, '../../reports', filename);
        
        doc.pipe(fs.createWriteStream(filepath));

        this.addHeader(doc, `Financial Year P&L Report (${options.financialYear})`);

        // FY Summary
        doc.fontSize(14).text('Financial Year Summary', 50, 120);
        const fyData = reportData.fyData;
        doc.fontSize(10)
           .text(`Total Invested: ₹${fyData.totalInvested.toLocaleString()}`, 50, 140)
           .text(`Redemption Value: ₹${fyData.redemptionValue.toLocaleString()}`, 50, 155)
           .text(`Realized Gains: ₹${fyData.realizedGains.toLocaleString()}`, 50, 170)
           .text(`Unrealized Gains: ₹${fyData.unrealizedGains.toLocaleString()}`, 50, 185)
           .text(`Dividend Income: ₹${fyData.dividendIncome.toLocaleString()}`, 50, 200);

        // Capital Gains Breakdown
        doc.fontSize(14).text('Capital Gains Analysis', 50, 240);
        doc.fontSize(10)
           .text(`Short-Term Capital Gains (STCG): ₹${fyData.stcg.toLocaleString()}`, 50, 260)
           .text(`Long-Term Capital Gains (LTCG): ₹${fyData.ltcg.toLocaleString()}`, 50, 275)
           .text(`Estimated Tax Liability: ₹${fyData.estimatedTax.toLocaleString()}`, 50, 290);

        // Monthly Investment vs Gain Chart Data
        doc.fontSize(14).text('Monthly Investment Pattern', 50, 330);
        this.addMonthlyInvestmentTable(doc, reportData.monthlyData, 350);

        // Tax Optimization Suggestions
        doc.fontSize(14).text('🧠 Tax Optimization Insights', 50, 500);
        reportData.taxOptimization.forEach((tip, index) => {
            doc.fontSize(10).text(`${tip.emoji} ${tip.text}`, 50, 520 + (index * 20));
        });

        // STCG Alerts
        if (reportData.stcgAlerts.length > 0) {
            doc.fontSize(14).text('⚠️ STCG Alerts', 50, 600);
            reportData.stcgAlerts.forEach((alert, index) => {
                doc.fontSize(10).fillColor('red').text(`• ${alert}`, 50, 620 + (index * 15));
            });
        }

        this.addFooter(doc);
        doc.end();
        return { filename, filepath };
    }

    // Helper methods for table generation
    addHoldingsTable(doc, holdings, startY) {
        const headers = ['Fund Name', 'NAV', 'Units', 'Invested', 'Current Value', 'Return %'];
        let yPos = startY;
        
        // Headers
        doc.fontSize(9);
        headers.forEach((header, index) => {
            doc.text(header, 50 + (index * 80), yPos);
        });
        
        yPos += 20;
        
        // Data rows
        holdings.forEach((holding, rowIndex) => {
            doc.text(holding.fundName.substring(0, 15), 50, yPos);
            doc.text(`₹${holding.nav}`, 130, yPos);
            doc.text(holding.units.toFixed(2), 210, yPos);
            doc.text(`₹${holding.invested.toLocaleString()}`, 290, yPos);
            doc.text(`₹${holding.currentValue.toLocaleString()}`, 370, yPos);
            doc.fillColor(holding.returnPercent >= 0 ? 'green' : 'red')
               .text(`${holding.returnPercent}%`, 450, yPos)
               .fillColor('black');
            yPos += 15;
        });
    }

    addSIPHistory(doc, sipHistory, startY) {
        let yPos = startY;
        sipHistory.forEach((sip, index) => {
            doc.fontSize(9)
               .text(`${sip.fundName}: ₹${sip.monthlyAmount}/month since ${sip.startDate}`, 50, yPos);
            if (sip.skippedMonths > 0) {
                doc.text(`(${sip.skippedMonths} months skipped)`, 300, yPos);
            }
            yPos += 15;
        });
    }

    addHeader(doc, title) {
        // SIP Brewery Logo and Header
        doc.fontSize(20).fillColor('#00FF87').text('SIP Brewery', 50, 50);
        doc.fontSize(16).fillColor('black').text(title, 50, 80);
        doc.fontSize(10).text(`Generated on: ${new Date().toLocaleDateString('en-IN')}`, 400, 50);
        
        // Add line separator
        doc.moveTo(50, 110).lineTo(550, 110).stroke();
    }

    addFooter(doc) {
        const pageHeight = doc.page.height;
        doc.fontSize(8)
           .text('AMFI Registered Mutual Fund Distributor | SEBI Compliant', 50, pageHeight - 80)
           .text('This report is for educational purposes only. Please consult your financial advisor.', 50, pageHeight - 65)
           .text('SIP Brewery - Powered by Artificial Super Intelligence', 50, pageHeight - 50);
    }

    async getUserData(userId) {
        // Mock user data - replace with actual database call
        return {
            userId: userId,
            name: 'Rajesh Kumar',
            pan: 'ABCDE1234F',
            folioNumber: 'SB123456789',
            email: 'rajesh@example.com',
            phone: '+91-9876543210'
        };
    }

    async getReportData(reportType, userId, options) {
        // Mock report data - replace with actual data fetching logic
        const baseData = {
            portfolioSummary: {
                totalInvested: 250000,
                currentValue: 287500,
                absoluteReturn: 15.0,
                xirr: 12.5
            },
            holdings: [
                {
                    fundName: 'SBI Blue Chip Fund',
                    nav: 45.67,
                    units: 1250.50,
                    invested: 50000,
                    currentValue: 57088,
                    returnPercent: 14.18
                },
                {
                    fundName: 'HDFC Top 100 Fund',
                    nav: 678.90,
                    units: 441.23,
                    invested: 300000,
                    currentValue: 299456,
                    returnPercent: -0.18
                }
            ],
            sipHistory: [
                {
                    fundName: 'SBI Blue Chip Fund',
                    monthlyAmount: 5000,
                    startDate: '01-Apr-2023',
                    skippedMonths: 0
                }
            ],
            aiInsights: {
                summary: 'Your portfolio shows strong performance with 15% absolute returns. The ASI system recommends maintaining current allocation with minor rebalancing.',
                topPerformer: 'SBI Blue Chip Fund has outperformed its benchmark by 3.2% this quarter',
                riskAlert: 'Consider reducing exposure to mid-cap funds as volatility may increase'
            },
            actionItems: [
                'Consider increasing SIP amount by ₹1000 to optimize tax benefits',
                'Switch underperforming HDFC fund to better alternatives',
                'Add ELSS fund to utilize remaining 80C limit'
            ]
        };

        // Add report-specific data based on type
        switch (reportType) {
            case this.reportTypes.ASI_DIAGNOSTIC:
                return {
                    ...baseData,
                    overallScore: 78,
                    subscores: {
                        'Return Efficiency': 85,
                        'Volatility Control': 72,
                        'Alpha Capture': 80,
                        'Drawdown Resistance': 75,
                        'Consistency Score': 78
                    },
                    fundAnalysis: [
                        { fundName: 'SBI Blue Chip', asiScore: 82, peerRank: 15 },
                        { fundName: 'HDFC Top 100', asiScore: 68, peerRank: 45 }
                    ],
                    recommendations: [
                        { emoji: '🚀', text: 'Switch to higher ASI score funds in large-cap category' },
                        { emoji: '⚖️', text: 'Rebalance portfolio to reduce overlap by 15%' }
                    ],
                    riskAlerts: [
                        'High concentration in banking sector (35%)',
                        'Low diversification score detected'
                    ]
                };
            
            default:
                return baseData;
        }
    }
}

module.exports = ComprehensiveReportService;
