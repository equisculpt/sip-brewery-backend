// Professional PDF Report Generation Service - Institutional Grade Reports
const PDFDocument = require('pdfkit');
const fs = require('fs').promises;
const path = require('path');
const Chart = require('chart.js/auto');
const { createCanvas } = require('canvas');

class PDFReportService {
    constructor() {
        this.reportTypes = {
            PORTFOLIO_STATEMENT: 'Portfolio Statement',
            PERFORMANCE_ANALYSIS: 'Performance Analysis Report',
            TAX_STATEMENT: 'Tax Statement',
            CAPITAL_GAINS: 'Capital Gains Report',
            TRANSACTION_STATEMENT: 'Transaction Statement',
            SIP_ANALYSIS: 'SIP Analysis Report',
            ASSET_ALLOCATION: 'Asset Allocation Report',
            RISK_ANALYSIS: 'Risk Analysis Report',
            DIVIDEND_STATEMENT: 'Dividend Statement',
            ANNUAL_REPORT: 'Annual Investment Report',
            QUARTERLY_REVIEW: 'Quarterly Portfolio Review',
            BENCHMARK_COMPARISON: 'Benchmark Comparison Report',
            GOAL_TRACKING: 'Goal Tracking Report',
            COMPLIANCE_REPORT: 'Compliance & Regulatory Report'
        };
        
        this.initializeChartDefaults();
    }

    initializeChartDefaults() {
        // Configure Chart.js defaults for PDF generation
        Chart.defaults.font.family = 'Helvetica';
        Chart.defaults.font.size = 10;
        Chart.defaults.color = '#333333';
    }

    // Generate Portfolio Statement - Most comprehensive report
    async generatePortfolioStatement(userId, options = {}) {
        const {
            dateRange = 'YTD',
            includeTransactions = true,
            includeTaxDetails = true,
            includePerformance = true,
            format = 'detailed' // detailed, summary, regulatory
        } = options;

        const doc = new PDFDocument({ 
            size: 'A4', 
            margin: 50,
            info: {
                Title: 'Portfolio Statement',
                Author: 'SIP Brewery',
                Subject: 'Investment Portfolio Statement',
                Keywords: 'portfolio, investment, mutual funds, statement'
            }
        });

        // Get user data
        const userData = await this.getUserData(userId);
        const portfolioData = await this.getPortfolioData(userId, dateRange);
        const transactionData = includeTransactions ? await this.getTransactionData(userId, dateRange) : null;

        // Generate report sections
        await this.addReportHeader(doc, 'PORTFOLIO STATEMENT', userData);
        await this.addPortfolioSummary(doc, portfolioData);
        await this.addHoldingsDetails(doc, portfolioData.holdings);
        
        if (includePerformance) {
            await this.addPerformanceAnalysis(doc, portfolioData);
        }
        
        if (includeTransactions && transactionData) {
            await this.addTransactionHistory(doc, transactionData);
        }
        
        if (includeTaxDetails) {
            await this.addTaxSummary(doc, portfolioData);
        }

        await this.addAssetAllocationChart(doc, portfolioData.assetAllocation);
        await this.addPerformanceChart(doc, portfolioData.performanceHistory);
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Generate Performance Analysis Report
    async generatePerformanceAnalysis(userId, options = {}) {
        const {
            period = '1Y',
            benchmarkComparison = true,
            riskMetrics = true,
            attribution = true
        } = options;

        const doc = new PDFDocument({ size: 'A4', margin: 50 });
        const userData = await this.getUserData(userId);
        const performanceData = await this.getPerformanceData(userId, period);

        await this.addReportHeader(doc, 'PERFORMANCE ANALYSIS REPORT', userData);
        
        // Executive Summary
        await this.addExecutiveSummary(doc, performanceData);
        
        // Performance Overview
        await this.addPerformanceOverview(doc, performanceData);
        
        // Benchmark Comparison
        if (benchmarkComparison) {
            await this.addBenchmarkComparison(doc, performanceData);
        }
        
        // Risk Metrics
        if (riskMetrics) {
            await this.addRiskMetrics(doc, performanceData);
        }
        
        // Performance Attribution
        if (attribution) {
            await this.addPerformanceAttribution(doc, performanceData);
        }
        
        // Charts and Graphs
        await this.addPerformanceCharts(doc, performanceData);
        await this.addRiskReturnScatter(doc, performanceData);
        
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Generate Tax Statement
    async generateTaxStatement(userId, financialYear) {
        const doc = new PDFDocument({ size: 'A4', margin: 50 });
        const userData = await this.getUserData(userId);
        const taxData = await this.getTaxData(userId, financialYear);

        await this.addReportHeader(doc, `TAX STATEMENT - FY ${financialYear}`, userData);
        
        // Tax Summary
        await this.addTaxSummarySection(doc, taxData);
        
        // Capital Gains Details
        await this.addCapitalGainsDetails(doc, taxData.capitalGains);
        
        // Dividend Income
        await this.addDividendIncome(doc, taxData.dividends);
        
        // ELSS Investments (80C)
        await this.addELSSInvestments(doc, taxData.elss);
        
        // TDS Details
        await this.addTDSDetails(doc, taxData.tds);
        
        // Tax Optimization Suggestions
        await this.addTaxOptimizationSuggestions(doc, taxData);
        
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Generate Capital Gains Report
    async generateCapitalGainsReport(userId, options = {}) {
        const {
            financialYear,
            gainType = 'ALL', // STCG, LTCG, ALL
            includeUnrealized = false
        } = options;

        const doc = new PDFDocument({ size: 'A4', margin: 50 });
        const userData = await this.getUserData(userId);
        const gainsData = await this.getCapitalGainsData(userId, financialYear, gainType);

        await this.addReportHeader(doc, 'CAPITAL GAINS REPORT', userData);
        
        // Gains Summary
        await this.addCapitalGainsSummary(doc, gainsData);
        
        // Short Term Capital Gains
        if (gainType === 'ALL' || gainType === 'STCG') {
            await this.addSTCGDetails(doc, gainsData.stcg);
        }
        
        // Long Term Capital Gains
        if (gainType === 'ALL' || gainType === 'LTCG') {
            await this.addLTCGDetails(doc, gainsData.ltcg);
        }
        
        // Unrealized Gains (if requested)
        if (includeUnrealized) {
            await this.addUnrealizedGains(doc, gainsData.unrealized);
        }
        
        // Tax Implications
        await this.addTaxImplications(doc, gainsData);
        
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Generate SIP Analysis Report
    async generateSIPAnalysisReport(userId, options = {}) {
        const {
            period = 'ALL',
            includeFutureProjections = true,
            includeOptimization = true
        } = options;

        const doc = new PDFDocument({ size: 'A4', margin: 50 });
        const userData = await this.getUserData(userId);
        const sipData = await this.getSIPData(userId, period);

        await this.addReportHeader(doc, 'SIP ANALYSIS REPORT', userData);
        
        // SIP Summary
        await this.addSIPSummary(doc, sipData);
        
        // SIP Performance Analysis
        await this.addSIPPerformanceAnalysis(doc, sipData);
        
        // Rupee Cost Averaging Analysis
        await this.addRupeeCostAveraging(doc, sipData);
        
        // SIP vs Lumpsum Comparison
        await this.addSIPvsLumpsumComparison(doc, sipData);
        
        // Future Projections
        if (includeFutureProjections) {
            await this.addSIPProjections(doc, sipData);
        }
        
        // SIP Optimization Recommendations
        if (includeOptimization) {
            await this.addSIPOptimizationRecommendations(doc, sipData);
        }
        
        // Charts
        await this.addSIPPerformanceChart(doc, sipData);
        await this.addSIPProjectionChart(doc, sipData);
        
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Generate Risk Analysis Report
    async generateRiskAnalysisReport(userId, options = {}) {
        const {
            includeStressTest = true,
            includeScenarioAnalysis = true,
            includeRiskRecommendations = true
        } = options;

        const doc = new PDFDocument({ size: 'A4', margin: 50 });
        const userData = await this.getUserData(userId);
        const riskData = await this.getRiskAnalysisData(userId);

        await this.addReportHeader(doc, 'RISK ANALYSIS REPORT', userData);
        
        // Risk Profile Summary
        await this.addRiskProfileSummary(doc, riskData);
        
        // Portfolio Risk Metrics
        await this.addPortfolioRiskMetrics(doc, riskData);
        
        // Volatility Analysis
        await this.addVolatilityAnalysis(doc, riskData);
        
        // Correlation Analysis
        await this.addCorrelationAnalysis(doc, riskData);
        
        // Stress Testing
        if (includeStressTest) {
            await this.addStressTesting(doc, riskData);
        }
        
        // Scenario Analysis
        if (includeScenarioAnalysis) {
            await this.addScenarioAnalysis(doc, riskData);
        }
        
        // Risk Recommendations
        if (includeRiskRecommendations) {
            await this.addRiskRecommendations(doc, riskData);
        }
        
        // Risk Charts
        await this.addRiskReturnChart(doc, riskData);
        await this.addVolatilityChart(doc, riskData);
        
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Generate Annual Investment Report
    async generateAnnualReport(userId, year) {
        const doc = new PDFDocument({ size: 'A4', margin: 50 });
        const userData = await this.getUserData(userId);
        const annualData = await this.getAnnualData(userId, year);

        await this.addReportHeader(doc, `ANNUAL INVESTMENT REPORT ${year}`, userData);
        
        // Year in Review
        await this.addYearInReview(doc, annualData);
        
        // Performance Highlights
        await this.addPerformanceHighlights(doc, annualData);
        
        // Investment Activity Summary
        await this.addInvestmentActivitySummary(doc, annualData);
        
        // Asset Allocation Evolution
        await this.addAssetAllocationEvolution(doc, annualData);
        
        // Goal Progress
        await this.addGoalProgress(doc, annualData);
        
        // Market Commentary
        await this.addMarketCommentary(doc, annualData);
        
        // Year Ahead Outlook
        await this.addYearAheadOutlook(doc, annualData);
        
        // Comprehensive Charts
        await this.addAnnualPerformanceChart(doc, annualData);
        await this.addAssetAllocationEvolutionChart(doc, annualData);
        
        await this.addDisclaimer(doc);
        await this.addReportFooter(doc, userData);

        return doc;
    }

    // Helper Methods for Report Sections

    async addReportHeader(doc, reportTitle, userData) {
        // Add company logo
        // doc.image('assets/logo.png', 50, 50, { width: 100 });
        
        // Company info
        doc.fontSize(20)
           .font('Helvetica-Bold')
           .text('SIP BREWERY', 50, 50);
        
        doc.fontSize(10)
           .font('Helvetica')
           .text('Premium Investment Platform', 50, 75)
           .text('SEBI Registered Investment Advisor', 50, 90);
        
        // Report title
        doc.fontSize(18)
           .font('Helvetica-Bold')
           .text(reportTitle, 50, 130, { align: 'center' });
        
        // User info
        doc.fontSize(12)
           .font('Helvetica')
           .text(`Client: ${userData.name}`, 400, 50)
           .text(`PAN: ${userData.pan || 'Not Provided'}`, 400, 65)
           .text(`Client ID: ${userData.clientId}`, 400, 80)
           .text(`Report Date: ${new Date().toLocaleDateString('en-IN')}`, 400, 95);
        
        // Add line separator
        doc.moveTo(50, 160)
           .lineTo(545, 160)
           .stroke();
        
        doc.y = 180;
    }

    async addPortfolioSummary(doc, portfolioData) {
        doc.fontSize(14)
           .font('Helvetica-Bold')
           .text('PORTFOLIO SUMMARY', 50, doc.y);
        
        doc.y += 20;
        
        // Summary table
        const summaryData = [
            ['Total Investment', `₹${portfolioData.totalInvested.toLocaleString('en-IN')}`],
            ['Current Value', `₹${portfolioData.currentValue.toLocaleString('en-IN')}`],
            ['Absolute Returns', `₹${portfolioData.absoluteReturns.toLocaleString('en-IN')}`],
            ['Returns %', `${portfolioData.returnsPercentage.toFixed(2)}%`],
            ['XIRR', `${portfolioData.xirr.toFixed(2)}%`],
            ['Number of Schemes', portfolioData.schemes.toString()],
            ['Active SIPs', portfolioData.activeSIPs.toString()]
        ];

        await this.addTable(doc, summaryData, { 
            columnWidths: [200, 150],
            headerBackground: '#f0f0f0'
        });
        
        doc.y += 30;
    }

    async addHoldingsDetails(doc, holdings) {
        doc.fontSize(14)
           .font('Helvetica-Bold')
           .text('HOLDINGS DETAILS', 50, doc.y);
        
        doc.y += 20;
        
        // Holdings table headers
        const headers = ['Scheme Name', 'Units', 'NAV', 'Investment', 'Current Value', 'Returns', 'Returns %'];
        const columnWidths = [140, 60, 50, 70, 80, 60, 60];
        
        // Add table header
        let startX = 50;
        doc.fontSize(10).font('Helvetica-Bold');
        
        headers.forEach((header, index) => {
            doc.text(header, startX, doc.y, { width: columnWidths[index], align: 'center' });
            startX += columnWidths[index];
        });
        
        doc.y += 15;
        
        // Add holdings data
        doc.font('Helvetica').fontSize(9);
        
        holdings.forEach(holding => {
            startX = 50;
            const rowData = [
                holding.schemeName,
                holding.units.toFixed(3),
                `₹${holding.nav.toFixed(2)}`,
                `₹${holding.investment.toLocaleString('en-IN')}`,
                `₹${holding.currentValue.toLocaleString('en-IN')}`,
                `₹${holding.returns.toLocaleString('en-IN')}`,
                `${holding.returnsPercentage.toFixed(2)}%`
            ];
            
            rowData.forEach((data, index) => {
                doc.text(data, startX, doc.y, { 
                    width: columnWidths[index], 
                    align: index === 0 ? 'left' : 'center' 
                });
                startX += columnWidths[index];
            });
            
            doc.y += 12;
            
            // Add new page if needed
            if (doc.y > 700) {
                doc.addPage();
                doc.y = 50;
            }
        });
        
        doc.y += 20;
    }

    async addPerformanceAnalysis(doc, portfolioData) {
        doc.fontSize(14)
           .font('Helvetica-Bold')
           .text('PERFORMANCE ANALYSIS', 50, doc.y);
        
        doc.y += 20;
        
        // Performance metrics
        const performanceData = [
            ['1 Month', `${portfolioData.performance['1M'].toFixed(2)}%`],
            ['3 Months', `${portfolioData.performance['3M'].toFixed(2)}%`],
            ['6 Months', `${portfolioData.performance['6M'].toFixed(2)}%`],
            ['1 Year', `${portfolioData.performance['1Y'].toFixed(2)}%`],
            ['3 Years (Annualized)', `${portfolioData.performance['3Y'].toFixed(2)}%`],
            ['5 Years (Annualized)', `${portfolioData.performance['5Y'].toFixed(2)}%`],
            ['Since Inception (Annualized)', `${portfolioData.performance.SI.toFixed(2)}%`]
        ];

        await this.addTable(doc, performanceData, { 
            columnWidths: [200, 150],
            headerBackground: '#e8f4f8'
        });
        
        doc.y += 30;
    }

    async addAssetAllocationChart(doc, assetAllocation) {
        doc.fontSize(14)
           .font('Helvetica-Bold')
           .text('ASSET ALLOCATION', 50, doc.y);
        
        doc.y += 20;
        
        // Create pie chart
        const canvas = createCanvas(400, 300);
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: Object.keys(assetAllocation),
                datasets: [{
                    data: Object.values(assetAllocation),
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
                    ]
                }]
            },
            options: {
                responsive: false,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
        
        // Convert canvas to buffer and add to PDF
        const buffer = canvas.toBuffer('image/png');
        doc.image(buffer, 50, doc.y, { width: 400, height: 300 });
        
        doc.y += 320;
    }

    async addTable(doc, data, options = {}) {
        const { columnWidths = [200, 200], headerBackground = '#f0f0f0' } = options;
        const startX = 50;
        let currentY = doc.y;
        
        data.forEach((row, rowIndex) => {
            let x = startX;
            
            // Add background for header row
            if (rowIndex === 0 && headerBackground) {
                doc.rect(startX, currentY - 2, columnWidths.reduce((a, b) => a + b, 0), 15)
                   .fill(headerBackground);
                doc.fillColor('black');
            }
            
            row.forEach((cell, cellIndex) => {
                doc.fontSize(10)
                   .font(rowIndex === 0 ? 'Helvetica-Bold' : 'Helvetica')
                   .text(cell, x, currentY, { 
                       width: columnWidths[cellIndex], 
                       align: cellIndex === 0 ? 'left' : 'right' 
                   });
                x += columnWidths[cellIndex];
            });
            
            currentY += 15;
        });
        
        doc.y = currentY;
    }

    async addDisclaimer(doc) {
        doc.addPage();
        doc.fontSize(14)
           .font('Helvetica-Bold')
           .text('IMPORTANT DISCLAIMERS', 50, 50);
        
        doc.fontSize(10)
           .font('Helvetica')
           .text(`
1. MUTUAL FUND INVESTMENTS ARE SUBJECT TO MARKET RISKS. READ ALL SCHEME RELATED DOCUMENTS CAREFULLY.

2. Past performance is not indicative of future results. The value of investments can go up as well as down.

3. This report is generated based on data available as of ${new Date().toLocaleDateString('en-IN')} and may not reflect real-time market conditions.

4. SIP Brewery is a SEBI registered Investment Advisor. Registration No: [Registration Number].

5. This report is for informational purposes only and should not be construed as investment advice.

6. Tax implications mentioned are based on current tax laws and may change. Please consult your tax advisor.

7. All calculations are based on the information provided by the investor and fund houses.

8. SIP Brewery does not guarantee the accuracy of third-party data used in this report.

9. Investors should carefully consider their investment objectives, risk tolerance, and financial situation before making investment decisions.

10. For any queries or clarifications, please contact our customer support at support@sipbrewery.com or call 1800-XXX-XXXX.
        `, 50, 100, { width: 495, align: 'justify' });
    }

    async addReportFooter(doc, userData) {
        doc.fontSize(8)
           .font('Helvetica')
           .text(`Report generated on ${new Date().toLocaleString('en-IN')} | SIP Brewery - Premium Investment Platform`, 
                  50, 750, { align: 'center' });
    }

    // Data fetching methods (mock implementations - replace with actual database queries)
    async getUserData(userId) {
        return {
            id: userId,
            name: 'Rajesh Kumar',
            email: 'rajesh@example.com',
            pan: 'ABCDE1234F',
            clientId: 'SB' + userId.slice(-6).toUpperCase(),
            phone: '+919876543210',
            address: 'Mumbai, Maharashtra',
            riskProfile: 'MODERATE'
        };
    }

    async getPortfolioData(userId, dateRange) {
        return {
            totalInvested: 500000,
            currentValue: 625000,
            absoluteReturns: 125000,
            returnsPercentage: 25.0,
            xirr: 18.5,
            schemes: 5,
            activeSIPs: 3,
            holdings: [
                {
                    schemeName: 'HDFC Top 100 Fund - Direct Growth',
                    units: 1180.5,
                    nav: 100.0,
                    investment: 100000,
                    currentValue: 118050,
                    returns: 18050,
                    returnsPercentage: 18.05
                },
                {
                    schemeName: 'SBI Blue Chip Fund - Direct Growth',
                    units: 2825.0,
                    nav: 60.0,
                    investment: 150000,
                    currentValue: 169500,
                    returns: 19500,
                    returnsPercentage: 13.0
                }
            ],
            performance: {
                '1M': 2.5,
                '3M': 8.2,
                '6M': 15.3,
                '1Y': 25.0,
                '3Y': 18.5,
                '5Y': 16.2,
                'SI': 18.5
            },
            assetAllocation: {
                'Large Cap': 45,
                'Mid Cap': 25,
                'Small Cap': 20,
                'Debt': 10
            }
        };
    }

    // Additional data fetching methods would be implemented here...
    async getTransactionData(userId, dateRange) { /* Implementation */ }
    async getPerformanceData(userId, period) { /* Implementation */ }
    async getTaxData(userId, financialYear) { /* Implementation */ }
    async getCapitalGainsData(userId, financialYear, gainType) { /* Implementation */ }
    async getSIPData(userId, period) { /* Implementation */ }
    async getRiskAnalysisData(userId) { /* Implementation */ }
    async getAnnualData(userId, year) { /* Implementation */ }
}

module.exports = new PDFReportService();
