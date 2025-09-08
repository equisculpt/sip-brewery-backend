// Sample PDF Report Generator - Test ASI Portfolio Analysis PDF Quality
const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');

class SamplePDFGenerator {
    constructor() {
        this.colors = {
            primary: '#1a365d',
            secondary: '#2d3748',
            accent: '#00ff87',
            success: '#38a169',
            warning: '#d69e2e',
            danger: '#e53e3e'
        };
    }

    async generateSampleASIReport() {
        console.log('🎯 Generating Sample ASI Portfolio Analysis Report...');
        
        const doc = new PDFDocument({ 
            size: 'A4', 
            margin: 50,
            info: {
                Title: 'ASI Portfolio Analysis Report - Sample',
                Author: 'SIP Brewery ASI Engine',
                Subject: 'AI-Powered Portfolio Analysis',
                Keywords: 'ASI, portfolio, analysis, AI, investment'
            }
        });

        const outputPath = path.join(__dirname, 'sample_reports', 'ASI_Portfolio_Analysis_Sample.pdf');
        
        // Ensure output directory exists
        const outputDir = path.dirname(outputPath);
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        // Create write stream
        const stream = fs.createWriteStream(outputPath);
        doc.pipe(stream);

        // Generate report content
        await this.addCoverPage(doc);
        await this.addExecutiveSummary(doc);
        await this.addASIScoreAnalysis(doc);
        await this.addPerformanceAttribution(doc);
        await this.addAIRecommendations(doc);
        await this.addComplianceDisclaimer(doc);

        // Finalize the PDF
        doc.end();

        return new Promise((resolve, reject) => {
            stream.on('finish', () => {
                console.log('✅ Sample PDF generated successfully!');
                console.log(`📄 File saved at: ${outputPath}`);
                resolve(outputPath);
            });
            stream.on('error', reject);
        });
    }

    async addCoverPage(doc) {
        // Header with gradient effect simulation
        doc.rect(0, 0, 612, 200)
           .fillColor('#1a365d')
           .fill();

        // Company logo area (simulated)
        doc.rect(50, 30, 60, 60)
           .fillColor('#00ff87')
           .fill();

        doc.fillColor('white')
           .fontSize(12)
           .font('Helvetica-Bold')
           .text('SIP', 65, 50)
           .text('BREWERY', 60, 65);

        // Main title
        doc.fillColor('white')
           .fontSize(32)
           .font('Helvetica-Bold')
           .text('ASI PORTFOLIO', 150, 40)
           .text('ANALYSIS REPORT', 150, 80);

        // Subtitle
        doc.fontSize(16)
           .font('Helvetica')
           .fillColor('#00ff87')
           .text('Artificial Super Intelligence Powered Investment Analysis', 150, 130);

        // Client information box
        doc.rect(50, 250, 512, 120)
           .fillColor('#f7fafc')
           .fill()
           .stroke('#e2e8f0');

        doc.fillColor('#1a365d')
           .fontSize(14)
           .font('Helvetica-Bold')
           .text('CLIENT INFORMATION', 70, 270);

        doc.fontSize(12)
           .font('Helvetica')
           .text('Client Name: Rajesh Kumar', 70, 300)
           .text('Client ID: SB001234', 70, 320)
           .text('Report Date: ' + new Date().toLocaleDateString('en-IN'), 70, 340)
           .text('Analysis Period: Last 12 Months', 300, 300)
           .text('ASI Model Version: 3.2.1', 300, 320)
           .text('Risk Profile: Moderate Aggressive', 300, 340);

        // ASI Score highlight
        doc.rect(50, 400, 512, 80)
           .fillColor('#38a169')
           .fill();

        doc.fillColor('white')
           .fontSize(20)
           .font('Helvetica-Bold')
           .text('OVERALL ASI SCORE', 70, 420);

        doc.fontSize(48)
           .font('Helvetica-Bold')
           .text('87.5', 450, 410);

        doc.fontSize(12)
           .font('Helvetica')
           .text('EXCELLENT - Strong alignment with AI recommendations', 70, 450);

        doc.addPage();
    }

    async addExecutiveSummary(doc) {
        this.addPageHeader(doc, 'EXECUTIVE SUMMARY');

        doc.fillColor('#1a365d')
           .fontSize(14)
           .font('Helvetica-Bold')
           .text('Portfolio Performance Overview', 50, 120);

        doc.fontSize(11)
           .font('Helvetica')
           .fillColor('#2d3748')
           .text(`Your portfolio has demonstrated exceptional performance over the analysis period, achieving a remarkable 25.8% return year-to-date. Our ASI engine has identified several key strengths in your investment strategy while highlighting specific opportunities for optimization.

The portfolio's risk-adjusted performance, measured by a Sharpe ratio of 1.85, significantly outperforms market benchmarks and peer portfolios.`, 50, 150, { width: 500, align: 'justify' });

        // Performance metrics table
        doc.rect(50, 220, 512, 200)
           .fillColor('#f7fafc')
           .fill()
           .stroke('#e2e8f0');

        doc.fillColor('#1a365d')
           .fontSize(12)
           .font('Helvetica-Bold')
           .text('KEY PERFORMANCE METRICS', 70, 240);

        const performanceData = [
            ['Metric', 'Your Portfolio', 'Benchmark', 'Peer Average'],
            ['Total Return (YTD)', '+25.8%', '+18.2%', '+21.3%'],
            ['Annualized Return (3Y)', '+18.5%', '+12.8%', '+15.2%'],
            ['Sharpe Ratio', '1.85', '1.12', '1.34'],
            ['Maximum Drawdown', '-8.2%', '-12.5%', '-10.8%'],
            ['Alpha Generation', '+4.2%', '0.0%', '+1.8%']
        ];

        this.addTable(doc, performanceData, { x: 70, y: 260, width: 470, cellHeight: 20 });

        doc.addPage();
    }

    async addASIScoreAnalysis(doc) {
        this.addPageHeader(doc, 'ASI SCORE BREAKDOWN');

        doc.fillColor('#1a365d')
           .fontSize(16)
           .font('Helvetica-Bold')
           .text('ARTIFICIAL SUPER INTELLIGENCE SCORE: 87.5/100', 70, 140);

        // Score components
        const scoreComponents = [
            { factor: 'Performance Excellence', score: 92, weight: '30%' },
            { factor: 'Risk-Adjusted Returns', score: 88, weight: '25%' },
            { factor: 'Diversification Quality', score: 85, weight: '20%' },
            { factor: 'Cost Efficiency', score: 90, weight: '15%' },
            { factor: 'Tax Optimization', score: 82, weight: '10%' }
        ];

        let yPos = 180;
        scoreComponents.forEach(component => {
            doc.fillColor('#2d3748')
               .fontSize(11)
               .font('Helvetica-Bold')
               .text(component.factor, 70, yPos);

            doc.fillColor('#1a365d')
               .fontSize(14)
               .font('Helvetica-Bold')
               .text(component.score.toString(), 400, yPos);

            doc.fillColor('#718096')
               .fontSize(9)
               .font('Helvetica')
               .text(`Weight: ${component.weight}`, 70, yPos + 15);

            // Score bar
            const barWidth = 250;
            const scoreWidth = (component.score / 100) * barWidth;
            
            doc.rect(150, yPos + 5, barWidth, 8)
               .fillColor('#e2e8f0')
               .fill();

            let barColor = component.score >= 80 ? '#38a169' : '#d69e2e';
            doc.rect(150, yPos + 5, scoreWidth, 8)
               .fillColor(barColor)
               .fill();

            yPos += 40;
        });

        doc.addPage();
    }

    async addPerformanceAttribution(doc) {
        this.addPageHeader(doc, 'PERFORMANCE ATTRIBUTION');

        doc.fillColor('#1a365d')
           .fontSize(14)
           .font('Helvetica-Bold')
           .text('Top Contributing Holdings', 50, 120);

        const topHoldings = [
            { name: 'ICICI Technology Fund', contribution: '+7.3%', allocation: '28.5%' },
            { name: 'Axis Small Cap Fund', contribution: '+4.3%', allocation: '22.0%' },
            { name: 'HDFC Top 100 Fund', contribution: '+4.0%', allocation: '25.0%' },
            { name: 'SBI Healthcare Fund', contribution: '+2.1%', allocation: '12.5%' },
            { name: 'Kotak Emerging Equity', contribution: '+1.8%', allocation: '12.0%' }
        ];

        let yPos = 150;
        topHoldings.forEach(holding => {
            doc.fillColor('#2d3748')
               .fontSize(11)
               .font('Helvetica')
               .text(holding.name, 50, yPos);

            doc.fillColor('#38a169')
               .fontSize(11)
               .font('Helvetica-Bold')
               .text(holding.contribution, 300, yPos);

            doc.fillColor('#718096')
               .fontSize(10)
               .font('Helvetica')
               .text(`(${holding.allocation})`, 400, yPos);

            yPos += 25;
        });

        doc.addPage();
    }

    async addAIRecommendations(doc) {
        this.addPageHeader(doc, 'AI RECOMMENDATIONS');

        doc.fillColor('#1a365d')
           .fontSize(14)
           .font('Helvetica-Bold')
           .text('🚨 IMMEDIATE ACTION ITEMS', 50, 120);

        const actions = [
            {
                action: 'Rebalance Technology Exposure',
                priority: 'HIGH',
                impact: '+0.8% risk-adjusted returns',
                confidence: '87%'
            },
            {
                action: 'Increase SIP Frequency',
                priority: 'MEDIUM',
                impact: '+0.3% annual returns',
                confidence: '72%'
            }
        ];

        let yPos = 150;
        actions.forEach(action => {
            const bgColor = action.priority === 'HIGH' ? '#fff5f5' : '#fffbf0';
            
            doc.rect(50, yPos - 5, 512, 60)
               .fillColor(bgColor)
               .fill()
               .stroke('#e2e8f0');

            doc.fillColor('#1a365d')
               .fontSize(12)
               .font('Helvetica-Bold')
               .text(action.action, 70, yPos);

            doc.fillColor('#e53e3e')
               .fontSize(10)
               .font('Helvetica-Bold')
               .text(`Priority: ${action.priority}`, 400, yPos);

            doc.fillColor('#38a169')
               .fontSize(9)
               .font('Helvetica-Bold')
               .text(`Impact: ${action.impact}`, 70, yPos + 20);

            doc.fillColor('#718096')
               .fontSize(9)
               .font('Helvetica')
               .text(`Confidence: ${action.confidence}`, 70, yPos + 35);

            yPos += 70;
        });

        doc.addPage();
    }

    async addComplianceDisclaimer(doc) {
        this.addPageHeader(doc, 'COMPLIANCE & DISCLAIMERS');

        doc.fillColor('#1a365d')
           .fontSize(12)
           .font('Helvetica-Bold')
           .text('IMPORTANT DISCLAIMERS', 50, 120);
        
        doc.fontSize(10)
           .font('Helvetica')
           .fillColor('#2d3748')
           .text(`1. MUTUAL FUND INVESTMENTS ARE SUBJECT TO MARKET RISKS. READ ALL SCHEME RELATED DOCUMENTS CAREFULLY.

2. Past performance is not indicative of future results. The value of investments can go up as well as down.

3. This report is generated based on data available as of ${new Date().toLocaleDateString('en-IN')} and may not reflect real-time market conditions.

4. SIP Brewery is a SEBI registered Investment Advisor. Registration No: [Registration Number].

5. This report is for informational purposes only and should not be construed as investment advice.

6. All calculations are based on the information provided by the investor and fund houses.

7. For any queries, please contact support@sipbrewery.com or call 1800-XXX-XXXX.`, 50, 150, { width: 500, align: 'justify' });

        doc.fontSize(8)
           .font('Helvetica')
           .fillColor('#718096')
           .text(`Report generated on ${new Date().toLocaleString('en-IN')} | SIP Brewery - Premium Investment Platform`, 
                  50, 750, { align: 'center' });
    }

    addPageHeader(doc, title) {
        doc.rect(0, 0, 612, 60)
           .fillColor('#1a365d')
           .fill();

        doc.fillColor('white')
           .fontSize(16)
           .font('Helvetica-Bold')
           .text(title, 50, 25);

        doc.fillColor('#00ff87')
           .fontSize(10)
           .font('Helvetica')
           .text('SIP Brewery ASI Engine', 450, 30);
    }

    addTable(doc, data, options = {}) {
        const { x = 50, y = 150, width = 500, cellHeight = 20 } = options;
        const colWidth = width / data[0].length;

        data.forEach((row, rowIndex) => {
            row.forEach((cell, colIndex) => {
                const cellX = x + (colIndex * colWidth);
                const cellY = y + (rowIndex * cellHeight);

                // Header row styling
                if (rowIndex === 0) {
                    doc.rect(cellX, cellY, colWidth, cellHeight)
                       .fillColor('#1a365d')
                       .fill();
                    
                    doc.fillColor('white')
                       .fontSize(9)
                       .font('Helvetica-Bold')
                       .text(cell, cellX + 5, cellY + 6, { width: colWidth - 10 });
                } else {
                    doc.rect(cellX, cellY, colWidth, cellHeight)
                       .fillColor(rowIndex % 2 === 0 ? '#f7fafc' : 'white')
                       .fill()
                       .stroke('#e2e8f0');
                    
                    doc.fillColor('#2d3748')
                       .fontSize(8)
                       .font('Helvetica')
                       .text(cell, cellX + 5, cellY + 6, { width: colWidth - 10 });
                }
            });
        });
    }
}

// Export for use
module.exports = SamplePDFGenerator;

// If run directly, generate sample
if (require.main === module) {
    const generator = new SamplePDFGenerator();
    generator.generateSampleASIReport()
        .then(filePath => {
            console.log(`🎉 Sample PDF generated successfully at: ${filePath}`);
        })
        .catch(error => {
            console.error('❌ Error generating PDF:', error);
        });
}
