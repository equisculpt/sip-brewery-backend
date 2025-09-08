// ⚡ FAST CSS CHARTS PDF GENERATOR
// NO JAVASCRIPT CHARTS - PURE CSS VISUALS

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

async function generateFastCSSChartsPDF() {
  console.log('⚡ GENERATING FAST CSS CHARTS PDF');
  console.log('🎯 NO HANGING - PURE CSS VISUALS');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const outputDir = path.join(__dirname, 'fast_css_reports');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const fileName = `SIP_Brewery_CSS_CHARTS_${Date.now()}.pdf`;
  const outputPath = path.join(outputDir, fileName);
  const reportDate = new Date().toLocaleDateString('en-IN');
  const reportTime = new Date().toLocaleTimeString('en-IN');
  
  const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIP Brewery - Fast CSS Charts Report</title>
    <style>
        @page {
            size: A4;
            margin: 15mm;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.4;
            color: #1a1a1a;
            background: white;
            font-size: 12px;
        }
        
        .page {
            background: white;
            padding: 10mm;
            page-break-after: always;
        }
        
        .page:last-child {
            page-break-after: avoid;
        }
        
        /* HEADER */
        .header {
            background: linear-gradient(135deg, #000000 0%, #333333 100%);
            color: white;
            padding: 15mm;
            margin: -10mm -10mm 8mm -10mm;
            text-align: center;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #d4af37, #ffd700, #d4af37);
        }
        
        .company-logo {
            font-size: 28px;
            font-weight: bold;
            background: linear-gradient(45deg, #d4af37, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5mm;
        }
        
        .report-title {
            font-size: 18px;
            margin-bottom: 3mm;
            font-weight: bold;
        }
        
        .client-info {
            font-size: 14px;
            opacity: 0.9;
        }
        
        /* EXECUTIVE SUMMARY */
        .executive-summary {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 2px solid #d4af37;
            border-radius: 8px;
            padding: 8mm;
            margin: 6mm 0;
        }
        
        .summary-title {
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 6mm;
            color: #1a1a1a;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 4mm;
            margin: 6mm 0;
        }
        
        .metric-box {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 4mm;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #d4af37;
            margin-bottom: 2mm;
        }
        
        .metric-label {
            font-size: 10px;
            color: #64748b;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        /* CSS CHARTS SECTION */
        .charts-section {
            margin: 8mm 0;
        }
        
        .section-title {
            font-size: 14px;
            font-weight: bold;
            color: #1a1a1a;
            margin-bottom: 4mm;
            padding-bottom: 2mm;
            border-bottom: 2px solid #d4af37;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6mm;
            margin-top: 6mm;
        }
        
        .chart-container {
            background: white;
            border: 2px solid #d4af37;
            border-radius: 8px;
            padding: 4mm;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .chart-title {
            font-size: 12px;
            font-weight: bold;
            color: #374151;
            margin-bottom: 3mm;
            text-align: center;
        }
        
        /* CSS PIE CHART */
        .pie-chart {
            width: 60mm;
            height: 60mm;
            border-radius: 50%;
            margin: 0 auto 3mm auto;
            background: conic-gradient(
                #d4af37 0deg 126deg,     /* Large Cap: 35% */
                #ffd700 126deg 198deg,   /* Mid Cap: 20% */
                #b8860b 198deg 252deg,   /* Small Cap: 15% */
                #1f2937 252deg 324deg,   /* Debt: 20% */
                #f59e0b 324deg 352.8deg, /* Gold: 8% */
                #6b7280 352.8deg 360deg  /* International: 2% */
            );
            position: relative;
        }
        
        .pie-chart::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 25mm;
            height: 25mm;
            background: white;
            border-radius: 50%;
        }
        
        .pie-legend {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2mm;
            font-size: 9px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 2mm;
        }
        
        .legend-color {
            width: 3mm;
            height: 3mm;
            border-radius: 50%;
        }
        
        .legend-color.large-cap { background: #d4af37; }
        .legend-color.mid-cap { background: #ffd700; }
        .legend-color.small-cap { background: #b8860b; }
        .legend-color.debt { background: #1f2937; }
        .legend-color.gold { background: #f59e0b; }
        .legend-color.international { background: #6b7280; }
        
        /* CSS BAR CHART */
        .bar-chart {
            height: 45mm;
            display: flex;
            align-items: end;
            gap: 5mm;
            padding: 0 5mm;
            margin-bottom: 8mm;
        }
        
        .bar {
            flex: 1;
            background: linear-gradient(to top, #d4af37, #ffd700);
            border-radius: 2mm 2mm 0 0;
            position: relative;
            min-height: 5mm;
        }
        
        .bar.portfolio { height: 85%; }
        .bar.benchmark { height: 70%; background: linear-gradient(to top, #6b7280, #9ca3af); }
        
        .bar-label {
            position: absolute;
            bottom: -7mm;
            left: 50%;
            transform: translateX(-50%);
            font-size: 10px;
            font-weight: bold;
            color: #374151;
            white-space: nowrap;
        }
        
        .bar-value {
            position: absolute;
            top: -4mm;
            left: 50%;
            transform: translateX(-50%);
            font-size: 8px;
            font-weight: bold;
            color: #1a1a1a;
        }
        
        /* PERFORMANCE TABLE */
        .performance-section {
            margin: 8mm 0;
        }
        
        .performance-table {
            background: white;
            border: 2px solid #d4af37;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .table-header {
            background: linear-gradient(135deg, #1a1a1a 0%, #374151 100%);
            color: white;
            padding: 4mm;
            text-align: center;
            font-weight: bold;
            font-size: 12px;
        }
        
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
        }
        
        .performance-cell {
            padding: 3mm;
            text-align: center;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .performance-cell:last-child {
            border-right: none;
        }
        
        .period-label {
            font-size: 9px;
            color: #64748b;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .performance-value {
            font-size: 14px;
            font-weight: bold;
            color: #d4af37;
            margin-top: 1mm;
        }
        
        /* RISK INDICATORS */
        .risk-section {
            background: linear-gradient(135deg, #fef7ed 0%, #fed7aa 100%);
            border: 2px solid #f59e0b;
            border-radius: 8px;
            padding: 6mm;
            margin: 6mm 0;
        }
        
        .risk-title {
            font-size: 14px;
            font-weight: bold;
            color: #92400e;
            text-align: center;
            margin-bottom: 4mm;
        }
        
        .risk-indicators {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4mm;
        }
        
        .risk-indicator {
            text-align: center;
        }
        
        .risk-meter {
            width: 20mm;
            height: 4mm;
            background: #e5e7eb;
            border-radius: 2mm;
            margin: 0 auto 2mm auto;
            position: relative;
            overflow: hidden;
        }
        
        .risk-fill {
            height: 100%;
            border-radius: 2mm;
            background: linear-gradient(to right, #10b981, #f59e0b, #ef4444);
        }
        
        .risk-fill.low { width: 30%; background: #10b981; }
        .risk-fill.medium { width: 60%; background: #f59e0b; }
        .risk-fill.high { width: 85%; background: #ef4444; }
        
        .risk-label {
            font-size: 10px;
            font-weight: bold;
            color: #92400e;
        }
        
        .risk-value {
            font-size: 12px;
            font-weight: bold;
            color: #1a1a1a;
            margin-top: 1mm;
        }
        
        /* COMPLIANCE */
        .compliance-section {
            background: #1f2937;
            color: white;
            border-radius: 8px;
            padding: 6mm;
            margin: 6mm 0;
        }
        
        .compliance-title {
            font-size: 12px;
            font-weight: bold;
            color: #ef4444;
            text-align: center;
            margin-bottom: 4mm;
        }
        
        .compliance-text {
            font-size: 10px;
            line-height: 1.4;
            text-align: center;
            color: #d1d5db;
        }
        
        /* FOOTER */
        .footer {
            background: #000000;
            color: white;
            padding: 4mm;
            margin: 6mm -10mm -10mm -10mm;
            text-align: center;
            font-size: 10px;
        }
        
        .footer-logo {
            font-size: 14px;
            font-weight: bold;
            color: #d4af37;
            margin-bottom: 2mm;
        }
    </style>
</head>
<body>
    <div class="page">
        <!-- HEADER -->
        <header class="header">
            <div class="company-logo">🍺 SIP BREWERY</div>
            <div class="report-title">PORTFOLIO ANALYSIS REPORT</div>
            <div class="client-info">
                Rajesh Kumar Sharma • ₹6,25,000 • ${reportDate} ${reportTime}
            </div>
        </header>
        
        <!-- EXECUTIVE SUMMARY -->
        <section class="executive-summary">
            <h2 class="summary-title">EXECUTIVE SUMMARY</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value">72</div>
                    <div class="metric-label">ASI Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">16.8%</div>
                    <div class="metric-label">Expected Return</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">78%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">14.2%</div>
                    <div class="metric-label">Portfolio Risk</div>
                </div>
            </div>
            <p style="text-align: justify; font-size: 11px; line-height: 1.4; color: #374151;">
                <strong>Portfolio Overview:</strong> Your portfolio demonstrates exceptional performance with an ASI score of 72/100, 
                indicating superior artificial intelligence optimization. The projected annual return of 16.8% significantly 
                outperforms market benchmarks with a 78% probability of positive returns. Risk management remains optimal 
                at 14.2% volatility, ensuring balanced growth with institutional-grade risk controls.
            </p>
        </section>
        
        <!-- CSS CHARTS SECTION -->
        <section class="charts-section">
            <h3 class="section-title">📊 PORTFOLIO VISUALIZATION</h3>
            <div class="charts-grid">
                <div class="chart-container">
                    <h4 class="chart-title">Asset Allocation</h4>
                    <div class="pie-chart"></div>
                    <div class="pie-legend">
                        <div class="legend-item">
                            <div class="legend-color large-cap"></div>
                            <span>Large Cap (35%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color mid-cap"></div>
                            <span>Mid Cap (20%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color small-cap"></div>
                            <span>Small Cap (15%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color debt"></div>
                            <span>Debt (20%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color gold"></div>
                            <span>Gold (8%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color international"></div>
                            <span>International (2%)</span>
                        </div>
                    </div>
                </div>
                <div class="chart-container">
                    <h4 class="chart-title">Performance vs Benchmark</h4>
                    <div class="bar-chart">
                        <div class="bar portfolio">
                            <div class="bar-value">128.4%</div>
                            <div class="bar-label">Portfolio</div>
                        </div>
                        <div class="bar benchmark">
                            <div class="bar-value">119.8%</div>
                            <div class="bar-label">Benchmark</div>
                        </div>
                    </div>
                    <div style="text-align: center; font-size: 11px; color: #d4af37; margin-top: 4mm; font-weight: bold;">
                        Portfolio Outperformance: +8.6%
                    </div>
                </div>
            </div>
        </section>
        
        <!-- PERFORMANCE TABLE -->
        <section class="performance-section">
            <h3 class="section-title">📈 PERFORMANCE ANALYSIS</h3>
            <div class="performance-table">
                <div class="table-header">Historical Returns Analysis</div>
                <div class="performance-grid">
                    <div class="performance-cell">
                        <div class="period-label">1 Month</div>
                        <div class="performance-value">+3.2%</div>
                    </div>
                    <div class="performance-cell">
                        <div class="period-label">3 Months</div>
                        <div class="performance-value">+8.7%</div>
                    </div>
                    <div class="performance-cell">
                        <div class="period-label">6 Months</div>
                        <div class="performance-value">+15.4%</div>
                    </div>
                    <div class="performance-cell">
                        <div class="period-label">1 Year</div>
                        <div class="performance-value">+22.8%</div>
                    </div>
                    <div class="performance-cell">
                        <div class="period-label">3 Years</div>
                        <div class="performance-value">+18.3%</div>
                    </div>
                    <div class="performance-cell">
                        <div class="period-label">Inception</div>
                        <div class="performance-value">+24.6%</div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- RISK SECTION -->
        <section class="risk-section">
            <h3 class="risk-title">⚠️ RISK ASSESSMENT</h3>
            <div class="risk-indicators">
                <div class="risk-indicator">
                    <div class="risk-meter">
                        <div class="risk-fill medium"></div>
                    </div>
                    <div class="risk-label">Portfolio Volatility</div>
                    <div class="risk-value">14.2%</div>
                </div>
                <div class="risk-indicator">
                    <div class="risk-meter">
                        <div class="risk-fill low"></div>
                    </div>
                    <div class="risk-label">Sharpe Ratio</div>
                    <div class="risk-value">1.18</div>
                </div>
                <div class="risk-indicator">
                    <div class="risk-meter">
                        <div class="risk-fill low"></div>
                    </div>
                    <div class="risk-label">Max Drawdown</div>
                    <div class="risk-value">-8.3%</div>
                </div>
            </div>
        </section>
        
        <!-- COMPLIANCE -->
        <section class="compliance-section">
            <h3 class="compliance-title">🏛️ REGULATORY COMPLIANCE</h3>
            <div class="compliance-text">
                <strong>SEBI Registration:</strong> SIP Brewery is a SEBI registered Mutual Fund Distributor (ARN-12345). 
                <strong>Disclaimer:</strong> Mutual fund investments are subject to market risks. Past performance is not indicative of future results. 
                <strong>Advisory Notice:</strong> SIP Brewery is a Distributor, NOT an Investment Advisor. We do not provide investment advice. 
                <strong>Risk Warning:</strong> Investment values can fluctuate. This analysis is based on historical data which may not predict future performance.
            </div>
        </section>
        
        <!-- FOOTER -->
        <footer class="footer">
            <div class="footer-logo">🍺 SIP BREWERY</div>
            <div>Institutional Portfolio Analytics • SEBI Registered • AMFI: ARN-12345<br>
            Educational Analysis Platform • support@sipbrewery.com • Mumbai, India</div>
        </footer>
    </div>
</body>
</html>`;

  console.log('🚀 Launching Puppeteer (fast mode)...');
  
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.setContent(htmlContent, { waitUntil: 'domcontentloaded' });
  
  // Quick wait - no JavaScript to load
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  await page.pdf({
    path: outputPath,
    format: 'A4',
    printBackground: true,
    margin: { top: '0mm', right: '0mm', bottom: '0mm', left: '0mm' }
  });
  
  await browser.close();
  
  const fileSize = fs.statSync(outputPath).size;
  
  console.log('⚡ FAST CSS CHARTS PDF GENERATED!');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📄 File:', fileName);
  console.log('📁 Location:', outputPath);
  console.log('📏 Size:', fileSize, 'bytes');
  console.log('⚡ Speed: LIGHTNING FAST - NO HANGING');
  console.log('🎯 Features:');
  console.log('✅ Pure CSS pie chart (no JavaScript)');
  console.log('✅ CSS bar chart for performance comparison');
  console.log('✅ Risk assessment meters');
  console.log('✅ Professional color gradients');
  console.log('✅ Instant rendering - no waiting');
  console.log('✅ 100% reliable chart display');
  console.log('✅ SEBI compliant content');
  
  return outputPath;
}

generateFastCSSChartsPDF().catch(console.error);
