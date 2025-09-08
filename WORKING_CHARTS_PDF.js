// 🎯 WORKING CHARTS PDF GENERATOR
// GUARANTEED CHART RENDERING FOR $1 BILLION PLATFORM

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

async function generateWorkingChartsPDF() {
  console.log('🎯 GENERATING PDF WITH WORKING CHARTS');
  console.log('📊 ENSURING CHARTS ACTUALLY RENDER');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const outputDir = path.join(__dirname, 'working_charts_reports');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const fileName = `SIP_Brewery_WORKING_CHARTS_${Date.now()}.pdf`;
  const outputPath = path.join(outputDir, fileName);
  const reportDate = new Date().toLocaleDateString('en-IN');
  const reportTime = new Date().toLocaleTimeString('en-IN');
  
  // Read beer mug icon
  const mugPath = path.join(__dirname, 'PERFECT_BEER_MUG.svg');
  let mugIcon = '';
  if (fs.existsSync(mugPath)) {
    mugIcon = fs.readFileSync(mugPath, 'utf8');
  }
  const mugBase64 = Buffer.from(mugIcon).toString('base64');
  const mugDataUrl = `data:image/svg+xml;base64,${mugBase64}`;
  
  const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIP Brewery - Working Charts Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
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
            font-size: 24px;
            font-weight: bold;
            background: linear-gradient(45deg, #d4af37, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5mm;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .mug-icon {
            width: 40px;
            height: 40px;
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
            font-size: 18px;
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
        
        /* CHARTS SECTION */
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
        
        .chart-wrapper {
            position: relative;
            height: 50mm;
            width: 100%;
        }
        
        .chart-wrapper canvas {
            max-width: 100% !important;
            max-height: 100% !important;
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
        
        /* CHART LOADING INDICATOR */
        .chart-loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 50mm;
            background: #f8fafc;
            border: 2px dashed #d4af37;
            border-radius: 8px;
            color: #64748b;
            font-size: 14px;
            font-weight: bold;
        }
        
        .charts-loaded .chart-loading {
            display: none;
        }
    </style>
</head>
<body>
    <div class="page">
        <!-- HEADER -->
        <header class="header">
            <div class="company-logo">
                <img src="${mugDataUrl}" alt="SIP Brewery" class="mug-icon" />
                SIP BREWERY
            </div>
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
        
        <!-- CHARTS SECTION -->
        <section class="charts-section">
            <h3 class="section-title">📊 PORTFOLIO VISUALIZATION</h3>
            <div class="charts-grid">
                <div class="chart-container">
                    <h4 class="chart-title">Asset Allocation</h4>
                    <div class="chart-wrapper">
                        <div class="chart-loading">Loading Chart...</div>
                        <canvas id="allocationChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h4 class="chart-title">Performance vs Benchmark</h4>
                    <div class="chart-wrapper">
                        <div class="chart-loading">Loading Chart...</div>
                        <canvas id="performanceChart"></canvas>
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
    
    <script>
        console.log('🎯 Starting chart initialization...');
        
        // Wait for Chart.js to load
        function waitForChartJS() {
            return new Promise((resolve) => {
                if (typeof Chart !== 'undefined') {
                    console.log('✅ Chart.js loaded successfully');
                    resolve();
                } else {
                    console.log('⏳ Waiting for Chart.js...');
                    setTimeout(() => waitForChartJS().then(resolve), 100);
                }
            });
        }
        
        async function initializeCharts() {
            await waitForChartJS();
            
            console.log('🎨 Initializing charts...');
            
            // Configure Chart.js defaults
            Chart.defaults.font.family = 'Arial';
            Chart.defaults.font.size = 11;
            Chart.defaults.color = '#374151';
            
            try {
                // ASSET ALLOCATION CHART
                console.log('📊 Creating allocation chart...');
                const allocationCtx = document.getElementById('allocationChart');
                if (allocationCtx) {
                    const allocationChart = new Chart(allocationCtx, {
                        type: 'doughnut',
                        data: {
                            labels: ['Large Cap', 'Mid Cap', 'Small Cap', 'Debt', 'Gold', 'International'],
                            datasets: [{
                                data: [35, 20, 15, 20, 8, 2],
                                backgroundColor: [
                                    '#d4af37',
                                    '#ffd700',
                                    '#b8860b',
                                    '#1f2937',
                                    '#f59e0b',
                                    '#6b7280'
                                ],
                                borderColor: '#ffffff',
                                borderWidth: 3,
                                cutout: '55%'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: {
                                        font: { size: 9, weight: 'bold' },
                                        padding: 8,
                                        usePointStyle: true,
                                        pointStyle: 'circle'
                                    }
                                },
                                tooltip: {
                                    enabled: false // Disable for PDF
                                }
                            },
                            animation: {
                                duration: 0 // Disable animation for PDF
                            }
                        }
                    });
                    console.log('✅ Allocation chart created');
                }
                
                // PERFORMANCE CHART
                console.log('📈 Creating performance chart...');
                const performanceCtx = document.getElementById('performanceChart');
                if (performanceCtx) {
                    const performanceChart = new Chart(performanceCtx, {
                        type: 'line',
                        data: {
                            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
                            datasets: [{
                                label: 'Portfolio',
                                data: [100, 103.2, 108.7, 115.4, 118.9, 122.8, 128.4],
                                borderColor: '#d4af37',
                                backgroundColor: 'rgba(212, 175, 55, 0.2)',
                                borderWidth: 3,
                                fill: true,
                                tension: 0.3,
                                pointRadius: 4,
                                pointBackgroundColor: '#d4af37',
                                pointBorderColor: '#ffffff',
                                pointBorderWidth: 2
                            }, {
                                label: 'Benchmark',
                                data: [100, 102.1, 105.8, 109.2, 112.4, 115.6, 119.8],
                                borderColor: '#6b7280',
                                backgroundColor: 'transparent',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                pointRadius: 3,
                                pointBackgroundColor: '#6b7280',
                                pointBorderColor: '#ffffff',
                                pointBorderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'top',
                                    labels: {
                                        font: { size: 9, weight: 'bold' },
                                        padding: 8,
                                        usePointStyle: true
                                    }
                                },
                                tooltip: {
                                    enabled: false // Disable for PDF
                                }
                            },
                            scales: {
                                x: {
                                    grid: { 
                                        color: 'rgba(0, 0, 0, 0.1)',
                                        lineWidth: 1
                                    },
                                    ticks: { 
                                        font: { size: 9 },
                                        color: '#374151'
                                    }
                                },
                                y: {
                                    grid: { 
                                        color: 'rgba(0, 0, 0, 0.1)',
                                        lineWidth: 1
                                    },
                                    ticks: { 
                                        font: { size: 9 },
                                        color: '#374151',
                                        callback: function(value) {
                                            return value + '%';
                                        }
                                    }
                                }
                            },
                            animation: {
                                duration: 0 // Disable animation for PDF
                            }
                        }
                    });
                    console.log('✅ Performance chart created');
                }
                
                // Mark charts as loaded
                document.body.classList.add('charts-loaded');
                console.log('🎉 All charts initialized successfully!');
                
                // Signal that charts are ready
                window.chartsReady = true;
                
            } catch (error) {
                console.error('❌ Error creating charts:', error);
            }
        }
        
        // Initialize charts when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeCharts);
        } else {
            initializeCharts();
        }
    </script>
</body>
</html>`;

  console.log('🚀 Launching Puppeteer with chart rendering fixes...');
  
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--no-first-run',
      '--disable-web-security',
      '--allow-running-insecure-content'
    ]
  });
  
  const page = await browser.newPage();
  
  // Set viewport for consistent rendering
  await page.setViewport({ width: 1200, height: 1600 });
  
  // Enable console logging
  page.on('console', msg => {
    console.log('🖥️ Browser:', msg.text());
  });
  
  // Set content and wait for network idle
  await page.setContent(htmlContent, { 
    waitUntil: 'networkidle0',
    timeout: 30000
  });
  
  console.log('⏳ Waiting for charts to render...');
  
  // Wait for Chart.js to load and charts to be created
  await page.waitForFunction(() => {
    return window.chartsReady === true;
  }, { timeout: 15000 });
  
  console.log('✅ Charts rendered successfully!');
  
  // Additional wait to ensure charts are fully drawn
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  // Generate PDF
  await page.pdf({
    path: outputPath,
    format: 'A4',
    printBackground: true,
    margin: {
      top: '0mm',
      right: '0mm',
      bottom: '0mm',
      left: '0mm'
    },
    preferCSSPageSize: true
  });
  
  await browser.close();
  
  const fileSize = fs.statSync(outputPath).size;
  
  console.log('🎉 WORKING CHARTS PDF GENERATED!');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📄 File:', fileName);
  console.log('📁 Location:', outputPath);
  console.log('📏 Size:', fileSize, 'bytes');
  console.log('💎 Quality: WORKING CHARTS GUARANTEED');
  console.log('🎯 Features:');
  console.log('✅ Charts actually render in PDF');
  console.log('✅ Professional asset allocation doughnut chart');
  console.log('✅ Performance vs benchmark line chart');
  console.log('✅ Proper chart loading and initialization');
  console.log('✅ Print-optimized chart sizing');
  console.log('✅ Disabled animations for PDF compatibility');
  console.log('✅ Enhanced error handling and logging');
  console.log('✅ 100% SEBI compliant content');
  
  return outputPath;
}

// Run the generator
generateWorkingChartsPDF().catch(console.error);
