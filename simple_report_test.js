const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

console.log('📊 SIMPLE REPORT GENERATION TEST');
console.log('================================');

async function simpleReportTest() {
    try {
        console.log('Step 1: Creating output directory...');
        await fs.mkdir('./test_reports', { recursive: true });
        console.log('✅ Directory created');

        console.log('Step 2: Preparing simple HTML...');
        const html = `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Simple Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .header { background: #4caf50; color: white; padding: 20px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎉 SIP Brewery Test Report</h1>
                <p>Generated at: ${new Date().toLocaleString()}</p>
            </div>
            <div style="padding: 20px;">
                <h2>Portfolio Summary</h2>
                <p>Total Investment: ₹1,00,000</p>
                <p>Current Value: ₹1,15,000</p>
                <p>Returns: 15%</p>
            </div>
        </body>
        </html>
        `;
        console.log('✅ HTML prepared');

        console.log('Step 3: Launching browser...');
        const browser = await puppeteer.launch({ 
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
            timeout: 10000
        });
        console.log('✅ Browser launched');

        console.log('Step 4: Creating page and setting content...');
        const page = await browser.newPage();
        await page.setContent(html, { waitUntil: 'domcontentloaded', timeout: 10000 });
        console.log('✅ Content set');

        console.log('Step 5: Generating PDF...');
        const filepath = path.join('./test_reports', `simple_test_${Date.now()}.pdf`);
        await page.pdf({
            path: filepath,
            format: 'A4',
            printBackground: true,
            margin: { top: '20px', bottom: '20px', left: '20px', right: '20px' }
        });
        console.log('✅ PDF generated');

        console.log('Step 6: Closing browser...');
        await browser.close();
        console.log('✅ Browser closed');

        console.log('Step 7: Checking file...');
        const stats = await fs.stat(filepath);
        console.log(`✅ File created: ${filepath} (${stats.size} bytes)`);

        console.log('🎉 SUCCESS: Simple report generation working!');
        console.log(`📄 Report saved at: ${filepath}`);
        
    } catch (error) {
        console.error('❌ ERROR:', error.message);
        console.error('Stack:', error.stack);
        process.exit(1);
    }
}

// Set timeout to prevent hanging
setTimeout(() => {
    console.error('⏰ TIMEOUT: Report generation took too long');
    process.exit(1);
}, 30000);

simpleReportTest();
