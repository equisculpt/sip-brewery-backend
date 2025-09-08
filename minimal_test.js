// Minimal test to identify the exact hanging point
console.log('🔍 MINIMAL PUPPETEER TEST');
console.log('========================');

async function minimalTest() {
    try {
        console.log('Step 1: Loading puppeteer...');
        const puppeteer = require('puppeteer');
        console.log('✅ Puppeteer loaded');

        console.log('Step 2: Launching browser...');
        const browser = await puppeteer.launch({ 
            headless: 'new',
            args: [
                '--no-sandbox', 
                '--disable-setuid-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ],
            timeout: 10000
        });
        console.log('✅ Browser launched');

        console.log('Step 3: Creating new page...');
        const page = await browser.newPage();
        console.log('✅ Page created');

        console.log('Step 4: Setting simple HTML content...');
        await page.setContent('<html><body><h1>Test</h1></body></html>', { 
            waitUntil: 'domcontentloaded', 
            timeout: 5000 
        });
        console.log('✅ Content set');

        console.log('Step 5: Closing browser...');
        await browser.close();
        console.log('✅ Browser closed');

        console.log('🎉 SUCCESS: Puppeteer is working correctly!');
        
    } catch (error) {
        console.error('❌ ERROR at step:', error.message);
        console.error('Stack:', error.stack);
        process.exit(1);
    }
}

// Set a global timeout to prevent infinite hanging
setTimeout(() => {
    console.error('⏰ TIMEOUT: Test took too long, likely hanging at browser launch');
    process.exit(1);
}, 15000);

minimalTest();
