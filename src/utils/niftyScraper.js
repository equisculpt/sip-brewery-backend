const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

async function fetchNifty50HistoricalData() {
  const downloadPath = path.resolve(__dirname, '../../downloads');
  if (!fs.existsSync(downloadPath)) fs.mkdirSync(downloadPath);

  console.log('[Scraper] Launching browser...');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1920,1080',
      '--proxy-server="direct://"',
      '--proxy-bypass-list=*'
    ]
  });
  const page = await browser.newPage();

  try {
    console.log('[Scraper] Navigating to NIFTY indices page...');
    await page.goto('https://www.niftyindices.com/reports/historical-data', { waitUntil: 'networkidle2', timeout: 60000 });
    console.log('[Scraper] Page loaded. Selecting NIFTY 50...');
    await page.select('#indexName', 'NIFTY 50');
    console.log('[Scraper] Clicking submit...');
    await page.click('#submitIndex');
    await page.waitForSelector('a#downloadIndex', { timeout: 15000 });
    console.log('[Scraper] Download link appeared. Clicking download...');
    await page.click('a#downloadIndex');

    // Wait for file to appear in downloadPath
    let csvFile;
    for (let i = 0; i < 20; i++) {
      const files = fs.readdirSync(downloadPath).filter(f => f.endsWith('.csv'));
      if (files.length > 0) {
        csvFile = path.join(downloadPath, files[0]);
        break;
      }
      console.log(`[Scraper] Waiting for CSV file... (${i + 1}/20)`);
      await new Promise(r => setTimeout(r, 1000));
    }
    if (!csvFile) throw new Error('CSV download failed');
    console.log(`[Scraper] CSV file found: ${csvFile}`);

    // Parse CSV
    const csvContent = fs.readFileSync(csvFile, 'utf8');
    const records = parse(csvContent, { columns: true, skip_empty_lines: true });
    // Clean up
    fs.unlinkSync(csvFile);
    console.log(`[Scraper] Parsed ${records.length} records from CSV.`);

    // Extract date and close price
    const data = records.map(row => ({
      date: row['Index Date'] || row['Date'],
      close: parseFloat(row['Closing Index Value'] || row['Close'])
    })).filter(d => d.date && !isNaN(d.close));
    console.log(`[Scraper] Returning ${data.length} cleaned records.`);

    await browser.close();
    return data;
  } catch (err) {
    await browser.close();
    console.error('[Scraper] Error:', err);
    throw err;
  }
}

module.exports = { fetchNifty50HistoricalData }; 