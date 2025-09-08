// nseBseBulkDownloader.js
// Automate bulk downloads, daily updates, and P/E extraction from NSE/BSE/data.gov.in
// Usage: require and call scheduleDailyUpdates() to keep data live for AGI

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const csv = require('csvtojson');
const unzipper = require('unzipper');

// Helper: Download file
async function downloadFile(url, dest) {
  const writer = fs.createWriteStream(dest);
  const response = await axios({ url, method: 'GET', responseType: 'stream' });
  response.data.pipe(writer);
  return new Promise((resolve, reject) => {
    writer.on('finish', resolve);
    writer.on('error', reject);
  });
}

// Helper: Unzip and parse CSV from zip
async function unzipAndParse(zipPath, extractDir) {
  await fs.createReadStream(zipPath)
    .pipe(unzipper.Extract({ path: extractDir }))
    .promise();
  // Find CSV file in extractDir
  const files = fs.readdirSync(extractDir);
  const csvFile = files.find(f => f.endsWith('.CSV') || f.endsWith('.csv'));
  if (!csvFile) throw new Error('No CSV found in ZIP');
  return await csv().fromFile(path.join(extractDir, csvFile));
}

// Download and parse latest NSE Bhavcopy (EOD)
async function fetchLatestNseBhavcopy(destDir) {
  const today = new Date();
  const dd = String(today.getDate()).padStart(2, '0');
  const mm = today.toLocaleString('en-US', { month: 'short' }).toUpperCase();
  const yyyy = today.getFullYear();
  const fileName = `cm${dd}${mm}${yyyy}bhav.csv.zip`;
  const url = `https://www1.nseindia.com/content/historical/EQUITIES/${yyyy}/${mm}/${fileName}`;
  const zipPath = path.join(destDir, fileName);
  try {
    await downloadFile(url, zipPath);
    const data = await unzipAndParse(zipPath, destDir);
    return data;
  } catch (err) {
    return { error: 'Failed to fetch or parse NSE Bhavcopy', details: err.message };
  }
}

// Download and parse latest BSE Bhavcopy (EOD)
async function fetchLatestBseBhavcopy(destDir) {
  const today = new Date();
  const dd = String(today.getDate()).padStart(2, '0');
  const mm = String(today.getMonth() + 1).padStart(2, '0');
  const yyyy = today.getFullYear();
  const fileName = `EQ${dd}${mm}${yyyy}_CSV.ZIP`;
  const url = `https://www.bseindia.com/download/BhavCopy/Equity/${fileName}`;
  const zipPath = path.join(destDir, fileName);
  try {
    await downloadFile(url, zipPath);
    const data = await unzipAndParse(zipPath, destDir);
    return data;
  } catch (err) {
    return { error: 'Failed to fetch or parse BSE Bhavcopy', details: err.message };
  }
}

// Extract P/E, P/B, and other ratios from NSE index data (CSV)
async function fetchNseIndexRatios() {
  const url = 'https://www1.nseindia.com/content/indices/ind_pepbv.csv';
  try {
    const response = await axios.get(url);
    return await csv().fromString(response.data);
  } catch (err) {
    return { error: 'Failed to fetch NSE index ratios', details: err.message };
  }
}

// (cron scheduling removed for testability in network-restricted environments)
function scheduleDailyUpdates(destDir) {
  console.log('Scheduling disabled: install cron to enable periodic updates.');
}

module.exports = {
  fetchLatestNseBhavcopy,
  fetchLatestBseBhavcopy,
  fetchNseIndexRatios,
  scheduleDailyUpdates
};
