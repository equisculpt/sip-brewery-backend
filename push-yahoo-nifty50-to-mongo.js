const fs = require('fs');
const path = require('path');
const mongoose = require('mongoose');
const BenchmarkIndex = require('./src/models/BenchmarkIndex');
const { parse } = require('csv-parse/sync');
const dayjs = require('dayjs');

const uri = 'mongodb+srv://admin:jgnDnev2mHVToKnJ@cluster0.1qcxe6p.mongodb.net/sip_brewery_dev?retryWrites=true&w=majority&appName=Cluster0';
const csvFile = path.resolve(__dirname, 'downloads', 'NIFTY50-Yahoo.csv');

async function main() {
  try {
    if (!fs.existsSync(csvFile)) {
      console.error('CSV file not found:', csvFile);
      process.exit(1);
    }
    const csvContent = fs.readFileSync(csvFile, 'utf8');
    const records = parse(csvContent, { columns: true, skip_empty_lines: true });
    // Yahoo CSV: Date, Open, High, Low, Close*, Adj Close*, Volume
    const data = records
      .filter(row => row['Close*'] && row['Date'] && !isNaN(parseFloat(row['Close*'])))
      .map(row => ({
        date: dayjs(row['Date']).format('YYYY-MM-DD'),
        close: parseFloat(row['Close*'])
      }));
    console.log(`Parsed ${data.length} records from Yahoo CSV.`);

    await mongoose.connect(uri, { useNewUrlParser: true, useUnifiedTopology: true });
    console.log('Connected to MongoDB');
    const result = await BenchmarkIndex.findOneAndUpdate(
      { indexId: 'NIFTY50' },
      {
        indexId: 'NIFTY50',
        name: 'NIFTY 50',
        data,
        lastUpdated: new Date()
      },
      { upsert: true, new: true }
    );
    console.log(`Upserted NIFTY 50 data. MongoDB _id: ${result && result._id}`);
    await mongoose.disconnect();
    console.log('Done.');
  } catch (err) {
    console.error('Error:', err);
    process.exit(1);
  }
}

main(); 