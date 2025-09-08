const mongoose = require('mongoose');
const BenchmarkIndex = require('./src/models/BenchmarkIndex');

const uri = 'mongodb+srv://admin:jgnDnev2mHVToKnJ@cluster0.1qcxe6p.mongodb.net/sip_brewery_dev?retryWrites=true&w=majority&appName=Cluster0';

async function main() {
  try {
    await mongoose.connect(uri, { useNewUrlParser: true, useUnifiedTopology: true });
    console.log('Connected to MongoDB');
    const doc = await BenchmarkIndex.findOne({ indexId: 'NIFTY50' });
    if (!doc) {
      console.log('No NIFTY50 data found.');
    } else {
      const latest = doc.data.slice(-5);
      console.log('Latest 5 NIFTY50 records:', latest);
    }
  } catch (err) {
    console.error('Error:', err);
  } finally {
    await mongoose.disconnect();
  }
}

main(); 