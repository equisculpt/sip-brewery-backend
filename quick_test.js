const ComprehensiveReportSuite = require('./COMPLETE_REPORT_SUITE');

async function quickTest() {
    try {
        console.log('🚀 QUICK TEST - TIMEOUT FIX VERIFICATION');
        console.log('========================================\n');
        
        console.log('📋 Step 1: Creating report suite instance...');
        const reportSuite = new ComprehensiveReportSuite();
        console.log('✅ Report suite created successfully');
        
        console.log('📋 Step 2: Preparing test data...');
    
    const clientData = {
        name: 'Test User',
        folio: 'TEST001',
        totalInvested: 100000,
        currentValue: 115000,
        absoluteReturn: 15.0,
        xirr: 18.5,
        holdings: [
            { fundName: 'Test Fund', amc: 'Test AMC', nav: 100, units: 1000, invested: 100000, currentValue: 115000, returnPct: 15.0 }
        ]
    };
    console.log('✅ Test data prepared');

    console.log('📊 Step 3: Testing Client Statement generation...');
    const startTime = Date.now();
    
    await reportSuite.generateClientStatement(clientData);
    
    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;
    
    console.log(`✅ SUCCESS! Report generated in ${duration.toFixed(2)} seconds`);
    console.log('🎉 Timeout fix working - no hanging issues!');
    
    } catch (error) {
        console.error('❌ Error occurred:', error.message);
        console.error('Stack trace:', error.stack);
        if (error.message.includes('timeout')) {
            console.log('⚠️  Still having timeout issues - may need further optimization');
        }
        process.exit(1);
    }
}

quickTest().catch(console.error);
