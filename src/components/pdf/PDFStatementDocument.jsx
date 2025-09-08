import React from 'react';
import { Document, Page, Text, View, StyleSheet, Font } from '@react-pdf/renderer';
import PDFHeader from './PDFHeader';
import PDFUserInfo from './PDFUserInfo';
import PDFPortfolioSummary from './PDFPortfolioSummary';
import PDFHoldingsTable from './PDFHoldingsTable';
import PDFAnalyticsInsight from './PDFAnalyticsInsight';
import PDFChart from './PDFChart';
import PDFFooter from './PDFFooter';

// Register fonts
Font.register({
  family: 'Helvetica',
  fonts: [
    { src: 'https://fonts.gstatic.com/s/helveticaneue/v70/1Ptsg8zYS_SKggPNyC0IT4ttDfA.ttf', fontWeight: 'normal' },
    { src: 'https://fonts.gstatic.com/s/helveticaneue/v70/1Ptsg8zYS_SKggPNyC0IT4ttDfB.ttf', fontWeight: 'bold' }
  ]
});

const styles = StyleSheet.create({
  page: {
    flexDirection: 'column',
    backgroundColor: '#ffffff',
    padding: 20,
    fontFamily: 'Helvetica'
  },
  pageContent: {
    flex: 1,
    marginBottom: 100 // Space for footer
  },
  section: {
    marginBottom: 20
  },
  pageBreak: {
    break: 'page'
  }
});

const PDFStatementDocument = ({ 
  statementType = 'comprehensive',
  user,
  portfolio,
  transactions = [],
  capitalGains,
  rewards = [],
  aiInsights = [],
  charts = {},
  metadata
}) => {
  
  const renderStatementContent = () => {
    switch (statementType) {
      case 'comprehensive':
        return (
          <>
            {/* Page 1: Overview */}
            <View style={styles.pageContent}>
              <PDFHeader metadata={metadata} statementType={statementType} />
              <PDFUserInfo user={user} />
              <PDFPortfolioSummary portfolio={portfolio} />
              <PDFAnalyticsInsight insights={aiInsights} portfolio={portfolio} />
              <PDFChart 
                chartImage={charts.performance}
                title="Portfolio Performance Trend"
                subtitle="6-Month Performance vs Nifty 50"
                type="performance"
              />
            </View>

            {/* Page 2: Holdings */}
            <View style={[styles.pageContent, styles.pageBreak]}>
              <PDFHeader metadata={metadata} statementType={statementType} />
              <PDFHoldingsTable funds={portfolio.funds} allocation={portfolio.allocation} />
              <PDFChart 
                chartImage={charts.allocation}
                title="Portfolio Allocation"
                subtitle="Current Fund Distribution"
                type="allocation"
              />
            </View>

            {/* Page 3: Transactions */}
            {transactions.length > 0 && (
              <View style={[styles.pageContent, styles.pageBreak]}>
                <PDFHeader metadata={metadata} statementType={statementType} />
                <View style={styles.section}>
                  <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                    Recent Transactions
                  </Text>
                  {/* Transaction table would go here */}
                  <Text style={{ fontSize: 10, color: '#6b7280', textAlign: 'center', marginTop: 20 }}>
                    Full transaction history available at www.sipbrewery.com
                  </Text>
                </View>
              </View>
            )}

            {/* Page 4: Capital Gains */}
            {capitalGains && (
              <View style={[styles.pageContent, styles.pageBreak]}>
                <PDFHeader metadata={metadata} statementType={statementType} />
                <View style={styles.section}>
                  <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                    Capital Gains Summary
                  </Text>
                  {/* Capital gains table would go here */}
                </View>
              </View>
            )}

            {/* Page 5: XIRR Analysis */}
            <View style={[styles.pageContent, styles.pageBreak]}>
              <PDFHeader metadata={metadata} statementType={statementType} />
              <PDFChart 
                chartImage={charts.xirr}
                title="XIRR vs Benchmark Returns"
                subtitle="Performance Comparison Across Time Periods"
                type="xirr"
              />
            </View>
          </>
        );

      case 'holdings':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <PDFHoldingsTable funds={portfolio.funds} allocation={portfolio.allocation} />
            <PDFChart 
              chartImage={charts.allocation}
              title="Portfolio Allocation"
              subtitle="Current Fund Distribution"
              type="allocation"
            />
          </View>
        );

      case 'transactions':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <View style={styles.section}>
              <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                Transaction Report
              </Text>
              {/* Transaction table would go here */}
            </View>
          </View>
        );

      case 'pnl':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <PDFPortfolioSummary portfolio={portfolio} />
            <PDFAnalyticsInsight insights={aiInsights} portfolio={portfolio} />
            <PDFChart 
              chartImage={charts.performance}
              title="Profit & Loss Analysis"
              subtitle="Portfolio Performance Over Time"
              type="performance"
            />
          </View>
        );

      case 'capital-gain':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <View style={styles.section}>
              <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                Capital Gains Statement
              </Text>
              {/* Capital gains details would go here */}
            </View>
          </View>
        );

      case 'tax':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <View style={styles.section}>
              <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                Tax Statement for CA/ITR Filing
              </Text>
              {/* Tax details would go here */}
            </View>
          </View>
        );

      case 'rewards':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <View style={styles.section}>
              <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                Rewards & Referral Summary
              </Text>
              {/* Rewards details would go here */}
            </View>
          </View>
        );

      case 'smart-sip':
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <View style={styles.section}>
              <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 15 }}>
                Smart SIP Summary
              </Text>
              {/* Smart SIP details would go here */}
            </View>
          </View>
        );

      default:
        return (
          <View style={styles.pageContent}>
            <PDFHeader metadata={metadata} statementType={statementType} />
            <PDFUserInfo user={user} />
            <PDFPortfolioSummary portfolio={portfolio} />
          </View>
        );
    }
  };

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {renderStatementContent()}
        <PDFFooter metadata={metadata} />
      </Page>
    </Document>
  );
};

export default PDFStatementDocument; 