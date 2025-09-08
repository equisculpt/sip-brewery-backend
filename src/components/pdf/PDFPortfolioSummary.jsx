import React from 'react';
import { View, Text } from '@react-pdf/renderer';

const PDFPortfolioSummary = ({ portfolio }) => {
  const styles = {
    container: {
      padding: 20,
      backgroundColor: '#ffffff',
      border: '1px solid #e5e7eb',
      borderRadius: 8,
      marginBottom: 20
    },
    title: {
      fontSize: 16,
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: 15,
      borderBottom: '2px solid #2563eb',
      paddingBottom: 5
    },
    metricsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      justifyContent: 'space-between'
    },
    metricCard: {
      width: '48%',
      backgroundColor: '#f8fafc',
      padding: 15,
      borderRadius: 6,
      marginBottom: 10,
      border: '1px solid #e5e7eb'
    },
    metricCardHighlight: {
      width: '48%',
      backgroundColor: '#eff6ff',
      padding: 15,
      borderRadius: 6,
      marginBottom: 10,
      border: '2px solid #2563eb'
    },
    metricLabel: {
      fontSize: 10,
      color: '#6b7280',
      marginBottom: 5,
      fontWeight: 'bold'
    },
    metricValue: {
      fontSize: 18,
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: 3
    },
    metricValuePositive: {
      fontSize: 18,
      fontWeight: 'bold',
      color: '#059669',
      marginBottom: 3
    },
    metricValueNegative: {
      fontSize: 18,
      fontWeight: 'bold',
      color: '#dc2626',
      marginBottom: 3
    },
    metricChange: {
      fontSize: 9,
      color: '#6b7280'
    },
    metricChangePositive: {
      fontSize: 9,
      color: '#059669'
    },
    metricChangeNegative: {
      fontSize: 9,
      color: '#dc2626'
    },
    xirrSection: {
      marginTop: 15,
      padding: 15,
      backgroundColor: '#fef3c7',
      borderRadius: 6,
      border: '1px solid #f59e0b'
    },
    xirrTitle: {
      fontSize: 12,
      fontWeight: 'bold',
      color: '#92400e',
      marginBottom: 10
    },
    xirrGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between'
    },
    xirrItem: {
      alignItems: 'center'
    },
    xirrPeriod: {
      fontSize: 9,
      color: '#92400e',
      marginBottom: 3
    },
    xirrValue: {
      fontSize: 14,
      fontWeight: 'bold',
      color: '#92400e'
    }
  };

  const formatCurrency = (amount) => {
    return `Rs. ${amount.toLocaleString('en-IN')}`;
  };

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  const isPositive = portfolio.percentageGain >= 0;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Portfolio Summary</Text>
      
      <View style={styles.metricsGrid}>
        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>Total Invested</Text>
          <Text style={styles.metricValue}>
            {formatCurrency(portfolio.totalInvested)}
          </Text>
          <Text style={styles.metricChange}>Principal Amount</Text>
        </View>

        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>Current Value</Text>
          <Text style={styles.metricValue}>
            {formatCurrency(portfolio.totalCurrentValue)}
          </Text>
          <Text style={styles.metricChange}>Market Value</Text>
        </View>

        <View style={isPositive ? styles.metricCardHighlight : styles.metricCard}>
          <Text style={styles.metricLabel}>Absolute Gain/Loss</Text>
          <Text style={isPositive ? styles.metricValuePositive : styles.metricValueNegative}>
            {formatCurrency(portfolio.absoluteGain)}
          </Text>
          <Text style={isPositive ? styles.metricChangePositive : styles.metricChangeNegative}>
            {formatPercentage(portfolio.percentageGain)}
          </Text>
        </View>

        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>Number of Funds</Text>
          <Text style={styles.metricValue}>
            {portfolio.funds.length}
          </Text>
          <Text style={styles.metricChange}>Active Holdings</Text>
        </View>
      </View>

      <View style={styles.xirrSection}>
        <Text style={styles.xirrTitle}>XIRR Performance</Text>
        <View style={styles.xirrGrid}>
          <View style={styles.xirrItem}>
            <Text style={styles.xirrPeriod}>1 Month</Text>
            <Text style={styles.xirrValue}>{formatPercentage(portfolio.xirr1M || 0)}</Text>
          </View>
          <View style={styles.xirrItem}>
            <Text style={styles.xirrPeriod}>3 Months</Text>
            <Text style={styles.xirrValue}>{formatPercentage(portfolio.xirr3M || 0)}</Text>
          </View>
          <View style={styles.xirrItem}>
            <Text style={styles.xirrPeriod}>6 Months</Text>
            <Text style={styles.xirrValue}>{formatPercentage(portfolio.xirr6M || 0)}</Text>
          </View>
          <View style={styles.xirrItem}>
            <Text style={styles.xirrPeriod}>1 Year</Text>
            <Text style={styles.xirrValue}>{formatPercentage(portfolio.xirr1Y || 0)}</Text>
          </View>
        </View>
      </View>
    </View>
  );
};

export default PDFPortfolioSummary; 