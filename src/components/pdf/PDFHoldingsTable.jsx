import React from 'react';
import { View, Text } from '@react-pdf/renderer';

const PDFHoldingsTable = ({ funds, allocation }) => {
  const styles = {
    container: {
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
    table: {
      border: '1px solid #e5e7eb',
      borderRadius: 6,
      overflow: 'hidden'
    },
    tableHeader: {
      backgroundColor: '#f8fafc',
      flexDirection: 'row',
      borderBottom: '1px solid #e5e7eb'
    },
    tableRow: {
      flexDirection: 'row',
      borderBottom: '1px solid #e5e7eb'
    },
    tableRowAlt: {
      flexDirection: 'row',
      borderBottom: '1px solid #e5e7eb',
      backgroundColor: '#f9fafb'
    },
    headerCell: {
      padding: 10,
      fontSize: 9,
      fontWeight: 'bold',
      color: '#374151',
      textAlign: 'center'
    },
    cell: {
      padding: 10,
      fontSize: 8,
      color: '#1f2937',
      textAlign: 'center'
    },
    cellLeft: {
      padding: 10,
      fontSize: 8,
      color: '#1f2937',
      textAlign: 'left'
    },
    cellRight: {
      padding: 10,
      fontSize: 8,
      color: '#1f2937',
      textAlign: 'right'
    },
    fundName: {
      width: '25%',
      borderRight: '1px solid #e5e7eb'
    },
    units: {
      width: '15%',
      borderRight: '1px solid #e5e7eb'
    },
    nav: {
      width: '15%',
      borderRight: '1px solid #e5e7eb'
    },
    invested: {
      width: '15%',
      borderRight: '1px solid #e5e7eb'
    },
    current: {
      width: '15%',
      borderRight: '1px solid #e5e7eb'
    },
    returns: {
      width: '15%'
    },
    positive: {
      color: '#059669',
      fontWeight: 'bold'
    },
    negative: {
      color: '#dc2626',
      fontWeight: 'bold'
    },
    allocationSection: {
      marginTop: 15,
      padding: 15,
      backgroundColor: '#f8fafc',
      borderRadius: 6,
      border: '1px solid #e5e7eb'
    },
    allocationTitle: {
      fontSize: 12,
      fontWeight: 'bold',
      color: '#374151',
      marginBottom: 10
    },
    allocationGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap'
    },
    allocationItem: {
      width: '33%',
      marginBottom: 8
    },
    allocationLabel: {
      fontSize: 9,
      color: '#6b7280',
      marginBottom: 2
    },
    allocationValue: {
      fontSize: 10,
      fontWeight: 'bold',
      color: '#1f2937'
    },
    summary: {
      marginTop: 15,
      padding: 10,
      backgroundColor: '#eff6ff',
      borderRadius: 6,
      border: '1px solid #2563eb'
    },
    summaryText: {
      fontSize: 9,
      color: '#1e40af',
      textAlign: 'center'
    }
  };

  const formatCurrency = (amount) => {
    return `Rs. ${amount.toLocaleString('en-IN')}`;
  };

  const formatNumber = (num) => {
    return num.toLocaleString('en-IN', { maximumFractionDigits: 2 });
  };

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  const calculateReturn = (fund) => {
    if (fund.investedValue <= 0) return 0;
    return ((fund.currentValue - fund.investedValue) / fund.investedValue) * 100;
  };

  const getTopHoldings = () => {
    return Object.entries(allocation)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Portfolio Holdings</Text>
      
      <View style={styles.table}>
        <View style={styles.tableHeader}>
          <Text style={[styles.headerCell, styles.fundName]}>Fund Name</Text>
          <Text style={[styles.headerCell, styles.units]}>Units</Text>
          <Text style={[styles.headerCell, styles.nav]}>NAV</Text>
          <Text style={[styles.headerCell, styles.invested]}>Invested</Text>
          <Text style={[styles.headerCell, styles.current]}>Current</Text>
          <Text style={[styles.headerCell, styles.returns]}>Returns</Text>
        </View>

        {funds.map((fund, index) => {
          const returnPercent = calculateReturn(fund);
          const isPositive = returnPercent >= 0;
          
          return (
            <View key={fund.schemeCode} style={index % 2 === 0 ? styles.tableRow : styles.tableRowAlt}>
              <Text style={[styles.cellLeft, styles.fundName]}>{fund.schemeName}</Text>
              <Text style={[styles.cellRight, styles.units]}>{formatNumber(fund.units)}</Text>
              <Text style={[styles.cellRight, styles.nav]}>{formatCurrency(fund.lastNav)}</Text>
              <Text style={[styles.cellRight, styles.invested]}>{formatCurrency(fund.investedValue)}</Text>
              <Text style={[styles.cellRight, styles.current]}>{formatCurrency(fund.currentValue)}</Text>
              <Text style={[
                styles.cellRight, 
                styles.returns, 
                isPositive ? styles.positive : styles.negative
              ]}>
                {formatPercentage(returnPercent)}
              </Text>
            </View>
          );
        })}
      </View>

      <View style={styles.allocationSection}>
        <Text style={styles.allocationTitle}>Portfolio Allocation</Text>
        <View style={styles.allocationGrid}>
          {getTopHoldings().map(([fundName, percentage]) => (
            <View key={fundName} style={styles.allocationItem}>
              <Text style={styles.allocationLabel}>{fundName}</Text>
              <Text style={styles.allocationValue}>{formatPercentage(percentage)}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.summary}>
        <Text style={styles.summaryText}>
          Total Holdings: {funds.length} funds | 
          Total Value: {formatCurrency(funds.reduce((sum, f) => sum + f.currentValue, 0))} |
          Average Return: {formatPercentage(funds.reduce((sum, f) => sum + calculateReturn(f), 0) / funds.length)}
        </Text>
      </View>
    </View>
  );
};

export default PDFHoldingsTable; 