import React from 'react';
import { View, Text, Image } from '@react-pdf/renderer';

const PDFChart = ({ chartImage, title, subtitle, type = 'performance' }) => {
  const styles = {
    container: {
      marginBottom: 20,
      padding: 15,
      backgroundColor: '#ffffff',
      border: '1px solid #e5e7eb',
      borderRadius: 8
    },
    title: {
      fontSize: 14,
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: 5,
      textAlign: 'center'
    },
    subtitle: {
      fontSize: 10,
      color: '#6b7280',
      marginBottom: 15,
      textAlign: 'center'
    },
    chartContainer: {
      alignItems: 'center',
      marginBottom: 10
    },
    chartImage: {
      width: 500,
      height: 300,
      objectFit: 'contain'
    },
    chartPlaceholder: {
      width: 500,
      height: 300,
      backgroundColor: '#f3f4f6',
      border: '2px dashed #d1d5db',
      borderRadius: 6,
      alignItems: 'center',
      justifyContent: 'center'
    },
    placeholderText: {
      fontSize: 12,
      color: '#9ca3af',
      textAlign: 'center'
    },
    legend: {
      flexDirection: 'row',
      justifyContent: 'center',
      flexWrap: 'wrap',
      marginTop: 10
    },
    legendItem: {
      flexDirection: 'row',
      alignItems: 'center',
      marginRight: 20,
      marginBottom: 5
    },
    legendColor: {
      width: 12,
      height: 12,
      borderRadius: 2,
      marginRight: 5
    },
    legendText: {
      fontSize: 8,
      color: '#6b7280'
    },
    performanceLegend: {
      primary: '#2563eb',
      secondary: '#7c3aed'
    },
    allocationLegend: {
      primary: '#2563eb',
      secondary: '#7c3aed',
      success: '#059669',
      warning: '#d97706',
      danger: '#dc2626',
      info: '#0891b2'
    },
    xirrLegend: {
      primary: '#2563eb',
      secondary: '#7c3aed'
    }
  };

  const getLegendColors = () => {
    switch (type) {
      case 'performance':
        return [
          { color: styles.performanceLegend.primary, label: 'Portfolio Value' },
          { color: styles.performanceLegend.secondary, label: 'Nifty 50' }
        ];
      case 'allocation':
        return [
          { color: styles.allocationLegend.primary, label: 'Large Cap' },
          { color: styles.allocationLegend.secondary, label: 'Mid Cap' },
          { color: styles.allocationLegend.success, label: 'Small Cap' },
          { color: styles.allocationLegend.warning, label: 'Debt' },
          { color: styles.allocationLegend.danger, label: 'International' },
          { color: styles.allocationLegend.info, label: 'Others' }
        ];
      case 'xirr':
        return [
          { color: styles.xirrLegend.primary, label: 'Portfolio XIRR' },
          { color: styles.xirrLegend.secondary, label: 'Nifty 50 Returns' }
        ];
      default:
        return [];
    }
  };

  const renderChart = () => {
    if (chartImage) {
      return (
        <Image 
          src={chartImage} 
          style={styles.chartImage}
        />
      );
    }

    return (
      <View style={styles.chartPlaceholder}>
        <Text style={styles.placeholderText}>
          Chart data will be generated here
        </Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
      
      <View style={styles.chartContainer}>
        {renderChart()}
      </View>

      <View style={styles.legend}>
        {getLegendColors().map((item, index) => (
          <View key={index} style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: item.color }]} />
            <Text style={styles.legendText}>{item.label}</Text>
          </View>
        ))}
      </View>
    </View>
  );
};

export default PDFChart; 