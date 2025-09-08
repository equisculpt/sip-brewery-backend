import React from 'react';
import { View, Text } from '@react-pdf/renderer';

const PDFAnalyticsInsight = ({ insights, portfolio }) => {
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
    insightsGrid: {
      flexDirection: 'column'
    },
    insightCard: {
      padding: 15,
      borderRadius: 6,
      marginBottom: 10,
      border: '1px solid #e5e7eb'
    },
    insightCardSuccess: {
      padding: 15,
      borderRadius: 6,
      marginBottom: 10,
      backgroundColor: '#f0fdf4',
      border: '1px solid #22c55e'
    },
    insightCardWarning: {
      padding: 15,
      borderRadius: 6,
      marginBottom: 10,
      backgroundColor: '#fffbeb',
      border: '1px solid #f59e0b'
    },
    insightCardInfo: {
      padding: 15,
      borderRadius: 6,
      marginBottom: 10,
      backgroundColor: '#eff6ff',
      border: '1px solid #3b82f6'
    },
    insightHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: 8
    },
    insightIcon: {
      width: 20,
      height: 20,
      borderRadius: 10,
      marginRight: 10,
      alignItems: 'center',
      justifyContent: 'center'
    },
    insightIconSuccess: {
      width: 20,
      height: 20,
      borderRadius: 10,
      marginRight: 10,
      backgroundColor: '#22c55e',
      alignItems: 'center',
      justifyContent: 'center'
    },
    insightIconWarning: {
      width: 20,
      height: 20,
      borderRadius: 10,
      marginRight: 10,
      backgroundColor: '#f59e0b',
      alignItems: 'center',
      justifyContent: 'center'
    },
    insightIconInfo: {
      width: 20,
      height: 20,
      borderRadius: 10,
      marginRight: 10,
      backgroundColor: '#3b82f6',
      alignItems: 'center',
      justifyContent: 'center'
    },
    insightIconText: {
      fontSize: 10,
      color: '#ffffff',
      fontWeight: 'bold'
    },
    insightTitle: {
      fontSize: 12,
      fontWeight: 'bold',
      color: '#1f2937'
    },
    insightMessage: {
      fontSize: 10,
      color: '#4b5563',
      lineHeight: 1.4,
      marginLeft: 30
    },
    performanceSection: {
      marginTop: 15,
      padding: 15,
      backgroundColor: '#f8fafc',
      borderRadius: 6,
      border: '1px solid #e5e7eb'
    },
    performanceTitle: {
      fontSize: 12,
      fontWeight: 'bold',
      color: '#374151',
      marginBottom: 10
    },
    performanceGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between'
    },
    performanceItem: {
      alignItems: 'center'
    },
    performanceLabel: {
      fontSize: 9,
      color: '#6b7280',
      marginBottom: 3
    },
    performanceValue: {
      fontSize: 11,
      fontWeight: 'bold',
      color: '#1f2937'
    },
    performanceValuePositive: {
      fontSize: 11,
      fontWeight: 'bold',
      color: '#059669'
    },
    performanceValueNegative: {
      fontSize: 11,
      fontWeight: 'bold',
      color: '#dc2626'
    },
    recommendations: {
      marginTop: 15,
      padding: 15,
      backgroundColor: '#fef3c7',
      borderRadius: 6,
      border: '1px solid #f59e0b'
    },
    recommendationsTitle: {
      fontSize: 12,
      fontWeight: 'bold',
      color: '#92400e',
      marginBottom: 10
    },
    recommendationItem: {
      flexDirection: 'row',
      marginBottom: 5
    },
    recommendationBullet: {
      fontSize: 8,
      color: '#92400e',
      marginRight: 8,
      marginTop: 2
    },
    recommendationText: {
      fontSize: 9,
      color: '#92400e',
      flex: 1
    }
  };

  const getInsightStyle = (type) => {
    switch (type) {
      case 'success':
        return styles.insightCardSuccess;
      case 'warning':
        return styles.insightCardWarning;
      case 'info':
        return styles.insightCardInfo;
      default:
        return styles.insightCard;
    }
  };

  const getInsightIconStyle = (type) => {
    switch (type) {
      case 'success':
        return styles.insightIconSuccess;
      case 'warning':
        return styles.insightIconWarning;
      case 'info':
        return styles.insightIconInfo;
      default:
        return styles.insightIcon;
    }
  };

  const getInsightIcon = (type) => {
    switch (type) {
      case 'success':
        return '✓';
      case 'warning':
        return '!';
      case 'info':
        return 'i';
      default:
        return '•';
    }
  };

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  const formatCurrency = (amount) => {
    return `Rs. ${amount.toLocaleString('en-IN')}`;
  };

  const generateRecommendations = () => {
    const recommendations = [];
    
    if (portfolio.xirr1Y < 10) {
      recommendations.push('Consider reviewing fund selection and diversifying into better-performing categories');
    }
    
    if (portfolio.funds.length < 3) {
      recommendations.push('Increase diversification by adding more funds across different categories');
    }
    
    if (portfolio.funds.length > 8) {
      recommendations.push('Consider consolidating similar funds to reduce portfolio complexity');
    }
    
    recommendations.push('Continue SIP discipline and consider increasing amounts during market dips');
    recommendations.push('Review portfolio allocation quarterly and rebalance if needed');
    
    return recommendations;
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Analytics & Insights</Text>
      
      <View style={styles.insightsGrid}>
        {insights.map((insight, index) => (
          <View key={index} style={getInsightStyle(insight.type)}>
            <View style={styles.insightHeader}>
              <View style={getInsightIconStyle(insight.type)}>
                <Text style={styles.insightIconText}>{getInsightIcon(insight.type)}</Text>
              </View>
              <Text style={styles.insightTitle}>{insight.title}</Text>
            </View>
            <Text style={styles.insightMessage}>{insight.message}</Text>
          </View>
        ))}
      </View>

      <View style={styles.performanceSection}>
        <Text style={styles.performanceTitle}>Performance Metrics</Text>
        <View style={styles.performanceGrid}>
          <View style={styles.performanceItem}>
            <Text style={styles.performanceLabel}>1Y XIRR</Text>
            <Text style={portfolio.xirr1Y >= 0 ? styles.performanceValuePositive : styles.performanceValueNegative}>
              {formatPercentage(portfolio.xirr1Y || 0)}
            </Text>
          </View>
          <View style={styles.performanceItem}>
            <Text style={styles.performanceLabel}>Total Gain</Text>
            <Text style={portfolio.percentageGain >= 0 ? styles.performanceValuePositive : styles.performanceValueNegative}>
              {formatPercentage(portfolio.percentageGain)}
            </Text>
          </View>
          <View style={styles.performanceItem}>
            <Text style={styles.performanceLabel}>Absolute Gain</Text>
            <Text style={portfolio.absoluteGain >= 0 ? styles.performanceValuePositive : styles.performanceValueNegative}>
              {formatCurrency(portfolio.absoluteGain)}
            </Text>
          </View>
          <View style={styles.performanceItem}>
            <Text style={styles.performanceLabel}>Fund Count</Text>
            <Text style={styles.performanceValue}>{portfolio.funds.length}</Text>
          </View>
        </View>
      </View>

      <View style={styles.recommendations}>
        <Text style={styles.recommendationsTitle}>Investment Recommendations</Text>
        {generateRecommendations().map((recommendation, index) => (
          <View key={index} style={styles.recommendationItem}>
            <Text style={styles.recommendationBullet}>•</Text>
            <Text style={styles.recommendationText}>{recommendation}</Text>
          </View>
        ))}
      </View>
    </View>
  );
};

export default PDFAnalyticsInsight; 