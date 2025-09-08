import React from 'react';
import { View, Text, Image } from '@react-pdf/renderer';

const PDFHeader = ({ metadata, statementType }) => {
  const styles = {
    header: {
      padding: 20,
      borderBottom: '1px solid #e5e7eb',
      marginBottom: 20
    },
    topRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 15
    },
    logoSection: {
      flexDirection: 'row',
      alignItems: 'center'
    },
    logo: {
      width: 40,
      height: 40,
      marginRight: 10
    },
    companyInfo: {
      flexDirection: 'column'
    },
    companyName: {
      fontSize: 18,
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: 2
    },
    companyTagline: {
      fontSize: 10,
      color: '#6b7280'
    },
    statementInfo: {
      alignItems: 'flex-end'
    },
    statementTitle: {
      fontSize: 16,
      fontWeight: 'bold',
      color: '#2563eb',
      marginBottom: 5
    },
    statementSubtitle: {
      fontSize: 10,
      color: '#6b7280',
      marginBottom: 3
    },
    statementDate: {
      fontSize: 9,
      color: '#9ca3af'
    },
    bottomRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    metadata: {
      flexDirection: 'column'
    },
    metadataRow: {
      flexDirection: 'row',
      marginBottom: 2
    },
    metadataLabel: {
      fontSize: 8,
      color: '#6b7280',
      width: 80
    },
    metadataValue: {
      fontSize: 8,
      color: '#1f2937',
      fontWeight: 'bold'
    },
    compliance: {
      alignItems: 'flex-end'
    },
    complianceText: {
      fontSize: 7,
      color: '#9ca3af',
      textAlign: 'right',
      marginBottom: 2
    }
  };

  return (
    <View style={styles.header}>
      <View style={styles.topRow}>
        <View style={styles.logoSection}>
          {/* Logo placeholder - replace with actual logo */}
          <View style={styles.logo}>
            <Text style={{ fontSize: 20, textAlign: 'center', color: '#2563eb' }}>SB</Text>
          </View>
          <View style={styles.companyInfo}>
            <Text style={styles.companyName}>SIP Brewery</Text>
            <Text style={styles.companyTagline}>Smart Investment Platform</Text>
          </View>
        </View>
        
        <View style={styles.statementInfo}>
          <Text style={styles.statementTitle}>{metadata.title}</Text>
          <Text style={styles.statementSubtitle}>{metadata.subtitle}</Text>
          <Text style={styles.statementDate}>Generated on: {metadata.generatedOn}</Text>
        </View>
      </View>

      <View style={styles.bottomRow}>
        <View style={styles.metadata}>
          <View style={styles.metadataRow}>
            <Text style={styles.metadataLabel}>ARN:</Text>
            <Text style={styles.metadataValue}>{metadata.arn}</Text>
          </View>
          <View style={styles.metadataRow}>
            <Text style={styles.metadataLabel}>SEBI Reg:</Text>
            <Text style={styles.metadataValue}>{metadata.sebiReg}</Text>
          </View>
          <View style={styles.metadataRow}>
            <Text style={styles.metadataLabel}>Period:</Text>
            <Text style={styles.metadataValue}>{metadata.dateRange}</Text>
          </View>
        </View>

        <View style={styles.compliance}>
          <Text style={styles.complianceText}>SEBI Registered Investment Advisor</Text>
          <Text style={styles.complianceText}>ARN: {metadata.arn}</Text>
          <Text style={styles.complianceText}>This is a digitally generated statement</Text>
        </View>
      </View>
    </View>
  );
};

export default PDFHeader; 