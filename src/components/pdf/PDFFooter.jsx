import React from 'react';
import { View, Text } from '@react-pdf/renderer';

const PDFFooter = ({ metadata }) => {
  const styles = {
    footer: {
      position: 'absolute',
      bottom: 30,
      left: 20,
      right: 20,
      padding: 15,
      borderTop: '1px solid #e5e7eb',
      backgroundColor: '#f8fafc'
    },
    disclaimer: {
      fontSize: 7,
      color: '#6b7280',
      textAlign: 'center',
      marginBottom: 10,
      lineHeight: 1.2
    },
    contactGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    contactSection: {
      alignItems: 'center'
    },
    contactLabel: {
      fontSize: 8,
      color: '#374151',
      fontWeight: 'bold',
      marginBottom: 2
    },
    contactValue: {
      fontSize: 7,
      color: '#6b7280',
      textAlign: 'center'
    },
    complianceSection: {
      alignItems: 'center'
    },
    complianceText: {
      fontSize: 7,
      color: '#6b7280',
      textAlign: 'center',
      marginBottom: 2
    },
    pageInfo: {
      alignItems: 'center'
    },
    pageNumber: {
      fontSize: 8,
      color: '#374151',
      fontWeight: 'bold'
    },
    generatedInfo: {
      fontSize: 7,
      color: '#9ca3af',
      marginTop: 2
    }
  };

  return (
    <View style={styles.footer} fixed>
      <Text style={styles.disclaimer}>
        This is a digitally generated statement for informational purposes only. 
        Past performance does not guarantee future results. 
        Mutual fund investments are subject to market risks. 
        Please read all scheme related documents carefully before investing.
      </Text>
      
      <View style={styles.contactGrid}>
        <View style={styles.contactSection}>
          <Text style={styles.contactLabel}>Contact Us</Text>
          <Text style={styles.contactValue}>{metadata.contact.email}</Text>
          <Text style={styles.contactValue}>{metadata.contact.phone}</Text>
        </View>
        
        <View style={styles.complianceSection}>
          <Text style={styles.complianceText}>SEBI Registered Investment Advisor</Text>
          <Text style={styles.complianceText}>ARN: {metadata.arn}</Text>
          <Text style={styles.complianceText}>{metadata.sebiReg}</Text>
        </View>
        
        <View style={styles.pageInfo}>
          <Text style={styles.pageNumber} render={({ pageNumber, totalPages }) => 
            `Page ${pageNumber} of ${totalPages}`
          } />
          <Text style={styles.generatedInfo}>
            Generated on {metadata.generatedOn}
          </Text>
        </View>
      </View>
    </View>
  );
};

export default PDFFooter; 