import React from 'react';
import { View, Text } from '@react-pdf/renderer';

const PDFUserInfo = ({ user }) => {
  const styles = {
    container: {
      padding: 20,
      backgroundColor: '#f8fafc',
      borderRadius: 8,
      marginBottom: 20
    },
    title: {
      fontSize: 14,
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: 15,
      borderBottom: '1px solid #e5e7eb',
      paddingBottom: 5
    },
    userGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap'
    },
    userSection: {
      width: '50%',
      marginBottom: 10
    },
    userSectionFull: {
      width: '100%',
      marginBottom: 10
    },
    label: {
      fontSize: 9,
      color: '#6b7280',
      marginBottom: 2,
      fontWeight: 'bold'
    },
    value: {
      fontSize: 10,
      color: '#1f2937',
      marginBottom: 5
    },
    highlight: {
      fontSize: 10,
      color: '#2563eb',
      fontWeight: 'bold'
    },
    divider: {
      borderRight: '1px solid #e5e7eb',
      paddingRight: 15,
      marginRight: 15
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Client Information</Text>
      
      <View style={styles.userGrid}>
        <View style={[styles.userSection, styles.divider]}>
          <Text style={styles.label}>Full Name</Text>
          <Text style={styles.value}>{user.name}</Text>
        </View>
        
        <View style={styles.userSection}>
          <Text style={styles.label}>Client Code</Text>
          <Text style={styles.highlight}>{user.clientCode}</Text>
        </View>
        
        <View style={[styles.userSection, styles.divider]}>
          <Text style={styles.label}>PAN Number</Text>
          <Text style={styles.value}>{user.pan}</Text>
        </View>
        
        <View style={styles.userSection}>
          <Text style={styles.label}>Mobile Number</Text>
          <Text style={styles.value}>{user.mobile}</Text>
        </View>
        
        <View style={styles.userSectionFull}>
          <Text style={styles.label}>Email Address</Text>
          <Text style={styles.value}>{user.email}</Text>
        </View>
      </View>
    </View>
  );
};

export default PDFUserInfo; 