const express = require('express');
const router = express.Router();

// Mock fund data for demonstration - in production, this would come from real APIs
const mockFunds = [
  {
    id: 'FUND001',
    name: 'SBI Bluechip Fund',
    category: 'Large Cap',
    nav: 58.45,
    returns: {
      '1Y': 12.5,
      '3Y': 15.2,
      '5Y': 14.8
    },
    rating: 4.5,
    riskLevel: 'Moderate',
    minSIP: 500,
    expenseRatio: 0.68,
    aum: 25000,
    fundManager: 'Dinesh Ahuja',
    launchDate: '2006-02-01',
    benchmark: 'S&P BSE 100',
    exitLoad: '1% if redeemed within 1 year'
  },
  {
    id: 'FUND002',
    name: 'HDFC Top 100 Fund',
    category: 'Large Cap',
    nav: 742.89,
    returns: {
      '1Y': 13.8,
      '3Y': 16.1,
      '5Y': 15.9
    },
    rating: 5.0,
    riskLevel: 'Moderate',
    minSIP: 1000,
    expenseRatio: 0.52,
    aum: 18500,
    fundManager: 'Prashant Jain',
    launchDate: '1996-10-01',
    benchmark: 'Nifty 100',
    exitLoad: '1% if redeemed within 1 year'
  },
  {
    id: 'FUND003',
    name: 'Axis Small Cap Fund',
    category: 'Small Cap',
    nav: 45.67,
    returns: {
      '1Y': 18.5,
      '3Y': 22.3,
      '5Y': 19.8
    },
    rating: 4.0,
    riskLevel: 'High',
    minSIP: 1000,
    expenseRatio: 1.25,
    aum: 8500,
    fundManager: 'Anupam Tiwari',
    launchDate: '2013-01-01',
    benchmark: 'Nifty Smallcap 100',
    exitLoad: '1% if redeemed within 1 year'
  },
  {
    id: 'FUND004',
    name: 'Mirae Asset Tax Saver Fund',
    category: 'ELSS',
    nav: 28.34,
    returns: {
      '1Y': 14.2,
      '3Y': 17.5,
      '5Y': 16.8
    },
    rating: 4.5,
    riskLevel: 'Moderate High',
    minSIP: 500,
    expenseRatio: 0.75,
    aum: 12000,
    fundManager: 'Neelesh Surana',
    launchDate: '2015-12-28',
    benchmark: 'Nifty 500',
    exitLoad: 'Lock-in period of 3 years',
    taxBenefit: '80C deduction up to ₹1.5 lakh'
  },
  {
    id: 'FUND005',
    name: 'Parag Parikh Flexi Cap Fund',
    category: 'Flexi Cap',
    nav: 52.18,
    returns: {
      '1Y': 16.8,
      '3Y': 19.2,
      '5Y': 18.5
    },
    rating: 5.0,
    riskLevel: 'Moderate High',
    minSIP: 1000,
    expenseRatio: 0.68,
    aum: 15500,
    fundManager: 'Rajeev Thakkar',
    launchDate: '2013-05-03',
    benchmark: 'Nifty 500',
    exitLoad: '1% if redeemed within 1 year'
  }
];

const topSIPs = [
  {
    id: 'SIP001',
    fundId: 'FUND002',
    name: 'HDFC Top 100 Fund',
    category: 'Large Cap',
    sipAmount: 5000,
    returns: {
      '1Y': 13.8,
      '3Y': 16.1,
      '5Y': 15.9
    },
    rating: 5.0,
    popularity: 95,
    investors: 125000,
    totalSIPValue: 2500000000
  },
  {
    id: 'SIP002',
    fundId: 'FUND005',
    name: 'Parag Parikh Flexi Cap Fund',
    category: 'Flexi Cap',
    sipAmount: 3000,
    returns: {
      '1Y': 16.8,
      '3Y': 19.2,
      '5Y': 18.5
    },
    rating: 5.0,
    popularity: 92,
    investors: 98000,
    totalSIPValue: 1800000000
  },
  {
    id: 'SIP003',
    fundId: 'FUND001',
    name: 'SBI Bluechip Fund',
    category: 'Large Cap',
    sipAmount: 2000,
    returns: {
      '1Y': 12.5,
      '3Y': 15.2,
      '5Y': 14.8
    },
    rating: 4.5,
    popularity: 88,
    investors: 156000,
    totalSIPValue: 3200000000
  }
];

const elssFunds = mockFunds.filter(fund => fund.category === 'ELSS').concat([
  {
    id: 'FUND006',
    name: 'Axis Long Term Equity Fund',
    category: 'ELSS',
    nav: 156.78,
    returns: {
      '1Y': 15.5,
      '3Y': 18.2,
      '5Y': 17.1
    },
    rating: 4.5,
    riskLevel: 'Moderate High',
    minSIP: 500,
    expenseRatio: 0.85,
    aum: 22000,
    fundManager: 'Jinesh Gopani',
    launchDate: '2009-12-30',
    benchmark: 'Nifty 500',
    exitLoad: 'Lock-in period of 3 years',
    taxBenefit: '80C deduction up to ₹1.5 lakh'
  },
  {
    id: 'FUND007',
    name: 'DSP Tax Saver Fund',
    category: 'ELSS',
    nav: 89.45,
    returns: {
      '1Y': 13.8,
      '3Y': 16.5,
      '5Y': 15.9
    },
    rating: 4.0,
    riskLevel: 'Moderate High',
    minSIP: 500,
    expenseRatio: 0.92,
    aum: 8500,
    fundManager: 'Apoorva Shah',
    launchDate: '2007-01-01',
    benchmark: 'Nifty 500',
    exitLoad: 'Lock-in period of 3 years',
    taxBenefit: '80C deduction up to ₹1.5 lakh'
  }
]);

// GET /api/funds/explore - Fund discovery & filtering
router.get('/explore', async (req, res) => {
  try {
    const { category, riskLevel, minSIP, sortBy, order = 'desc' } = req.query;
    
    let filteredFunds = [...mockFunds];
    
    // Apply filters
    if (category) {
      filteredFunds = filteredFunds.filter(fund => 
        fund.category.toLowerCase().includes(category.toLowerCase())
      );
    }
    
    if (riskLevel) {
      filteredFunds = filteredFunds.filter(fund => 
        fund.riskLevel.toLowerCase().includes(riskLevel.toLowerCase())
      );
    }
    
    if (minSIP) {
      filteredFunds = filteredFunds.filter(fund => 
        fund.minSIP <= parseInt(minSIP)
      );
    }
    
    // Apply sorting
    if (sortBy) {
      filteredFunds.sort((a, b) => {
        let aValue, bValue;
        
        switch (sortBy) {
          case 'returns1Y':
            aValue = a.returns['1Y'];
            bValue = b.returns['1Y'];
            break;
          case 'returns3Y':
            aValue = a.returns['3Y'];
            bValue = b.returns['3Y'];
            break;
          case 'rating':
            aValue = a.rating;
            bValue = b.rating;
            break;
          case 'aum':
            aValue = a.aum;
            bValue = b.aum;
            break;
          case 'expenseRatio':
            aValue = a.expenseRatio;
            bValue = b.expenseRatio;
            break;
          default:
            aValue = a.name;
            bValue = b.name;
        }
        
        if (order === 'asc') {
          return aValue > bValue ? 1 : -1;
        } else {
          return aValue < bValue ? 1 : -1;
        }
      });
    }
    
    res.json({
      success: true,
      data: {
        funds: filteredFunds,
        totalCount: filteredFunds.length,
        filters: {
          categories: ['Large Cap', 'Mid Cap', 'Small Cap', 'Flexi Cap', 'ELSS', 'Hybrid'],
          riskLevels: ['Low', 'Moderate', 'Moderate High', 'High'],
          sortOptions: ['returns1Y', 'returns3Y', 'rating', 'aum', 'expenseRatio']
        }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in fund exploration:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch funds',
      message: error.message
    });
  }
});

// GET /api/funds/top-sips - Top performing SIP funds
router.get('/top-sips', async (req, res) => {
  try {
    const { limit = 10 } = req.query;
    
    const limitedSIPs = topSIPs.slice(0, parseInt(limit));
    
    res.json({
      success: true,
      data: {
        topSIPs: limitedSIPs,
        totalCount: topSIPs.length,
        averageReturns: {
          '1Y': topSIPs.reduce((sum, sip) => sum + sip.returns['1Y'], 0) / topSIPs.length,
          '3Y': topSIPs.reduce((sum, sip) => sum + sip.returns['3Y'], 0) / topSIPs.length,
          '5Y': topSIPs.reduce((sum, sip) => sum + sip.returns['5Y'], 0) / topSIPs.length
        },
        totalInvestors: topSIPs.reduce((sum, sip) => sum + sip.investors, 0),
        totalSIPValue: topSIPs.reduce((sum, sip) => sum + sip.totalSIPValue, 0)
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in top SIPs:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch top SIPs',
      message: error.message
    });
  }
});

// GET /api/funds/elss - Tax saving ELSS funds
router.get('/elss', async (req, res) => {
  try {
    const { sortBy = 'returns3Y', order = 'desc' } = req.query;
    
    let sortedELSS = [...elssFunds];
    
    // Apply sorting
    sortedELSS.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'returns1Y':
          aValue = a.returns['1Y'];
          bValue = b.returns['1Y'];
          break;
        case 'returns3Y':
          aValue = a.returns['3Y'];
          bValue = b.returns['3Y'];
          break;
        case 'rating':
          aValue = a.rating;
          bValue = b.rating;
          break;
        case 'aum':
          aValue = a.aum;
          bValue = b.aum;
          break;
        default:
          aValue = a.returns['3Y'];
          bValue = b.returns['3Y'];
      }
      
      if (order === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
    
    res.json({
      success: true,
      data: {
        elssFunds: sortedELSS,
        totalCount: sortedELSS.length,
        taxBenefits: {
          section80C: 'Deduction up to ₹1.5 lakh under Section 80C',
          lockInPeriod: '3 years mandatory lock-in period',
          ltcg: 'Long-term capital gains tax applicable after ₹1 lakh'
        },
        averageReturns: {
          '1Y': sortedELSS.reduce((sum, fund) => sum + fund.returns['1Y'], 0) / sortedELSS.length,
          '3Y': sortedELSS.reduce((sum, fund) => sum + fund.returns['3Y'], 0) / sortedELSS.length,
          '5Y': sortedELSS.reduce((sum, fund) => sum + fund.returns['5Y'], 0) / sortedELSS.length
        }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in ELSS funds:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch ELSS funds',
      message: error.message
    });
  }
});

// GET /api/funds/goals - Goal-based fund recommendations
router.get('/goals', async (req, res) => {
  try {
    const { goal, timeHorizon, riskTolerance, amount } = req.query;
    
    const goalBasedRecommendations = {
      'retirement': {
        timeHorizon: '20+ years',
        recommendedFunds: mockFunds.filter(fund => 
          fund.category === 'Large Cap' || fund.category === 'Flexi Cap'
        ),
        strategy: 'Long-term wealth creation with moderate risk',
        sipAmount: Math.max(5000, (amount || 1000000) / 240) // 20 years
      },
      'child-education': {
        timeHorizon: '10-15 years',
        recommendedFunds: mockFunds.filter(fund => 
          fund.category === 'Large Cap' || fund.category === 'Mid Cap'
        ),
        strategy: 'Balanced growth with capital protection',
        sipAmount: Math.max(3000, (amount || 500000) / 120) // 10 years
      },
      'house-purchase': {
        timeHorizon: '5-10 years',
        recommendedFunds: mockFunds.filter(fund => 
          fund.category === 'Large Cap' || fund.category === 'Hybrid'
        ),
        strategy: 'Moderate growth with stability',
        sipAmount: Math.max(8000, (amount || 2000000) / 60) // 5 years
      },
      'emergency-fund': {
        timeHorizon: '1-3 years',
        recommendedFunds: mockFunds.filter(fund => 
          fund.riskLevel === 'Low' || fund.riskLevel === 'Moderate'
        ),
        strategy: 'Capital preservation with liquidity',
        sipAmount: Math.max(2000, (amount || 300000) / 36) // 3 years
      }
    };
    
    const selectedGoal = goal ? goalBasedRecommendations[goal.toLowerCase()] : null;
    
    res.json({
      success: true,
      data: {
        availableGoals: Object.keys(goalBasedRecommendations),
        selectedGoal: selectedGoal,
        allRecommendations: goal ? null : goalBasedRecommendations,
        sipCalculator: {
          formula: 'SIP Amount = Target Amount / (Time Horizon in months)',
          factors: ['Target Amount', 'Time Horizon', 'Expected Returns', 'Risk Tolerance']
        }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in goal-based recommendations:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch goal-based recommendations',
      message: error.message
    });
  }
});

// GET /api/funds/:id - Get specific fund details
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const fund = mockFunds.find(f => f.id === id);
    
    if (!fund) {
      return res.status(404).json({
        success: false,
        error: 'Fund not found',
        message: `Fund with ID ${id} does not exist`
      });
    }
    
    // Add additional details for specific fund
    const detailedFund = {
      ...fund,
      portfolio: {
        topHoldings: [
          { name: 'Reliance Industries', percentage: 8.5 },
          { name: 'HDFC Bank', percentage: 7.2 },
          { name: 'Infosys', percentage: 6.8 },
          { name: 'TCS', percentage: 6.1 },
          { name: 'ICICI Bank', percentage: 5.9 }
        ],
        sectorAllocation: [
          { sector: 'Financial Services', percentage: 28.5 },
          { sector: 'Information Technology', percentage: 22.1 },
          { sector: 'Oil & Gas', percentage: 12.8 },
          { sector: 'Consumer Goods', percentage: 10.2 },
          { sector: 'Healthcare', percentage: 8.4 }
        ]
      },
      performance: {
        monthlyReturns: [
          { month: 'Jan 2024', return: 2.1 },
          { month: 'Feb 2024', return: 1.8 },
          { month: 'Mar 2024', return: 3.2 },
          { month: 'Apr 2024', return: -0.5 },
          { month: 'May 2024', return: 2.8 }
        ],
        riskMetrics: {
          standardDeviation: 15.2,
          sharpeRatio: 1.25,
          beta: 0.95,
          alpha: 2.1
        }
      }
    };
    
    res.json({
      success: true,
      data: detailedFund,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in fund details:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch fund details',
      message: error.message
    });
  }
});

module.exports = router;
