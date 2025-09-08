const axios = require('axios');
const dayjs = require('dayjs');
const { xirr, xirrResult } = require('xirr');
require('dotenv').config();

// Gemini API Configuration
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || 'AIzaSyDc63xZUJktleMdGwfGILfp5oUITQ3znpM';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';

function calculateCAGR(startNav, endNav, years) {
  if (!startNav || !endNav || years <= 0) return null;
  return Math.pow(endNav / startNav, 1 / years) - 1;
}

function calculateReturns(navArray) {
  // Returns as percentage change between consecutive NAVs
  let returns = [];
  for (let i = 1; i < navArray.length; i++) {
    const prev = navArray[i - 1].nav;
    const curr = navArray[i].nav;
    if (prev && curr) {
      returns.push((curr - prev) / prev);
    }
  }
  return returns;
}

function calculateVolatility(navArray) {
  const returns = calculateReturns(navArray);
  if (returns.length === 0) return null;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
  return Math.sqrt(variance);
}

function calculateMaxDrawdown(navArray) {
  let maxDrawdown = 0;
  let peak = navArray[0]?.nav || 0;
  for (let i = 1; i < navArray.length; i++) {
    if (navArray[i].nav > peak) peak = navArray[i].nav;
    const drawdown = (peak - navArray[i].nav) / peak;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }
  return maxDrawdown;
}

function calculateBreakout(navArray, months) {
  if (navArray.length === 0) return { breakout: false, arrow: '', reason: 'No data available' };
  const latest = navArray[navArray.length - 1].nav;
  const periodData = navArray.slice(-months * 30);
  const periodHigh = Math.max(...periodData.map(d => d.nav));
  const breakout = latest > periodHigh;
  return {
    breakout,
    arrow: breakout ? '↗️' : '',
    reason: breakout ? `Latest NAV (${latest}) > ${months}M high (${periodHigh.toFixed(4)})` : `Latest NAV (${latest}) ≤ ${months}M high (${periodHigh.toFixed(4)})`
  };
}

function calculateBreakdown(navArray, months) {
  if (navArray.length === 0) return { breakdown: false, arrow: '', reason: 'No data available' };
  const latest = navArray[navArray.length - 1].nav;
  const periodData = navArray.slice(-months * 30);
  const periodLow = Math.min(...periodData.map(d => d.nav));
  const breakdown = latest < periodLow;
  return {
    breakdown,
    arrow: breakdown ? '↘️' : '',
    reason: breakdown ? `Latest NAV (${latest}) < ${months}M low (${periodLow.toFixed(4)})` : `Latest NAV (${latest}) ≥ ${months}M low (${periodLow.toFixed(4)})`
  };
}

function groupByPeriod(navHistory, months) {
  const cutoff = dayjs().subtract(months, 'month');
  return navHistory.filter(d => dayjs(d.date).isAfter(cutoff));
}

function groupByYears(navHistory, years) {
  const cutoff = dayjs().subtract(years, 'year');
  return navHistory.filter(d => dayjs(d.date).isAfter(cutoff));
}

function getSinceInception(navHistory) {
  return navHistory;
}

function getXIRR(navHistory) {
  // Assume monthly inflow of 1000 at each month, last date is redemption
  if (navHistory.length < 2) return null;
  const inflows = [];
  let lastDate = navHistory[navHistory.length - 1].date;
  for (let i = 0; i < navHistory.length - 1; i++) {
    if (i % 30 === 0) {
      inflows.push({
        amount: -1000,
        when: new Date(navHistory[i].date)
      });
    }
  }
  // Redemption: all units at last NAV
  inflows.push({
    amount: 1000 * (navHistory.length / 30) * (navHistory[navHistory.length - 1].nav / navHistory[0].nav),
    when: new Date(lastDate)
  });
  try {
    return xirr(inflows);
  } catch {
    return null;
  }
}

function chartData(navArray) {
  return navArray.map(d => ({ label: d.date, value: d.nav }));
}

async function analyzeFundWithNAV(schemeCodes, userQuery) {
  try {
    const results = [];
    for (const schemeCode of schemeCodes) {
      // Fetch NAV data
      const response = await axios.get(`https://api.mfapi.in/mf/${schemeCode}`);
      const fundData = response.data;
      if (!fundData || !fundData.data || fundData.data.length === 0) {
        results.push({ schemeCode, error: 'No NAV data found' });
        continue;
      }
      const navHistory = fundData.data.map(item => ({ date: item.date, nav: parseFloat(item.nav) })).reverse();
      const latestNAV = navHistory[navHistory.length - 1];
      // Group NAVs - Fix the date filtering issue
      const currentDate = dayjs();
      const periods = {
        oneMonth: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(1, 'month'))),
        threeMonth: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(3, 'month'))),
        sixMonth: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(6, 'month'))),
        oneYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(1, 'year'))),
        twoYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(2, 'year'))),
        threeYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(3, 'year'))),
        fiveYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(5, 'year'))),
        tenYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(10, 'year'))),
        sinceInception: getSinceInception(navHistory)
      };
      // Calculate stats
      const navStats = {
        cagr1Y: periods.oneYear.length > 1 ? calculateCAGR(periods.oneYear[0].nav, periods.oneYear[periods.oneYear.length - 1].nav, 1) : null,
        cagr3Y: periods.threeYear.length > 1 ? calculateCAGR(periods.threeYear[0].nav, periods.threeYear[periods.threeYear.length - 1].nav, 3) : null,
        cagr5Y: periods.fiveYear.length > 1 ? calculateCAGR(periods.fiveYear[0].nav, periods.fiveYear[periods.fiveYear.length - 1].nav, 5) : null,
        cagr10Y: periods.tenYear.length > 1 ? calculateCAGR(periods.tenYear[0].nav, periods.tenYear[periods.tenYear.length - 1].nav, 10) : null,
        cagrSI: periods.sinceInception.length > 1 ? calculateCAGR(periods.sinceInception[0].nav, periods.sinceInception[periods.sinceInception.length - 1].nav, (dayjs(periods.sinceInception[periods.sinceInception.length - 1].date, 'DD-MM-YYYY').diff(dayjs(periods.sinceInception[0].date, 'DD-MM-YYYY'), 'year', true))) : null,
        xirr: getXIRR(navHistory),
        drawdown: calculateMaxDrawdown(navHistory),
        volatility: calculateVolatility(navHistory),
        breakout: calculateBreakout(navHistory, 6), // 6 months
        breakdown: calculateBreakdown(navHistory, 6) // 6 months
      };
      // Chart data
      const charts = {
        oneMonth: chartData(periods.oneMonth),
        threeMonth: chartData(periods.threeMonth),
        sixMonth: chartData(periods.sixMonth),
        oneYear: chartData(periods.oneYear),
        twoYear: chartData(periods.twoYear),
        threeYear: chartData(periods.threeYear),
        fiveYear: chartData(periods.fiveYear),
        tenYear: chartData(periods.tenYear),
        sinceInception: chartData(periods.sinceInception)
      };
      results.push({
        schemeCode,
        schemeName: fundData.meta?.scheme_name || 'Unknown Fund',
        metadata: fundData.meta || {},
        navStats,
        charts,
        latestNAV
      });
    }
    // Prepare AI input
    const aiInput = results.map(fund => ({
      schemeName: fund.schemeName,
      metadata: fund.metadata,
      navStats: fund.navStats,
      latestNAV: fund.latestNAV,
      charts: fund.charts
    }));
    // Prepare prompt
    const prompt = `You are a financial analyst. Analyze the following mutual funds based on their NAV history and statistics. Provide:
- Performance summary
- Risk assessment
- Investment recommendations
- Notable trends or breakouts/breakdowns

Data:
${JSON.stringify(aiInput, null, 2)}

User Query: ${userQuery}
`;
    // Call Gemini
    const geminiPayload = {
      contents: [
        {
          parts: [
            {
              text: prompt
            }
          ]
        }
      ]
    };
    const geminiResponse = await axios.post(GEMINI_API_URL, geminiPayload, {
      headers: {
        'Content-Type': 'application/json',
        'X-goog-api-key': GEMINI_API_KEY
      }
    });
    const analysis = geminiResponse.data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    return {
      success: true,
      analysis,
      funds: results
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Analyze mutual fund data directly (without fetching from API)
 * @param {Array} fundData - Array of mutual fund data objects
 * @param {string} userQuery - User's query for analysis
 * @returns {Promise<Object>} - Complete analysis response
 */
async function analyzeFundData(fundData, userQuery) {
  try {
    const results = [];
    
    for (const fund of fundData) {
      try {
        // Process NAV history
        const navHistory = fund.navHistory.map(item => ({
          date: item.date,
          nav: parseFloat(item.nav)
        })).reverse();
        
        if (navHistory.length === 0) {
          results.push({
            schemeCode: fund.schemeCode || 'Unknown',
            schemeName: fund.schemeName,
            error: 'No NAV history provided'
          });
          continue;
        }

        const latestNAV = navHistory[navHistory.length - 1];
        
        // Group NAVs by periods
        const currentDate = dayjs();
        const periods = {
          oneMonth: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(1, 'month'))),
          threeMonth: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(3, 'month'))),
          sixMonth: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(6, 'month'))),
          oneYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(1, 'year'))),
          twoYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(2, 'year'))),
          threeYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(3, 'year'))),
          fiveYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(5, 'year'))),
          tenYear: navHistory.filter(d => dayjs(d.date, 'DD-MM-YYYY').isAfter(currentDate.subtract(10, 'year'))),
          sinceInception: getSinceInception(navHistory)
        };

        // Calculate stats
        const navStats = {
          cagr1Y: periods.oneYear.length > 1 ? calculateCAGR(periods.oneYear[0].nav, periods.oneYear[periods.oneYear.length - 1].nav, 1) : null,
          cagr3Y: periods.threeYear.length > 1 ? calculateCAGR(periods.threeYear[0].nav, periods.threeYear[periods.threeYear.length - 1].nav, 3) : null,
          cagr5Y: periods.fiveYear.length > 1 ? calculateCAGR(periods.fiveYear[0].nav, periods.fiveYear[periods.fiveYear.length - 1].nav, 5) : null,
          cagr10Y: periods.tenYear.length > 1 ? calculateCAGR(periods.tenYear[0].nav, periods.tenYear[periods.tenYear.length - 1].nav, 10) : null,
          cagrSI: periods.sinceInception.length > 1 ? calculateCAGR(periods.sinceInception[0].nav, periods.sinceInception[periods.sinceInception.length - 1].nav, (dayjs(periods.sinceInception[periods.sinceInception.length - 1].date, 'DD-MM-YYYY').diff(dayjs(periods.sinceInception[0].date, 'DD-MM-YYYY'), 'year', true))) : null,
          xirr: getXIRR(navHistory),
          drawdown: calculateMaxDrawdown(navHistory),
          volatility: calculateVolatility(navHistory),
          breakout: calculateBreakout(navHistory, 6),
          breakdown: calculateBreakdown(navHistory, 6)
        };

        // Chart data
        const charts = {
          oneMonth: chartData(periods.oneMonth),
          threeMonth: chartData(periods.threeMonth),
          sixMonth: chartData(periods.sixMonth),
          oneYear: chartData(periods.oneYear),
          twoYear: chartData(periods.twoYear),
          threeYear: chartData(periods.threeYear),
          fiveYear: chartData(periods.fiveYear),
          tenYear: chartData(periods.tenYear),
          sinceInception: chartData(periods.sinceInception)
        };

        results.push({
          schemeCode: fund.schemeCode || 'Unknown',
          schemeName: fund.schemeName,
          metadata: fund.metadata || {},
          navStats,
          charts,
          latestNAV
        });

      } catch (error) {
        results.push({
          schemeCode: fund.schemeCode || 'Unknown',
          schemeName: fund.schemeName,
          error: error.message
        });
      }
    }

    // Prepare AI input
    const aiInput = results.map(fund => ({
      schemeName: fund.schemeName,
      metadata: fund.metadata,
      navStats: fund.navStats,
      latestNAV: fund.latestNAV,
      charts: fund.charts
    }));

    // Prepare prompt
    const prompt = `You are a financial analyst. Analyze the following mutual funds based on their NAV history and statistics. Provide:
- Performance summary
- Risk assessment
- Investment recommendations
- Notable trends or breakouts/breakdowns

Data:
${JSON.stringify(aiInput, null, 2)}

User Query: ${userQuery}
`;

    // Call Gemini
    const geminiPayload = {
      contents: [
        {
          parts: [
            {
              text: prompt
            }
          ]
        }
      ]
    };

    const geminiResponse = await axios.post(GEMINI_API_URL, geminiPayload, {
      headers: {
        'Content-Type': 'application/json',
        'X-goog-api-key': GEMINI_API_KEY
      }
    });

    const analysis = geminiResponse.data?.candidates?.[0]?.content?.parts?.[0]?.text || '';

    return {
      success: true,
      analysis,
      funds: results
    };

  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Fetch NAV data for a specific scheme code
 * @param {string} schemeCode - The mutual fund scheme code
 * @returns {Promise<Object>} - NAV data for the scheme
 */
async function fetchNAVData(schemeCode) {
  try {
    const response = await axios.get(`https://api.mfapi.in/mf/${schemeCode}`);
    const fundData = response.data;
    
    if (!fundData || !fundData.data || fundData.data.length === 0) {
      throw new Error('No NAV data found for the given scheme code');
    }

    const navHistory = fundData.data.map(item => ({
      date: item.date,
      nav: parseFloat(item.nav)
    })).reverse();

    return {
      success: true,
      schemeCode,
      schemeName: fundData.meta?.scheme_name || 'Unknown Fund',
      metadata: fundData.meta || {},
      navHistory,
      latestNAV: navHistory[navHistory.length - 1]
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

// Only require Gemini client if not in test mode
let geminiClient = null;
if (process.env.NODE_ENV !== 'test') {
  geminiClient = require('../ai/geminiClient');
} else {
  geminiClient = {
    generate: async () => 'Simulated AI response (test mode)'
  };
}

module.exports = {
  analyzeFundWithNAV,
  analyzeFundData,
  fetchNAVData,
  geminiClient: geminiClient || { generate: async () => 'Simulated AI response (test mode)' }
}; 