/**
 * 🏦 ASSET MANAGEMENT COMPANY (AMC) DATA SOURCES
 * 
 * Comprehensive list of AMC websites and data sources for financial search engine
 * Based on AMFI data with AUM figures and growth metrics
 * 
 * @author Financial Data Integration Team
 * @version 1.0.0 - AMC Integration
 */

const amcDataSources = {
  // Top 10 AMCs by AUM
  "ICICI Prudential Asset Management Company Limited": {
    aum_march_2025: 912509.39,
    aum_june_2025: 947144.81,
    change_percent: 3.80,
    websites: [
      "https://www.icicipruamc.com",
      "https://www.iciciprudential.com/mutual-fund"
    ],
    search_keywords: ["icici prudential", "icici amc", "icici mutual fund", "icici mf"],
    category: "large_amc",
    rank: 1
  },

  "SBI Funds Management Ltd": {
    aum_march_2025: 1076151.71,
    aum_june_2025: 935796.32,
    change_percent: -13.04,
    websites: [
      "https://www.sbimf.com",
      "https://www.sbifundsmanagement.com"
    ],
    search_keywords: ["sbi mutual fund", "sbi mf", "sbi funds", "sbi amc"],
    category: "large_amc",
    rank: 2
  },

  "HDFC Asset Management Company Limited": {
    aum_march_2025: 767267.40,
    aum_june_2025: 832483.59,
    change_percent: 8.50,
    websites: [
      "https://www.hdfcfund.com",
      "https://www.hdfcamc.com"
    ],
    search_keywords: ["hdfc mutual fund", "hdfc amc", "hdfc mf", "hdfc fund"],
    category: "large_amc",
    rank: 3
  },

  "Nippon Life India Asset Management Ltd": {
    aum_march_2025: 561105.45,
    aum_june_2025: 498202.04,
    change_percent: -11.21,
    websites: [
      "https://www.nipponindiamutualf und.com",
      "https://www.reliancemutual.com"
    ],
    search_keywords: ["nippon india", "reliance mutual fund", "nippon amc", "reliance mf"],
    category: "large_amc",
    rank: 4
  },

  "Kotak Mahindra Asset Management Co Ltd": {
    aum_march_2025: 487046.59,
    aum_june_2025: 513231.45,
    change_percent: 5.38,
    websites: [
      "https://www.kotakmf.com",
      "https://www.kotakmutualf und.com"
    ],
    search_keywords: ["kotak mutual fund", "kotak mf", "kotak amc", "kotak mahindra mf"],
    category: "large_amc",
    rank: 5
  },

  "Aditya Birla Sun Life AMC Ltd": {
    aum_march_2025: 381840.82,
    aum_june_2025: 402938.10,
    change_percent: 5.53,
    websites: [
      "https://www.birlasunlife.com",
      "https://www.adityabirlacapital.com/mutual-funds"
    ],
    search_keywords: ["aditya birla", "birla sun life", "absl", "birla mutual fund"],
    category: "large_amc",
    rank: 6
  },

  "UTI Asset Management Company Ltd": {
    aum_march_2025: 339916.88,
    aum_june_2025: 307704.32,
    change_percent: -9.48,
    websites: [
      "https://www.utimf.com",
      "https://www.utimutualf und.com"
    ],
    search_keywords: ["uti mutual fund", "uti mf", "uti amc", "unit trust of india"],
    category: "large_amc",
    rank: 7
  },

  "Axis Asset Management Company Limited": {
    aum_march_2025: 322461.66,
    aum_june_2025: 337086.65,
    change_percent: 4.54,
    websites: [
      "https://www.axismf.com",
      "https://www.axismutualf und.com"
    ],
    search_keywords: ["axis mutual fund", "axis mf", "axis amc", "axis bank mf"],
    category: "large_amc",
    rank: 8
  },

  "Mirae Asset Investment Managers (India) Private Limited": {
    aum_march_2025: 189592.70,
    aum_june_2025: 195463.09,
    change_percent: 3.10,
    websites: [
      "https://www.miraeassetmf.co.in",
      "https://www.miraeasset.com/in"
    ],
    search_keywords: ["mirae asset", "mirae mutual fund", "mirae mf", "mirae amc"],
    category: "mid_amc",
    rank: 9
  },

  "Tata Asset Management Limited": {
    aum_march_2025: 187619.94,
    aum_june_2025: 196351.52,
    change_percent: 4.65,
    websites: [
      "https://www.tatamutualfund.com",
      "https://www.tatamutualf und.com"
    ],
    search_keywords: ["tata mutual fund", "tata mf", "tata amc", "tata asset management"],
    category: "mid_amc",
    rank: 10
  }
};

// Generate all AMC websites for search engine integration
const getAllAMCWebsites = () => {
  const websites = [];
  Object.values(amcDataSources).forEach(amc => {
    websites.push(...amc.websites);
  });
  return [...new Set(websites)]; // Remove duplicates
};

// Generate search keywords for all AMCs
const getAllAMCKeywords = () => {
  const keywords = [];
  Object.values(amcDataSources).forEach(amc => {
    keywords.push(...amc.search_keywords);
  });
  return [...new Set(keywords)];
};

// Get AMC data by category
const getAMCsByCategory = (category) => {
  return Object.entries(amcDataSources)
    .filter(([name, data]) => data.category === category)
    .map(([name, data]) => ({ name, ...data }));
};

// Get top performing AMCs by growth
const getTopPerformingAMCs = (limit = 10) => {
  return Object.entries(amcDataSources)
    .map(([name, data]) => ({ name, ...data }))
    .sort((a, b) => b.change_percent - a.change_percent)
    .slice(0, limit);
};

module.exports = {
  amcDataSources,
  getAllAMCWebsites,
  getAllAMCKeywords,
  getAMCsByCategory,
  getTopPerformingAMCs
};
