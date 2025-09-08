/**
 * 🔍 AMC SEARCH INTEGRATION
 * 
 * Integrates all Asset Management Company websites into the financial search engine
 * Provides enhanced search capabilities for mutual fund data
 * 
 * @author Financial Search Integration Team
 * @version 1.0.0 - AMC Search Integration
 */

const { amcDataSources, getAllAMCWebsites, getAllAMCKeywords } = require('./parsers/amc-data-sources');
const { client, ensureIndex } = require('./elasticsearch');
const axios = require('axios');
const cheerio = require('cheerio');
const logger = require('../utils/logger');

class AMCSearchIntegration {
  constructor(options = {}) {
    this.config = {
      indexName: options.indexName || 'amc-data',
      batchSize: options.batchSize || 50,
      requestDelay: options.requestDelay || 2000,
      maxRetries: options.maxRetries || 3,
      ...options
    };
    
    this.amcWebsites = getAllAMCWebsites();
    this.amcKeywords = getAllAMCKeywords();
    this.crawlResults = new Map();
  }

  async initialize() {
    try {
      logger.info('🏦 Initializing AMC Search Integration...');
      
      // Ensure Elasticsearch index exists
      await ensureIndex(this.config.indexName);
      
      // Index AMC data
      await this.indexAMCData();
      
      // Setup search mappings
      await this.setupSearchMappings();
      
      logger.info('✅ AMC Search Integration initialized successfully');
      
    } catch (error) {
      logger.error('❌ AMC Search Integration initialization failed:', error);
      throw error;
    }
  }

  async indexAMCData() {
    try {
      logger.info('📊 Indexing AMC data to Elasticsearch...');
      
      const bulkOps = [];
      
      for (const [amcName, amcData] of Object.entries(amcDataSources)) {
        // Index AMC basic information
        bulkOps.push({
          index: {
            _index: this.config.indexName,
            _id: `amc_${amcData.rank}`
          }
        });
        
        bulkOps.push({
          name: amcName,
          type: 'amc',
          aum_march_2025: amcData.aum_march_2025,
          aum_june_2025: amcData.aum_june_2025,
          change_percent: amcData.change_percent,
          growth_category: this.categorizeGrowth(amcData.change_percent),
          websites: amcData.websites,
          search_keywords: amcData.search_keywords,
          category: amcData.category,
          rank: amcData.rank,
          indexed_at: new Date().toISOString()
        });
        
        // Index individual websites
        for (let i = 0; i < amcData.websites.length; i++) {
          bulkOps.push({
            index: {
              _index: this.config.indexName,
              _id: `website_${amcData.rank}_${i}`
            }
          });
          
          bulkOps.push({
            name: amcName,
            type: 'website',
            url: amcData.websites[i],
            amc_rank: amcData.rank,
            amc_category: amcData.category,
            search_keywords: amcData.search_keywords,
            indexed_at: new Date().toISOString()
          });
        }
        
        // Index search keywords
        for (let j = 0; j < amcData.search_keywords.length; j++) {
          bulkOps.push({
            index: {
              _index: this.config.indexName,
              _id: `keyword_${amcData.rank}_${j}`
            }
          });
          
          bulkOps.push({
            name: amcName,
            type: 'keyword',
            keyword: amcData.search_keywords[j],
            amc_rank: amcData.rank,
            amc_category: amcData.category,
            aum_current: amcData.aum_june_2025,
            change_percent: amcData.change_percent,
            indexed_at: new Date().toISOString()
          });
        }
      }
      
      // Bulk index to Elasticsearch
      if (bulkOps.length > 0) {
        const response = await client.bulk({
          refresh: true,
          body: bulkOps
        });
        
        if (response.body.errors) {
          logger.warn('⚠️ Some AMC data indexing errors occurred');
        } else {
          logger.info(`✅ Successfully indexed ${bulkOps.length / 2} AMC records`);
        }
      }
      
    } catch (error) {
      logger.error('❌ AMC data indexing failed:', error);
      throw error;
    }
  }

  categorizeGrowth(changePercent) {
    if (changePercent > 20) return 'high_growth';
    if (changePercent > 10) return 'good_growth';
    if (changePercent > 0) return 'positive_growth';
    if (changePercent > -5) return 'slight_decline';
    return 'significant_decline';
  }

  async setupSearchMappings() {
    try {
      // Create mapping for better search performance
      await client.indices.putMapping({
        index: this.config.indexName,
        body: {
          properties: {
            name: {
              type: 'text',
              analyzer: 'standard',
              fields: {
                keyword: { type: 'keyword' }
              }
            },
            type: { type: 'keyword' },
            search_keywords: {
              type: 'text',
              analyzer: 'standard'
            },
            aum_june_2025: { type: 'float' },
            change_percent: { type: 'float' },
            rank: { type: 'integer' },
            category: { type: 'keyword' },
            growth_category: { type: 'keyword' },
            url: { type: 'keyword' },
            indexed_at: { type: 'date' }
          }
        }
      });
      
      logger.info('✅ AMC search mappings configured');
      
    } catch (error) {
      logger.warn('⚠️ Search mapping setup failed:', error.message);
    }
  }

  async searchAMCs(query, options = {}) {
    try {
      const searchParams = {
        index: this.config.indexName,
        body: {
          query: {
            bool: {
              should: [
                {
                  multi_match: {
                    query: query,
                    fields: ['name^3', 'search_keywords^2', 'keyword^2'],
                    type: 'best_fields',
                    fuzziness: 'AUTO'
                  }
                },
                {
                  wildcard: {
                    'name.keyword': `*${query.toLowerCase()}*`
                  }
                }
              ],
              minimum_should_match: 1
            }
          },
          sort: [
            { rank: { order: 'asc' } },
            { aum_june_2025: { order: 'desc' } },
            { _score: { order: 'desc' } }
          ],
          size: options.limit || 20
        }
      };
      
      // Add filters if specified
      if (options.category) {
        searchParams.body.query.bool.filter = [
          { term: { category: options.category } }
        ];
      }
      
      if (options.growth_category) {
        if (!searchParams.body.query.bool.filter) {
          searchParams.body.query.bool.filter = [];
        }
        searchParams.body.query.bool.filter.push({
          term: { growth_category: options.growth_category }
        });
      }
      
      const response = await client.search(searchParams);
      
      return {
        total: response.body.hits.total.value,
        results: response.body.hits.hits.map(hit => ({
          id: hit._id,
          score: hit._score,
          ...hit._source
        }))
      };
      
    } catch (error) {
      logger.error('❌ AMC search failed:', error);
      throw error;
    }
  }

  async getAMCByRank(rank) {
    try {
      const response = await client.search({
        index: this.config.indexName,
        body: {
          query: {
            bool: {
              must: [
                { term: { rank: rank } },
                { term: { type: 'amc' } }
              ]
            }
          }
        }
      });
      
      if (response.body.hits.hits.length > 0) {
        return response.body.hits.hits[0]._source;
      }
      
      return null;
      
    } catch (error) {
      logger.error('❌ Get AMC by rank failed:', error);
      return null;
    }
  }

  async getTopAMCs(limit = 10, category = null) {
    try {
      const searchParams = {
        index: this.config.indexName,
        body: {
          query: {
            bool: {
              must: [
                { term: { type: 'amc' } }
              ]
            }
          },
          sort: [
            { aum_june_2025: { order: 'desc' } }
          ],
          size: limit
        }
      };
      
      if (category) {
        searchParams.body.query.bool.must.push({
          term: { category: category }
        });
      }
      
      const response = await client.search(searchParams);
      
      return response.body.hits.hits.map(hit => hit._source);
      
    } catch (error) {
      logger.error('❌ Get top AMCs failed:', error);
      return [];
    }
  }

  async getGrowthLeaders(limit = 10) {
    try {
      const response = await client.search({
        index: this.config.indexName,
        body: {
          query: {
            bool: {
              must: [
                { term: { type: 'amc' } },
                { range: { change_percent: { gt: 0 } } }
              ]
            }
          },
          sort: [
            { change_percent: { order: 'desc' } }
          ],
          size: limit
        }
      });
      
      return response.body.hits.hits.map(hit => hit._source);
      
    } catch (error) {
      logger.error('❌ Get growth leaders failed:', error);
      return [];
    }
  }

  async crawlAMCWebsites(amcNames = null) {
    try {
      logger.info('🕷️ Starting AMC website crawling...');
      
      const amcsToCrawl = amcNames ? 
        Object.entries(amcDataSources).filter(([name]) => amcNames.includes(name)) :
        Object.entries(amcDataSources);
      
      for (const [amcName, amcData] of amcsToCrawl) {
        logger.info(`🔍 Crawling ${amcName}...`);
        
        for (const website of amcData.websites) {
          try {
            const crawlData = await this.crawlWebsite(website, amcName, amcData);
            this.crawlResults.set(`${amcName}_${website}`, crawlData);
            
            // Index crawled data
            await this.indexCrawledData(amcName, amcData, website, crawlData);
            
            // Delay between requests
            await new Promise(resolve => setTimeout(resolve, this.config.requestDelay));
            
          } catch (error) {
            logger.warn(`⚠️ Failed to crawl ${website} for ${amcName}:`, error.message);
          }
        }
      }
      
      logger.info('✅ AMC website crawling completed');
      
    } catch (error) {
      logger.error('❌ AMC website crawling failed:', error);
      throw error;
    }
  }

  async crawlWebsite(url, amcName, amcData) {
    try {
      const response = await axios.get(url, {
        timeout: 10000,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      
      const $ = cheerio.load(response.data);
      
      // Extract relevant content
      const crawlData = {
        url: url,
        title: $('title').text().trim(),
        description: $('meta[name="description"]').attr('content') || '',
        keywords: $('meta[name="keywords"]').attr('content') || '',
        content: this.extractMainContent($),
        nav_links: this.extractNavLinks($),
        fund_data: this.extractFundData($),
        crawled_at: new Date().toISOString()
      };
      
      return crawlData;
      
    } catch (error) {
      throw new Error(`Website crawl failed: ${error.message}`);
    }
  }

  extractMainContent($) {
    // Remove unwanted elements
    $('script, style, nav, header, footer, .advertisement').remove();
    
    // Extract main content
    const contentSelectors = [
      'main', '.main-content', '.content', 
      '.fund-data', '.scheme-data', '.nav-data'
    ];
    
    let content = '';
    for (const selector of contentSelectors) {
      const element = $(selector);
      if (element.length > 0) {
        content += element.text().trim() + ' ';
      }
    }
    
    return content.trim().substring(0, 5000); // Limit content length
  }

  extractNavLinks($) {
    const links = [];
    $('nav a, .navigation a, .menu a').each((i, elem) => {
      const href = $(elem).attr('href');
      const text = $(elem).text().trim();
      if (href && text) {
        links.push({ href, text });
      }
    });
    return links.slice(0, 20); // Limit number of links
  }

  extractFundData($) {
    const fundData = [];
    
    // Look for fund/scheme data in tables or lists
    $('table tr, .fund-list li, .scheme-list li').each((i, elem) => {
      const text = $(elem).text().trim();
      if (text.includes('NAV') || text.includes('Fund') || text.includes('Scheme')) {
        fundData.push(text);
      }
    });
    
    return fundData.slice(0, 10); // Limit fund data entries
  }

  async indexCrawledData(amcName, amcData, website, crawlData) {
    try {
      await client.index({
        index: this.config.indexName,
        id: `crawled_${amcData.rank}_${Date.now()}`,
        body: {
          name: amcName,
          type: 'crawled_data',
          amc_rank: amcData.rank,
          amc_category: amcData.category,
          website: website,
          title: crawlData.title,
          description: crawlData.description,
          content: crawlData.content,
          fund_data: crawlData.fund_data,
          search_keywords: amcData.search_keywords,
          indexed_at: new Date().toISOString()
        }
      });
      
    } catch (error) {
      logger.warn(`⚠️ Failed to index crawled data for ${amcName}:`, error.message);
    }
  }

  // Integration with existing financial search
  async enhanceFinancialSearch(query, originalResults) {
    try {
      // Search AMC data
      const amcResults = await this.searchAMCs(query, { limit: 10 });
      
      // Enhance original results with AMC data
      const enhancedResults = [...originalResults];
      
      for (const amcResult of amcResults.results) {
        enhancedResults.push({
          title: `${amcResult.name} - AUM: ₹${amcResult.aum_june_2025} Cr`,
          url: amcResult.websites ? amcResult.websites[0] : '#',
          snippet: `${amcResult.name} | AUM Growth: ${amcResult.change_percent}% | Rank: ${amcResult.rank}`,
          source: 'AMC Database',
          type: 'amc_data',
          relevanceScore: 0.9, // High relevance for AMC data
          amcData: amcResult
        });
      }
      
      return enhancedResults;
      
    } catch (error) {
      logger.error('❌ AMC search enhancement failed:', error);
      return originalResults;
    }
  }

  getSearchStats() {
    return {
      totalAMCs: Object.keys(amcDataSources).length,
      totalWebsites: this.amcWebsites.length,
      totalKeywords: this.amcKeywords.length,
      crawlResults: this.crawlResults.size,
      indexName: this.config.indexName
    };
  }
}

module.exports = { AMCSearchIntegration };
