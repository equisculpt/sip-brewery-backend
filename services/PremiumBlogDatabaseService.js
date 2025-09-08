const mongoose = require('mongoose');
const logger = require('../utils/logger');

/**
 * PREMIUM BLOG DATABASE SERVICE
 * Goldman Sachs-Quality Blog Management System
 * 
 * Manages world-class financial blogs with institutional-grade
 * metadata, analytics, and professional presentation.
 */

// Blog Schema for Goldman Sachs-quality content
const BlogSchema = new mongoose.Schema({
  id: { type: String, required: true, unique: true },
  type: { 
    type: String, 
    enum: ['INVESTOR_EDUCATION', 'IPO_ANALYSIS', 'MUTUAL_FUND_ANALYSIS', 'MARKET_INSIGHTS'],
    required: true 
  },
  title: { type: String, required: true },
  slug: { type: String, required: true, unique: true },
  
  // Content Structure
  content: {
    executive_summary: String,
    main_content: String,
    key_insights: [String],
    asi_predictions: Object,
    risk_warnings: [String],
    investment_thesis: String,
    conclusion: String,
    disclaimer: String
  },
  
  // Premium Features
  interactive_elements: {
    charts: [Object],
    tables: [Object],
    calculators: [Object],
    visualizations: [Object]
  },
  
  // Institutional Metadata
  metadata: {
    goldman_sachs_rating: { type: String, enum: ['SUPERIOR', 'EXCELLENT', 'GOOD'] },
    complexity_score: Number,
    target_audience: String,
    research_depth: String,
    asi_confidence: Number,
    word_count: Number,
    reading_time: Number,
    data_sources: [String],
    analysis_date: Date
  },
  
  // SEO & Professional Optimization
  seo: {
    meta_title: String,
    meta_description: String,
    keywords: [String],
    professional_tags: [String],
    linkedin_optimized: Boolean,
    goldman_sachs_keywords: [String]
  },
  
  // Analytics & Engagement
  analytics: {
    views: { type: Number, default: 0 },
    professional_views: { type: Number, default: 0 },
    goldman_sachs_visits: { type: Number, default: 0 },
    engagement_score: { type: Number, default: 0 },
    share_count: { type: Number, default: 0 },
    professional_shares: { type: Number, default: 0 },
    time_on_page: { type: Number, default: 0 },
    bounce_rate: { type: Number, default: 0 }
  },
  
  // Publication Details
  publication: {
    status: { type: String, enum: ['DRAFT', 'SCHEDULED', 'PUBLISHED'], default: 'DRAFT' },
    published_date: Date,
    scheduled_date: Date,
    last_updated: Date,
    version: { type: Number, default: 1 },
    auto_generated: { type: Boolean, default: true },
    asi_generated: { type: Boolean, default: true }
  },
  
  // Professional Features
  professional_features: {
    downloadable_pdf: Boolean,
    executive_summary_pdf: Boolean,
    data_export: Boolean,
    api_access: Boolean,
    institutional_access: Boolean,
    goldman_sachs_exclusive: Boolean
  },
  
  // Related Content
  related_content: {
    similar_blogs: [String],
    referenced_ipos: [String],
    referenced_mutual_funds: [String],
    market_data_used: [String]
  }
}, {
  timestamps: true,
  collection: 'premium_blogs'
});

// Indexes for performance
BlogSchema.index({ type: 1, 'publication.published_date': -1 });
BlogSchema.index({ 'metadata.goldman_sachs_rating': 1 });
BlogSchema.index({ 'analytics.professional_views': -1 });
BlogSchema.index({ slug: 1 });

const PremiumBlog = mongoose.model('PremiumBlog', BlogSchema);

class PremiumBlogDatabaseService {
  constructor() {
    this.blogModel = PremiumBlog;
    this.initializeService();
  }

  async initializeService() {
    logger.info('💎 Initializing Premium Blog Database Service...');
    logger.info('📚 Goldman Sachs-quality blog management ready');
    logger.info('🎯 Professional audience targeting enabled');
  }

  /**
   * SAVE GOLDMAN SACHS-QUALITY BLOG
   */
  async savePremiumBlog(blogData) {
    try {
      logger.info(`💎 Saving Goldman Sachs-quality blog: ${blogData.title}`);
      
      // Generate professional slug
      const slug = this.generateProfessionalSlug(blogData.title);
      
      // Enhance with premium features
      const premiumBlogData = {
        ...blogData,
        slug,
        metadata: {
          ...blogData.metadata,
          goldman_sachs_rating: this.calculateGoldmanSachsRating(blogData),
          complexity_score: this.calculateComplexityScore(blogData.content),
          target_audience: 'PROFESSIONAL_INVESTORS',
          research_depth: 'INSTITUTIONAL_SUPERIOR'
        },
        seo: await this.generateProfessionalSEO(blogData),
        professional_features: {
          downloadable_pdf: true,
          executive_summary_pdf: true,
          data_export: true,
          api_access: true,
          institutional_access: true,
          goldman_sachs_exclusive: blogData.metadata?.goldman_sachs_rating === 'SUPERIOR'
        }
      };
      
      const blog = new this.blogModel(premiumBlogData);
      await blog.save();
      
      logger.info('✅ Premium blog saved with institutional-grade features');
      
      return {
        success: true,
        blog_id: blog.id,
        slug: blog.slug,
        goldman_sachs_rating: blog.metadata.goldman_sachs_rating,
        url: `https://sipbrewery.com/blog/${blog.slug}`,
        professional_features_enabled: true
      };
      
    } catch (error) {
      logger.error('Failed to save premium blog:', error);
      throw error;
    }
  }

  /**
   * GET BLOGS FOR GOLDMAN SACHS PROFESSIONALS
   */
  async getBlogsForProfessionals(filters = {}) {
    try {
      const query = {
        'publication.status': 'PUBLISHED',
        'metadata.goldman_sachs_rating': { $in: ['SUPERIOR', 'EXCELLENT'] }
      };
      
      if (filters.type) query.type = filters.type;
      if (filters.date_from) query['publication.published_date'] = { $gte: new Date(filters.date_from) };
      
      const blogs = await this.blogModel
        .find(query)
        .sort({ 'publication.published_date': -1, 'analytics.professional_views': -1 })
        .limit(filters.limit || 20)
        .select({
          id: 1,
          title: 1,
          slug: 1,
          type: 1,
          'content.executive_summary': 1,
          'metadata.goldman_sachs_rating': 1,
          'metadata.reading_time': 1,
          'metadata.complexity_score': 1,
          'analytics.professional_views': 1,
          'analytics.engagement_score': 1,
          'publication.published_date': 1
        });
      
      return {
        success: true,
        blogs,
        total_count: blogs.length,
        professional_quality: 'GOLDMAN_SACHS_GRADE',
        target_audience: 'INSTITUTIONAL_INVESTORS'
      };
      
    } catch (error) {
      logger.error('Failed to get professional blogs:', error);
      throw error;
    }
  }

  /**
   * TRACK GOLDMAN SACHS ENGAGEMENT
   */
  async trackProfessionalEngagement(blogId, engagementData) {
    try {
      const updateData = {
        $inc: {
          'analytics.views': 1,
          'analytics.professional_views': engagementData.is_professional ? 1 : 0,
          'analytics.goldman_sachs_visits': engagementData.is_goldman_sachs ? 1 : 0
        },
        $set: {
          'analytics.last_engagement': new Date()
        }
      };
      
      if (engagementData.time_on_page) {
        updateData.$inc['analytics.time_on_page'] = engagementData.time_on_page;
      }
      
      if (engagementData.shared) {
        updateData.$inc['analytics.share_count'] = 1;
        updateData.$inc['analytics.professional_shares'] = engagementData.is_professional ? 1 : 0;
      }
      
      await this.blogModel.updateOne({ id: blogId }, updateData);
      
      // Calculate engagement score
      await this.updateEngagementScore(blogId);
      
      logger.info(`📊 Professional engagement tracked for blog ${blogId}`);
      
    } catch (error) {
      logger.error('Failed to track professional engagement:', error);
      throw error;
    }
  }

  /**
   * GENERATE DAILY BLOG ANALYTICS
   */
  async generateDailyAnalytics() {
    try {
      logger.info('📊 Generating daily blog analytics...');
      
      const today = new Date();
      const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
      
      const analytics = await this.blogModel.aggregate([
        {
          $match: {
            'publication.published_date': { $gte: yesterday }
          }
        },
        {
          $group: {
            _id: '$type',
            total_blogs: { $sum: 1 },
            total_views: { $sum: '$analytics.views' },
            professional_views: { $sum: '$analytics.professional_views' },
            goldman_sachs_visits: { $sum: '$analytics.goldman_sachs_visits' },
            avg_engagement: { $avg: '$analytics.engagement_score' },
            avg_time_on_page: { $avg: '$analytics.time_on_page' }
          }
        }
      ]);
      
      const overallStats = await this.blogModel.aggregate([
        {
          $group: {
            _id: null,
            total_published: { $sum: 1 },
            superior_rated: {
              $sum: {
                $cond: [{ $eq: ['$metadata.goldman_sachs_rating', 'SUPERIOR'] }, 1, 0]
              }
            },
            total_professional_views: { $sum: '$analytics.professional_views' },
            total_goldman_sachs_visits: { $sum: '$analytics.goldman_sachs_visits' }
          }
        }
      ]);
      
      return {
        date: today.toISOString().split('T')[0],
        daily_analytics: analytics,
        overall_stats: overallStats[0],
        professional_engagement: {
          goldman_sachs_visits: overallStats[0]?.total_goldman_sachs_visits || 0,
          professional_view_ratio: (overallStats[0]?.total_professional_views || 0) / (overallStats[0]?.total_published || 1),
          superior_content_ratio: (overallStats[0]?.superior_rated || 0) / (overallStats[0]?.total_published || 1)
        }
      };
      
    } catch (error) {
      logger.error('Failed to generate daily analytics:', error);
      throw error;
    }
  }

  /**
   * GET TOP PERFORMING BLOGS FOR PROFESSIONALS
   */
  async getTopPerformingBlogs(timeframe = '30d') {
    try {
      const dateFilter = this.getDateFilter(timeframe);
      
      const topBlogs = await this.blogModel
        .find({
          'publication.status': 'PUBLISHED',
          'publication.published_date': dateFilter
        })
        .sort({
          'analytics.professional_views': -1,
          'analytics.engagement_score': -1,
          'analytics.goldman_sachs_visits': -1
        })
        .limit(10)
        .select({
          title: 1,
          slug: 1,
          type: 1,
          'metadata.goldman_sachs_rating': 1,
          'analytics.professional_views': 1,
          'analytics.goldman_sachs_visits': 1,
          'analytics.engagement_score': 1,
          'publication.published_date': 1
        });
      
      return {
        success: true,
        timeframe,
        top_blogs: topBlogs,
        professional_quality: 'INSTITUTIONAL_GRADE',
        goldman_sachs_appeal: 'MAXIMUM'
      };
      
    } catch (error) {
      logger.error('Failed to get top performing blogs:', error);
      throw error;
    }
  }

  // Helper methods
  generateProfessionalSlug(title) {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .substring(0, 100);
  }

  calculateGoldmanSachsRating(blogData) {
    let score = 0;
    
    // Content depth
    if (blogData.content?.main_content?.length > 5000) score += 30;
    else if (blogData.content?.main_content?.length > 3000) score += 20;
    else score += 10;
    
    // ASI insights
    if (blogData.content?.asi_predictions) score += 25;
    
    // Interactive elements
    if (blogData.interactive_elements?.charts?.length > 0) score += 20;
    if (blogData.interactive_elements?.tables?.length > 0) score += 15;
    
    // Data sources
    if (blogData.metadata?.data_sources?.length > 5) score += 10;
    
    if (score >= 85) return 'SUPERIOR';
    if (score >= 70) return 'EXCELLENT';
    return 'GOOD';
  }

  calculateComplexityScore(content) {
    if (!content?.main_content) return 0;
    
    const text = content.main_content;
    const sentences = text.split(/[.!?]+/).length;
    const words = text.split(/\s+/).length;
    const avgWordsPerSentence = words / sentences;
    
    // Complexity based on sentence length and financial terminology
    const financialTerms = ['valuation', 'dcf', 'ebitda', 'roe', 'pe ratio', 'beta', 'alpha'];
    const termCount = financialTerms.reduce((count, term) => {
      return count + (text.toLowerCase().includes(term) ? 1 : 0);
    }, 0);
    
    return Math.min((avgWordsPerSentence * 2 + termCount * 5) / 10, 10);
  }

  async generateProfessionalSEO(blogData) {
    return {
      meta_title: `${blogData.title} | SIP Brewery Professional Analysis`,
      meta_description: blogData.content?.executive_summary?.substring(0, 160) || '',
      keywords: this.extractKeywords(blogData),
      professional_tags: ['institutional-analysis', 'goldman-sachs-quality', 'asi-powered'],
      linkedin_optimized: true,
      goldman_sachs_keywords: ['institutional', 'professional', 'analysis', 'investment', 'financial']
    };
  }

  extractKeywords(blogData) {
    const commonKeywords = ['investment', 'analysis', 'ipo', 'mutual fund', 'market', 'financial'];
    const titleWords = blogData.title.toLowerCase().split(/\s+/);
    return [...new Set([...commonKeywords, ...titleWords])].slice(0, 10);
  }

  async updateEngagementScore(blogId) {
    const blog = await this.blogModel.findOne({ id: blogId });
    if (!blog) return;
    
    const engagementScore = (
      (blog.analytics.professional_views * 2) +
      (blog.analytics.goldman_sachs_visits * 5) +
      (blog.analytics.share_count * 3) +
      (blog.analytics.time_on_page / 1000)
    ) / 10;
    
    await this.blogModel.updateOne(
      { id: blogId },
      { $set: { 'analytics.engagement_score': Math.round(engagementScore) } }
    );
  }

  getDateFilter(timeframe) {
    const now = new Date();
    const days = parseInt(timeframe.replace('d', ''));
    return { $gte: new Date(now.getTime() - days * 24 * 60 * 60 * 1000) };
  }
}

module.exports = { PremiumBlogDatabaseService, PremiumBlog };
