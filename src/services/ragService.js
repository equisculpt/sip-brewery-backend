const axios = require('axios');
const logger = require('../utils/logger');
const fs = require('fs').promises;
const path = require('path');

class RAGService {
  constructor() {
    this.ollamaBaseUrl = 'http://localhost:11434';
    this.vectorDb = new Map(); // In-memory vector store (replace with ChromaDB/Weaviate in production)
    this.embeddings = new Map();
    this.documents = [];
  }

  /**
   * Initialize RAG service
   */
  async initialize() {
    try {
      // Test Ollama connection
      await this.testOllamaConnection();
      
      // Load existing documents
      await this.loadDocuments();
      
      logger.info('RAG Service initialized successfully');
      return true;
    } catch (error) {
      logger.error('Error initializing RAG service:', error);
      return false;
    }
  }

  /**
   * Test Ollama connection
   */
  async testOllamaConnection() {
    try {
      const response = await axios.get(`${this.ollamaBaseUrl}/api/tags`);
      logger.info('Ollama connection successful');
      return response.data;
    } catch (error) {
      logger.error('Ollama connection failed:', error.message);
      throw new Error('Ollama service not available');
    }
  }

  /**
   * Generate embeddings using Ollama
   */
  async generateEmbedding(text) {
    try {
      const response = await axios.post(`${this.ollamaBaseUrl}/api/embeddings`, {
        model: 'mistral',
        prompt: text
      });

      return response.data.embedding;
    } catch (error) {
      logger.error('Error generating embedding:', error);
      // Fallback to simple hash-based embedding
      return this.simpleEmbedding(text);
    }
  }

  /**
   * Simple fallback embedding (for when Ollama is not available)
   */
  simpleEmbedding(text) {
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(384).fill(0);
    
    words.forEach((word, index) => {
      const hash = this.hashString(word);
      embedding[index % 384] += hash;
    });
    
    // Normalize
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / magnitude);
  }

  /**
   * Simple hash function
   */
  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  cosineSimilarity(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error('Vector dimensions must match');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Add document to vector database
   */
  async addDocument(document) {
    try {
      const { id, content, metadata = {} } = document;
      
      // Generate embedding
      const embedding = await this.generateEmbedding(content);
      
      // Store document and embedding
      this.documents.push({
        id,
        content,
        metadata,
        embedding
      });

      // Index by category for faster retrieval
      const category = metadata.category || 'general';
      if (!this.vectorDb.has(category)) {
        this.vectorDb.set(category, []);
      }
      this.vectorDb.get(category).push({
        id,
        content,
        metadata,
        embedding
      });

      logger.info(`Added document to RAG: ${id}`);
      return true;
    } catch (error) {
      logger.error('Error adding document to RAG:', error);
      return false;
    }
  }

  /**
   * Search for relevant documents
   */
  async searchDocuments(query, category = null, limit = 5) {
    try {
      const queryEmbedding = await this.generateEmbedding(query);
      const results = [];

      // Search in specific category or all categories
      const searchCategories = category ? [category] : Array.from(this.vectorDb.keys());

      for (const cat of searchCategories) {
        const documents = this.vectorDb.get(cat) || [];
        
        for (const doc of documents) {
          const similarity = this.cosineSimilarity(queryEmbedding, doc.embedding);
          results.push({
            ...doc,
            similarity,
            category: cat
          });
        }
      }

      // Sort by similarity and return top results
      results.sort((a, b) => b.similarity - a.similarity);
      return results.slice(0, limit);
    } catch (error) {
      logger.error('Error searching documents:', error);
      return [];
    }
  }

  /**
   * Generate answer using RAG
   */
  async generateAnswer(query, context = null) {
    try {
      // Search for relevant documents
      const relevantDocs = await this.searchDocuments(query);
      
      // Build context from relevant documents
      const contextText = context || relevantDocs
        .map(doc => doc.content)
        .join('\n\n');

      // Create prompt with context
      const prompt = this.buildRAGPrompt(query, contextText);

      // Generate answer using Ollama
      const response = await axios.post(`${this.ollamaBaseUrl}/api/generate`, {
        model: 'mistral',
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 1000
        }
      });

      const answer = response.data.response;
      
      logger.info(`Generated RAG answer for query: ${query.substring(0, 50)}...`);
      
      return {
        answer,
        sources: relevantDocs.map(doc => ({
          id: doc.id,
          title: doc.metadata.title,
          category: doc.category,
          similarity: doc.similarity
        })),
        context: contextText.substring(0, 500) + '...'
      };
    } catch (error) {
      logger.error('Error generating RAG answer:', error);
      
      // Fallback to simple keyword-based response
      return this.generateFallbackAnswer(query);
    }
  }

  /**
   * Build RAG prompt with context
   */
  buildRAGPrompt(query, context) {
    return `You are a financial advisor specializing in Indian mutual funds. Use the following context to answer the user's question accurately and helpfully.

Context:
${context}

User Question: ${query}

Instructions:
1. Answer based on the provided context
2. Focus only on Indian mutual funds and SEBI regulations
3. Do not provide investment advice, only educational information
4. If the context doesn't contain relevant information, say so
5. Be clear, concise, and accurate
6. Include relevant SEBI guidelines when applicable

Answer:`;
  }

  /**
   * Fallback answer generation
   */
  generateFallbackAnswer(query) {
    const fallbackResponses = {
      'xirr': 'XIRR (Extended Internal Rate of Return) is used to calculate returns on SIP investments with irregular cash flows. It provides a more accurate measure of actual returns compared to simple returns.',
      'nav': 'NAV (Net Asset Value) is the per-unit market value of a mutual fund scheme, calculated daily for open-ended funds.',
      'sip': 'SIP (Systematic Investment Plan) allows regular investments in mutual funds, providing benefits like rupee cost averaging and disciplined investing.',
      'elss': 'ELSS (Equity Linked Savings Scheme) offers tax deduction under Section 80C up to ₹1.5 lakh with a 3-year lock-in period.',
      'sebi': 'SEBI regulates mutual funds in India, ensuring investor protection and market integrity through various guidelines and regulations.',
      'tax': 'Mutual fund gains are taxed based on fund type and holding period. Equity funds have LTCG at 10% after 1 year, while debt funds have different tax structures.'
    };

    const queryLower = query.toLowerCase();
    for (const [keyword, response] of Object.entries(fallbackResponses)) {
      if (queryLower.includes(keyword)) {
        return {
          answer: response,
          sources: [],
          context: 'Fallback response based on keyword matching'
        };
      }
    }

    return {
      answer: 'I apologize, but I need more specific information about Indian mutual funds to provide an accurate answer. Please ask about topics like NAV, SIP, XIRR, ELSS, SEBI regulations, or mutual fund taxation.',
      sources: [],
      context: 'No relevant context found'
    };
  }

  /**
   * Load documents from storage
   */
  async loadDocuments() {
    try {
      const documentsPath = path.join(__dirname, '../../training-data/documents.json');
      
      try {
        const data = await fs.readFile(documentsPath, 'utf8');
        const documents = JSON.parse(data);
        
        for (const doc of documents) {
          await this.addDocument(doc);
        }
        
        logger.info(`Loaded ${documents.length} documents into RAG`);
      } catch (error) {
        logger.info('No existing documents found, starting with empty RAG');
      }
    } catch (error) {
      logger.error('Error loading documents:', error);
    }
  }

  /**
   * Save documents to storage
   */
  async saveDocuments() {
    try {
      const documentsPath = path.join(__dirname, '../../training-data/documents.json');
      const documentsData = this.documents.map(doc => ({
        id: doc.id,
        content: doc.content,
        metadata: doc.metadata
      }));
      
      await fs.writeFile(documentsPath, JSON.stringify(documentsData, null, 2));
      logger.info(`Saved ${documentsData.length} documents to storage`);
    } catch (error) {
      logger.error('Error saving documents:', error);
    }
  }

  /**
   * Add SEBI circulars and regulations
   */
  async addSEBIDocuments() {
    const sebiDocs = [
      {
        id: 'sebi-mf-regulations',
        content: 'SEBI Mutual Fund Regulations cover fund structure, investment limits, disclosure requirements, and investor protection measures. All mutual funds must comply with SEBI guidelines for transparency and fair practices.',
        metadata: {
          title: 'SEBI Mutual Fund Regulations',
          category: 'sebi',
          source: 'SEBI Guidelines',
          date: '2024'
        }
      },
      {
        id: 'sebi-expense-ratio-caps',
        content: 'SEBI has capped mutual fund expense ratios: Equity funds maximum 2.5%, Debt funds maximum 2.25%, Index funds maximum 1.5%, and ETFs maximum 1%. These limits ensure reasonable costs for investors.',
        metadata: {
          title: 'SEBI Expense Ratio Caps',
          category: 'sebi',
          source: 'SEBI Circular',
          date: '2024'
        }
      },
      {
        id: 'sebi-kyc-requirements',
        content: 'SEBI requires KYC for all mutual fund investments. This includes PAN card, Aadhaar, address proof, and bank account details. KYC is mandatory for investments above ₹50,000.',
        metadata: {
          title: 'SEBI KYC Requirements',
          category: 'sebi',
          source: 'SEBI Guidelines',
          date: '2024'
        }
      }
    ];

    for (const doc of sebiDocs) {
      await this.addDocument(doc);
    }
  }

  /**
   * Add mutual fund educational content
   */
  async addMFEducationalContent() {
    const mfDocs = [
      {
        id: 'mf-basics',
        content: 'Mutual funds pool money from multiple investors to invest in diversified portfolios. They offer professional management, diversification, and various investment options suitable for different risk profiles and goals.',
        metadata: {
          title: 'Mutual Fund Basics',
          category: 'education',
          source: 'AMFI Guidelines',
          date: '2024'
        }
      },
      {
        id: 'sip-benefits',
        content: 'SIP benefits include rupee cost averaging, disciplined investing, power of compounding, lower minimum investments, and reduced market timing risk. SIPs are ideal for long-term wealth creation.',
        metadata: {
          title: 'SIP Benefits',
          category: 'education',
          source: 'AMFI Guidelines',
          date: '2024'
        }
      },
      {
        id: 'fund-categories',
        content: 'Indian mutual funds include equity funds (large, mid, small cap), debt funds (gilt, corporate bond), hybrid funds, solution-oriented funds (ELSS), and sectoral funds. Each category has different risk-return profiles.',
        metadata: {
          title: 'Fund Categories',
          category: 'education',
          source: 'AMFI Guidelines',
          date: '2024'
        }
      }
    ];

    for (const doc of mfDocs) {
      await this.addDocument(doc);
    }
  }

  /**
   * Get RAG statistics
   */
  getRAGStats() {
    const stats = {
      totalDocuments: this.documents.length,
      categories: {},
      totalEmbeddings: this.embeddings.size
    };

    for (const [category, docs] of this.vectorDb.entries()) {
      stats.categories[category] = docs.length;
    }

    return stats;
  }
}

module.exports = RAGService; 