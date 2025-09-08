const logger = require('../utils/logger');
const axios = require('axios');

class OllamaService {
  constructor() {
    this.baseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
    this.model = process.env.OLLAMA_MODEL || 'mistral';
    this.timeout = parseInt(process.env.OLLAMA_TIMEOUT) || 30000;
    this.maxRetries = parseInt(process.env.OLLAMA_MAX_RETRIES) || 3;
  }

  /**
   * Generate response using Ollama Mistral model
   */
  async generateResponse(prompt, options = {}) {
    try {
      logger.info('Generating Ollama response', { 
        model: this.model, 
        promptLength: prompt.length,
        options: Object.keys(options)
      });

      const requestData = {
        model: this.model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: options.temperature || 0.7,
          top_p: options.top_p || 0.9,
          top_k: options.top_k || 40,
          repeat_penalty: options.repeat_penalty || 1.1,
          max_tokens: options.max_tokens || 2048
        }
      };

      const response = await this.makeRequest(requestData);
      
      if (response && response.response) {
        logger.info('Ollama response generated successfully', { 
          responseLength: response.response.length 
        });
        return response.response;
      } else {
        throw new Error('Invalid response format from Ollama');
      }
    } catch (error) {
      logger.error('Failed to generate Ollama response', { 
        error: error.message, 
        prompt: prompt.substring(0, 100) + '...' 
      });
      
      // Return fallback response for testing
      return this.generateFallbackResponse(prompt, options);
    }
  }

  /**
   * Make HTTP request to Ollama API with retry logic
   */
  async makeRequest(requestData, retryCount = 0) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/api/generate`,
        requestData,
        {
          timeout: this.timeout,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      return response.data;
    } catch (error) {
      if (retryCount < this.maxRetries && this.isRetryableError(error)) {
        logger.warn('Ollama request failed, retrying', { 
          retryCount: retryCount + 1, 
          error: error.message 
        });
        
        // Exponential backoff
        await this.delay(Math.pow(2, retryCount) * 1000);
        return this.makeRequest(requestData, retryCount + 1);
      }
      
      throw error;
    }
  }

  /**
   * Check if error is retryable
   */
  isRetryableError(error) {
    return (
      error.code === 'ECONNRESET' ||
      error.code === 'ETIMEDOUT' ||
      error.code === 'ENOTFOUND' ||
      (error.response && error.response.status >= 500)
    );
  }

  /**
   * Delay function for retry backoff
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate fallback response for testing when Ollama is not available
   */
  generateFallbackResponse(prompt, options = {}) {
    logger.info('Generating fallback response for testing');
    
    // Simple fallback responses based on prompt content
    if (prompt.includes('portfolio') && prompt.includes('recommendation')) {
      return `Based on your portfolio analysis, you may consider:
1. Consider rebalancing your equity allocation to 60%
2. Add some debt funds for stability
3. Review your SIP amounts monthly
4. Monitor fund performance quarterly`;
    }
    
    if (prompt.includes('tax') && prompt.includes('optimization')) {
      return `Tax optimization recommendations:
1. Consider ELSS funds for tax savings under Section 80C
2. Monitor LTCG gains and plan exits strategically
3. Use tax-loss harvesting for underperforming funds
4. Optimize SIP timing for tax efficiency`;
    }
    
    if (prompt.includes('risk') && prompt.includes('management')) {
      return `Risk management information:
1. Diversify across asset classes
2. Maintain emergency fund equivalent to 6 months expenses
3. Review risk tolerance annually
4. Consider insurance coverage adequacy`;
    }
    
    if (prompt.includes('learning') && prompt.includes('path')) {
      return `Personalized learning path:
1. Start with basic mutual fund concepts
2. Learn about different fund categories
3. Understand risk and return relationship
4. Practice with virtual portfolio
5. Advanced topics: tax optimization and rebalancing`;
    }
    
    if (prompt.includes('financial') && prompt.includes('planning')) {
      return `Financial planning guidance:
1. Set clear financial goals with timelines
2. Calculate required corpus for each goal
3. Choose appropriate investment vehicles
4. Regular review and adjustment of plans
5. Consider inflation and tax implications`;
    }
    
    if (prompt.includes('insurance') && prompt.includes('coverage')) {
      return `Insurance coverage analysis:
1. Life insurance: 10-15 times annual income
2. Health insurance: Comprehensive coverage for family
3. Critical illness: Additional protection
4. Disability insurance: Income protection
5. Review coverage every 2-3 years`;
    }
    
    // Default fallback response
    return `Thank you for your query. I'm here to help with your mutual fund investments. 
Please provide more specific details about what you'd like to know, and I'll provide 
personalized recommendations based on your financial goals and risk profile.`;
  }

  /**
   * Check if Ollama service is available
   */
  async checkHealth() {
    try {
      const response = await axios.get(`${this.baseUrl}/api/tags`, {
        timeout: 5000
      });
      
      return {
        available: true,
        models: response.data.models || [],
        baseUrl: this.baseUrl
      };
    } catch (error) {
      logger.warn('Ollama service not available', { error: error.message });
      return {
        available: false,
        error: error.message,
        baseUrl: this.baseUrl
      };
    }
  }

  /**
   * Get available models
   */
  async getAvailableModels() {
    try {
      const response = await axios.get(`${this.baseUrl}/api/tags`, {
        timeout: 5000
      });
      
      return response.data.models || [];
    } catch (error) {
      logger.error('Failed to get available models', { error: error.message });
      return [];
    }
  }

  /**
   * Generate structured response for specific use cases
   */
  async generateStructuredResponse(prompt, structure, options = {}) {
    try {
      const structuredPrompt = `${prompt}\n\nPlease respond in the following JSON structure:\n${JSON.stringify(structure, null, 2)}`;
      
      const response = await this.generateResponse(structuredPrompt, {
        ...options,
        temperature: 0.3, // Lower temperature for more consistent structured output
        max_tokens: options.max_tokens || 4096
      });
      
      // Try to parse JSON response
      try {
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          return JSON.parse(jsonMatch[0]);
        }
      } catch (parseError) {
        logger.warn('Failed to parse structured response as JSON', { parseError: parseError.message });
      }
      
      return {
        raw_response: response,
        structured: false
      };
    } catch (error) {
      logger.error('Failed to generate structured response', { error: error.message });
      return {
        error: error.message,
        structured: false
      };
    }
  }

  /**
   * Generate multiple responses for comparison
   */
  async generateMultipleResponses(prompt, count = 3, options = {}) {
    const responses = [];
    
    for (let i = 0; i < count; i++) {
      try {
        const response = await this.generateResponse(prompt, {
          ...options,
          temperature: (options.temperature || 0.7) + (i * 0.1) // Vary temperature for diversity
        });
        
        responses.push({
          index: i,
          response: response,
          temperature: (options.temperature || 0.7) + (i * 0.1)
        });
      } catch (error) {
        logger.warn(`Failed to generate response ${i + 1}`, { error: error.message });
      }
    }
    
    return responses;
  }
}

module.exports = new OllamaService(); 