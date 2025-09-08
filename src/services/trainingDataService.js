const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');
const axios = require('axios');

class TrainingDataService {
  constructor() {
    this.trainingDataPath = path.join(__dirname, '../../training-data');
    this.categories = {
      mutualFunds: 'fund-portfolios',
      sebiRules: 'sebi-compliance',
      taxation: 'tax-rules',
      userQueries: 'user-qa',
      marketAnalysis: 'market-data'
    };
  }

  /**
   * Initialize training data structure
   */
  async initializeTrainingData() {
    try {
      // Create main training data directory
      await fs.mkdir(this.trainingDataPath, { recursive: true });
      
      // Create category directories
      for (const [category, dirName] of Object.entries(this.categories)) {
        const categoryPath = path.join(this.trainingDataPath, dirName);
        await fs.mkdir(categoryPath, { recursive: true });
        logger.info(`Created training data directory: ${dirName}`);
      }

      // Create JSONL output directory
      const jsonlPath = path.join(this.trainingDataPath, 'jsonl-output');
      await fs.mkdir(jsonlPath, { recursive: true });

      logger.info('Training data structure initialized successfully');
      return true;
    } catch (error) {
      logger.error('Error initializing training data:', error);
      return false;
    }
  }

  /**
   * Generate comprehensive Indian mutual fund Q&A dataset
   */
  async generateMFQADataset() {
    const qaData = [
      // Basic Mutual Fund Concepts
      {
        question: "What is a mutual fund?",
        answer: "A mutual fund is a type of investment vehicle that pools money from multiple investors to invest in a diversified portfolio of stocks, bonds, or other securities. In India, mutual funds are regulated by SEBI and offer various categories like equity, debt, hybrid, and solution-oriented funds."
      },
      {
        question: "What is NAV in mutual funds?",
        answer: "NAV (Net Asset Value) is the per-unit market value of a mutual fund scheme. It is calculated by dividing the total net assets of the fund by the total number of units outstanding. NAV is declared daily for open-ended funds and is used for buying and selling fund units."
      },
      {
        question: "What is the difference between direct and regular plans?",
        answer: "Direct plans are bought directly from the fund house without intermediaries, resulting in lower expense ratios and higher returns. Regular plans are bought through distributors/agents who charge commission, leading to higher expense ratios. Direct plans typically offer 0.5-1% better returns due to lower costs."
      },
      {
        question: "What is XIRR in SIP investments?",
        answer: "XIRR (Extended Internal Rate of Return) is a method to calculate returns on investments with irregular cash flows, commonly used for SIPs. It accounts for the timing of each investment and provides a more accurate measure of actual returns compared to simple returns."
      },
      {
        question: "What are the different types of mutual funds in India?",
        answer: "Indian mutual funds are categorized into: 1) Equity funds (Large, Mid, Small cap), 2) Debt funds (Gilt, Corporate bond, Liquid), 3) Hybrid funds (Balanced, Conservative), 4) Solution-oriented funds (ELSS, Children's, Retirement), 5) International funds, and 6) Sectoral/Thematic funds."
      },
      {
        question: "What is ELSS and its tax benefits?",
        answer: "ELSS (Equity Linked Savings Scheme) is a tax-saving mutual fund that offers tax deduction under Section 80C up to ₹1.5 lakh per year. It has a 3-year lock-in period and invests primarily in equity. ELSS typically offers better returns than traditional tax-saving instruments like PPF or FD."
      },
      {
        question: "How is expense ratio calculated?",
        answer: "Expense ratio is the annual fee charged by the fund house to manage the mutual fund, expressed as a percentage of the fund's average net assets. It includes management fees, administrative costs, and other operational expenses. Lower expense ratios generally lead to better returns for investors."
      },
      {
        question: "What is the difference between growth and dividend options?",
        answer: "Growth option reinvests profits back into the fund, increasing the NAV and unit value over time. Dividend option distributes profits to investors periodically. Growth option is better for long-term wealth creation, while dividend option provides regular income but may have tax implications."
      },
      {
        question: "What is SIP and its benefits?",
        answer: "SIP (Systematic Investment Plan) allows investors to invest a fixed amount regularly in mutual funds. Benefits include: 1) Rupee cost averaging, 2) Disciplined investing, 3) Power of compounding, 4) Lower minimum investment amounts, 5) Reduces market timing risk."
      },
      {
        question: "How to calculate mutual fund returns?",
        answer: "Mutual fund returns can be calculated using: 1) Absolute returns (simple percentage change), 2) CAGR (Compound Annual Growth Rate) for long-term performance, 3) XIRR for SIP investments, 4) Rolling returns for consistency analysis. Always consider expense ratios and taxes for actual returns."
      },
      {
        question: "What are the tax implications of mutual fund investments?",
        answer: "Equity funds: LTCG tax of 10% after 1 year, STCG tax of 15% for less than 1 year. Debt funds: LTCG at 20% with indexation after 3 years, STCG at slab rate for less than 3 years. Dividend income is taxable at slab rates. ELSS offers 80C deduction."
      },
      {
        question: "What is the role of SEBI in mutual funds?",
        answer: "SEBI (Securities and Exchange Board of India) regulates mutual funds in India. It ensures investor protection, sets disclosure norms, monitors fund performance, approves fund launches, and maintains market integrity. SEBI guidelines cover fund structure, investment limits, and risk management."
      },
      {
        question: "How to choose the right mutual fund?",
        answer: "Consider: 1) Investment goals and time horizon, 2) Risk tolerance, 3) Fund category and investment style, 4) Past performance and consistency, 5) Expense ratio, 6) Fund manager track record, 7) Fund house reputation, 8) Asset under management (AUM)."
      },
      {
        question: "What is asset allocation in mutual funds?",
        answer: "Asset allocation is the distribution of investments across different asset classes (equity, debt, gold, etc.) based on risk profile and investment goals. It helps in risk diversification and optimal returns. Younger investors can have higher equity allocation, while older investors should prefer debt-heavy allocation."
      },
      {
        question: "What is rebalancing in mutual fund portfolios?",
        answer: "Rebalancing is adjusting portfolio allocation back to target weights when market movements cause deviations. It helps maintain risk profile and can improve returns by selling high and buying low. Should be done annually or when allocation drifts by more than 5-10%."
      },
      {
        question: "What are the risks in mutual fund investments?",
        answer: "Key risks include: 1) Market risk (equity funds), 2) Interest rate risk (debt funds), 3) Credit risk (corporate bonds), 4) Liquidity risk, 5) Concentration risk, 6) Fund manager risk, 7) Regulatory risk. Diversification helps mitigate these risks."
      },
      {
        question: "What is the difference between index funds and actively managed funds?",
        answer: "Index funds passively track market indices (like Nifty 50) with minimal management, resulting in lower expense ratios. Actively managed funds have fund managers making investment decisions to outperform the market, but with higher costs. Index funds often outperform active funds over long periods."
      },
      {
        question: "How does rupee cost averaging work in SIPs?",
        answer: "Rupee cost averaging reduces average purchase price by buying more units when prices are low and fewer units when prices are high. This happens automatically in SIPs as you invest the same amount regularly, regardless of market conditions, potentially leading to better average returns."
      },
      {
        question: "What is the power of compounding in mutual funds?",
        answer: "Compounding is earning returns on both principal and accumulated returns. In mutual funds, reinvested dividends and capital gains compound over time, leading to exponential growth. The longer the investment horizon, the more powerful the compounding effect becomes."
      },
      {
        question: "What are the exit load charges in mutual funds?",
        answer: "Exit load is a fee charged when redeeming fund units within a specified period (usually 1-3 years). It discourages short-term trading and compensates the fund for transaction costs. Exit loads typically range from 0.5% to 2% and decrease over time."
      },
      {
        question: "How to track mutual fund performance?",
        answer: "Track performance using: 1) Fund house statements, 2) Online portals (AMFI, Value Research), 3) Mobile apps, 4) Regular NAV updates, 5) Benchmark comparisons, 6) Peer group analysis, 7) Rolling returns for consistency evaluation."
      }
    ];

    return qaData;
  }

  /**
   * Generate SEBI compliance Q&A
   */
  async generateSEBIComplianceData() {
    const sebiData = [
      {
        question: "What are SEBI's KYC requirements for mutual fund investments?",
        answer: "SEBI requires KYC (Know Your Customer) for all mutual fund investments. This includes PAN card, Aadhaar, address proof, and bank account details. KYC can be done online through KRA (KYC Registration Agency) or offline through fund houses. KYC is mandatory for investments above ₹50,000."
      },
      {
        question: "What are the SEBI guidelines for mutual fund expense ratios?",
        answer: "SEBI has capped expense ratios based on fund type: Equity funds max 2.5%, Debt funds max 2.25%, Index funds max 1.5%, and ETFs max 1%. For funds with AUM above ₹500 crore, additional caps apply. These limits ensure reasonable costs for investors."
      },
      {
        question: "What is SEBI's role in protecting mutual fund investors?",
        answer: "SEBI protects investors through: 1) Regulation of fund houses and distributors, 2) Mandatory disclosures and transparency, 3) Grievance redressal mechanisms, 4) Investor education initiatives, 5) Monitoring fund performance and compliance, 6) Setting investment limits and risk management norms."
      },
      {
        question: "What are the SEBI guidelines for mutual fund advertising?",
        answer: "SEBI guidelines require: 1) Past performance disclaimers, 2) Risk warnings, 3) No guaranteed returns, 4) Fair comparison with benchmarks, 5) Disclosure of expense ratios, 6) No misleading claims about returns or safety. All advertisements must be pre-approved by fund houses."
      },
      {
        question: "What is SEBI's stance on mutual fund commission structures?",
        answer: "SEBI has abolished upfront commissions and introduced trail commission model. Distributors earn based on assets under management (AUM) rather than transaction volumes. This aligns distributor interests with long-term investor benefits and reduces mis-selling."
      }
    ];

    return sebiData;
  }

  /**
   * Generate taxation Q&A
   */
  async generateTaxationData() {
    const taxData = [
      {
        question: "How are mutual fund capital gains taxed in India?",
        answer: "Equity funds: LTCG (Long Term Capital Gains) of 10% after 1 year, STCG (Short Term Capital Gains) of 15% for less than 1 year. Debt funds: LTCG at 20% with indexation after 3 years, STCG at individual slab rates for less than 3 years. No tax on gains up to ₹1 lakh for equity funds."
      },
      {
        question: "What is indexation benefit in debt mutual funds?",
        answer: "Indexation adjusts the purchase price for inflation using Cost Inflation Index (CII). This reduces taxable gains and lowers tax liability for long-term debt fund investments. Indexation benefit is available only for investments held for more than 3 years."
      },
      {
        question: "How are mutual fund dividends taxed?",
        answer: "Dividend income from mutual funds is taxable at individual slab rates. Fund houses deduct TDS at 10% if dividend exceeds ₹5,000. Investors can claim credit for TDS while filing income tax returns. Dividend option may be less tax-efficient than growth option for high-tax-bracket investors."
      },
      {
        question: "What are the tax benefits of ELSS funds?",
        answer: "ELSS offers tax deduction under Section 80C up to ₹1.5 lakh per year. It has a 3-year lock-in period and invests primarily in equity. LTCG tax of 10% applies after 1 year, with ₹1 lakh exemption. ELSS is one of the most tax-efficient investment options available."
      },
      {
        question: "How to calculate tax on mutual fund redemptions?",
        answer: "Calculate gains by subtracting purchase price from redemption price. For equity funds: apply 10% LTCG after 1 year or 15% STCG for less than 1 year. For debt funds: apply 20% LTCG with indexation after 3 years or slab rate STCG for less than 3 years. Consider ₹1 lakh exemption for equity funds."
      }
    ];

    return taxData;
  }

  /**
   * Convert Q&A data to JSONL format for fine-tuning
   */
  async convertToJSONL(qaData, outputFile) {
    try {
      const jsonlPath = path.join(this.trainingDataPath, 'jsonl-output', outputFile);
      const jsonlContent = qaData.map(qa => 
        JSON.stringify({ role: "user", content: qa.question }) + '\n' +
        JSON.stringify({ role: "assistant", content: qa.answer })
      ).join('\n');

      await fs.writeFile(jsonlPath, jsonlContent, 'utf8');
      logger.info(`Generated JSONL file: ${outputFile} with ${qaData.length} Q&A pairs`);
      return jsonlPath;
    } catch (error) {
      logger.error('Error converting to JSONL:', error);
      throw error;
    }
  }

  /**
   * Generate complete training dataset
   */
  async generateCompleteDataset() {
    try {
      logger.info('Starting complete training dataset generation...');

      // Initialize directory structure
      await this.initializeTrainingData();

      // Generate different categories of data
      const mfData = await this.generateMFQADataset();
      const sebiData = await this.generateSEBIComplianceData();
      const taxData = await this.generateTaxationData();

      // Convert to JSONL format
      await this.convertToJSONL(mfData, 'mutual_fund_qa.jsonl');
      await this.convertToJSONL(sebiData, 'sebi_compliance.jsonl');
      await this.convertToJSONL(taxData, 'taxation_rules.jsonl');

      // Combine all data
      const allData = [...mfData, ...sebiData, ...taxData];
      await this.convertToJSONL(allData, 'complete_training_data.jsonl');

      logger.info(`Generated complete training dataset with ${allData.length} Q&A pairs`);
      return {
        totalPairs: allData.length,
        categories: {
          mutualFunds: mfData.length,
          sebiCompliance: sebiData.length,
          taxation: taxData.length
        },
        files: [
          'mutual_fund_qa.jsonl',
          'sebi_compliance.jsonl',
          'taxation_rules.jsonl',
          'complete_training_data.jsonl'
        ]
      };
    } catch (error) {
      logger.error('Error generating complete dataset:', error);
      throw error;
    }
  }

  /**
   * Validate training data quality
   */
  async validateTrainingData(jsonlFile) {
    try {
      const filePath = path.join(this.trainingDataPath, 'jsonl-output', jsonlFile);
      const content = await fs.readFile(filePath, 'utf8');
      const lines = content.trim().split('\n');

      let userCount = 0;
      let assistantCount = 0;
      let errors = [];

      for (let i = 0; i < lines.length; i++) {
        try {
          const line = JSON.parse(lines[i]);
          if (line.role === 'user') userCount++;
          if (line.role === 'assistant') assistantCount++;

          // Validate content
          if (!line.content || line.content.length < 10) {
            errors.push(`Line ${i + 1}: Content too short`);
          }
        } catch (parseError) {
          errors.push(`Line ${i + 1}: Invalid JSON`);
        }
      }

      const validation = {
        totalLines: lines.length,
        userMessages: userCount,
        assistantMessages: assistantCount,
        pairs: Math.min(userCount, assistantCount),
        errors: errors,
        isValid: errors.length === 0 && userCount === assistantCount
      };

      logger.info(`Training data validation for ${jsonlFile}:`, validation);
      return validation;
    } catch (error) {
      logger.error('Error validating training data:', error);
      throw error;
    }
  }
}

module.exports = TrainingDataService; 