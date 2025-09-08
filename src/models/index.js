const BenchmarkIndex = require('./BenchmarkIndex');
const User = require('./User');
const Holding = require('./Holding');
const Transaction = require('./Transaction');
const Reward = require('./Reward');
const RewardSummary = require('./RewardSummary');
const Referral = require('./Referral');
const AIInsight = require('./AIInsight');
const SmartSip = require('./SmartSip');
const UserPortfolio = require('./UserPortfolio');
const Leaderboard = require('./Leaderboard');
const PortfolioCopy = require('./PortfolioCopy');
const WhatsAppSession = require('./WhatsAppSession');
const SipOrder = require('./SipOrder');
const WhatsAppMessage = require('./WhatsAppMessage');
const AGIInsight = require('./AGIInsight');
const UserBehavior = require('./UserBehavior');
const MarketData = require('./MarketData');
const EconomicIndicator = require('./EconomicIndicator');

if (process.env.NODE_ENV === 'test') {
  module.exports = {
    BenchmarkIndex: require('./BenchmarkIndex.mock'),
    User: require('./User.mock'),
    Holding: require('./Holding.mock'),
    Transaction: require('./Transaction.mock'),
    Reward: require('./Reward.mock'),
    RewardSummary: require('./RewardSummary.mock'),
    Referral: require('./Referral.mock'),
    AIInsight: require('./AIInsight.mock'),
    SmartSip: require('./SmartSip.mock'),
    UserPortfolio: require('./UserPortfolio.mock'),
    Leaderboard: require('./Leaderboard.mock'),
    PortfolioCopy: require('./PortfolioCopy.mock'),
    WhatsAppSession: require('./WhatsAppSession.mock'),
    SipOrder: require('./SipOrder.mock'),
    WhatsAppMessage: require('./WhatsAppMessage.mock'),
    AGIInsight: require('./AGIInsight.mock'),
    UserBehavior: require('./UserBehavior.mock'),
    MarketData: require('./MarketData.mock'),
    EconomicIndicator: require('./EconomicIndicator.mock')
  };
} else {
  module.exports = {
    BenchmarkIndex,
    User,
    Holding,
    Transaction,
    Reward,
    RewardSummary,
    Referral,
    AIInsight,
    SmartSip,
    UserPortfolio,
    Leaderboard,
    PortfolioCopy,
    WhatsAppSession,
    SipOrder,
    WhatsAppMessage,
    AGIInsight,
    UserBehavior,
    MarketData,
    EconomicIndicator
  };
}