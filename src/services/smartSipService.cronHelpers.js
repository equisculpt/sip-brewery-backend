// Helper methods for cron jobs in smartSipService.js
const SmartSip = require('../models/SmartSip');
const marketScoreService = require('./marketScoreService');
const mfApiClient = require('../utils/mfApiClient');
const agiPipelineOrchestrator = require('../utils/agiPipelineOrchestrator');
const logger = require('../utils/logger');

/**
 * Get SIPs due in N days, filtered by SIP-only/normal
 * @param {number} days - Number of days from today
 * @param {object} filter - { isSipOnly: true/false }
 * @returns {Promise<Array>}
 */
async function getSIPsDueInDays(days, filter) {
  const targetDate = new Date();
  targetDate.setDate(targetDate.getDate() + days);
  targetDate.setHours(0, 0, 0, 0);
  const nextDay = new Date(targetDate);
  nextDay.setDate(targetDate.getDate() + 1);

  const query = {
    nextSIPDate: { $gte: targetDate, $lt: nextDay },
    status: 'ACTIVE',
    isActive: true,
    ...filter
  };
  return SmartSip.find(query).lean();
}

/**
 * Consult AGI for SIP-only SIPs (pause/resume/start decision)
 * @param {Array} sipList
 */
async function consultAGIForSIPOnly(sipList) {
  if (!sipList.length) return;
  logger.info(`[AGI] Consulting AGI for SIP-only pause/resume decisions for ${sipList.length} SIPs`);
  // Example: Use market regime to decide which SIPs to keep active
  const market = await marketScoreService.calculateMarketScore();
  // Mark all as PAUSED by default, then activate as per regime
  let nActive = 0;
  if (market.score > 0.5) nActive = sipList.length; // bullish
  else if (market.score < -0.5) nActive = 0; // bearish
  else nActive = Math.round(sipList.length / 2 + sipList.length / 2 * market.score);

  sipList.forEach((sip, idx) => {
    sip._agiDecision = idx < nActive ? 'ACTIVE' : 'PAUSED';
  });
  // Store AGI decision in DB or cache if needed
}

/**
 * Consult AGI for normal funds (lumpsum amount decision)
 * @param {Array} sipList
 */
async function consultAGIForNormal(sipList) {
  if (!sipList.length) return;
  logger.info(`[AGI] Consulting AGI for normal fund lumpsum for ${sipList.length} SIPs`);
  // Example: Use AGI pipeline to recommend amount
  for (const sip of sipList) {
    const agiResult = await agiPipelineOrchestrator.getLumpsumRecommendation(sip.userId);
    sip._agiLumpsum = agiResult.recommendedLumpsum || 0;
  }
  // Store AGI decision in DB or cache if needed
}

/**
 * Initiate pause/resume/start for SIP-only SIPs (7 days ahead)
 * @param {Array} sipList
 */
async function initiatePauseResumeForSIPOnly(sipList) {
  if (!sipList.length) return;
  logger.info(`[BSE] Initiating pause/resume/start for SIP-only: ${sipList.length} SIPs`);
  for (const sip of sipList) {
    // Use AGI decision (should be attached as _agiDecision)
    if (sip._agiDecision === 'ACTIVE') {
      await mfApiClient.resumeSIP(sip.bseSIPRegistrationId);
      logger.info(`[BSE] Resumed SIP ${sip.bseSIPRegistrationId}`);
    } else {
      await mfApiClient.pauseSIP(sip.bseSIPRegistrationId);
      logger.info(`[BSE] Paused SIP ${sip.bseSIPRegistrationId}`);
    }
  }
}

/**
 * Place lumpsum orders for normal funds (today)
 * @param {Array} sipList
 */
async function placeLumpsumOrdersForNormal(sipList) {
  if (!sipList.length) return;
  logger.info(`[BSE] Placing lumpsum orders for normal funds: ${sipList.length} SIPs`);
  for (const sip of sipList) {
    if (sip._agiLumpsum && sip._agiLumpsum > 0) {
      await mfApiClient.placeLumpsumOrder({
        userId: sip.userId,
        schemeCode: sip.fundSelection[0]?.schemeCode,
        amount: sip._agiLumpsum,
        // Add any other required fields
      });
      logger.info(`[BSE] Placed lumpsum order for user ${sip.userId} amount ₹${sip._agiLumpsum}`);
    }
  }
}

module.exports = {
  getSIPsDueInDays,
  consultAGIForSIPOnly,
  consultAGIForNormal,
  initiatePauseResumeForSIPOnly,
  placeLumpsumOrdersForNormal
};
