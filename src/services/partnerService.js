const mongoose = require('mongoose');
const Partner = require('../models/Partner');
const PartnerClientMap = require('../models/PartnerClientMap');
const Consent = require('../models/Consent');
const User = require('../models/User');
const UserPortfolio = require('../models/UserPortfolio');

const PARTNER_TYPES = ['IFA', 'SUB_DISTRIBUTOR', 'BROKER', 'INSTITUTIONAL'];
const CONSENT_CHANNELS = ['OTP', 'ESIGN', 'VOICE', 'CHECKBOX', 'DOCUMENT'];
const CONSENT_SCOPES = {
  referral: 'PARTNER_REFERRAL',
  dashboard: 'PARTNER_DASHBOARD_ACCESS'
};

class PartnerService {
  buildReferralLink(baseUrl, partnerCode) {
    const trimmed = (baseUrl || '').replace(/\/$/, '');
    if (!trimmed) {
      return `/partner/ref/${partnerCode}`;
    }
    return `${trimmed}/partner/ref/${partnerCode}`;
  }

  async createPartner({ ownerUser, payload, baseUrl }) {
    if (!ownerUser) {
      throw new Error('Owner user is required');
    }

    if (ownerUser.partnerId) {
      throw new Error('User already linked to a partner');
    }

    const {
      name,
      code,
      type,
      contact = {},
      metadata = {},
      onboardingCompleted = false,
      kycStatus
    } = payload;

    if (!PARTNER_TYPES.includes(type)) {
      throw new Error('Invalid partner type');
    }

    const partner = await Partner.create({
      name,
      code,
      type,
      status: 'ACTIVE',
      contact,
      onboarding: {
        referralLink: null,
        onboardingCompleted: Boolean(onboardingCompleted),
        kycStatus: kycStatus || 'PENDING'
      },
      metadata
    });

    const referralLink = this.buildReferralLink(baseUrl, partner.code);
    partner.onboarding.referralLink = referralLink;
    await partner.save();

    const ownerUpdates = {
      partnerId: partner._id,
      partnerRole: 'OWNER',
      partnerStatus: 'ACTIVE'
    };

    if (!ownerUser.role || ownerUser.role === 'user') {
      ownerUpdates.role = 'partner';
    }

    const updatedOwner = await User.findByIdAndUpdate(
      ownerUser._id,
      ownerUpdates,
      { new: true }
    );

    return {
      partner,
      owner: updatedOwner,
      referralLink
    };
  }

  async getPartnerByUser(user) {
    if (!user?.partnerId) {
      throw new Error('Partner context missing');
    }

    const partner = await Partner.findById(user.partnerId);
    if (!partner) {
      throw new Error('Partner not found');
    }

    return partner;
  }

  async mapClient({
    partnerId,
    investorId,
    relationshipType = 'DASHBOARD_ONBOARDING',
    consentId,
    consentScope,
    consentChannel,
    consentArtifactUri,
    metadata = {}
  }) {
    const partner = await Partner.findById(partnerId);
    if (!partner) {
      throw new Error('Partner not found');
    }

    const investor = await this.resolveInvestor(investorId);
    if (!investor) {
      throw new Error('Investor not found');
    }

    let consent = null;

    if (consentId) {
      consent = await Consent.findById(consentId);
      if (!consent) {
        throw new Error('Consent not found');
      }
    } else if (consentScope || consentChannel || consentArtifactUri) {
      if (!consentScope || !consentChannel || !consentArtifactUri) {
        throw new Error('Consent scope, channel, and artifactUri are required');
      }

      if (!CONSENT_CHANNELS.includes(consentChannel)) {
        throw new Error('Invalid consent channel');
      }

      consent = await Consent.create({
        investorId: investor._id,
        partnerId,
        scope: consentScope,
        channel: consentChannel,
        artifactUri: consentArtifactUri,
        status: 'ACTIVE'
      });
    }

    const mapping = await PartnerClientMap.findOneAndUpdate(
      { partnerId, investorId: investor._id },
      {
        partnerId,
        investorId: investor._id,
        relationshipType,
        status: 'ACTIVE',
        consentId: consent ? consent._id : null,
        metadata
      },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );

    return { partner, investor, mapping, consent };
  }

  async mapClientByReferral({ investorUser, partnerCode, consentPayload = {} }) {
    const partner = await Partner.findOne({ code: partnerCode });
    if (!partner) {
      throw new Error('Partner not found');
    }

    const consentChannel = consentPayload.consentChannel;
    const consentArtifactUri = consentPayload.consentArtifactUri;
    const consentScope = consentChannel && consentArtifactUri
      ? (consentPayload.consentScope || CONSENT_SCOPES.referral)
      : undefined;

    return this.mapClient({
      partnerId: partner._id,
      investorId: investorUser._id,
      relationshipType: 'REFERRAL',
      consentScope,
      consentChannel,
      consentArtifactUri,
      metadata: consentPayload.metadata || {}
    });
  }

  async listClients({ partnerId, status, limit = 20, offset = 0 }) {
    const query = { partnerId };
    if (status) {
      query.status = status;
    }

    const [total, mappings] = await Promise.all([
      PartnerClientMap.countDocuments(query),
      PartnerClientMap.find(query)
        .sort({ createdAt: -1 })
        .skip(offset)
        .limit(limit)
        .populate('investorId', 'supabaseId name email phone kycStatus partnerStatus partnerRole')
    ]);

    return {
      total,
      mappings
    };
  }

  async getClientDetail({ partnerId, investorId }) {
    const investor = await this.resolveInvestor(investorId);
    if (!investor) {
      throw new Error('Investor not found');
    }

    const mapping = await PartnerClientMap.findOne({ partnerId, investorId: investor._id })
      .populate('investorId', 'supabaseId name email phone kycStatus partnerStatus partnerRole');

    if (!mapping) {
      throw new Error('Client mapping not found');
    }

    return mapping;
  }

  async getDashboardSummary({ partnerId }) {
    const mappings = await PartnerClientMap.find({ partnerId, status: 'ACTIVE' })
      .select('investorId consentId relationshipType status')
      .lean();

    const investorIds = mappings.map(mapping => mapping.investorId);

    const [portfolios, pendingKycCount] = await Promise.all([
      investorIds.length
        ? UserPortfolio.find({ userId: { $in: investorIds }, isActive: true }).lean()
        : [],
      investorIds.length
        ? User.countDocuments({ _id: { $in: investorIds }, kycStatus: { $ne: 'VERIFIED' } })
        : 0
    ]);

    const totals = portfolios.reduce(
      (acc, portfolio) => {
        acc.totalAum += portfolio.totalCurrentValue || 0;
        acc.totalInvested += portfolio.totalInvested || 0;
        return acc;
      },
      { totalAum: 0, totalInvested: 0 }
    );

    const relationshipBreakdown = mappings.reduce((acc, mapping) => {
      const key = mapping.relationshipType || 'UNKNOWN';
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});

    const noConsentCount = mappings.filter(mapping => !mapping.consentId).length;

    return {
      totalClients: mappings.length,
      activeClients: mappings.length,
      totalAum: totals.totalAum,
      totalInvested: totals.totalInvested,
      pendingKycCount,
      noConsentCount,
      relationshipBreakdown
    };
  }

  async resolveInvestor(investorId) {
    if (!investorId) {
      return null;
    }

    if (mongoose.Types.ObjectId.isValid(investorId)) {
      const byId = await User.findById(investorId);
      if (byId) {
        return byId;
      }
    }

    return User.findOne({ supabaseId: investorId });
  }
}

module.exports = new PartnerService();
