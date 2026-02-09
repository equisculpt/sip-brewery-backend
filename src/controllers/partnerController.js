const partnerService = require('../services/partnerService');
const response = require('../utils/response');
const logger = require('../utils/logger');

class PartnerController {
  async createPartner(req, res) {
    try {
      const ownerUser = req.user;
      const baseUrl = req.body?.baseUrl || process.env.PARTNER_PORTAL_URL || process.env.FRONTEND_URL || process.env.CLIENT_URL || '';

      const result = await partnerService.createPartner({
        ownerUser,
        payload: req.body,
        baseUrl
      });

      return response.successResponse(res, 'Partner created successfully', result, 201);
    } catch (error) {
      logger.error('Partner creation failed', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to create partner', error, 400);
    }
  }

  async getPartnerProfile(req, res) {
    try {
      const partner = await partnerService.getPartnerByUser(req.user);
      return response.successResponse(res, 'Partner profile retrieved', partner);
    } catch (error) {
      logger.error('Partner profile error', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to fetch partner profile', error, 404);
    }
  }

  async updatePartnerProfile(req, res) {
    try {
      const partner = await partnerService.getPartnerByUser(req.user);
      const updates = req.body || {};

      if (updates.contact) {
        partner.contact = { ...partner.contact, ...updates.contact };
      }

      if (updates.metadata) {
        partner.metadata = { ...partner.metadata, ...updates.metadata };
      }

      if (updates.onboarding) {
        partner.onboarding = { ...partner.onboarding, ...updates.onboarding };
      }

      await partner.save();
      return response.successResponse(res, 'Partner profile updated', partner);
    } catch (error) {
      logger.error('Partner update error', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to update partner profile', error, 400);
    }
  }

  async mapClient(req, res) {
    try {
      const partnerId = req.user?.partnerId;
      const {
        investorId,
        relationshipType,
        consentId,
        consentScope,
        consentChannel,
        consentArtifactUri,
        metadata
      } = req.body || {};

      const result = await partnerService.mapClient({
        partnerId,
        investorId,
        relationshipType,
        consentId,
        consentScope,
        consentChannel,
        consentArtifactUri,
        metadata
      });

      return response.successResponse(res, 'Client mapped successfully', result, 201);
    } catch (error) {
      logger.error('Client mapping failed', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to map client', error, 400);
    }
  }

  async mapClientByReferral(req, res) {
    try {
      const investorUser = req.user;
      const { partnerCode } = req.params;
      const result = await partnerService.mapClientByReferral({
        investorUser,
        partnerCode,
        consentPayload: req.body || {}
      });

      return response.successResponse(res, 'Referral mapping created', result, 201);
    } catch (error) {
      logger.error('Referral mapping failed', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to map referral', error, 400);
    }
  }

  async listClients(req, res) {
    try {
      const partnerId = req.user?.partnerId;
      const { status, limit = 20, offset = 0 } = req.query;

      const result = await partnerService.listClients({
        partnerId,
        status,
        limit: parseInt(limit, 10),
        offset: parseInt(offset, 10)
      });

      return response.successResponse(res, 'Client list retrieved', result);
    } catch (error) {
      logger.error('Client list error', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to fetch clients', error, 400);
    }
  }

  async getClientDetail(req, res) {
    try {
      const partnerId = req.user?.partnerId;
      const { investorId } = req.params;

      const mapping = await partnerService.getClientDetail({ partnerId, investorId });

      return response.successResponse(res, 'Client details retrieved', mapping);
    } catch (error) {
      logger.error('Client detail error', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to fetch client details', error, 404);
    }
  }

  async getDashboardSummary(req, res) {
    try {
      const partnerId = req.user?.partnerId;
      const summary = await partnerService.getDashboardSummary({ partnerId });
      return response.successResponse(res, 'Partner dashboard summary', summary);
    } catch (error) {
      logger.error('Partner dashboard error', { error: error.message });
      return response.errorResponse(res, error.message || 'Failed to fetch dashboard summary', error, 400);
    }
  }
}

module.exports = new PartnerController();
