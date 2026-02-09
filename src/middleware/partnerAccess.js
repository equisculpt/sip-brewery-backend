const mongoose = require('mongoose');
const PartnerClientMap = require('../models/PartnerClientMap');
const User = require('../models/User');

const requirePartnerRole = (allowedRoles = []) => (req, res, next) => {
  const partnerId = req.user?.partnerId;
  const partnerRole = req.user?.partnerRole;
  const partnerStatus = req.user?.partnerStatus;
  const fallbackRole = req.user?.role;
  const role = partnerRole || fallbackRole;

  if (!partnerId) {
    return res.status(403).json({ success: false, message: 'Partner context missing' });
  }

  if (partnerStatus && partnerStatus !== 'ACTIVE') {
    return res.status(403).json({ success: false, message: 'Partner access suspended' });
  }

  if (!role || (allowedRoles.length > 0 && !allowedRoles.includes(role))) {
    return res.status(403).json({ success: false, message: 'Partner access denied' });
  }

  return next();
};

const resolveInvestorId = async (investorId) => {
  if (!investorId) {
    return null;
  }

  if (mongoose.Types.ObjectId.isValid(investorId)) {
    return investorId;
  }

  const user = await User.findOne({ supabaseId: investorId });
  return user?._id || null;
};

const enforcePartnerClientAccess = async (req, res, next) => {
  const partnerId = req.user?.partnerId;
  const investorId = req.params.investorId || req.body?.investorId;

  if (!partnerId || !investorId) {
    return res.status(400).json({ success: false, message: 'Missing partner or investor context' });
  }

  const resolvedInvestorId = await resolveInvestorId(investorId);
  if (!resolvedInvestorId) {
    return res.status(404).json({ success: false, message: 'Investor not found' });
  }

  const mapping = await PartnerClientMap.findOne({
    partnerId,
    investorId: resolvedInvestorId,
    status: 'ACTIVE'
  });

  if (!mapping) {
    return res.status(403).json({ success: false, message: 'Client not mapped to partner' });
  }

  return next();
};

module.exports = { requirePartnerRole, enforcePartnerClientAccess };
