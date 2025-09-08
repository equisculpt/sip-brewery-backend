const { createClient } = require('@supabase/supabase-js');
const { User } = require('../models');
const logger = require('../utils/logger');

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  if (process.env.NODE_ENV === 'test') {
    logger.warn('Missing Supabase environment variables in test mode, using dummy values');
  } else {
    logger.error('Missing Supabase environment variables');
    process.exit(1);
  }
}

const supabase = (supabaseUrl && supabaseServiceKey)
  ? createClient(supabaseUrl, supabaseServiceKey)
  : {
      auth: {
        getUser: async () => ({ data: { user: { id: 'dummy-id' } }, error: null })
      }
    };

const authenticateUser = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        success: false,
        message: 'Authorization header required'
      });
    }
    
    const token = authHeader.substring(7); // Remove 'Bearer ' prefix
    
    // Verify JWT token with Supabase
    const { data: { user }, error } = await supabase.auth.getUser(token);
    
    if (error || !user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }
    
    // Find user in our database
    const dbUser = await User.findOne({ supabaseId: user.id });
    
    if (!dbUser) {
      return res.status(404).json({
        success: false,
        message: 'User not found in database'
      });
    }
    
    // Check if user is active
    if (!dbUser.isActive) {
      return res.status(403).json({
        success: false,
        message: 'User account is deactivated'
      });
    }
    
    // Check KYC status for reward-related endpoints
    if (req.path.includes('/rewards/') && dbUser.kycStatus !== 'VERIFIED') {
      return res.status(403).json({
        success: false,
        message: 'KYC verification required to access rewards'
      });
    }
    
    // Attach user to request
    req.user = dbUser;
    req.supabaseUser = user;
    
    next();
  } catch (error) {
    logger.error('Authentication error:', error);
    return res.status(500).json({
      success: false,
      message: 'Authentication failed'
    });
  }
};

module.exports = authenticateUser; 