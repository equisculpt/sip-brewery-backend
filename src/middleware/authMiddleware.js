const User = require('../models/User');
const { verifyToken } = require('../utils/auth');
const AuthService = require('../services/AuthService');

async function authenticateToken(req, res, next) {
  try {
    const authHeader = req.headers['authorization'];
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ success: false, message: 'No token provided' });
    }
    const token = authHeader.split(' ')[1];

    const decoded = verifyToken(token);

    if (decoded.token_use !== 'access') {
      return res.status(401).json({ success: false, message: 'Invalid token type' });
    }
    if (!decoded.jti || !decoded.sid) {
      return res.status(401).json({ success: false, message: 'Invalid token' });
    }

    // Validate session in DB
    const sessionQuery = `
      SELECT * FROM user_sessions 
      WHERE user_id = $1 AND session_token = $2 AND status = 'ACTIVE' AND expires_at > NOW()
      LIMIT 1
    `;
    const sessionRes = await AuthService.db.query(sessionQuery, [decoded.userId, decoded.sid]);
    if (sessionRes.rows.length === 0) {
      return res.status(401).json({ success: false, message: 'Session expired or revoked' });
    }
    const session = sessionRes.rows[0];

    // Load user
    const user = await User.findById(decoded.userId);
    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid token' });
    }

    req.user = user;
    req.session = { id: session.id, token: session.session_token, last_activity_at: session.last_activity_at };

    // Fire-and-forget activity update
    AuthService.db.query('UPDATE user_sessions SET last_activity_at = NOW() WHERE id = $1', [session.id]).catch(() => {});

    next();
  } catch (err) {
    return res.status(401).json({ success: false, message: 'Invalid or expired token' });
  }
}

function getClientInfo(req) {
  return {
    ipAddress: req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket?.remoteAddress || '',
    userAgent: req.headers['user-agent'] || ''
  };
}

module.exports = { authenticateToken, getClientInfo };
