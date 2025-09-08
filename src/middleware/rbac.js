// Role-Based Access Control middleware
module.exports = function rbac(requiredRole = 'user') {
  return function (req, res, next) {
    const user = req.user;
    if (!user || !user.role) {
      return res.status(403).json({ success: false, message: 'Access denied: No user role.' });
    }
    const roles = ['user', 'analyst', 'admin', 'superadmin'];
    const userIdx = roles.indexOf(user.role);
    const reqIdx = roles.indexOf(requiredRole);
    if (userIdx === -1 || userIdx < reqIdx) {
      return res.status(403).json({ success: false, message: 'Access denied: Insufficient role.' });
    }
    next();
  };
};
