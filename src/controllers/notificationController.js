module.exports = {
  getNotifications: async (req, res) => res.status(200).json({ message: 'Stub: getNotifications' }),
  markAsRead: async (req, res) => res.status(200).json({ message: 'Stub: markAsRead' }),
  markAllAsRead: async (req, res) => res.status(200).json({ message: 'Stub: markAllAsRead' })
};
