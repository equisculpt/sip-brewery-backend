module.exports = {
  getAllAgents: async (req, res) => res.status(200).json({ message: 'Stub: getAllAgents' }),
  getAgentById: async (req, res) => res.status(200).json({ message: 'Stub: getAgentById' }),
  createAgent: async (req, res) => res.status(200).json({ message: 'Stub: createAgent' }),
  updateAgent: async (req, res) => res.status(200).json({ message: 'Stub: updateAgent' }),
  deactivateAgent: async (req, res) => res.status(200).json({ message: 'Stub: deactivateAgent' }),
  getAgentDashboard: async (req, res) => res.status(200).json({ message: 'Stub: getAgentDashboard' })
};
