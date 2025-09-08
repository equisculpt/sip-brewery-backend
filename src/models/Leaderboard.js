const mongoose = require('mongoose');

const leaderboardSchema = new mongoose.Schema({
  duration: {
    type: String,
    enum: ['1M', '3M', '6M', '1Y', '3Y'],
    required: true,
    index: true
  },
  leaders: [{
    secretCode: {
      type: String,
      required: true
    },
    returnPercent: {
      type: Number,
      required: true
    },
    allocation: {
      type: Map,
      of: Number
    },
    rank: {
      type: Number,
      required: true
    },
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true
    },
    portfolioId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'UserPortfolio',
      required: true
    }
  }],
  generatedAt: {
    type: Date,
    default: Date.now,
    index: true
  },
  totalParticipants: {
    type: Number,
    default: 0
  },
  averageReturn: {
    type: Number,
    default: 0
  },
  medianReturn: {
    type: Number,
    default: 0
  },
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Method to get allocation as object
leaderboardSchema.methods.getLeadersWithAllocation = function() {
  return this.leaders.map(leader => ({
    secretCode: leader.secretCode,
    returnPercent: leader.returnPercent,
    rank: leader.rank,
    allocation: this.mapToObject(leader.allocation)
  }));
};

// Helper method to convert Map to object
leaderboardSchema.methods.mapToObject = function(map) {
  const obj = {};
  map.forEach((value, key) => {
    obj[key] = value;
  });
  return obj;
};

// Method to add leader
leaderboardSchema.methods.addLeader = function(leaderData) {
  this.leaders.push(leaderData);
  // Sort by return percentage in descending order
  this.leaders.sort((a, b) => b.returnPercent - a.returnPercent);
  // Update ranks
  this.leaders.forEach((leader, index) => {
    leader.rank = index + 1;
  });
  return this.save();
};

// Method to update leaderboard statistics
leaderboardSchema.methods.updateStatistics = function(totalParticipants, averageReturn, medianReturn) {
  this.totalParticipants = totalParticipants;
  this.averageReturn = averageReturn;
  this.medianReturn = medianReturn;
  return this.save();
};

// Method to get top N leaders
leaderboardSchema.methods.getTopLeaders = function(limit = 20) {
  return this.leaders.slice(0, limit).map(leader => ({
    secretCode: leader.secretCode,
    returnPercent: leader.returnPercent,
    rank: leader.rank,
    allocation: this.mapToObject(leader.allocation)
  }));
};

// Method to find user's rank
leaderboardSchema.methods.findUserRank = function(userId) {
  const leader = this.leaders.find(l => l.userId.toString() === userId.toString());
  return leader ? leader.rank : null;
};

// Indexes for performance optimization
// leaderboardSchema.index({ leaderboardId: 1 }); // Duplicate of inline index: true
// leaderboardSchema.index({ 'leaders.secretCode': 1 }); // secretCode has inline index: true

leaderboardSchema.index({ duration: 1, generatedAt: -1 });
leaderboardSchema.index({ 'leaders.userId': 1 });
leaderboardSchema.index({ isActive: 1 });

if (process.env.NODE_ENV === 'test') {
  const mockLeaderboardModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockLeaderboardId', ...data }),
    };
  };
  mockLeaderboardModel.find = jest.fn().mockResolvedValue([]);
  mockLeaderboardModel.findOne = jest.fn().mockResolvedValue(null);
  mockLeaderboardModel.findById = jest.fn().mockResolvedValue(null);
  mockLeaderboardModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockLeaderboardModel.create = jest.fn().mockResolvedValue({ _id: 'mockLeaderboardId' });
  module.exports = mockLeaderboardModel;
} else {
  module.exports = mongoose.model('Leaderboard', leaderboardSchema);
}