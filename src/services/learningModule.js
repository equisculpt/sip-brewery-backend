const logger = require('../utils/logger');
const { User, UserLearning, LearningPath, Quiz, Lesson, Achievement } = require('../models');
const ollamaService = require('./ollamaService');

class LearningModule {
  constructor() {
    this.learningLevels = {
      BEGINNER: {
        name: 'Beginner',
        description: 'New to mutual funds and investing',
        topics: ['basics', 'sip', 'risk', 'goals'],
        duration: '2-4 weeks',
        lessonsCount: 12
      },
      INTERMEDIATE: {
        name: 'Intermediate',
        description: 'Basic understanding, ready for advanced concepts',
        topics: ['asset_allocation', 'fund_selection', 'tax_optimization', 'rebalancing'],
        duration: '4-6 weeks',
        lessonsCount: 16
      },
      ADVANCED: {
        name: 'Advanced',
        description: 'Experienced investor seeking expert knowledge',
        topics: ['portfolio_optimization', 'market_timing', 'alternative_investments', 'estate_planning'],
        duration: '6-8 weeks',
        lessonsCount: 20
      },
      EXPERT: {
        name: 'Expert',
        description: 'Professional-level knowledge and insights',
        topics: ['quantitative_analysis', 'derivatives', 'international_markets', 'hedge_strategies'],
        duration: '8-12 weeks',
        lessonsCount: 24
      }
    };

    this.learningTopics = {
      BASICS: {
        name: 'Mutual Fund Basics',
        level: 'BEGINNER',
        lessons: [
          'What are Mutual Funds?',
          'Types of Mutual Funds',
          'NAV and Units',
          'Expense Ratio and Charges'
        ],
        duration: '90 seconds',
        difficulty: 1
      },
      SIP: {
        name: 'Systematic Investment Planning',
        level: 'BEGINNER',
        lessons: [
          'What is SIP?',
          'Benefits of SIP',
          'SIP vs Lump Sum',
          'Power of Compounding'
        ],
        duration: '90 seconds',
        difficulty: 1
      },
      RISK: {
        name: 'Risk Management',
        level: 'BEGINNER',
        lessons: [
          'Understanding Investment Risk',
          'Risk-Return Relationship',
          'Diversification',
          'Risk Tolerance Assessment'
        ],
        duration: '90 seconds',
        difficulty: 2
      },
      GOALS: {
        name: 'Goal-Based Investing',
        level: 'BEGINNER',
        lessons: [
          'Setting Financial Goals',
          'Goal Timeline Planning',
          'Goal Amount Calculation',
          'Goal Achievement Tracking'
        ],
        duration: '90 seconds',
        difficulty: 2
      },
      ASSET_ALLOCATION: {
        name: 'Asset Allocation',
        level: 'INTERMEDIATE',
        lessons: [
          'Asset Classes Overview',
          'Allocation Strategies',
          'Age-Based Allocation',
          'Goal-Based Allocation'
        ],
        duration: '90 seconds',
        difficulty: 3
      },
      FUND_SELECTION: {
        name: 'Fund Selection',
        level: 'INTERMEDIATE',
        lessons: [
          'Fund Performance Analysis',
          'Fund Manager Track Record',
          'Fund House Reputation',
          'Fund Size and Liquidity'
        ],
        duration: '90 seconds',
        difficulty: 3
      },
      TAX_OPTIMIZATION: {
        name: 'Tax Optimization',
        level: 'INTERMEDIATE',
        lessons: [
          'Tax on Mutual Funds',
          'ELSS for Tax Saving',
          'LTCG and STCG',
          'Tax Loss Harvesting'
        ],
        duration: '90 seconds',
        difficulty: 4
      },
      REBALANCING: {
        name: 'Portfolio Rebalancing',
        level: 'INTERMEDIATE',
        lessons: [
          'Why Rebalancing?',
          'Rebalancing Strategies',
          'Rebalancing Frequency',
          'Tax Implications of Rebalancing'
        ],
        duration: '90 seconds',
        difficulty: 4
      }
    };

    this.quizTypes = {
      MULTIPLE_CHOICE: {
        name: 'Multiple Choice',
        description: 'Single correct answer from multiple options',
        points: 10
      },
      TRUE_FALSE: {
        name: 'True/False',
        description: 'Binary choice questions',
        points: 5
      },
      FILL_BLANK: {
        name: 'Fill in the Blank',
        description: 'Complete the statement',
        points: 15
      },
      SCENARIO: {
        name: 'Scenario Based',
        description: 'Real-world investment scenarios',
        points: 20
      }
    };

    this.achievementTypes = {
      LESSON_COMPLETION: {
        name: 'Lesson Completion',
        description: 'Complete a lesson successfully',
        points: 10,
        icon: '📚'
      },
      QUIZ_MASTERY: {
        name: 'Quiz Mastery',
        description: 'Score 90% or above in a quiz',
        points: 25,
        icon: '🏆'
      },
      STREAK_MAINTENANCE: {
        name: 'Learning Streak',
        description: 'Maintain daily learning streak',
        points: 50,
        icon: '🔥'
      },
      TOPIC_MASTERY: {
        name: 'Topic Mastery',
        description: 'Complete all lessons in a topic',
        points: 100,
        icon: '🎯'
      },
      LEVEL_UPGRADE: {
        name: 'Level Upgrade',
        description: 'Advance to next learning level',
        points: 200,
        icon: '⭐'
      }
    };
  }

  /**
   * Initialize learning path for user
   */
  async initializeLearningPath(userId) {
    try {
      logger.info('Initializing learning path', { userId });

      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Assess user's current knowledge level
      const currentLevel = await this.assessUserLevel(userId);
      
      // Create personalized learning path
      const learningPath = new LearningPath({
        userId,
        currentLevel,
        targetLevel: this.getNextLevel(currentLevel),
        topics: this.getTopicsForLevel(currentLevel),
        progress: {
          completedLessons: 0,
          totalLessons: this.getTotalLessonsForLevel(currentLevel),
          completedTopics: 0,
          totalTopics: this.getTopicsForLevel(currentLevel).length,
          currentStreak: 0,
          longestStreak: 0,
          totalPoints: 0
        },
        preferences: {
          preferredTime: 'morning',
          lessonDuration: '90 seconds',
          quizFrequency: 'after_topic',
          reminderFrequency: 'daily'
        }
      });

      await learningPath.save();

      // Create initial user learning record
      const userLearning = new UserLearning({
        userId,
        learningPathId: learningPath._id,
        currentTopic: this.getTopicsForLevel(currentLevel)[0],
        currentLesson: 0,
        achievements: [],
        quizScores: [],
        learningHistory: []
      });

      await userLearning.save();

      return {
        success: true,
        data: {
          learningPath,
          userLearning,
          recommendedLessons: await this.getRecommendedLessons(userId, currentLevel)
        }
      };
    } catch (error) {
      logger.error('Failed to initialize learning path', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to initialize learning path',
        error: error.message
      };
    }
  }

  /**
   * Get personalized lesson content
   */
  async getPersonalizedLesson(userId, topic, lessonIndex) {
    try {
      logger.info('Getting personalized lesson', { userId, topic, lessonIndex });

      const user = await User.findById(userId);
      const userLearning = await UserLearning.findOne({ userId });
      const userPortfolio = await this.getUserPortfolioContext(userId);

      if (!user || !userLearning) {
        throw new Error('User or learning data not found');
      }

      // Generate personalized lesson content using Ollama
      const lessonContent = await this.generateLessonContent(topic, lessonIndex, user, userPortfolio);

      // Create lesson record
      const lesson = new Lesson({
        userId,
        topic,
        lessonIndex,
        content: lessonContent,
        duration: '90 seconds',
        difficulty: this.learningTopics[topic]?.difficulty || 1,
        personalized: true
      });

      await lesson.save();

      return {
        success: true,
        data: {
          lesson,
          nextLesson: await this.getNextLesson(userId, topic, lessonIndex),
          relatedTopics: await this.getRelatedTopics(topic),
          estimatedTime: '90 seconds'
        }
      };
    } catch (error) {
      logger.error('Failed to get personalized lesson', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get personalized lesson',
        error: error.message
      };
    }
  }

  /**
   * Generate personalized lesson content using AGI
   */
  async generateLessonContent(topic, lessonIndex, user, userPortfolio) {
    try {
      const topicInfo = this.learningTopics[topic];
      const lessonTitle = topicInfo?.lessons[lessonIndex] || 'Custom Lesson';

      const prompt = `
You are SipBrewery's AI Financial Educator. Create a personalized 90-second lesson for an Indian investor.

TOPIC: ${topicInfo?.name || topic}
LESSON: ${lessonTitle}
USER CONTEXT:
- Age: ${user.age || 30}
- Income: ₹${user.income || 500000}/year
- Portfolio Value: ₹${userPortfolio?.totalValue || 0}
- Risk Profile: ${userPortfolio?.riskProfile || 'moderate'}
- Goals: ${userPortfolio?.goals?.map(g => g.name).join(', ') || 'wealth creation'}

Create a lesson that:
1. Explains the concept in simple terms (60 seconds)
2. Provides a real Indian example (20 seconds)
3. Relates to user's personal situation (10 seconds)

Format as JSON:
{
  "title": "Lesson Title",
  "content": "Main lesson content...",
  "example": "Real Indian example...",
  "personalizedTip": "Tip specific to this user...",
  "keyTakeaways": ["Point 1", "Point 2", "Point 3"],
  "nextSteps": "What to do next..."
}
      `;

      const ollamaResponse = await ollamaService.generateResponse(prompt, {
        model: 'mistral',
        temperature: 0.3,
        max_tokens: 1000
      });

      return this.parseLessonResponse(ollamaResponse);
    } catch (error) {
      logger.error('Failed to generate lesson content', { error: error.message });
      return this.getFallbackLessonContent(topic, lessonIndex);
    }
  }

  /**
   * Start a quiz for user
   */
  async startQuiz(userId, topic) {
    try {
      logger.info('Starting quiz', { userId, topic });

      const userLearning = await UserLearning.findOne({ userId });
      if (!userLearning) {
        throw new Error('User learning data not found');
      }

      // Generate personalized quiz questions
      const quizQuestions = await this.generateQuizQuestions(topic, userId);

      // Create quiz record
      const quiz = new Quiz({
        userId,
        topic,
        questions: quizQuestions,
        status: 'in_progress',
        startTime: new Date(),
        timeLimit: 300 // 5 minutes
      });

      await quiz.save();

      return {
        success: true,
        data: {
          quizId: quiz._id,
          questions: quizQuestions.map(q => ({
            id: q.id,
            question: q.question,
            options: q.options,
            type: q.type
          })),
          timeLimit: 300,
          totalQuestions: quizQuestions.length
        }
      };
    } catch (error) {
      logger.error('Failed to start quiz', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to start quiz',
        error: error.message
      };
    }
  }

  /**
   * Submit quiz answers and get results
   */
  async submitQuiz(userId, quizId, answers) {
    try {
      logger.info('Submitting quiz', { userId, quizId });

      const quiz = await Quiz.findById(quizId);
      if (!quiz || quiz.userId.toString() !== userId) {
        throw new Error('Quiz not found');
      }

      // Evaluate answers
      const results = await this.evaluateQuizAnswers(quiz.questions, answers);
      
      // Update quiz with results
      quiz.answers = answers;
      quiz.results = results;
      quiz.status = 'completed';
      quiz.endTime = new Date();
      quiz.score = results.score;
      quiz.timeTaken = quiz.endTime - quiz.startTime;

      await quiz.save();

      // Update user learning progress
      await this.updateLearningProgress(userId, results);

      // Check for achievements
      const achievements = await this.checkAchievements(userId, results);

      return {
        success: true,
        data: {
          results,
          achievements,
          nextSteps: await this.getNextSteps(userId, results.score)
        }
      };
    } catch (error) {
      logger.error('Failed to submit quiz', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to submit quiz',
        error: error.message
      };
    }
  }

  /**
   * Get daily learning nudge
   */
  async getDailyNudge(userId) {
    try {
      logger.info('Getting daily nudge', { userId });

      const userLearning = await UserLearning.findOne({ userId });
      const learningPath = await LearningPath.findOne({ userId });

      if (!userLearning || !learningPath) {
        throw new Error('Learning data not found');
      }

      // Generate personalized nudge using AGI
      const nudge = await this.generatePersonalizedNudge(userId, userLearning, learningPath);

      return {
        success: true,
        data: {
          nudge,
          learningStreak: learningPath.progress.currentStreak,
          nextMilestone: await this.getNextMilestone(userId),
          todayGoal: await this.getTodayGoal(userId)
        }
      };
    } catch (error) {
      logger.error('Failed to get daily nudge', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get daily nudge',
        error: error.message
      };
    }
  }

  /**
   * Track learning progress
   */
  async trackLearningProgress(userId, action, details) {
    try {
      logger.info('Tracking learning progress', { userId, action });

      const userLearning = await UserLearning.findOne({ userId });
      const learningPath = await LearningPath.findOne({ userId });

      if (!userLearning || !learningPath) {
        throw new Error('Learning data not found');
      }

      // Update learning history
      userLearning.learningHistory.push({
        action,
        details,
        timestamp: new Date()
      });

      // Update progress based on action
      switch (action) {
        case 'lesson_completed':
          learningPath.progress.completedLessons++;
          await this.checkLessonAchievement(userId);
          break;
        case 'quiz_completed':
          learningPath.progress.totalPoints += details.score || 0;
          await this.checkQuizAchievement(userId, details.score);
          break;
        case 'topic_completed':
          learningPath.progress.completedTopics++;
          await this.checkTopicAchievement(userId, details.topic);
          break;
        case 'daily_login':
          learningPath.progress.currentStreak++;
          if (learningPath.progress.currentStreak > learningPath.progress.longestStreak) {
            learningPath.progress.longestStreak = learningPath.progress.currentStreak;
          }
          await this.checkStreakAchievement(userId, learningPath.progress.currentStreak);
          break;
      }

      await userLearning.save();
      await learningPath.save();

      return {
        success: true,
        data: {
          updatedProgress: learningPath.progress,
          newAchievements: await this.getRecentAchievements(userId)
        }
      };
    } catch (error) {
      logger.error('Failed to track learning progress', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to track learning progress',
        error: error.message
      };
    }
  }

  /**
   * Get learning analytics and insights
   */
  async getLearningAnalytics(userId) {
    try {
      logger.info('Getting learning analytics', { userId });

      const userLearning = await UserLearning.findOne({ userId });
      const learningPath = await LearningPath.findOne({ userId });
      const quizzes = await Quiz.find({ userId });
      const achievements = await Achievement.find({ userId });

      if (!userLearning || !learningPath) {
        throw new Error('Learning data not found');
      }

      const analytics = {
        progress: {
          overall: (learningPath.progress.completedLessons / learningPath.progress.totalLessons) * 100,
          topics: (learningPath.progress.completedTopics / learningPath.progress.totalTopics) * 100,
          currentLevel: learningPath.currentLevel,
          targetLevel: learningPath.targetLevel
        },
        performance: {
          averageQuizScore: this.calculateAverageQuizScore(quizzes),
          totalPoints: learningPath.progress.totalPoints,
          currentStreak: learningPath.progress.currentStreak,
          longestStreak: learningPath.progress.longestStreak
        },
        achievements: {
          total: achievements.length,
          recent: achievements.slice(-5),
          categories: this.categorizeAchievements(achievements)
        },
        recommendations: await this.getLearningRecommendations(userId, analytics)
      };

      return {
        success: true,
        data: analytics
      };
    } catch (error) {
      logger.error('Failed to get learning analytics', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get learning analytics',
        error: error.message
      };
    }
  }

  // Helper methods
  async assessUserLevel(userId) {
    // Simple assessment based on user profile and portfolio
    const user = await User.findById(userId);
    const userPortfolio = await this.getUserPortfolioContext(userId);

    let level = 'BEGINNER';

    if (userPortfolio?.totalValue > 1000000) level = 'INTERMEDIATE';
    if (userPortfolio?.totalValue > 5000000) level = 'ADVANCED';
    if (userPortfolio?.totalValue > 10000000) level = 'EXPERT';

    return level;
  }

  getNextLevel(currentLevel) {
    const levels = Object.keys(this.learningLevels);
    const currentIndex = levels.indexOf(currentLevel);
    return levels[Math.min(currentIndex + 1, levels.length - 1)];
  }

  getTopicsForLevel(level) {
    return Object.keys(this.learningTopics).filter(topic => 
      this.learningTopics[topic].level === level
    );
  }

  getTotalLessonsForLevel(level) {
    const topics = this.getTopicsForLevel(level);
    return topics.reduce((total, topic) => 
      total + (this.learningTopics[topic]?.lessons?.length || 0), 0
    );
  }

  async getRecommendedLessons(userId, level) {
    const topics = this.getTopicsForLevel(level);
    return topics.slice(0, 3).map(topic => ({
      topic,
      name: this.learningTopics[topic].name,
      lessons: this.learningTopics[topic].lessons.slice(0, 2)
    }));
  }

  async getUserPortfolioContext(userId) {
    // Get user portfolio context for personalization
    return {
      totalValue: 500000,
      riskProfile: 'moderate',
      goals: [{ name: 'Retirement', amount: 10000000 }]
    };
  }

  parseLessonResponse(ollamaResponse) {
    try {
      const jsonMatch = ollamaResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No valid JSON found in response');
      }

      return JSON.parse(jsonMatch[0]);
    } catch (error) {
      logger.error('Failed to parse lesson response', { error: error.message });
      return this.getFallbackLessonContent();
    }
  }

  getFallbackLessonContent(topic, lessonIndex) {
    return {
      title: `Lesson ${lessonIndex + 1}`,
      content: 'This is a fallback lesson content. Please try again later.',
      example: 'Example will be provided in the next update.',
      personalizedTip: 'Complete your profile for personalized tips.',
      keyTakeaways: ['Key point 1', 'Key point 2', 'Key point 3'],
      nextSteps: 'Continue with the next lesson.'
    };
  }

  async generateQuizQuestions(topic, userId) {
    // Generate personalized quiz questions using AGI
    const questions = [
      {
        id: 1,
        question: 'What is the primary benefit of SIP?',
        options: ['Higher returns', 'Rupee cost averaging', 'Tax benefits', 'Lower risk'],
        correctAnswer: 1,
        type: 'MULTIPLE_CHOICE',
        explanation: 'SIP helps in rupee cost averaging by buying more units when prices are low.'
      }
    ];

    return questions;
  }

  async evaluateQuizAnswers(questions, answers) {
    let correctAnswers = 0;
    const results = [];

    questions.forEach((question, index) => {
      const userAnswer = answers[index];
      const isCorrect = userAnswer === question.correctAnswer;
      
      if (isCorrect) correctAnswers++;

      results.push({
        questionId: question.id,
        userAnswer,
        correctAnswer: question.correctAnswer,
        isCorrect,
        explanation: question.explanation
      });
    });

    const score = (correctAnswers / questions.length) * 100;

    return {
      score: Math.round(score),
      correctAnswers,
      totalQuestions: questions.length,
      results
    };
  }

  async updateLearningProgress(userId, results) {
    const userLearning = await UserLearning.findOne({ userId });
    if (userLearning) {
      userLearning.quizScores.push({
        score: results.score,
        timestamp: new Date()
      });
      await userLearning.save();
    }
  }

  async checkAchievements(userId, results) {
    const achievements = [];

    if (results.score >= 90) {
      achievements.push(await this.createAchievement(userId, 'QUIZ_MASTERY', results));
    }

    return achievements;
  }

  async createAchievement(userId, type, context) {
    const achievement = new Achievement({
      userId,
      type,
      title: this.achievementTypes[type].name,
      description: this.achievementTypes[type].description,
      points: this.achievementTypes[type].points,
      icon: this.achievementTypes[type].icon,
      context,
      earnedAt: new Date()
    });

    await achievement.save();
    return achievement;
  }

  async getNextSteps(userId, score) {
    if (score >= 90) {
      return 'Excellent! You can move to the next topic.';
    } else if (score >= 70) {
      return 'Good job! Review the incorrect answers and try again.';
    } else {
      return 'Consider reviewing the lesson before retaking the quiz.';
    }
  }

  async generatePersonalizedNudge(userId, userLearning, learningPath) {
    const nudge = {
      title: '📚 Your Daily Learning Reminder',
      message: `Keep your ${learningPath.progress.currentStreak}-day streak alive!`,
      action: 'Start today\'s lesson',
      estimatedTime: '90 seconds',
      topic: userLearning.currentTopic
    };

    return nudge;
  }

  async getNextMilestone(userId) {
    return {
      type: 'lesson_completion',
      target: 10,
      current: 7,
      reward: '50 points'
    };
  }

  async getTodayGoal(userId) {
    return {
      action: 'Complete 1 lesson',
      topic: 'Mutual Fund Basics',
      estimatedTime: '90 seconds'
    };
  }

  async checkLessonAchievement(userId) {
    // Check for lesson completion achievements
  }

  async checkQuizAchievement(userId, score) {
    // Check for quiz achievements
  }

  async checkTopicAchievement(userId, topic) {
    // Check for topic completion achievements
  }

  async checkStreakAchievement(userId, streak) {
    // Check for streak achievements
  }

  async getRecentAchievements(userId) {
    const achievements = await Achievement.find({ userId }).sort({ earnedAt: -1 }).limit(5);
    return achievements;
  }

  calculateAverageQuizScore(quizzes) {
    if (quizzes.length === 0) return 0;
    const totalScore = quizzes.reduce((sum, quiz) => sum + (quiz.score || 0), 0);
    return Math.round(totalScore / quizzes.length);
  }

  categorizeAchievements(achievements) {
    const categories = {};
    achievements.forEach(achievement => {
      const type = achievement.type;
      categories[type] = (categories[type] || 0) + 1;
    });
    return categories;
  }

  async getLearningRecommendations(userId, analytics) {
    return [
      'Complete the next lesson in your current topic',
      'Review previous lessons if quiz scores are low',
      'Try a different topic to diversify knowledge'
    ];
  }

  async getNextLesson(userId, topic, lessonIndex) {
    const topicInfo = this.learningTopics[topic];
    if (lessonIndex + 1 < topicInfo.lessons.length) {
      return {
        topic,
        lessonIndex: lessonIndex + 1,
        title: topicInfo.lessons[lessonIndex + 1]
      };
    }
    return null;
  }

  async getRelatedTopics(topic) {
    const currentLevel = this.learningTopics[topic]?.level;
    return this.getTopicsForLevel(currentLevel).filter(t => t !== topic).slice(0, 3);
  }
}

module.exports = new LearningModule(); 