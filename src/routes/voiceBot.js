const express = require('express');
const router = express.Router();
const voiceBotController = require('../controllers/voiceBotController');
const { authenticateUser } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');

/**
 * @swagger
 * components:
 *   schemas:
 *     AudioData:
 *       type: object
 *       required:
 *         - content
 *         - format
 *         - duration
 *       properties:
 *         content:
 *           type: string
 *           description: Base64 encoded audio content
 *         format:
 *           type: string
 *           enum: [wav, mp3, m4a]
 *           description: Audio format
 *         duration:
 *           type: number
 *           description: Audio duration in seconds
 *         quality:
 *           type: number
 *           minimum: 0
 *           maximum: 1
 *           description: Audio quality score
 *         metadata:
 *           type: object
 *           description: Additional audio metadata
 *     
 *     VoiceCommand:
 *       type: object
 *       required:
 *         - command
 *       properties:
 *         command:
 *           type: string
 *           description: Voice command text
 *         language:
 *           type: string
 *           enum: [en, hi, ta, bn, te, mr, gu, kn, ml, pa]
 *           default: en
 *           description: Language code
 *     
 *     VoicePreferences:
 *       type: object
 *       properties:
 *         preferredLanguage:
 *           type: string
 *           enum: [en, hi, ta, bn, te, mr, gu, kn, ml, pa]
 *           description: Preferred language for voice processing
 *         voiceSpeed:
 *           type: number
 *           minimum: 0.5
 *           maximum: 2.0
 *           default: 1.0
 *           description: Voice response speed
 *         voiceGender:
 *           type: string
 *           enum: [male, female, neutral]
 *           default: neutral
 *           description: Voice gender preference
 *         autoLanguageDetection:
 *           type: boolean
 *           default: true
 *           description: Enable automatic language detection
 */

/**
 * @swagger
 * /api/voice/analyze:
 *   post:
 *     summary: Analyze voice input and extract investment intent
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - audioData
 *             properties:
 *               audioData:
 *                 $ref: '#/components/schemas/AudioData'
 *               language:
 *                 type: string
 *                 enum: [en, hi, ta, bn, te, mr, gu, kn, ml, pa]
 *                 default: en
 *                 description: Language for voice processing
 *     responses:
 *       200:
 *         description: Voice analysis completed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   type: object
 *                   properties:
 *                     transcription:
 *                       type: string
 *                       description: Transcribed text from audio
 *                     intent:
 *                       type: object
 *                       description: Extracted intent and entities
 *                     response:
 *                       type: object
 *                       description: Generated response based on intent
 *                     confidence:
 *                       type: number
 *                       description: Confidence score of the analysis
 *                     language:
 *                       type: string
 *                       description: Detected or specified language
 *       400:
 *         description: Invalid input data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/analyze', 
  authenticateUser, 
  validateRequest({
    body: {
      audioData: { type: 'object', required: true },
      language: { type: 'string', enum: ['en', 'hi', 'ta', 'bn', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'], default: 'en' }
    }
  }),
  voiceBotController.analyzeVoice
);

/**
 * @swagger
 * /api/voice/command:
 *   post:
 *     summary: Process voice command for investment actions
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/VoiceCommand'
 *     responses:
 *       200:
 *         description: Voice command processed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   type: object
 *                   properties:
 *                     action:
 *                       type: string
 *                       description: Action type (BUY, SELL, PORTFOLIO, etc.)
 *                     message:
 *                       type: string
 *                       description: Response message
 *                     data:
 *                       type: object
 *                       description: Action-specific data
 *       400:
 *         description: Invalid input data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/command',
  authenticateUser,
  validateRequest({
    body: {
      command: { type: 'string', required: true },
      language: { type: 'string', enum: ['en', 'hi', 'ta', 'bn', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'], default: 'en' }
    }
  }),
  voiceBotController.processVoiceCommand
);

/**
 * @swagger
 * /api/voice/hindi:
 *   post:
 *     summary: Handle Hindi voice input specifically
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - audioData
 *             properties:
 *               audioData:
 *                 $ref: '#/components/schemas/AudioData'
 *     responses:
 *       200:
 *         description: Hindi voice processed successfully
 *       400:
 *         description: Invalid input data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/hindi',
  authenticateUser,
  validateRequest({
    body: {
      audioData: { type: 'object', required: true }
    }
  }),
  voiceBotController.handleHindiVoice
);

/**
 * @swagger
 * /api/voice/multilanguage:
 *   post:
 *     summary: Multi-language voice support with automatic language detection
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - audioData
 *             properties:
 *               audioData:
 *                 $ref: '#/components/schemas/AudioData'
 *               preferredLanguage:
 *                 type: string
 *                 enum: [en, hi, ta, bn, te, mr, gu, kn, ml, pa]
 *                 default: en
 *                 description: Preferred language if detection fails
 *     responses:
 *       200:
 *         description: Multi-language voice processed successfully
 *       400:
 *         description: Invalid input data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/multilanguage',
  authenticateUser,
  validateRequest({
    body: {
      audioData: { type: 'object', required: true },
      preferredLanguage: { type: 'string', enum: ['en', 'hi', 'ta', 'bn', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'], default: 'en' }
    }
  }),
  voiceBotController.handleMultiLanguageVoice
);

/**
 * @swagger
 * /api/voice/languages:
 *   get:
 *     summary: Get supported languages
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Supported languages retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   type: object
 *                   properties:
 *                     languages:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           code:
 *                             type: string
 *                             description: Language code
 *                           name:
 *                             type: string
 *                             description: Language name in English
 *                           nativeName:
 *                             type: string
 *                             description: Language name in native script
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/languages',
  authenticateUser,
  voiceBotController.getSupportedLanguages
);

/**
 * @swagger
 * /api/voice/config:
 *   get:
 *     summary: Get voice processing configuration
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Voice configuration retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   type: object
 *                   properties:
 *                     maxDuration:
 *                       type: number
 *                       description: Maximum audio duration in seconds
 *                     supportedFormats:
 *                       type: array
 *                       items:
 *                         type: string
 *                       description: Supported audio formats
 *                     qualityThreshold:
 *                       type: number
 *                       description: Minimum audio quality threshold
 *                     languageDetection:
 *                       type: boolean
 *                       description: Whether language detection is enabled
 *                     supportedLanguages:
 *                       type: array
 *                       items:
 *                         type: string
 *                       description: Supported language codes
 *                     actionIntents:
 *                       type: array
 *                       items:
 *                         type: string
 *                       description: Supported action intents
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/config',
  authenticateUser,
  voiceBotController.getVoiceConfig
);

/**
 * @swagger
 * /api/voice/test:
 *   post:
 *     summary: Test voice processing with sample data
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - testCommand
 *             properties:
 *               testCommand:
 *                 type: string
 *                 description: Test command to process
 *               language:
 *                 type: string
 *                 enum: [en, hi, ta, bn, te, mr, gu, kn, ml, pa]
 *                 default: en
 *                 description: Language for processing
 *     responses:
 *       200:
 *         description: Voice processing test completed successfully
 *       400:
 *         description: Invalid input data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/test',
  authenticateUser,
  validateRequest({
    body: {
      testCommand: { type: 'string', required: true },
      language: { type: 'string', enum: ['en', 'hi', 'ta', 'bn', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'], default: 'en' }
    }
  }),
  voiceBotController.testVoiceProcessing
);

/**
 * @swagger
 * /api/voice/history:
 *   get:
 *     summary: Get voice interaction history
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 10
 *           minimum: 1
 *           maximum: 100
 *         description: Number of records to return
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *           minimum: 0
 *         description: Number of records to skip
 *     responses:
 *       200:
 *         description: Voice history retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   type: object
 *                   properties:
 *                     history:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           id:
 *                             type: string
 *                           timestamp:
 *                             type: string
 *                             format: date-time
 *                           command:
 *                             type: string
 *                           action:
 *                             type: string
 *                           language:
 *                             type: string
 *                           confidence:
 *                             type: number
 *                           response:
 *                             type: string
 *                     total:
 *                       type: integer
 *                     limit:
 *                       type: integer
 *                     offset:
 *                       type: integer
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/history',
  authenticateUser,
  validateRequest({
    query: {
      limit: { type: 'integer', min: 1, max: 100, default: 10 },
      offset: { type: 'integer', min: 0, default: 0 }
    }
  }),
  voiceBotController.getVoiceHistory
);

/**
 * @swagger
 * /api/voice/analytics:
 *   get:
 *     summary: Get voice analytics
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [7d, 30d, 90d, 1y]
 *           default: 30d
 *         description: Analytics period
 *     responses:
 *       200:
 *         description: Voice analytics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   type: object
 *                   properties:
 *                     totalInteractions:
 *                       type: integer
 *                     successfulInteractions:
 *                       type: integer
 *                     successRate:
 *                       type: number
 *                     averageConfidence:
 *                       type: number
 *                     mostUsedLanguage:
 *                       type: string
 *                     mostUsedAction:
 *                       type: string
 *                     languageBreakdown:
 *                       type: object
 *                     actionBreakdown:
 *                       type: object
 *                     timeOfDayUsage:
 *                       type: object
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/analytics',
  authenticateUser,
  validateRequest({
    query: {
      period: { type: 'string', enum: ['7d', '30d', '90d', '1y'], default: '30d' }
    }
  }),
  voiceBotController.getVoiceAnalytics
);

/**
 * @swagger
 * /api/voice/preferences:
 *   get:
 *     summary: Get voice preferences
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Voice preferences retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   $ref: '#/components/schemas/VoicePreferences'
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/preferences',
  authenticateUser,
  voiceBotController.getVoicePreferences
);

/**
 * @swagger
 * /api/voice/preferences:
 *   put:
 *     summary: Update voice preferences
 *     tags: [Voice Bot]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/VoicePreferences'
 *     responses:
 *       200:
 *         description: Voice preferences updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 data:
 *                   $ref: '#/components/schemas/VoicePreferences'
 *       400:
 *         description: Invalid input data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.put('/preferences',
  authenticateUser,
  validateRequest({
    body: {
      preferredLanguage: { type: 'string', enum: ['en', 'hi', 'ta', 'bn', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'], optional: true },
      voiceSpeed: { type: 'number', min: 0.5, max: 2.0, optional: true },
      voiceGender: { type: 'string', enum: ['male', 'female', 'neutral'], optional: true },
      autoLanguageDetection: { type: 'boolean', optional: true }
    }
  }),
  voiceBotController.updateVoicePreferences
);

module.exports = router; 