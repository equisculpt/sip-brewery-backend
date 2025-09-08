const express = require('express');
const router = express.Router();
const regionalLanguageController = require('../controllers/regionalLanguageController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * components:
 *   schemas:
 *     LanguagePreference:
 *       type: object
 *       properties:
 *         languageCode:
 *           type: string
 *           description: Language code (e.g., 'hi', 'ta', 'te')
 *         preferences:
 *           type: object
 *           properties:
 *             voiceEnabled:
 *               type: boolean
 *             textEnabled:
 *               type: boolean
 *     TranslationRequest:
 *       type: object
 *       properties:
 *         terms:
 *           type: array
 *           items:
 *             type: string
 *         targetLanguage:
 *           type: string
 *     VoiceCommandRequest:
 *       type: object
 *       properties:
 *         audioData:
 *           type: string
 *         languageCode:
 *           type: string
 *     LocalizedContentRequest:
 *       type: object
 *       properties:
 *         contentType:
 *           type: string
 *           enum: [greeting, investment_advice, market_update, portfolio_summary, educational_content]
 *         languageCode:
 *           type: string
 *         context:
 *           type: object
 */

/**
 * @swagger
 * /api/regional/languages:
 *   get:
 *     summary: Get supported languages
 *     tags: [Regional Language]
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
 *                           name:
 *                             type: string
 *                           nativeName:
 *                             type: string
 *                           script:
 *                             type: string
 *                           regions:
 *                             type: array
 *                             items:
 *                               type: string
 *                           voiceSupport:
 *                             type: boolean
 *                           textSupport:
 *                             type: boolean
 *                     totalLanguages:
 *                       type: number
 *                     voiceSupported:
 *                       type: number
 *                     textSupported:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/languages', regionalLanguageController.getSupportedLanguages);

/**
 * @swagger
 * /api/regional/language-preference:
 *   post:
 *     summary: Set user language preference
 *     tags: [Regional Language]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/LanguagePreference'
 *     responses:
 *       200:
 *         description: Language preference set successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/language-preference', authenticateUser, regionalLanguageController.setLanguagePreference);

/**
 * @swagger
 * /api/regional/language-preference:
 *   get:
 *     summary: Get user language preferences
 *     tags: [Regional Language]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Language preferences retrieved successfully
 *       401:
 *         description: Unauthorized
 */
router.get('/language-preference', authenticateUser, regionalLanguageController.getLanguagePreference);

/**
 * @swagger
 * /api/regional/translate:
 *   post:
 *     summary: Translate investment terms
 *     tags: [Regional Language]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/TranslationRequest'
 *     responses:
 *       200:
 *         description: Terms translated successfully
 *       400:
 *         description: Bad request
 */
router.post('/translate', regionalLanguageController.translateTerms);

/**
 * @swagger
 * /api/regional/cultural-context/{languageCode}:
 *   get:
 *     summary: Get cultural context for language
 *     tags: [Regional Language]
 *     parameters:
 *       - in: path
 *         name: languageCode
 *         required: true
 *         schema:
 *           type: string
 *         description: Language code
 *     responses:
 *       200:
 *         description: Cultural context retrieved successfully
 *       400:
 *         description: Bad request
 */
router.get('/cultural-context/:languageCode', regionalLanguageController.getCulturalContext);

/**
 * @swagger
 * /api/regional/voice-command:
 *   post:
 *     summary: Process voice command
 *     tags: [Regional Language]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/VoiceCommandRequest'
 *     responses:
 *       200:
 *         description: Voice command processed successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/voice-command', authenticateUser, regionalLanguageController.processVoiceCommand);

/**
 * @swagger
 * /api/regional/localized-content:
 *   post:
 *     summary: Generate localized content
 *     tags: [Regional Language]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/LocalizedContentRequest'
 *     responses:
 *       200:
 *         description: Localized content generated successfully
 *       400:
 *         description: Bad request
 */
router.post('/localized-content', regionalLanguageController.generateLocalizedContent);

/**
 * @swagger
 * /api/regional/investment-preferences/{languageCode}:
 *   get:
 *     summary: Get regional investment preferences
 *     tags: [Regional Language]
 *     parameters:
 *       - in: path
 *         name: languageCode
 *         required: true
 *         schema:
 *           type: string
 *         description: Language code
 *     responses:
 *       200:
 *         description: Regional investment preferences retrieved successfully
 *       400:
 *         description: Bad request
 */
router.get('/investment-preferences/:languageCode', regionalLanguageController.getRegionalInvestmentPreferences);

/**
 * @swagger
 * /api/regional/voice-commands:
 *   post:
 *     summary: Create voice commands for language
 *     tags: [Regional Language]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               languageCode:
 *                 type: string
 *     responses:
 *       200:
 *         description: Voice commands created successfully
 *       400:
 *         description: Bad request
 */
router.post('/voice-commands', regionalLanguageController.createVoiceCommands);

module.exports = router; 