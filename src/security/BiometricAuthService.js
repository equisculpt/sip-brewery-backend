/**
 * 🔐 Biometric Authentication Service - Military Grade Security
 * Implements fingerprint, face recognition, and behavioral biometrics
 */

const crypto = require('crypto');
const { Pool } = require('pg');

class BiometricAuthService {
    constructor() {
        this.db = new Pool({
            connectionString: process.env.DATABASE_URL,
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        });

        this.config = {
            fingerprint: { minMatchScore: 0.85, maxAttempts: 3 },
            face: { minMatchScore: 0.90, maxAttempts: 3 },
            behavioral: { keystrokeDynamics: true, mouseDynamics: true, minSamples: 10 },
            security: { encryptionAlgorithm: 'aes-256-gcm', templateEncryption: true }
        };
    }

    async registerBiometricTemplate(userId, biometricType, templateData, metadata = {}) {
        try {
            const encryptedTemplate = this.encryptBiometricData(templateData);
            const templateHash = crypto.createHash('sha256').update(JSON.stringify(templateData)).digest('hex');

            const duplicateCheck = await this.db.query(
                'SELECT user_id FROM biometric_templates WHERE template_hash = $1 AND biometric_type = $2',
                [templateHash, biometricType]
            );

            if (duplicateCheck.rows.length > 0) {
                throw new Error('Biometric template already exists');
            }

            const query = `
                INSERT INTO biometric_templates (
                    user_id, biometric_type, encrypted_template, 
                    template_hash, quality_score, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                RETURNING template_id
            `;

            const result = await this.db.query(query, [
                userId, biometricType, encryptedTemplate,
                templateHash, metadata.qualityScore || 0.9, JSON.stringify(metadata)
            ]);

            await this.logBiometricEvent(userId, 'BIOMETRIC_REGISTERED', 'INFO', { biometricType });

            return { success: true, templateId: result.rows[0].template_id };
        } catch (error) {
            await this.logBiometricEvent(userId, 'BIOMETRIC_REGISTRATION_FAILED', 'ERROR', { error: error.message });
            throw error;
        }
    }

    async authenticateBiometric(userId, biometricType, templateData) {
        try {
            const storedTemplates = await this.db.query(
                'SELECT template_id, encrypted_template FROM biometric_templates WHERE user_id = $1 AND biometric_type = $2 AND is_active = true',
                [userId, biometricType]
            );

            if (storedTemplates.rows.length === 0) {
                throw new Error('No biometric templates found');
            }

            let bestScore = 0;
            let bestMatch = null;

            for (const template of storedTemplates.rows) {
                const decryptedTemplate = this.decryptBiometricData(template.encrypted_template);
                const matchScore = this.compareBiometricTemplates(biometricType, templateData, decryptedTemplate);

                if (matchScore > bestScore) {
                    bestScore = matchScore;
                    bestMatch = template;
                }
            }

            const threshold = this.config[biometricType]?.minMatchScore || 0.85;
            const isAuthenticated = bestScore >= threshold;

            await this.logBiometricEvent(userId, 'BIOMETRIC_AUTH_ATTEMPT', 'INFO', {
                biometricType, matchScore: bestScore, success: isAuthenticated
            });

            if (!isAuthenticated) {
                await this.updateFailedAttempts(userId, biometricType);
                throw new Error('Biometric authentication failed');
            }

            await this.updateSuccessfulAuth(bestMatch.template_id);
            return { success: true, matchScore: bestScore, templateId: bestMatch.template_id };

        } catch (error) {
            await this.logBiometricEvent(userId, 'BIOMETRIC_AUTH_FAILED', 'WARNING', { error: error.message });
            throw error;
        }
    }

    async registerBehavioralProfile(userId, behaviorData) {
        try {
            const { keystroke, mouse } = behaviorData;

            if (keystroke) {
                const keystrokeProfile = this.processKeystrokeDynamics(keystroke);
                await this.storeBehavioralProfile(userId, 'keystroke', keystrokeProfile);
            }

            if (mouse) {
                const mouseProfile = this.processMouseDynamics(mouse);
                await this.storeBehavioralProfile(userId, 'mouse', mouseProfile);
            }

            return { success: true, message: 'Behavioral profile registered' };
        } catch (error) {
            throw error;
        }
    }

    processKeystrokeDynamics(keystrokeData) {
        const { keystrokes } = keystrokeData;
        const profile = { dwellTimes: [], flightTimes: [], averageSpeed: 0 };

        for (let i = 0; i < keystrokes.length; i++) {
            const keystroke = keystrokes[i];
            const dwellTime = keystroke.keyup - keystroke.keydown;
            profile.dwellTimes.push(dwellTime);

            if (i > 0) {
                const flightTime = keystroke.keydown - keystrokes[i - 1].keyup;
                profile.flightTimes.push(flightTime);
            }
        }

        profile.averageSpeed = this.calculateAverage(profile.dwellTimes);
        return profile;
    }

    processMouseDynamics(mouseData) {
        const { movements, clicks } = mouseData;
        const profile = { velocity: [], clickDuration: [] };

        for (let i = 1; i < movements.length; i++) {
            const prev = movements[i - 1];
            const curr = movements[i];
            const distance = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));
            const time = curr.timestamp - prev.timestamp;
            profile.velocity.push(distance / time);
        }

        for (const click of clicks) {
            profile.clickDuration.push(click.duration);
        }

        return profile;
    }

    compareBiometricTemplates(biometricType, template1, template2) {
        if (biometricType === 'fingerprint') {
            return this.compareFingerprints(template1, template2);
        } else if (biometricType === 'face') {
            return this.compareFaceTemplates(template1, template2);
        }
        return 0;
    }

    compareFingerprints(template1, template2) {
        const minutiae1 = template1.minutiae || [];
        const minutiae2 = template2.minutiae || [];
        
        if (minutiae1.length === 0 || minutiae2.length === 0) return 0;

        let matchCount = 0;
        const tolerance = 10;

        for (const point1 of minutiae1) {
            for (const point2 of minutiae2) {
                const distance = Math.sqrt(Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2));
                if (distance <= tolerance) {
                    matchCount++;
                    break;
                }
            }
        }

        return matchCount / Math.max(minutiae1.length, minutiae2.length);
    }

    compareFaceTemplates(template1, template2) {
        const features1 = template1.features || [];
        const features2 = template2.features || [];
        
        if (features1.length !== features2.length || features1.length === 0) return 0;

        let dotProduct = 0, norm1 = 0, norm2 = 0;

        for (let i = 0; i < features1.length; i++) {
            dotProduct += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }

        return norm1 === 0 || norm2 === 0 ? 0 : dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    encryptBiometricData(data) {
        const key = crypto.scryptSync(process.env.BIOMETRIC_ENCRYPTION_KEY || 'default-key', 'salt', 32);
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipher('aes-256-gcm', key);
        
        let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
        encrypted += cipher.final('hex');
        
        return { encrypted, iv: iv.toString('hex'), authTag: cipher.getAuthTag().toString('hex') };
    }

    decryptBiometricData(encryptedData) {
        const key = crypto.scryptSync(process.env.BIOMETRIC_ENCRYPTION_KEY || 'default-key', 'salt', 32);
        const decipher = crypto.createDecipher('aes-256-gcm', key);
        decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
        
        let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        
        return JSON.parse(decrypted);
    }

    async storeBehavioralProfile(userId, profileType, profileData) {
        const query = `
            INSERT INTO behavioral_profiles (user_id, profile_type, profile_data, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id, profile_type)
            DO UPDATE SET profile_data = $3, updated_at = NOW()
        `;
        await this.db.query(query, [userId, profileType, JSON.stringify(profileData)]);
    }

    async updateFailedAttempts(userId, biometricType) {
        await this.db.query(
            'UPDATE biometric_templates SET failed_attempts = failed_attempts + 1 WHERE user_id = $1 AND biometric_type = $2',
            [userId, biometricType]
        );
    }

    async updateSuccessfulAuth(templateId) {
        await this.db.query(
            'UPDATE biometric_templates SET failed_attempts = 0, last_successful_auth = NOW() WHERE template_id = $1',
            [templateId]
        );
    }

    async logBiometricEvent(userId, eventType, severity, metadata) {
        try {
            await this.db.query(
                'INSERT INTO biometric_events (user_id, event_type, severity, metadata, created_at) VALUES ($1, $2, $3, $4, NOW())',
                [userId, eventType, severity, JSON.stringify(metadata)]
            );
        } catch (error) {
            console.error('Error logging biometric event:', error);
        }
    }

    calculateAverage(arr) {
        return arr.length > 0 ? arr.reduce((sum, val) => sum + val, 0) / arr.length : 0;
    }

    generateBiometricChallenge(biometricType) {
        const challenge = {
            challengeId: crypto.randomUUID(),
            biometricType,
            timestamp: Date.now(),
            expiresAt: Date.now() + (5 * 60 * 1000),
            nonce: crypto.randomBytes(32).toString('hex')
        };

        if (biometricType === 'face') {
            challenge.livenessChallenge = {
                actions: ['blink', 'turn_left', 'turn_right'],
                sequence: this.generateRandomSequence(['blink', 'turn_left', 'turn_right'])
            };
        }

        return challenge;
    }

    generateRandomSequence(actions) {
        const sequence = [];
        const count = Math.floor(Math.random() * 3) + 2;
        
        for (let i = 0; i < count; i++) {
            sequence.push(actions[Math.floor(Math.random() * actions.length)]);
        }
        
        return sequence;
    }
}

module.exports = new BiometricAuthService();
