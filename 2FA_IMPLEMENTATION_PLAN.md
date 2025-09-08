# 🔐 Two-Factor Authentication (2FA) Implementation Plan

## 🎯 **COMPLETE 2FA SYSTEM ARCHITECTURE**

---

## 📋 **SYSTEM OVERVIEW**

### **Authentication Flow**:
1. **Primary Login**: Email + Password
2. **2FA Challenge**: TOTP/SMS verification
3. **Session Management**: JWT tokens with refresh
4. **Backup Options**: Recovery codes + SMS fallback

### **Security Features**:
- **TOTP-based**: Google Authenticator/Authy compatible
- **Backup Codes**: 10 single-use recovery codes
- **SMS Fallback**: For lost authenticator devices
- **Rate Limiting**: Prevent brute force attacks
- **Session Management**: Secure token handling

---

## 🛠️ **TECHNICAL IMPLEMENTATION**

### **1. Required Dependencies**

```bash
# Core 2FA Libraries
npm install speakeasy qrcode
npm install twilio  # for SMS backup
npm install jsonwebtoken bcryptjs
npm install express-rate-limit
npm install redis  # for session management

# Additional Security
npm install helmet cors
npm install express-validator
npm install crypto-js
```

### **2. Database Schema**

```sql
-- Users table with 2FA fields
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    is_2fa_enabled BOOLEAN DEFAULT FALSE,
    totp_secret VARCHAR(255),  -- encrypted TOTP secret
    backup_codes TEXT[],       -- array of hashed backup codes
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2FA Sessions table
CREATE TABLE two_fa_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE,
    attempts INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Login attempts tracking
CREATE TABLE login_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255),
    ip_address INET,
    attempt_type VARCHAR(50), -- 'password', 'totp', 'sms', 'backup_code'
    success BOOLEAN,
    attempted_at TIMESTAMP DEFAULT NOW()
);

-- Backup codes usage tracking
CREATE TABLE backup_code_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    code_hash VARCHAR(255),
    used_at TIMESTAMP DEFAULT NOW(),
    ip_address INET
);
```

### **3. Core 2FA Service Implementation**

```javascript
// services/TwoFactorService.js
const speakeasy = require('speakeasy');
const QRCode = require('qrcode');
const crypto = require('crypto');
const bcrypt = require('bcryptjs');

class TwoFactorService {
    constructor() {
        this.issuer = 'SIP Brewery';
        this.algorithm = 'sha1';
        this.digits = 6;
        this.period = 30;
    }

    // Generate TOTP secret for new user
    generateSecret(userEmail) {
        const secret = speakeasy.generateSecret({
            issuer: this.issuer,
            name: userEmail,
            length: 32
        });

        return {
            secret: secret.base32,
            otpauth_url: secret.otpauth_url,
            manual_entry_key: secret.base32
        };
    }

    // Generate QR code for secret
    async generateQRCode(otpauth_url) {
        try {
            const qrCodeDataURL = await QRCode.toDataURL(otpauth_url);
            return qrCodeDataURL;
        } catch (error) {
            throw new Error('Failed to generate QR code');
        }
    }

    // Verify TOTP token
    verifyToken(secret, token) {
        return speakeasy.totp.verify({
            secret: secret,
            encoding: 'base32',
            token: token,
            window: 2  // Allow 2 time steps (60 seconds) tolerance
        });
    }

    // Generate backup codes
    generateBackupCodes(count = 10) {
        const codes = [];
        for (let i = 0; i < count; i++) {
            const code = crypto.randomBytes(4).toString('hex').toUpperCase();
            codes.push(code);
        }
        return codes;
    }

    // Hash backup codes for storage
    async hashBackupCodes(codes) {
        const hashedCodes = [];
        for (const code of codes) {
            const hash = await bcrypt.hash(code, 12);
            hashedCodes.push(hash);
        }
        return hashedCodes;
    }

    // Verify backup code
    async verifyBackupCode(providedCode, hashedCodes) {
        for (const hashedCode of hashedCodes) {
            const isValid = await bcrypt.compare(providedCode, hashedCode);
            if (isValid) {
                return { valid: true, codeHash: hashedCode };
            }
        }
        return { valid: false, codeHash: null };
    }

    // Generate SMS OTP (fallback)
    generateSMSOTP() {
        return Math.floor(100000 + Math.random() * 900000).toString();
    }
}

module.exports = new TwoFactorService();
```

### **4. SMS Service Integration**

```javascript
// services/SMSService.js
const twilio = require('twilio');

class SMSService {
    constructor() {
        this.client = twilio(
            process.env.TWILIO_ACCOUNT_SID,
            process.env.TWILIO_AUTH_TOKEN
        );
        this.fromNumber = process.env.TWILIO_PHONE_NUMBER;
    }

    async sendOTP(phoneNumber, otp) {
        try {
            const message = await this.client.messages.create({
                body: `Your SIP Brewery verification code is: ${otp}. Valid for 5 minutes.`,
                from: this.fromNumber,
                to: phoneNumber
            });

            return {
                success: true,
                messageId: message.sid,
                status: message.status
            };
        } catch (error) {
            console.error('SMS sending failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async sendSecurityAlert(phoneNumber, message) {
        try {
            await this.client.messages.create({
                body: `SIP Brewery Security Alert: ${message}`,
                from: this.fromNumber,
                to: phoneNumber
            });
        } catch (error) {
            console.error('Security alert SMS failed:', error);
        }
    }
}

module.exports = new SMSService();
```

### **5. Authentication Middleware**

```javascript
// middleware/auth2FA.js
const jwt = require('jsonwebtoken');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');

// Rate limiting for 2FA attempts
const twoFALimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // limit each IP to 5 requests per windowMs
    message: 'Too many 2FA attempts, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
});

// Validation rules for 2FA
const validate2FAToken = [
    body('token')
        .isLength({ min: 6, max: 6 })
        .isNumeric()
        .withMessage('Token must be 6 digits'),
    body('sessionToken')
        .isLength({ min: 32 })
        .withMessage('Invalid session token')
];

// Middleware to verify JWT tokens
const verifyJWT = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({
            success: false,
            message: 'Access token required'
        });
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({
            success: false,
            message: 'Invalid or expired token'
        });
    }
};

// Middleware to check if 2FA is required
const require2FA = async (req, res, next) => {
    try {
        const user = await User.findById(req.user.id);
        
        if (user.is_2fa_enabled && !req.user.is_2fa_verified) {
            return res.status(403).json({
                success: false,
                message: '2FA verification required',
                requires_2fa: true
            });
        }
        
        next();
    } catch (error) {
        return res.status(500).json({
            success: false,
            message: 'Authentication error'
        });
    }
};

module.exports = {
    twoFALimiter,
    validate2FAToken,
    verifyJWT,
    require2FA
};
```

---

## 🔗 **API ENDPOINTS IMPLEMENTATION**

### **1. Setup 2FA Endpoint**

```javascript
// POST /api/auth/2fa/setup
app.post('/api/auth/2fa/setup', verifyJWT, async (req, res) => {
    try {
        const user = await User.findById(req.user.id);
        
        if (user.is_2fa_enabled) {
            return res.status(400).json({
                success: false,
                message: '2FA is already enabled'
            });
        }

        // Generate new secret
        const secretData = TwoFactorService.generateSecret(user.email);
        const qrCode = await TwoFactorService.generateQRCode(secretData.otpauth_url);

        // Store encrypted secret temporarily (not enabled yet)
        const encryptedSecret = encrypt(secretData.secret);
        await User.updateOne(
            { _id: user._id },
            { totp_secret: encryptedSecret }
        );

        res.json({
            success: true,
            data: {
                qr_code: qrCode,
                manual_entry_key: secretData.manual_entry_key,
                backup_codes: null // Will be provided after verification
            }
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Failed to setup 2FA'
        });
    }
});
```

### **2. Verify and Enable 2FA**

```javascript
// POST /api/auth/2fa/verify-setup
app.post('/api/auth/2fa/verify-setup', 
    verifyJWT, 
    validate2FAToken, 
    async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                success: false,
                errors: errors.array()
            });
        }

        const { token } = req.body;
        const user = await User.findById(req.user.id);
        
        if (!user.totp_secret) {
            return res.status(400).json({
                success: false,
                message: 'No 2FA setup in progress'
            });
        }

        // Decrypt and verify token
        const secret = decrypt(user.totp_secret);
        const isValid = TwoFactorService.verifyToken(secret, token);

        if (!isValid) {
            return res.status(400).json({
                success: false,
                message: 'Invalid verification code'
            });
        }

        // Generate backup codes
        const backupCodes = TwoFactorService.generateBackupCodes();
        const hashedBackupCodes = await TwoFactorService.hashBackupCodes(backupCodes);

        // Enable 2FA
        await User.updateOne(
            { _id: user._id },
            {
                is_2fa_enabled: true,
                backup_codes: hashedBackupCodes
            }
        );

        res.json({
            success: true,
            message: '2FA enabled successfully',
            data: {
                backup_codes: backupCodes
            }
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Failed to verify 2FA setup'
        });
    }
});
```

### **3. Login with 2FA**

```javascript
// POST /api/auth/login
app.post('/api/auth/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        
        // Verify email and password
        const user = await User.findOne({ email });
        if (!user || !await bcrypt.compare(password, user.password_hash)) {
            return res.status(401).json({
                success: false,
                message: 'Invalid credentials'
            });
        }

        // Check if 2FA is enabled
        if (user.is_2fa_enabled) {
            // Create 2FA session
            const sessionToken = crypto.randomBytes(32).toString('hex');
            const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

            await TwoFASession.create({
                user_id: user.id,
                session_token: sessionToken,
                expires_at: expiresAt
            });

            return res.json({
                success: true,
                requires_2fa: true,
                session_token: sessionToken,
                message: 'Please provide 2FA verification'
            });
        }

        // Generate JWT for non-2FA users
        const token = jwt.sign(
            { id: user.id, email: user.email, is_2fa_verified: true },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );

        res.json({
            success: true,
            token,
            user: {
                id: user.id,
                email: user.email,
                is_2fa_enabled: false
            }
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Login failed'
        });
    }
});
```

### **4. Verify 2FA Token**

```javascript
// POST /api/auth/2fa/verify
app.post('/api/auth/2fa/verify', 
    twoFALimiter,
    validate2FAToken,
    async (req, res) => {
    try {
        const { token, sessionToken, type = 'totp' } = req.body;
        
        // Find and validate session
        const session = await TwoFASession.findOne({
            session_token: sessionToken,
            expires_at: { $gt: new Date() },
            is_verified: false
        });

        if (!session) {
            return res.status(400).json({
                success: false,
                message: 'Invalid or expired session'
            });
        }

        const user = await User.findById(session.user_id);
        let isValid = false;

        if (type === 'totp') {
            // Verify TOTP token
            const secret = decrypt(user.totp_secret);
            isValid = TwoFactorService.verifyToken(secret, token);
        } else if (type === 'backup_code') {
            // Verify backup code
            const result = await TwoFactorService.verifyBackupCode(
                token, 
                user.backup_codes
            );
            isValid = result.valid;
            
            if (isValid) {
                // Remove used backup code
                await User.updateOne(
                    { _id: user._id },
                    { $pull: { backup_codes: result.codeHash } }
                );
                
                // Log backup code usage
                await BackupCodeUsage.create({
                    user_id: user._id,
                    code_hash: result.codeHash,
                    ip_address: req.ip
                });
            }
        }

        if (!isValid) {
            // Increment attempt counter
            await TwoFASession.updateOne(
                { _id: session._id },
                { $inc: { attempts: 1 } }
            );

            return res.status(400).json({
                success: false,
                message: 'Invalid verification code'
            });
        }

        // Mark session as verified
        await TwoFASession.updateOne(
            { _id: session._id },
            { is_verified: true }
        );

        // Generate JWT token
        const jwtToken = jwt.sign(
            { 
                id: user.id, 
                email: user.email, 
                is_2fa_verified: true 
            },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );

        res.json({
            success: true,
            token: jwtToken,
            user: {
                id: user.id,
                email: user.email,
                is_2fa_enabled: true
            }
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: '2FA verification failed'
        });
    }
});
```

---

## 🎯 **FRONTEND INTEGRATION**

### **1. React 2FA Setup Component**

```jsx
// components/TwoFactorSetup.jsx
import React, { useState } from 'react';
import QRCode from 'qrcode.react';

const TwoFactorSetup = () => {
    const [step, setStep] = useState(1); // 1: QR, 2: Verify, 3: Backup Codes
    const [qrCode, setQrCode] = useState('');
    const [manualKey, setManualKey] = useState('');
    const [verificationCode, setVerificationCode] = useState('');
    const [backupCodes, setBackupCodes] = useState([]);

    const setup2FA = async () => {
        try {
            const response = await fetch('/api/auth/2fa/setup', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            
            const data = await response.json();
            if (data.success) {
                setQrCode(data.data.qr_code);
                setManualKey(data.data.manual_entry_key);
                setStep(2);
            }
        } catch (error) {
            console.error('2FA setup failed:', error);
        }
    };

    const verify2FA = async () => {
        try {
            const response = await fetch('/api/auth/2fa/verify-setup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ token: verificationCode })
            });
            
            const data = await response.json();
            if (data.success) {
                setBackupCodes(data.data.backup_codes);
                setStep(3);
            }
        } catch (error) {
            console.error('2FA verification failed:', error);
        }
    };

    return (
        <div className="2fa-setup">
            {step === 1 && (
                <div>
                    <h3>Enable Two-Factor Authentication</h3>
                    <p>Secure your account with 2FA</p>
                    <button onClick={setup2FA}>Setup 2FA</button>
                </div>
            )}
            
            {step === 2 && (
                <div>
                    <h3>Scan QR Code</h3>
                    <div className="qr-code">
                        <img src={qrCode} alt="2FA QR Code" />
                    </div>
                    <p>Manual Entry Key: <code>{manualKey}</code></p>
                    
                    <div>
                        <input
                            type="text"
                            placeholder="Enter 6-digit code"
                            value={verificationCode}
                            onChange={(e) => setVerificationCode(e.target.value)}
                            maxLength={6}
                        />
                        <button onClick={verify2FA}>Verify</button>
                    </div>
                </div>
            )}
            
            {step === 3 && (
                <div>
                    <h3>Save Your Backup Codes</h3>
                    <p>Store these codes safely. Each can only be used once.</p>
                    <div className="backup-codes">
                        {backupCodes.map((code, index) => (
                            <div key={index} className="backup-code">{code}</div>
                        ))}
                    </div>
                    <button onClick={() => window.print()}>Print Codes</button>
                </div>
            )}
        </div>
    );
};

export default TwoFactorSetup;
```

---

## 📱 **MOBILE APP INTEGRATION**

### **1. React Native 2FA Component**

```jsx
// components/TwoFactorAuth.jsx (React Native)
import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const TwoFactorAuth = ({ sessionToken, onSuccess }) => {
    const [code, setCode] = useState('');
    const [loading, setLoading] = useState(false);

    const verify2FA = async () => {
        if (code.length !== 6) {
            Alert.alert('Error', 'Please enter a 6-digit code');
            return;
        }

        setLoading(true);
        try {
            const response = await fetch('/api/auth/2fa/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    token: code,
                    sessionToken: sessionToken,
                    type: 'totp'
                })
            });

            const data = await response.json();
            if (data.success) {
                await AsyncStorage.setItem('authToken', data.token);
                onSuccess(data.user);
            } else {
                Alert.alert('Error', data.message);
            }
        } catch (error) {
            Alert.alert('Error', 'Verification failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Two-Factor Authentication</Text>
            <Text style={styles.subtitle}>
                Enter the 6-digit code from your authenticator app
            </Text>
            
            <TextInput
                style={styles.input}
                value={code}
                onChangeText={setCode}
                placeholder="000000"
                keyboardType="numeric"
                maxLength={6}
                autoFocus
            />
            
            <TouchableOpacity 
                style={styles.button} 
                onPress={verify2FA}
                disabled={loading || code.length !== 6}
            >
                <Text style={styles.buttonText}>
                    {loading ? 'Verifying...' : 'Verify'}
                </Text>
            </TouchableOpacity>
        </View>
    );
};
```

---

## ⚙️ **ENVIRONMENT CONFIGURATION**

```bash
# .env file
# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-here
JWT_REFRESH_SECRET=your-refresh-token-secret-here

# Twilio Configuration (for SMS)
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=+1234567890

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/sipbrewery

# Redis Configuration (for sessions)
REDIS_URL=redis://localhost:6379

# Encryption Key (for TOTP secrets)
ENCRYPTION_KEY=your-32-character-encryption-key

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000  # 15 minutes
RATE_LIMIT_MAX_ATTEMPTS=5
```

---

## 🧪 **TESTING STRATEGY**

### **1. Unit Tests**

```javascript
// tests/twoFactor.test.js
const TwoFactorService = require('../services/TwoFactorService');

describe('TwoFactorService', () => {
    test('should generate valid TOTP secret', () => {
        const secret = TwoFactorService.generateSecret('test@example.com');
        expect(secret.secret).toBeDefined();
        expect(secret.otpauth_url).toContain('test@example.com');
    });

    test('should verify valid TOTP token', () => {
        const secret = 'JBSWY3DPEHPK3PXP';
        const token = speakeasy.totp({
            secret: secret,
            encoding: 'base32'
        });
        
        const isValid = TwoFactorService.verifyToken(secret, token);
        expect(isValid).toBe(true);
    });

    test('should generate backup codes', () => {
        const codes = TwoFactorService.generateBackupCodes(10);
        expect(codes).toHaveLength(10);
        expect(codes[0]).toMatch(/^[A-F0-9]{8}$/);
    });
});
```

### **2. Integration Tests**

```javascript
// tests/auth2FA.integration.test.js
const request = require('supertest');
const app = require('../app');

describe('2FA Authentication', () => {
    let userToken;
    let sessionToken;

    beforeEach(async () => {
        // Setup test user with 2FA enabled
        userToken = await createTestUserWithToken();
    });

    test('should require 2FA for enabled users', async () => {
        const response = await request(app)
            .post('/api/auth/login')
            .send({
                email: 'test@example.com',
                password: 'password123'
            });

        expect(response.status).toBe(200);
        expect(response.body.requires_2fa).toBe(true);
        expect(response.body.session_token).toBeDefined();
    });

    test('should verify valid 2FA token', async () => {
        const response = await request(app)
            .post('/api/auth/2fa/verify')
            .send({
                token: '123456',
                sessionToken: sessionToken,
                type: 'totp'
            });

        expect(response.status).toBe(200);
        expect(response.body.token).toBeDefined();
    });
});
```

---

## 🚀 **DEPLOYMENT CHECKLIST**

### **Pre-Deployment**:
- [ ] Environment variables configured
- [ ] Database schema deployed
- [ ] Redis server configured
- [ ] Twilio account setup and verified
- [ ] SSL certificates installed
- [ ] Rate limiting configured

### **Security Checklist**:
- [ ] TOTP secrets encrypted at rest
- [ ] Backup codes properly hashed
- [ ] Rate limiting on all 2FA endpoints
- [ ] Session tokens have proper expiration
- [ ] Audit logging implemented
- [ ] Failed attempt monitoring

### **Testing Checklist**:
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Load testing completed
- [ ] Security penetration testing
- [ ] Mobile app testing (iOS/Android)

This comprehensive 2FA implementation provides enterprise-grade security with multiple fallback options and robust testing coverage.
