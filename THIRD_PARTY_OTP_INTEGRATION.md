# 📱 Third-Party OTP Integration Guide

## 🎯 **RECOMMENDED PROVIDER: MSG91**

**Why MSG91?**
- ✅ **Cost**: ₹0.20-0.25 per SMS (vs Twilio ₹0.50)
- ✅ **Reliability**: 98%+ delivery rate in India
- ✅ **Compliance**: TRAI registered, DLT compliant
- ✅ **Features**: SMS, Email, Voice, WhatsApp OTP
- ✅ **Support**: 24/7 Indian customer support

---

## 🚀 **QUICK IMPLEMENTATION**

### **1. Install Dependencies**
```bash
npm install msg91-nodejs axios validator express-validator express-rate-limit
```

### **2. Environment Setup**
```bash
# Add to .env
MSG91_API_KEY=your_msg91_api_key
MSG91_SENDER_ID=SIPBREW
MSG91_ROUTE=4
OTP_SALT=your_secure_salt_here
```

### **3. OTP Service**
```javascript
// src/services/MSG91OTPService.js
const axios = require('axios');
const crypto = require('crypto');

class MSG91OTPService {
    constructor() {
        this.apiKey = process.env.MSG91_API_KEY;
        this.senderId = process.env.MSG91_SENDER_ID;
        this.otpStorage = new Map(); // Use Redis in production
    }

    async sendSMSOTP(phoneNumber, purpose = 'LOGIN') {
        const otp = crypto.randomInt(100000, 999999).toString();
        const otpId = crypto.randomUUID();
        
        try {
            // Send via MSG91
            const response = await axios.post('https://api.msg91.com/api/v5/otp', {
                template_id: 'your_template_id',
                mobile: phoneNumber,
                authkey: this.apiKey,
                otp: otp,
                sender: this.senderId
            });

            if (response.data.type === 'success') {
                // Store OTP
                this.otpStorage.set(otpId, {
                    otp: this.hashOTP(otp),
                    phoneNumber,
                    purpose,
                    expiresAt: new Date(Date.now() + 5 * 60 * 1000),
                    attempts: 0
                });

                return {
                    success: true,
                    otpId,
                    message: 'OTP sent successfully',
                    expiresAt: new Date(Date.now() + 5 * 60 * 1000)
                };
            }
        } catch (error) {
            console.error('MSG91 Error:', error);
            throw new Error('Failed to send OTP');
        }
    }

    async verifyOTP(otpId, providedOTP, phoneNumber) {
        const otpData = this.otpStorage.get(otpId);
        
        if (!otpData || new Date() > otpData.expiresAt) {
            return { success: false, message: 'OTP expired or invalid' };
        }

        if (otpData.attempts >= 3) {
            this.otpStorage.delete(otpId);
            return { success: false, message: 'Maximum attempts exceeded' };
        }

        const hashedOTP = this.hashOTP(providedOTP);
        const isValid = crypto.timingSafeEqual(
            Buffer.from(otpData.otp, 'hex'),
            Buffer.from(hashedOTP, 'hex')
        );

        if (isValid && otpData.phoneNumber === phoneNumber) {
            this.otpStorage.delete(otpId);
            return { success: true, message: 'OTP verified successfully' };
        } else {
            otpData.attempts += 1;
            return { success: false, message: 'Invalid OTP' };
        }
    }

    hashOTP(otp) {
        return crypto.createHash('sha256')
            .update(otp + process.env.OTP_SALT)
            .digest('hex');
    }
}

module.exports = new MSG91OTPService();
```

### **4. API Routes**
```javascript
// src/routes/otpRoutes.js
const express = require('express');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const MSG91OTPService = require('../services/MSG91OTPService');

const router = express.Router();

// Rate limiting
const otpLimit = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 3, // 3 OTP requests per minute
    message: { success: false, message: 'Too many OTP requests' }
});

// Send OTP
router.post('/otp/send', 
    otpLimit,
    [body('phoneNumber').isMobilePhone('en-IN')],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                success: false,
                message: 'Invalid phone number'
            });
        }

        try {
            const { phoneNumber, purpose } = req.body;
            const result = await MSG91OTPService.sendSMSOTP(phoneNumber, purpose);
            res.json(result);
        } catch (error) {
            res.status(500).json({
                success: false,
                message: 'Failed to send OTP'
            });
        }
    }
);

// Verify OTP
router.post('/otp/verify',
    [
        body('otpId').isUUID(),
        body('otp').isLength({ min: 6, max: 6 }).isNumeric(),
        body('phoneNumber').isMobilePhone('en-IN')
    ],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                success: false,
                message: 'Invalid input'
            });
        }

        try {
            const { otpId, otp, phoneNumber } = req.body;
            const result = await MSG91OTPService.verifyOTP(otpId, otp, phoneNumber);
            res.json(result);
        } catch (error) {
            res.status(500).json({
                success: false,
                message: 'Verification failed'
            });
        }
    }
);

module.exports = router;
```

### **5. Frontend Component**
```javascript
// components/OTPVerification.jsx
import React, { useState } from 'react';

const OTPVerification = ({ onVerified }) => {
    const [step, setStep] = useState('phone');
    const [phoneNumber, setPhoneNumber] = useState('');
    const [otp, setOtp] = useState('');
    const [otpId, setOtpId] = useState('');
    const [loading, setLoading] = useState(false);

    const sendOTP = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/otp/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phoneNumber, purpose: 'LOGIN' })
            });
            
            const data = await response.json();
            if (data.success) {
                setOtpId(data.otpId);
                setStep('verify');
            }
        } catch (error) {
            console.error('Send OTP failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const verifyOTP = async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/otp/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ otpId, otp, phoneNumber })
            });
            
            const data = await response.json();
            if (data.success) {
                onVerified && onVerified(phoneNumber);
            }
        } catch (error) {
            console.error('Verify OTP failed:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="otp-verification">
            {step === 'phone' ? (
                <div>
                    <input
                        type="tel"
                        value={phoneNumber}
                        onChange={(e) => setPhoneNumber(e.target.value)}
                        placeholder="Enter phone number"
                    />
                    <button onClick={sendOTP} disabled={loading}>
                        {loading ? 'Sending...' : 'Send OTP'}
                    </button>
                </div>
            ) : (
                <div>
                    <input
                        type="text"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value)}
                        placeholder="Enter 6-digit OTP"
                        maxLength="6"
                    />
                    <button onClick={verifyOTP} disabled={loading}>
                        {loading ? 'Verifying...' : 'Verify OTP'}
                    </button>
                </div>
            )}
        </div>
    );
};

export default OTPVerification;
```

---

## 💰 **COST BENEFITS**

### **MSG91 vs Competitors**
- **MSG91**: ₹0.22 per SMS
- **Twilio**: ₹0.50 per SMS
- **TextLocal**: ₹0.30 per SMS
- **Savings**: 56% cheaper than Twilio

### **Monthly Cost (50K SMS)**
- **MSG91**: ₹11,000
- **Twilio**: ₹25,000
- **Annual Savings**: ₹1,68,000

---

## 🔧 **INTEGRATION STEPS**

1. **Register MSG91 Account**: Get API key and sender ID
2. **Install Dependencies**: `npm install msg91-nodejs`
3. **Add Environment Variables**: API key and configuration
4. **Deploy Service**: Copy MSG91OTPService.js
5. **Add Routes**: Mount OTP routes in main server
6. **Frontend Integration**: Use OTPVerification component
7. **Test**: Verify SMS delivery and OTP verification

---

## 🎯 **FEATURES**

✅ **SMS OTP**: 6-digit secure OTP via MSG91
✅ **Rate Limiting**: 3 OTP requests per minute
✅ **Secure Storage**: Hashed OTP with salt
✅ **Timing-Safe Verification**: Prevents timing attacks
✅ **Automatic Expiry**: 5-minute OTP validity
✅ **Attempt Limiting**: Max 3 verification attempts
✅ **Indian Compliance**: TRAI registered provider

**Ready for production with enterprise-grade reliability!** 🚀
