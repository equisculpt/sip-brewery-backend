-- 📧📱 Email Verification & Custom OTP Database Schema
-- Support for email verification and self-hosted OTP service

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- EMAIL VERIFICATION TABLES
-- =====================================================

-- Email verification requests
CREATE TABLE IF NOT EXISTS email_verifications (
    verification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    hashed_code VARCHAR(64) NOT NULL, -- Hashed OTP or token
    purpose VARCHAR(50) NOT NULL CHECK (purpose IN (
        'EMAIL_VERIFICATION', 'PASSWORD_RESET', 'LOGIN_VERIFICATION', 
        'ACCOUNT_SECURITY', 'TWO_FACTOR_AUTH'
    )),
    verification_type VARCHAR(20) NOT NULL CHECK (verification_type IN ('OTP', 'MAGIC_LINK')),
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 5,
    is_verified BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    is_locked BOOLEAN DEFAULT false,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN (
        'PENDING', 'VERIFIED', 'EXPIRED', 'LOCKED', 'CANCELLED'
    )),
    expires_at TIMESTAMP NOT NULL,
    verified_at TIMESTAMP,
    last_attempt TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_email_verifications_email (email),
    INDEX idx_email_verifications_user_id (user_id),
    INDEX idx_email_verifications_status (status, is_active),
    INDEX idx_email_verifications_expires (expires_at, is_active)
);

-- Email verification events log
CREATE TABLE IF NOT EXISTS email_verification_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    verification_id UUID REFERENCES email_verifications(verification_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_email_events_verification (verification_id),
    INDEX idx_email_events_user (user_id, created_at),
    INDEX idx_email_events_type (event_type, created_at)
);

-- =====================================================
-- CUSTOM OTP SERVICE TABLES
-- =====================================================

-- Custom OTP requests (SMS & Email)
CREATE TABLE IF NOT EXISTS custom_otps (
    otp_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    otp_type VARCHAR(10) NOT NULL CHECK (otp_type IN ('SMS', 'EMAIL')),
    recipient VARCHAR(255) NOT NULL, -- Phone number or email
    hashed_otp VARCHAR(64) NOT NULL, -- Securely hashed OTP
    purpose VARCHAR(50) NOT NULL CHECK (purpose IN (
        'LOGIN', 'SIGNUP', 'PASSWORD_RESET', 'TRANSACTION', 
        'TWO_FACTOR', 'EMAIL_VERIFICATION', 'PHONE_VERIFICATION'
    )),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3, -- SMS: 3, Email: 5
    is_verified BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    is_locked BOOLEAN DEFAULT false,
    status VARCHAR(20) DEFAULT 'SENT' CHECK (status IN (
        'SENT', 'DELIVERED', 'VERIFIED', 'EXPIRED', 'LOCKED', 'FAILED'
    )),
    delivery_status VARCHAR(20) DEFAULT 'PENDING' CHECK (delivery_status IN (
        'PENDING', 'SENT', 'DELIVERED', 'FAILED', 'UNKNOWN'
    )),
    provider_name VARCHAR(50), -- SMS provider used
    provider_message_id VARCHAR(100), -- Provider's message ID
    delivery_attempts INTEGER DEFAULT 0,
    expires_at TIMESTAMP NOT NULL,
    verified_at TIMESTAMP,
    last_attempt TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_custom_otps_recipient (recipient, otp_type),
    INDEX idx_custom_otps_user_id (user_id),
    INDEX idx_custom_otps_status (status, is_active),
    INDEX idx_custom_otps_expires (expires_at, is_active),
    INDEX idx_custom_otps_purpose (purpose, created_at)
);

-- OTP delivery tracking
CREATE TABLE IF NOT EXISTS otp_delivery_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    otp_id UUID NOT NULL REFERENCES custom_otps(otp_id) ON DELETE CASCADE,
    provider_name VARCHAR(50) NOT NULL,
    delivery_attempt INTEGER NOT NULL,
    request_payload JSONB,
    response_payload JSONB,
    http_status INTEGER,
    delivery_status VARCHAR(20) NOT NULL,
    error_message TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_otp_delivery_otp_id (otp_id),
    INDEX idx_otp_delivery_provider (provider_name, created_at),
    INDEX idx_otp_delivery_status (delivery_status, created_at)
);

-- SMS provider configuration and status
CREATE TABLE IF NOT EXISTS sms_providers (
    provider_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider_name VARCHAR(50) NOT NULL UNIQUE,
    provider_type VARCHAR(20) NOT NULL CHECK (provider_type IN (
        'PRIMARY', 'BACKUP', 'FALLBACK'
    )),
    endpoint_url VARCHAR(500) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 1, -- Lower number = higher priority
    success_rate DECIMAL(5,2) DEFAULT 0.00, -- Percentage
    avg_response_time INTEGER DEFAULT 0, -- Milliseconds
    last_success TIMESTAMP,
    last_failure TIMESTAMP,
    total_sent INTEGER DEFAULT 0,
    total_delivered INTEGER DEFAULT 0,
    total_failed INTEGER DEFAULT 0,
    configuration JSONB DEFAULT '{}', -- Provider-specific config
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_sms_providers_active (is_active, priority),
    INDEX idx_sms_providers_success_rate (success_rate DESC)
);

-- Email templates for different purposes
CREATE TABLE IF NOT EXISTS email_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    template_name VARCHAR(100) NOT NULL UNIQUE,
    purpose VARCHAR(50) NOT NULL,
    subject VARCHAR(200) NOT NULL,
    html_content TEXT NOT NULL,
    text_content TEXT NOT NULL,
    variables JSONB DEFAULT '{}', -- Available template variables
    is_active BOOLEAN DEFAULT true,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_email_templates_purpose (purpose, is_active),
    INDEX idx_email_templates_name (template_name)
);

-- =====================================================
-- RATE LIMITING TABLES
-- =====================================================

-- Rate limiting for OTP requests
CREATE TABLE IF NOT EXISTS otp_rate_limits (
    limit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL, -- Phone, email, or IP
    identifier_type VARCHAR(20) NOT NULL CHECK (identifier_type IN (
        'PHONE', 'EMAIL', 'IP', 'USER_ID'
    )),
    otp_type VARCHAR(10) NOT NULL CHECK (otp_type IN ('SMS', 'EMAIL')),
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMP DEFAULT NOW(),
    window_duration INTEGER DEFAULT 3600, -- Seconds (1 hour)
    max_requests INTEGER DEFAULT 5,
    is_blocked BOOLEAN DEFAULT false,
    blocked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(identifier, identifier_type, otp_type),
    INDEX idx_otp_rate_limits_identifier (identifier, identifier_type),
    INDEX idx_otp_rate_limits_blocked (is_blocked, blocked_until)
);

-- =====================================================
-- SECURITY AND AUDIT TABLES
-- =====================================================

-- Comprehensive OTP audit log
CREATE TABLE IF NOT EXISTS otp_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    otp_id UUID REFERENCES custom_otps(otp_id) ON DELETE SET NULL,
    verification_id UUID REFERENCES email_verifications(verification_id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN (
        'SEND', 'VERIFY', 'EXPIRE', 'LOCK', 'RESEND'
    )),
    recipient VARCHAR(255),
    recipient_type VARCHAR(10) CHECK (recipient_type IN ('SMS', 'EMAIL')),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    ip_address INET,
    user_agent TEXT,
    geo_location JSONB DEFAULT '{}',
    device_fingerprint VARCHAR(64),
    session_id UUID,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_otp_audit_user (user_id, created_at),
    INDEX idx_otp_audit_recipient (recipient, recipient_type),
    INDEX idx_otp_audit_action (action, created_at),
    INDEX idx_otp_audit_success (success, created_at)
);

-- Suspicious OTP activity detection
CREATE TABLE IF NOT EXISTS suspicious_otp_activity (
    activity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL, -- Phone, email, or IP
    identifier_type VARCHAR(20) NOT NULL,
    activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN (
        'EXCESSIVE_REQUESTS', 'RAPID_ATTEMPTS', 'MULTIPLE_FAILURES',
        'UNUSUAL_PATTERN', 'POTENTIAL_ABUSE'
    )),
    severity VARCHAR(20) NOT NULL CHECK (severity IN (
        'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    )),
    description TEXT,
    evidence JSONB DEFAULT '{}',
    risk_score DECIMAL(3,2) DEFAULT 0.00,
    is_resolved BOOLEAN DEFAULT false,
    resolution_notes TEXT,
    first_detected TIMESTAMP DEFAULT NOW(),
    last_detected TIMESTAMP DEFAULT NOW(),
    detection_count INTEGER DEFAULT 1,
    
    INDEX idx_suspicious_otp_identifier (identifier, identifier_type),
    INDEX idx_suspicious_otp_severity (severity, is_resolved),
    INDEX idx_suspicious_otp_detected (last_detected DESC)
);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update OTP provider statistics
CREATE OR REPLACE FUNCTION update_sms_provider_stats() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        -- Update provider statistics based on delivery status
        UPDATE sms_providers 
        SET 
            total_sent = total_sent + 1,
            total_delivered = CASE 
                WHEN NEW.delivery_status = 'DELIVERED' THEN total_delivered + 1 
                ELSE total_delivered 
            END,
            total_failed = CASE 
                WHEN NEW.delivery_status = 'FAILED' THEN total_failed + 1 
                ELSE total_failed 
            END,
            success_rate = CASE 
                WHEN total_sent + 1 > 0 THEN 
                    (CASE WHEN NEW.delivery_status = 'DELIVERED' THEN total_delivered + 1 ELSE total_delivered END) * 100.0 / (total_sent + 1)
                ELSE 0 
            END,
            last_success = CASE 
                WHEN NEW.delivery_status = 'DELIVERED' THEN NOW() 
                ELSE last_success 
            END,
            last_failure = CASE 
                WHEN NEW.delivery_status = 'FAILED' THEN NOW() 
                ELSE last_failure 
            END,
            updated_at = NOW()
        WHERE provider_name = NEW.provider_name;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update SMS provider statistics
CREATE TRIGGER trigger_update_sms_provider_stats
    AFTER INSERT OR UPDATE ON custom_otps
    FOR EACH ROW
    WHEN (NEW.otp_type = 'SMS' AND NEW.provider_name IS NOT NULL)
    EXECUTE FUNCTION update_sms_provider_stats();

-- Function to detect suspicious OTP activity
CREATE OR REPLACE FUNCTION detect_suspicious_otp_activity() RETURNS TRIGGER AS $$
DECLARE
    recent_count INTEGER;
    failure_count INTEGER;
    risk_score DECIMAL(3,2) := 0.00;
    activity_type VARCHAR(50);
    severity VARCHAR(20) := 'LOW';
BEGIN
    -- Check for excessive requests in last hour
    SELECT COUNT(*) INTO recent_count
    FROM custom_otps 
    WHERE recipient = NEW.recipient 
    AND otp_type = NEW.otp_type
    AND created_at > NOW() - INTERVAL '1 hour';
    
    -- Check for multiple failures
    SELECT COUNT(*) INTO failure_count
    FROM custom_otps 
    WHERE recipient = NEW.recipient 
    AND otp_type = NEW.otp_type
    AND is_verified = false
    AND created_at > NOW() - INTERVAL '1 hour';
    
    -- Determine activity type and risk score
    IF recent_count > 10 THEN
        activity_type := 'EXCESSIVE_REQUESTS';
        risk_score := 0.8;
        severity := 'HIGH';
    ELSIF failure_count > 5 THEN
        activity_type := 'MULTIPLE_FAILURES';
        risk_score := 0.6;
        severity := 'MEDIUM';
    ELSIF recent_count > 5 THEN
        activity_type := 'RAPID_ATTEMPTS';
        risk_score := 0.4;
        severity := 'MEDIUM';
    END IF;
    
    -- Insert suspicious activity record if risk score is significant
    IF risk_score > 0.3 THEN
        INSERT INTO suspicious_otp_activity (
            identifier, identifier_type, activity_type, severity,
            description, risk_score, evidence
        ) VALUES (
            NEW.recipient,
            NEW.otp_type,
            activity_type,
            severity,
            format('Detected %s: %s requests in last hour, %s failures', 
                   activity_type, recent_count, failure_count),
            risk_score,
            jsonb_build_object(
                'recent_requests', recent_count,
                'failed_attempts', failure_count,
                'time_window', '1 hour'
            )
        ) ON CONFLICT (identifier, identifier_type, activity_type) 
        DO UPDATE SET 
            detection_count = suspicious_otp_activity.detection_count + 1,
            last_detected = NOW(),
            risk_score = GREATEST(suspicious_otp_activity.risk_score, EXCLUDED.risk_score);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to detect suspicious activity
CREATE TRIGGER trigger_detect_suspicious_otp_activity
    AFTER INSERT ON custom_otps
    FOR EACH ROW
    EXECUTE FUNCTION detect_suspicious_otp_activity();

-- Function to cleanup expired records
CREATE OR REPLACE FUNCTION cleanup_expired_otp_records() RETURNS INTEGER AS $$
DECLARE
    cleanup_count INTEGER := 0;
BEGIN
    -- Mark expired OTPs as inactive
    UPDATE custom_otps 
    SET is_active = false, status = 'EXPIRED'
    WHERE expires_at < NOW() AND is_active = true;
    
    GET DIAGNOSTICS cleanup_count = ROW_COUNT;
    
    -- Mark expired email verifications as inactive
    UPDATE email_verifications 
    SET is_active = false, status = 'EXPIRED'
    WHERE expires_at < NOW() AND is_active = true;
    
    -- Clean old rate limit records (older than 24 hours)
    DELETE FROM otp_rate_limits 
    WHERE window_start < NOW() - INTERVAL '24 hours';
    
    -- Clean old audit logs (older than 90 days)
    DELETE FROM otp_audit_log 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    RETURN cleanup_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- VIEWS FOR MONITORING AND ANALYTICS
-- =====================================================

-- OTP success rate by type and purpose
CREATE OR REPLACE VIEW otp_success_rates AS
SELECT 
    otp_type,
    purpose,
    COUNT(*) as total_sent,
    COUNT(CASE WHEN is_verified = true THEN 1 END) as verified,
    COUNT(CASE WHEN is_locked = true THEN 1 END) as locked,
    COUNT(CASE WHEN status = 'EXPIRED' THEN 1 END) as expired,
    ROUND(
        COUNT(CASE WHEN is_verified = true THEN 1 END) * 100.0 / COUNT(*), 2
    ) as success_rate_percent,
    AVG(attempts) as avg_attempts,
    AVG(EXTRACT(EPOCH FROM (verified_at - created_at))) as avg_verification_time_seconds
FROM custom_otps 
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY otp_type, purpose
ORDER BY success_rate_percent DESC;

-- Email verification statistics
CREATE OR REPLACE VIEW email_verification_stats AS
SELECT 
    purpose,
    verification_type,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN is_verified = true THEN 1 END) as verified,
    COUNT(CASE WHEN is_locked = true THEN 1 END) as locked,
    ROUND(
        COUNT(CASE WHEN is_verified = true THEN 1 END) * 100.0 / COUNT(*), 2
    ) as success_rate_percent,
    AVG(attempts) as avg_attempts
FROM email_verifications 
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY purpose, verification_type
ORDER BY success_rate_percent DESC;

-- SMS provider performance comparison
CREATE OR REPLACE VIEW sms_provider_performance AS
SELECT 
    provider_name,
    provider_type,
    is_active,
    success_rate,
    avg_response_time,
    total_sent,
    total_delivered,
    total_failed,
    CASE 
        WHEN total_sent > 0 THEN ROUND(total_delivered * 100.0 / total_sent, 2)
        ELSE 0 
    END as calculated_success_rate,
    last_success,
    last_failure
FROM sms_providers 
ORDER BY success_rate DESC, avg_response_time ASC;

-- Suspicious activity summary
CREATE OR REPLACE VIEW suspicious_activity_summary AS
SELECT 
    activity_type,
    severity,
    COUNT(*) as incident_count,
    COUNT(CASE WHEN is_resolved = false THEN 1 END) as unresolved_count,
    AVG(risk_score) as avg_risk_score,
    MAX(last_detected) as most_recent_incident
FROM suspicious_otp_activity 
WHERE first_detected > NOW() - INTERVAL '7 days'
GROUP BY activity_type, severity
ORDER BY severity DESC, incident_count DESC;

-- =====================================================
-- INITIAL DATA SETUP
-- =====================================================

-- Insert default SMS providers
INSERT INTO sms_providers (provider_name, provider_type, endpoint_url, priority, configuration) VALUES
('BSNL_API', 'PRIMARY', 'https://bulksms.bsnl.in/api/send', 1, '{"timeout": 10000, "retry_count": 2}'),
('AIRTEL_BUSINESS', 'BACKUP', 'https://api.airtel.in/sms/send', 2, '{"timeout": 10000, "retry_count": 2}'),
('JIO_BUSINESS', 'BACKUP', 'https://jioapi.jio.com/sms/v1/send', 3, '{"timeout": 10000, "retry_count": 2}'),
('HTTP_GATEWAY', 'FALLBACK', 'http://localhost:8080/send-sms', 4, '{"timeout": 15000, "retry_count": 3}')
ON CONFLICT (provider_name) DO NOTHING;

-- Insert default email templates
INSERT INTO email_templates (template_name, purpose, subject, html_content, text_content, variables) VALUES
(
    'otp_verification',
    'EMAIL_VERIFICATION',
    '🔐 Your SIP Brewery Verification Code',
    '<!DOCTYPE html><html><head><style>body{font-family:Arial,sans-serif;margin:0;padding:20px;background-color:#f5f5f5}.container{max-width:600px;margin:0 auto;background-color:#fff;border-radius:10px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.1)}.header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:30px 20px;text-align:center;color:white}.content{padding:30px 20px}.otp-box{background-color:#f8f9fa;border:2px dashed #667eea;border-radius:8px;padding:20px;text-align:center;margin:20px 0}.otp-code{font-size:32px;font-weight:bold;color:#667eea;letter-spacing:4px;margin:10px 0}.footer{background-color:#f8f9fa;padding:20px;text-align:center;color:#666;font-size:14px}</style></head><body><div class="container"><div class="header"><h1>🔐 SIP Brewery</h1><p>Secure Investment Platform</p></div><div class="content"><h2>Your Verification Code</h2><p>Use the following code to complete your verification:</p><div class="otp-box"><div class="otp-code">{{OTP}}</div><p style="margin:0;color:#888">Valid for {{EXPIRY}} minutes</p></div><p><strong>Security Notice:</strong> Never share this code with anyone.</p></div><div class="footer"><p>© 2024 SIP Brewery. All rights reserved.</p></div></div></body></html>',
    'Your SIP Brewery verification code is: {{OTP}}. Valid for {{EXPIRY}} minutes. Never share this code with anyone.',
    '{"OTP": "6-digit verification code", "EXPIRY": "expiry time in minutes"}'
)
ON CONFLICT (template_name) DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_custom_otps_created_at ON custom_otps(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_email_verifications_created_at ON email_verifications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_otp_audit_log_created_at ON otp_audit_log(created_at DESC);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON custom_otps TO authenticated_users;
GRANT SELECT, INSERT, UPDATE ON email_verifications TO authenticated_users;
GRANT SELECT, INSERT ON otp_delivery_log TO authenticated_users;
GRANT SELECT, INSERT ON otp_audit_log TO authenticated_users;
GRANT SELECT ON sms_providers TO authenticated_users;
GRANT SELECT ON email_templates TO authenticated_users;

-- Grant admin permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO security_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO security_admin;
