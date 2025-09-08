-- 🔐 SIP Brewery Authentication Database Schema
-- Production-ready schema with enterprise security features

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table with comprehensive security features
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Basic Information
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(15) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    
    -- Authentication
    password_hash VARCHAR(255), -- Optional for OTP-only users
    phone_verified BOOLEAN DEFAULT FALSE,
    email_verified BOOLEAN DEFAULT FALSE,
    
    -- Security Features
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP NULL,
    password_changed_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP NULL,
    last_login_ip INET NULL,
    
    -- 2FA Configuration
    two_fa_enabled BOOLEAN DEFAULT FALSE,
    totp_secret TEXT NULL, -- Encrypted TOTP secret
    backup_codes TEXT[], -- Array of hashed backup codes
    
    -- Account Status
    status VARCHAR(20) DEFAULT 'ACTIVE', -- ACTIVE, SUSPENDED, CLOSED
    kyc_status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, VERIFIED, REJECTED
    risk_profile VARCHAR(20) DEFAULT 'MODERATE', -- LOW, MODERATE, HIGH
    
    -- BSE Integration
    bse_client_code VARCHAR(20) UNIQUE,
    pan VARCHAR(10) UNIQUE,
    
    -- Audit Fields
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by UUID NULL,
    updated_by UUID NULL
);

-- OTP Management table
CREATE TABLE otp_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- OTP Details
    phone VARCHAR(15) NOT NULL,
    otp_code VARCHAR(6) NOT NULL,
    otp_hash VARCHAR(255) NOT NULL, -- Hashed OTP for security
    
    -- OTP Type and Purpose
    otp_type VARCHAR(20) NOT NULL, -- SIGNUP, LOGIN, RESET_PASSWORD, TRANSACTION
    purpose VARCHAR(50) NOT NULL,
    
    -- Security Features
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    
    -- Expiry and Status
    expires_at TIMESTAMP NOT NULL,
    verified_at TIMESTAMP NULL,
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, VERIFIED, EXPIRED, BLOCKED
    
    -- Tracking
    ip_address INET NOT NULL,
    user_agent TEXT,
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    verified_by UUID NULL REFERENCES users(id)
);

-- Login sessions for JWT management
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session Details
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token_hash VARCHAR(255) NOT NULL,
    
    -- Device Information
    device_id VARCHAR(255),
    device_name VARCHAR(100),
    device_type VARCHAR(50), -- MOBILE, DESKTOP, TABLET
    browser VARCHAR(100),
    os VARCHAR(100),
    
    -- Location and Security
    ip_address INET NOT NULL,
    location JSONB, -- Store city, country, etc.
    is_trusted_device BOOLEAN DEFAULT FALSE,
    
    -- Session Status
    status VARCHAR(20) DEFAULT 'ACTIVE', -- ACTIVE, EXPIRED, REVOKED
    expires_at TIMESTAMP NOT NULL,
    last_activity_at TIMESTAMP DEFAULT NOW(),
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    revoked_at TIMESTAMP NULL,
    revoked_reason VARCHAR(100) NULL
);

-- Security events logging
CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    
    -- Event Details
    event_type VARCHAR(50) NOT NULL, -- LOGIN_SUCCESS, LOGIN_FAILED, OTP_SENT, PASSWORD_CHANGED, etc.
    event_category VARCHAR(30) NOT NULL, -- AUTHENTICATION, AUTHORIZATION, SECURITY
    severity VARCHAR(20) DEFAULT 'INFO', -- INFO, WARNING, ERROR, CRITICAL
    
    -- Event Data
    description TEXT NOT NULL,
    metadata JSONB, -- Additional event data
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(255),
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW()
);

-- Failed login attempts tracking
CREATE TABLE failed_login_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Attempt Details
    identifier VARCHAR(255) NOT NULL, -- phone or email
    identifier_type VARCHAR(10) NOT NULL, -- PHONE, EMAIL
    
    -- Failure Details
    failure_reason VARCHAR(100) NOT NULL,
    ip_address INET NOT NULL,
    user_agent TEXT,
    
    -- Tracking
    attempt_count INTEGER DEFAULT 1,
    first_attempt_at TIMESTAMP DEFAULT NOW(),
    last_attempt_at TIMESTAMP DEFAULT NOW(),
    
    -- Blocking
    blocked_until TIMESTAMP NULL,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- User profiles for additional information
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Personal Information
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    date_of_birth DATE,
    gender VARCHAR(10),
    
    -- Contact Information
    alternate_phone VARCHAR(15),
    address JSONB, -- Store complete address
    
    -- Financial Information
    annual_income DECIMAL(15,2),
    occupation VARCHAR(100),
    employer VARCHAR(100),
    
    -- Investment Preferences
    investment_experience VARCHAR(20), -- BEGINNER, INTERMEDIATE, EXPERT
    risk_tolerance VARCHAR(20), -- CONSERVATIVE, MODERATE, AGGRESSIVE
    investment_goals TEXT[],
    
    -- KYC Documents
    kyc_documents JSONB, -- Store document references
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Notification preferences
CREATE TABLE notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Notification Channels
    sms_enabled BOOLEAN DEFAULT TRUE,
    email_enabled BOOLEAN DEFAULT TRUE,
    push_enabled BOOLEAN DEFAULT TRUE,
    whatsapp_enabled BOOLEAN DEFAULT FALSE,
    
    -- Notification Types
    security_alerts BOOLEAN DEFAULT TRUE,
    transaction_alerts BOOLEAN DEFAULT TRUE,
    market_updates BOOLEAN DEFAULT FALSE,
    promotional BOOLEAN DEFAULT FALSE,
    
    -- Preferences
    preferred_language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'Asia/Kolkata',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX idx_users_phone ON users(phone);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_pan ON users(pan);
CREATE INDEX idx_users_bse_client_code ON users(bse_client_code);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE INDEX idx_otp_phone ON otp_verifications(phone);
CREATE INDEX idx_otp_expires_at ON otp_verifications(expires_at);
CREATE INDEX idx_otp_status ON otp_verifications(status);
CREATE INDEX idx_otp_type_purpose ON otp_verifications(otp_type, purpose);

CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_sessions_status ON user_sessions(status);
CREATE INDEX idx_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX idx_security_events_user_id ON security_events(user_id);
CREATE INDEX idx_security_events_type ON security_events(event_type);
CREATE INDEX idx_security_events_created_at ON security_events(created_at);

CREATE INDEX idx_failed_attempts_identifier ON failed_login_attempts(identifier);
CREATE INDEX idx_failed_attempts_ip ON failed_login_attempts(ip_address);
CREATE INDEX idx_failed_attempts_blocked_until ON failed_login_attempts(blocked_until);

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_notification_preferences_updated_at BEFORE UPDATE ON notification_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY users_policy ON users
    FOR ALL USING (id = current_setting('app.current_user_id')::UUID);

CREATE POLICY user_profiles_policy ON user_profiles
    FOR ALL USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY user_sessions_policy ON user_sessions
    FOR ALL USING (user_id = current_setting('app.current_user_id')::UUID);

-- Function to clean expired OTPs
CREATE OR REPLACE FUNCTION cleanup_expired_otps()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM otp_verifications 
    WHERE expires_at < NOW() - INTERVAL '1 hour';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    UPDATE user_sessions 
    SET status = 'EXPIRED', revoked_at = NOW()
    WHERE expires_at < NOW() AND status = 'ACTIVE';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create scheduled jobs for cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-expired-otps', '*/15 * * * *', 'SELECT cleanup_expired_otps();');
-- SELECT cron.schedule('cleanup-expired-sessions', '0 */6 * * *', 'SELECT cleanup_expired_sessions();');

-- Initial admin user (optional)
-- INSERT INTO users (
--     phone, name, email, phone_verified, email_verified, 
--     status, kyc_status, two_fa_enabled
-- ) VALUES (
--     '+919999999999', 'System Admin', 'admin@sipbrewery.com', 
--     TRUE, TRUE, 'ACTIVE', 'VERIFIED', TRUE
-- );

-- Grant permissions (adjust based on your application user)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO sipbrewery_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO sipbrewery_app;
