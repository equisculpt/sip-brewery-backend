-- 🛡️ Advanced Security Database Schema - FBI Level Protection
-- Military-grade security tables and functions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =====================================================
-- BIOMETRIC AUTHENTICATION TABLES
-- =====================================================

-- Biometric templates storage (encrypted)
CREATE TABLE IF NOT EXISTS biometric_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    biometric_type VARCHAR(20) NOT NULL CHECK (biometric_type IN ('fingerprint', 'face', 'iris', 'voice')),
    encrypted_template JSONB NOT NULL, -- Encrypted biometric data
    template_hash VARCHAR(64) NOT NULL, -- Hash for duplicate detection
    quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0 AND quality_score <= 1),
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    is_locked BOOLEAN DEFAULT false,
    failed_attempts INTEGER DEFAULT 0,
    auth_count INTEGER DEFAULT 0,
    last_successful_auth TIMESTAMP,
    last_failed_attempt TIMESTAMP,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(user_id, biometric_type, template_hash)
);

-- Behavioral biometrics profiles
CREATE TABLE IF NOT EXISTS behavioral_profiles (
    profile_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    profile_type VARCHAR(20) NOT NULL CHECK (profile_type IN ('keystroke', 'mouse', 'touch', 'gait')),
    profile_data JSONB NOT NULL,
    sample_count INTEGER DEFAULT 1,
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(user_id, profile_type)
);

-- Biometric authentication events
CREATE TABLE IF NOT EXISTS biometric_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    template_id UUID REFERENCES biometric_templates(template_id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    metadata JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- ADVANCED THREAT DETECTION TABLES
-- =====================================================

-- Threat intelligence database
CREATE TABLE IF NOT EXISTS threat_intelligence (
    threat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_type VARCHAR(50) NOT NULL,
    indicator_value TEXT NOT NULL, -- IP, domain, hash, etc.
    indicator_type VARCHAR(20) NOT NULL CHECK (indicator_type IN ('ip', 'domain', 'hash', 'url', 'email')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    source VARCHAR(100),
    description TEXT,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    UNIQUE(indicator_value, indicator_type)
);

-- Device fingerprinting and tracking
CREATE TABLE IF NOT EXISTS device_fingerprints (
    fingerprint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    fingerprint_hash VARCHAR(64) NOT NULL UNIQUE,
    device_info JSONB NOT NULL,
    browser_info JSONB NOT NULL,
    screen_info JSONB DEFAULT '{}',
    timezone_info JSONB DEFAULT '{}',
    plugin_info JSONB DEFAULT '{}',
    canvas_fingerprint TEXT,
    webgl_fingerprint TEXT,
    audio_fingerprint TEXT,
    risk_score DECIMAL(3,2) DEFAULT 0.0,
    is_trusted BOOLEAN DEFAULT false,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

-- Behavioral analysis profiles
CREATE TABLE IF NOT EXISTS behavior_profiles (
    profile_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fingerprint VARCHAR(64) NOT NULL,
    request_count INTEGER DEFAULT 0,
    endpoint_count INTEGER DEFAULT 0,
    avg_request_interval BIGINT DEFAULT 0,
    anomaly_score DECIMAL(3,2) DEFAULT 0.0,
    suspicious_actions INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(fingerprint)
);

-- Advanced security events with ML scoring
CREATE TABLE IF NOT EXISTS advanced_security_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    session_id UUID,
    event_type VARCHAR(50) NOT NULL,
    event_category VARCHAR(30) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    risk_score DECIMAL(3,2) DEFAULT 0.0,
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    endpoint TEXT,
    request_method VARCHAR(10),
    response_status INTEGER,
    response_time INTEGER,
    geo_location JSONB DEFAULT '{}',
    device_fingerprint VARCHAR(64),
    is_blocked BOOLEAN DEFAULT false,
    is_false_positive BOOLEAN DEFAULT false,
    analyst_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_security_events_user_time (user_id, created_at),
    INDEX idx_security_events_severity (severity, created_at),
    INDEX idx_security_events_ip (ip_address, created_at),
    INDEX idx_security_events_type (event_type, created_at)
);

-- =====================================================
-- ENCRYPTION AND KEY MANAGEMENT
-- =====================================================

-- Encryption keys management
CREATE TABLE IF NOT EXISTS encryption_keys (
    key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(100) NOT NULL UNIQUE,
    key_type VARCHAR(20) NOT NULL CHECK (key_type IN ('AES', 'RSA', 'ECDSA')),
    key_size INTEGER NOT NULL,
    encrypted_key TEXT NOT NULL, -- Key encrypted with master key
    key_hash VARCHAR(64) NOT NULL,
    purpose VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    rotation_interval INTEGER DEFAULT 30, -- days
    last_rotated TIMESTAMP DEFAULT NOW(),
    next_rotation TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(user_id)
);

-- Audit trail for key usage
CREATE TABLE IF NOT EXISTS key_usage_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_id UUID NOT NULL REFERENCES encryption_keys(key_id),
    user_id UUID REFERENCES users(user_id),
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('ENCRYPT', 'DECRYPT', 'SIGN', 'VERIFY')),
    purpose TEXT,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- COMPLIANCE AND AUDIT TABLES
-- =====================================================

-- Comprehensive audit log
CREATE TABLE IF NOT EXISTS comprehensive_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    session_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id TEXT,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    geo_location JSONB DEFAULT '{}',
    compliance_flags JSONB DEFAULT '{}',
    retention_period INTEGER DEFAULT 2555, -- 7 years in days
    is_sensitive BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_audit_user_time (user_id, created_at),
    INDEX idx_audit_action (action, created_at),
    INDEX idx_audit_resource (resource_type, resource_id, created_at)
);

-- Data access tracking
CREATE TABLE IF NOT EXISTS data_access_log (
    access_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    data_type VARCHAR(50) NOT NULL,
    data_classification VARCHAR(20) NOT NULL CHECK (data_classification IN ('PUBLIC', 'INTERNAL', 'CONFIDENTIAL', 'RESTRICTED')),
    access_type VARCHAR(20) NOT NULL CHECK (access_type IN ('READ', 'write', 'delete', 'export')),
    record_count INTEGER DEFAULT 1,
    purpose TEXT,
    legal_basis VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_data_access_user_time (user_id, created_at),
    INDEX idx_data_access_type (data_type, access_type, created_at)
);

-- =====================================================
-- ADVANCED SECURITY FUNCTIONS
-- =====================================================

-- Function to calculate risk score
CREATE OR REPLACE FUNCTION calculate_risk_score(
    p_user_id UUID,
    p_ip_address INET,
    p_device_fingerprint VARCHAR(64),
    p_behavior_score DECIMAL(3,2) DEFAULT 0.0
) RETURNS DECIMAL(3,2) AS $$
DECLARE
    v_risk_score DECIMAL(3,2) := 0.0;
    v_threat_score DECIMAL(3,2) := 0.0;
    v_device_score DECIMAL(3,2) := 0.0;
    v_behavior_score DECIMAL(3,2) := p_behavior_score;
    v_history_score DECIMAL(3,2) := 0.0;
BEGIN
    -- Check threat intelligence
    SELECT COALESCE(MAX(confidence_score), 0.0) INTO v_threat_score
    FROM threat_intelligence 
    WHERE indicator_value = p_ip_address::TEXT 
    AND indicator_type = 'ip' 
    AND is_active = true;
    
    -- Check device trust
    SELECT COALESCE(1.0 - AVG(risk_score), 0.0) INTO v_device_score
    FROM device_fingerprints 
    WHERE fingerprint_hash = p_device_fingerprint 
    AND is_trusted = true;
    
    -- Check user history
    SELECT COALESCE(COUNT(*) * 0.1, 0.0) INTO v_history_score
    FROM advanced_security_events 
    WHERE user_id = p_user_id 
    AND severity IN ('ERROR', 'CRITICAL')
    AND created_at > NOW() - INTERVAL '24 hours';
    
    -- Calculate weighted risk score
    v_risk_score := (v_threat_score * 0.4) + 
                   (v_device_score * 0.2) + 
                   (v_behavior_score * 0.3) + 
                   (LEAST(v_history_score, 1.0) * 0.1);
    
    RETURN LEAST(v_risk_score, 1.0);
END;
$$ LANGUAGE plpgsql;

-- Function to detect anomalous behavior
CREATE OR REPLACE FUNCTION detect_behavioral_anomaly(
    p_fingerprint VARCHAR(64),
    p_current_interval BIGINT,
    p_endpoint TEXT
) RETURNS DECIMAL(3,2) AS $$
DECLARE
    v_profile RECORD;
    v_anomaly_score DECIMAL(3,2) := 0.0;
    v_frequency_anomaly DECIMAL(3,2) := 0.0;
    v_pattern_anomaly DECIMAL(3,2) := 0.0;
BEGIN
    -- Get existing behavior profile
    SELECT * INTO v_profile
    FROM behavior_profiles 
    WHERE fingerprint = p_fingerprint;
    
    IF NOT FOUND THEN
        RETURN 0.0; -- No baseline to compare
    END IF;
    
    -- Calculate frequency anomaly
    IF v_profile.avg_request_interval > 0 THEN
        IF p_current_interval < (v_profile.avg_request_interval * 0.1) THEN
            v_frequency_anomaly := 0.8; -- Very fast requests
        ELSIF p_current_interval < (v_profile.avg_request_interval * 0.5) THEN
            v_frequency_anomaly := 0.4; -- Fast requests
        END IF;
    END IF;
    
    -- Calculate pattern anomaly (simplified)
    IF v_profile.request_count > 10 THEN
        v_pattern_anomaly := LEAST(v_profile.suspicious_actions / v_profile.request_count, 0.5);
    END IF;
    
    v_anomaly_score := v_frequency_anomaly + v_pattern_anomaly;
    
    RETURN LEAST(v_anomaly_score, 1.0);
END;
$$ LANGUAGE plpgsql;

-- Function to encrypt sensitive data
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(
    p_data TEXT,
    p_key_name VARCHAR(100)
) RETURNS TEXT AS $$
DECLARE
    v_encrypted_key TEXT;
    v_result TEXT;
BEGIN
    -- Get encryption key
    SELECT encrypted_key INTO v_encrypted_key
    FROM encryption_keys 
    WHERE key_name = p_key_name 
    AND is_active = true;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Encryption key not found: %', p_key_name;
    END IF;
    
    -- Encrypt data (simplified - in production use proper key decryption)
    v_result := encode(digest(p_data || v_encrypted_key, 'sha256'), 'hex');
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SECURITY TRIGGERS
-- =====================================================

-- Trigger to update behavior profiles
CREATE OR REPLACE FUNCTION update_behavior_profile() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO behavior_profiles (
        fingerprint, request_count, last_seen
    ) VALUES (
        NEW.device_fingerprint, 1, NOW()
    )
    ON CONFLICT (fingerprint) 
    DO UPDATE SET 
        request_count = behavior_profiles.request_count + 1,
        last_seen = NOW(),
        anomaly_score = detect_behavioral_anomaly(
            NEW.device_fingerprint,
            EXTRACT(EPOCH FROM (NOW() - behavior_profiles.last_seen)) * 1000,
            NEW.endpoint
        );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to security events
CREATE TRIGGER trigger_update_behavior_profile
    AFTER INSERT ON advanced_security_events
    FOR EACH ROW
    WHEN (NEW.device_fingerprint IS NOT NULL)
    EXECUTE FUNCTION update_behavior_profile();

-- Trigger for automatic threat intelligence updates
CREATE OR REPLACE FUNCTION update_threat_intelligence() RETURNS TRIGGER AS $$
BEGIN
    -- Update last_seen for existing threats
    IF TG_OP = 'INSERT' THEN
        UPDATE threat_intelligence 
        SET last_seen = NOW(),
            confidence_score = LEAST(confidence_score + 0.1, 1.0)
        WHERE indicator_value = NEW.ip_address::TEXT 
        AND indicator_type = 'ip'
        AND NEW.severity IN ('ERROR', 'CRITICAL');
        
        -- Insert new threat if high severity and not exists
        IF NEW.severity = 'CRITICAL' AND NOT FOUND THEN
            INSERT INTO threat_intelligence (
                threat_type, indicator_value, indicator_type,
                severity, confidence_score, source, description
            ) VALUES (
                'MALICIOUS_IP', NEW.ip_address::TEXT, 'ip',
                'HIGH', 0.7, 'AUTO_DETECTED', 
                'Automatically detected from security events'
            ) ON CONFLICT (indicator_value, indicator_type) DO NOTHING;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to security events
CREATE TRIGGER trigger_update_threat_intelligence
    AFTER INSERT ON advanced_security_events
    FOR EACH ROW
    EXECUTE FUNCTION update_threat_intelligence();

-- =====================================================
-- SECURITY VIEWS FOR MONITORING
-- =====================================================

-- Real-time threat dashboard
CREATE OR REPLACE VIEW security_dashboard AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    severity,
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT ip_address) as unique_ips,
    COUNT(DISTINCT user_id) as affected_users,
    AVG(risk_score) as avg_risk_score
FROM advanced_security_events 
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), severity, event_type
ORDER BY hour DESC, event_count DESC;

-- High-risk users view
CREATE OR REPLACE VIEW high_risk_users AS
SELECT 
    u.user_id,
    u.phone_number,
    u.email,
    COUNT(ase.event_id) as security_events,
    AVG(ase.risk_score) as avg_risk_score,
    MAX(ase.created_at) as last_incident,
    CASE 
        WHEN AVG(ase.risk_score) > 0.8 THEN 'CRITICAL'
        WHEN AVG(ase.risk_score) > 0.6 THEN 'HIGH'
        WHEN AVG(ase.risk_score) > 0.4 THEN 'MEDIUM'
        ELSE 'LOW'
    END as risk_level
FROM users u
JOIN advanced_security_events ase ON u.user_id = ase.user_id
WHERE ase.created_at > NOW() - INTERVAL '7 days'
GROUP BY u.user_id, u.phone_number, u.email
HAVING AVG(ase.risk_score) > 0.4
ORDER BY avg_risk_score DESC;

-- Threat intelligence summary
CREATE OR REPLACE VIEW threat_summary AS
SELECT 
    threat_type,
    indicator_type,
    severity,
    COUNT(*) as indicator_count,
    AVG(confidence_score) as avg_confidence,
    MAX(last_seen) as most_recent
FROM threat_intelligence 
WHERE is_active = true
GROUP BY threat_type, indicator_type, severity
ORDER BY indicator_count DESC;

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Biometric templates indexes
CREATE INDEX IF NOT EXISTS idx_biometric_templates_user_type ON biometric_templates(user_id, biometric_type);
CREATE INDEX IF NOT EXISTS idx_biometric_templates_hash ON biometric_templates(template_hash);
CREATE INDEX IF NOT EXISTS idx_biometric_templates_active ON biometric_templates(is_active, user_id);

-- Behavioral profiles indexes
CREATE INDEX IF NOT EXISTS idx_behavioral_profiles_user_type ON behavioral_profiles(user_id, profile_type);
CREATE INDEX IF NOT EXISTS idx_behavioral_profiles_updated ON behavioral_profiles(last_updated);

-- Threat intelligence indexes
CREATE INDEX IF NOT EXISTS idx_threat_intelligence_indicator ON threat_intelligence(indicator_value, indicator_type);
CREATE INDEX IF NOT EXISTS idx_threat_intelligence_active ON threat_intelligence(is_active, severity);
CREATE INDEX IF NOT EXISTS idx_threat_intelligence_seen ON threat_intelligence(last_seen);

-- Device fingerprints indexes
CREATE INDEX IF NOT EXISTS idx_device_fingerprints_hash ON device_fingerprints(fingerprint_hash);
CREATE INDEX IF NOT EXISTS idx_device_fingerprints_user ON device_fingerprints(user_id, is_trusted);
CREATE INDEX IF NOT EXISTS idx_device_fingerprints_risk ON device_fingerprints(risk_score, last_seen);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_user_time ON comprehensive_audit_log(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON comprehensive_audit_log(action, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_sensitive ON comprehensive_audit_log(is_sensitive, created_at);

-- Data access log indexes
CREATE INDEX IF NOT EXISTS idx_data_access_user ON data_access_log(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_data_access_classification ON data_access_log(data_classification, access_type);

-- =====================================================
-- ROW LEVEL SECURITY POLICIES
-- =====================================================

-- Enable RLS on sensitive tables
ALTER TABLE biometric_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE behavioral_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE comprehensive_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_access_log ENABLE ROW LEVEL SECURITY;

-- Biometric templates RLS policy
CREATE POLICY biometric_templates_policy ON biometric_templates
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- Behavioral profiles RLS policy
CREATE POLICY behavioral_profiles_policy ON behavioral_profiles
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- Audit log RLS policy (users can only see their own audit logs)
CREATE POLICY audit_log_policy ON comprehensive_audit_log
    FOR SELECT TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- Data access log RLS policy
CREATE POLICY data_access_policy ON data_access_log
    FOR SELECT TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- =====================================================
-- SECURITY ROLES AND PERMISSIONS
-- =====================================================

-- Create security roles
CREATE ROLE security_analyst;
CREATE ROLE security_admin;
CREATE ROLE compliance_officer;
CREATE ROLE authenticated_users;

-- Grant permissions to security analyst
GRANT SELECT ON ALL TABLES IN SCHEMA public TO security_analyst;
GRANT SELECT ON security_dashboard TO security_analyst;
GRANT SELECT ON high_risk_users TO security_analyst;
GRANT SELECT ON threat_summary TO security_analyst;

-- Grant permissions to security admin
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO security_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO security_admin;

-- Grant permissions to compliance officer
GRANT SELECT ON comprehensive_audit_log TO compliance_officer;
GRANT SELECT ON data_access_log TO compliance_officer;
GRANT SELECT ON biometric_events TO compliance_officer;

-- Grant basic permissions to authenticated users
GRANT SELECT, INSERT, UPDATE ON users TO authenticated_users;
GRANT SELECT, INSERT, UPDATE ON user_sessions TO authenticated_users;
GRANT SELECT, INSERT ON biometric_templates TO authenticated_users;
GRANT SELECT, INSERT ON behavioral_profiles TO authenticated_users;

-- =====================================================
-- MONITORING AND ALERTING FUNCTIONS
-- =====================================================

-- Function to check for security anomalies
CREATE OR REPLACE FUNCTION check_security_anomalies() RETURNS TABLE(
    alert_type TEXT,
    severity TEXT,
    description TEXT,
    count BIGINT
) AS $$
BEGIN
    -- High-risk events in last hour
    RETURN QUERY
    SELECT 
        'HIGH_RISK_EVENTS'::TEXT,
        'CRITICAL'::TEXT,
        'High number of critical security events'::TEXT,
        COUNT(*)
    FROM advanced_security_events 
    WHERE severity = 'CRITICAL' 
    AND created_at > NOW() - INTERVAL '1 hour'
    HAVING COUNT(*) > 10;
    
    -- Unusual login patterns
    RETURN QUERY
    SELECT 
        'UNUSUAL_LOGIN_PATTERN'::TEXT,
        'WARNING'::TEXT,
        'Unusual number of failed login attempts'::TEXT,
        COUNT(*)
    FROM failed_login_attempts 
    WHERE created_at > NOW() - INTERVAL '1 hour'
    HAVING COUNT(*) > 100;
    
    -- Biometric authentication failures
    RETURN QUERY
    SELECT 
        'BIOMETRIC_FAILURES'::TEXT,
        'WARNING'::TEXT,
        'High number of biometric authentication failures'::TEXT,
        COUNT(*)
    FROM biometric_events 
    WHERE event_type = 'BIOMETRIC_AUTH_FAILED'
    AND created_at > NOW() - INTERVAL '1 hour'
    HAVING COUNT(*) > 50;
    
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on monitoring functions
GRANT EXECUTE ON FUNCTION check_security_anomalies() TO security_analyst;
GRANT EXECUTE ON FUNCTION calculate_risk_score(UUID, INET, VARCHAR(64), DECIMAL(3,2)) TO authenticated_users;
GRANT EXECUTE ON FUNCTION detect_behavioral_anomaly(VARCHAR(64), BIGINT, TEXT) TO authenticated_users;
