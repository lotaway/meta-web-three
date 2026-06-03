-- Developer Portal Database Schema
-- Creates tables for API developer management, API keys, subscriptions, and usage tracking

-- API Developer Table
CREATE TABLE IF NOT EXISTS api_developer (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    developer_id VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    name VARCHAR(128) NOT NULL,
    phone VARCHAR(32),
    description TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    review_note TEXT,
    reviewed_by VARCHAR(64),
    reviewed_at DATETIME,
    daily_quota INT NOT NULL DEFAULT 10000,
    monthly_quota INT NOT NULL DEFAULT 100000,
    billing_plan VARCHAR(32) NOT NULL DEFAULT 'FREE',
    balance BIGINT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_developer_id (developer_id),
    INDEX idx_email (email),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Developer Registration';

-- API Key Table
CREATE TABLE IF NOT EXISTS api_key (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    key_id VARCHAR(64) UNIQUE NOT NULL,
    key_secret VARCHAR(128) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    expires_at DATETIME,
    scopes TEXT,
    allowed_ips TEXT,
    allowed_domains TEXT,
    rate_limit INT DEFAULT 100,
    last_used_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_key_id (key_id),
    INDEX idx_developer_id (developer_id),
    INDEX idx_status (status),
    FOREIGN KEY (developer_id) REFERENCES api_developer(developer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Key Management';

-- API Subscription Table
CREATE TABLE IF NOT EXISTS api_subscription (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    subscription_id VARCHAR(64) UNIQUE NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    api_pattern VARCHAR(256) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    review_note TEXT,
    reviewed_by VARCHAR(64),
    reviewed_at DATETIME,
    started_at DATETIME,
    ended_at DATETIME,
    reason TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_subscription_id (subscription_id),
    INDEX idx_developer_id (developer_id),
    INDEX idx_status (status),
    FOREIGN KEY (developer_id) REFERENCES api_developer(developer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Subscription Management';

-- API Usage Statistics Table
CREATE TABLE IF NOT EXISTS api_usage_stats (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    developer_id VARCHAR(64) NOT NULL,
    key_id VARCHAR(64),
    api_endpoint VARCHAR(256) NOT NULL,
    http_method VARCHAR(16) NOT NULL,
    stat_time DATETIME NOT NULL,
    request_count BIGINT NOT NULL DEFAULT 0,
    success_count BIGINT NOT NULL DEFAULT 0,
    error_count BIGINT NOT NULL DEFAULT 0,
    avg_response_time_ms DOUBLE,
    data_transferred_bytes BIGINT DEFAULT 0,
    billing_amount_cents BIGINT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    INDEX idx_developer_time (developer_id, stat_time),
    INDEX idx_api_endpoint (api_endpoint, stat_time),
    INDEX idx_key_id (key_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Usage Statistics';

-- OAuth Application Table
CREATE TABLE IF NOT EXISTS oauth_application (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    client_id VARCHAR(64) UNIQUE NOT NULL,
    client_secret VARCHAR(128) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    description TEXT,
    redirect_uris TEXT,
    app_type VARCHAR(32) NOT NULL DEFAULT 'CONFIDENTIAL',
    grant_types TEXT,
    scopes TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_client_id (client_id),
    INDEX idx_developer_id (developer_id),
    INDEX idx_status (status),
    FOREIGN KEY (developer_id) REFERENCES api_developer(developer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='OAuth Application Registration';

-- OAuth Authorization Code Table (for OAuth 2.0 flow)
CREATE TABLE IF NOT EXISTS oauth_authorization_code (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    code VARCHAR(128) UNIQUE NOT NULL,
    client_id VARCHAR(64) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    redirect_uri VARCHAR(512),
    scopes TEXT,
    expires_at DATETIME NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at DATETIME NOT NULL,
    INDEX idx_code (code),
    INDEX idx_client_id (client_id),
    FOREIGN KEY (client_id) REFERENCES oauth_application(client_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='OAuth Authorization Code';

-- OAuth Access Token Table
CREATE TABLE IF NOT EXISTS oauth_access_token (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    token_id VARCHAR(128) UNIQUE NOT NULL,
    access_token VARCHAR(512) NOT NULL,
    refresh_token VARCHAR(512),
    client_id VARCHAR(64) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    scopes TEXT,
    expires_at DATETIME NOT NULL,
    refresh_expires_at DATETIME,
    revoked BOOLEAN DEFAULT FALSE,
    created_at DATETIME NOT NULL,
    INDEX idx_access_token (access_token(255)),
    INDEX idx_refresh_token (refresh_token(255)),
    INDEX idx_client_id (client_id),
    INDEX idx_developer_id (developer_id),
    FOREIGN KEY (client_id) REFERENCES oauth_application(client_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='OAuth Access Token';

-- API Billing Record Table
CREATE TABLE IF NOT EXISTS api_billing_record (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    developer_id VARCHAR(64) NOT NULL,
    billing_period VARCHAR(32) NOT NULL,
    total_requests BIGINT NOT NULL DEFAULT 0,
    total_billing_cents BIGINT NOT NULL DEFAULT 0,
    paid BOOLEAN DEFAULT FALSE,
    paid_at DATETIME,
    invoice_url VARCHAR(512),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_developer_period (developer_id, billing_period),
    FOREIGN KEY (developer_id) REFERENCES api_developer(developer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Billing Records';

-- API Webhook Configuration Table
CREATE TABLE IF NOT EXISTS api_webhook_config (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    developer_id VARCHAR(64) NOT NULL,
    webhook_url VARCHAR(512) NOT NULL,
    secret VARCHAR(128),
    events TEXT NOT NULL COMMENT 'Comma-separated event types',
    active BOOLEAN DEFAULT TRUE,
    last_triggered_at DATETIME,
    failure_count INT DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_developer_id (developer_id),
    FOREIGN KEY (developer_id) REFERENCES api_developer(developer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API Webhook Configuration';
