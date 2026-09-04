-- Developer Portal Database Schema
-- Creates tables for API developer management, API keys, subscriptions, and usage tracking

-- API Developer Table
CREATE TABLE IF NOT EXISTS api_developer (
    id BIGSERIAL PRIMARY KEY,
    developer_id VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    name VARCHAR(128) NOT NULL,
    phone VARCHAR(32),
    description TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    review_note TEXT,
    reviewed_by VARCHAR(64),
    reviewed_at TIMESTAMP,
    daily_quota INT NOT NULL DEFAULT 10000,
    monthly_quota INT NOT NULL DEFAULT 100000,
    billing_plan VARCHAR(32) NOT NULL DEFAULT 'FREE',
    balance BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_developer_developer_id ON api_developer (developer_id);
CREATE INDEX IF NOT EXISTS idx_api_developer_email ON api_developer (email);
CREATE INDEX IF NOT EXISTS idx_api_developer_status ON api_developer (status);

-- API Key Table
CREATE TABLE IF NOT EXISTS api_key (
    id BIGSERIAL PRIMARY KEY,
    key_id VARCHAR(64) UNIQUE NOT NULL,
    key_secret VARCHAR(128) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    expires_at TIMESTAMP,
    scopes TEXT,
    allowed_ips TEXT,
    allowed_domains TEXT,
    rate_limit INT DEFAULT 100,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_api_key_developer FOREIGN KEY (developer_id) REFERENCES api_developer (developer_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_api_key_key_id ON api_key (key_id);
CREATE INDEX IF NOT EXISTS idx_api_key_developer_id ON api_key (developer_id);
CREATE INDEX IF NOT EXISTS idx_api_key_status ON api_key (status);

-- API Subscription Table
CREATE TABLE IF NOT EXISTS api_subscription (
    id BIGSERIAL PRIMARY KEY,
    subscription_id VARCHAR(64) UNIQUE NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    api_pattern VARCHAR(256) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    review_note TEXT,
    reviewed_by VARCHAR(64),
    reviewed_at TIMESTAMP,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    reason TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_api_subscription_developer FOREIGN KEY (developer_id) REFERENCES api_developer (developer_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_api_subscription_id ON api_subscription (subscription_id);
CREATE INDEX IF NOT EXISTS idx_api_subscription_developer_id ON api_subscription (developer_id);
CREATE INDEX IF NOT EXISTS idx_api_subscription_status ON api_subscription (status);

-- API Usage Statistics Table
CREATE TABLE IF NOT EXISTS api_usage_stats (
    id BIGSERIAL PRIMARY KEY,
    developer_id VARCHAR(64) NOT NULL,
    key_id VARCHAR(64),
    api_endpoint VARCHAR(256) NOT NULL,
    http_method VARCHAR(16) NOT NULL,
    stat_time TIMESTAMP NOT NULL,
    request_count BIGINT NOT NULL DEFAULT 0,
    success_count BIGINT NOT NULL DEFAULT 0,
    error_count BIGINT NOT NULL DEFAULT 0,
    avg_response_time_ms DOUBLE PRECISION,
    data_transferred_bytes BIGINT DEFAULT 0,
    billing_amount_cents BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_usage_developer_time ON api_usage_stats (developer_id, stat_time);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint_time ON api_usage_stats (api_endpoint, stat_time);
CREATE INDEX IF NOT EXISTS idx_api_usage_key_id ON api_usage_stats (key_id);

-- OAuth Application Table
CREATE TABLE IF NOT EXISTS oauth_application (
    id BIGSERIAL PRIMARY KEY,
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
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_oauth_application_developer FOREIGN KEY (developer_id) REFERENCES api_developer (developer_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_oauth_application_client_id ON oauth_application (client_id);
CREATE INDEX IF NOT EXISTS idx_oauth_application_developer_id ON oauth_application (developer_id);
CREATE INDEX IF NOT EXISTS idx_oauth_application_status ON oauth_application (status);

-- OAuth Authorization Code Table (for OAuth 2.0 flow)
CREATE TABLE IF NOT EXISTS oauth_authorization_code (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(128) UNIQUE NOT NULL,
    client_id VARCHAR(64) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    redirect_uri VARCHAR(512),
    scopes TEXT,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_oauth_code_client FOREIGN KEY (client_id) REFERENCES oauth_application (client_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_oauth_code_code ON oauth_authorization_code (code);
CREATE INDEX IF NOT EXISTS idx_oauth_code_client_id ON oauth_authorization_code (client_id);

-- OAuth Access Token Table
CREATE TABLE IF NOT EXISTS oauth_access_token (
    id BIGSERIAL PRIMARY KEY,
    token_id VARCHAR(128) UNIQUE NOT NULL,
    access_token VARCHAR(512) NOT NULL,
    refresh_token VARCHAR(512),
    client_id VARCHAR(64) NOT NULL,
    developer_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    scopes TEXT,
    expires_at TIMESTAMP NOT NULL,
    refresh_expires_at TIMESTAMP,
    revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_oauth_token_client FOREIGN KEY (client_id) REFERENCES oauth_application (client_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_oauth_token_access ON oauth_access_token (access_token);
CREATE INDEX IF NOT EXISTS idx_oauth_token_refresh ON oauth_access_token (refresh_token);
CREATE INDEX IF NOT EXISTS idx_oauth_token_client_id ON oauth_access_token (client_id);
CREATE INDEX IF NOT EXISTS idx_oauth_token_developer_id ON oauth_access_token (developer_id);

-- API Billing Record Table
CREATE TABLE IF NOT EXISTS api_billing_record (
    id BIGSERIAL PRIMARY KEY,
    developer_id VARCHAR(64) NOT NULL,
    billing_period VARCHAR(32) NOT NULL,
    total_requests BIGINT NOT NULL DEFAULT 0,
    total_billing_cents BIGINT NOT NULL DEFAULT 0,
    paid BOOLEAN DEFAULT FALSE,
    paid_at TIMESTAMP,
    invoice_url VARCHAR(512),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_api_billing_developer FOREIGN KEY (developer_id) REFERENCES api_developer (developer_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_api_billing_developer_period ON api_billing_record (developer_id, billing_period);

-- API Webhook Configuration Table
CREATE TABLE IF NOT EXISTS api_webhook_config (
    id BIGSERIAL PRIMARY KEY,
    developer_id VARCHAR(64) NOT NULL,
    webhook_url VARCHAR(512) NOT NULL,
    secret VARCHAR(128),
    events TEXT NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    last_triggered_at TIMESTAMP,
    failure_count INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT fk_api_webhook_developer FOREIGN KEY (developer_id) REFERENCES api_developer (developer_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_api_webhook_developer_id ON api_webhook_config (developer_id);