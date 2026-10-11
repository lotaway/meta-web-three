-- ===== platform-domain\message-service\src\main\resources\db\schema.sql =====
-- Message Service Schema

CREATE TABLE IF NOT EXISTS tb_notification (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    icon VARCHAR(100),
    image_url VARCHAR(500),
    type VARCHAR(32) NOT NULL,
    related_id VARCHAR(64),
    read_status INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_notification_user_id ON tb_notification(user_id);
CREATE INDEX IF NOT EXISTS idx_notification_type ON tb_notification(type);
CREATE INDEX IF NOT EXISTS idx_notification_read_status ON tb_notification(read_status);

COMMENT ON COLUMN tb_notification.type IS 'SYSTEM/ORDER/PROMOTION';


-- ===== platform-domain\media-service\src\main\resources\db\schema.sql =====
-- Postgre SQL init, warning: create "metawebthree" database before start this script.

-- CREATE DATABASE IF NOT EXISTS metawebthree;
-- \c metawebthree;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

DROP TABLE IF EXISTS "Artwork";

DROP TABLE IF EXISTS "People_Type";

DROP TABLE IF EXISTS "People";

DROP TABLE IF EXISTS "Artwork_Tag";

DROP TABLE IF EXISTS "Artwork_Category";

CREATE TABLE IF NOT EXISTS "Artwork_Category" (
    "id" BIGSERIAL PRIMARY KEY,
    "name" VARCHAR NOT NULL,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_video_category_updated_at
BEFORE UPDATE ON "Artwork_Category"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS "Artwork_Tag" (
    "id" BIGSERIAL PRIMARY KEY,
    "tag" VARCHAR NOT NULL,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "People" (
    "id" BIGSERIAL PRIMARY KEY,
    "name" VARCHAR NOT NULL,
    "types" BIGINT[] DEFAULT '{}', -- List of Table People_Type IDs,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "People_Type" (
    "id" BIGSERIAL PRIMARY KEY,
    "type" VARCHAR NOT NULL UNIQUE, -- Such as Director, Editor, Actor, Voicer,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "Artwork" (
    "id" BIGSERIAL PRIMARY KEY,
    "series" VARCHAR(255) DEFAULT '', -- Batman Prequel,
    "title" VARCHAR NOT NULL, -- Batman Begins,
    "cover" VARCHAR(255) DEFAULT '', -- cover image URL,
    "link" VARCHAR(255) DEFAULT '',
    "subtitle" VARCHAR(255) DEFAULT '', -- Behind the scenes,
    "season" SMALLINT DEFAULT 1,
    "episode" SMALLINT DEFAULT 1,
    "category_id" BIGINT REFERENCES "Artwork_Category" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    "tags" BIGINT[] DEFAULT '{}', -- List of Table Tag IDs,
    "year_tag" SMALLINT DEFAULT 0,
    "acts" BIGINT[] DEFAULT '{}', -- List of Table People_Type type=Actor IDs,
    "director" BIGINT REFERENCES "People" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_Artwork_updated_at
BEFORE UPDATE ON "Artwork"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE FUNCTION update_year_tag_column()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.created_at IS NOT NULL THEN
        NEW.year_tag = EXTRACT(YEAR FROM NEW.created_at);
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER update_video_tag_created_at
BEFORE INSERT ON "Artwork"
FOR EACH ROW
EXECUTE FUNCTION update_year_tag_column();

CREATE INDEX IF NOT EXISTS idx_Artwork_updated_at ON "Artwork" ("updated_at");

CREATE INDEX IF NOT EXISTS idx_video_tags ON "Artwork" USING GIN (tags);

CREATE TABLE IF NOT EXISTS tb_user_storage (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE,
    total_used BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_storage_user_id ON tb_user_storage (user_id);

-- ===== platform-domain\commission-service\src\main\resources\db\schema.sql =====
CREATE TABLE IF NOT EXISTS commission_relation (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    parent_user_id BIGINT NOT NULL,
    depth INT NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_commission_relation_user UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_commission_relation_parent ON commission_relation(parent_user_id);

CREATE TABLE IF NOT EXISTS commission_config (
    id BIGINT PRIMARY KEY,
    buy_rate DECIMAL(10,4) NOT NULL,
    level_rates VARCHAR(255) NOT NULL,
    max_levels INT NOT NULL,
    return_window_days INT NOT NULL,
    confirm_method VARCHAR(20) NOT NULL DEFAULT 'RECEIVE',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS commission_record (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    from_user_id BIGINT NOT NULL,
    level INT NOT NULL,
    amount DECIMAL(18,4) NOT NULL,
    status VARCHAR(20) NOT NULL,
    available_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_commission_record_order ON commission_record(order_id);
CREATE INDEX IF NOT EXISTS idx_commission_record_user ON commission_record(user_id);
CREATE INDEX IF NOT EXISTS idx_commission_record_available ON commission_record(available_at);

CREATE TABLE IF NOT EXISTS commission_main (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    total_amount DECIMAL(18,4) NOT NULL DEFAULT 0,
    available_amount DECIMAL(18,4) NOT NULL DEFAULT 0,
    frozen_amount DECIMAL(18,4) NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_commission_main_user UNIQUE (user_id)
);


-- ===== platform-domain\cs-service\src\main\resources\db\schema.sql =====
CREATE TABLE IF NOT EXISTS cs_agent (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    admin_id BIGINT NOT NULL UNIQUE,
    nickname VARCHAR(100),
    avatar VARCHAR(500),
    status VARCHAR(20) DEFAULT 'OFFLINE',
    max_concurrent INT DEFAULT 5,
    current_load INT DEFAULT 0,
    group_id BIGINT,
    working_hours JSONB,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cs_group (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    skill_tags VARCHAR(255),
    priority INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cs_quick_reply (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    group_id BIGINT,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    msg_type VARCHAR(20) DEFAULT 'TEXT',
    sort INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cs_transfer_log (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    session_id VARCHAR(64) NOT NULL,
    from_agent_id BIGINT,
    to_agent_id BIGINT,
    reason VARCHAR(500),
    transfer_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cs_faq (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    question VARCHAR(500) NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100),
    keywords VARCHAR(1000),
    hit_count INT DEFAULT 0,
    relevance_score DOUBLE PRECISION DEFAULT 0.0,
    enabled BOOLEAN DEFAULT TRUE,
    priority INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cs_work_order (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    order_no VARCHAR(64) NOT NULL UNIQUE,
    customer_id BIGINT NOT NULL,
    agent_id BIGINT,
    category VARCHAR(50),
    ai_suggested_category VARCHAR(50),
    confidence_score DOUBLE PRECISION DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'PENDING',
    priority VARCHAR(20) DEFAULT 'MEDIUM',
    title VARCHAR(200) NOT NULL,
    description TEXT,
    resolution TEXT,
    metadata JSONB,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolve_time TIMESTAMP
);


-- ===== platform-domain\developer-portal-service\src\main\resources\db\schema.sql =====
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

-- ===== platform-domain\tenant-service\src\main\resources\db\schema.sql =====
CREATE TABLE IF NOT EXISTS tenant (
    id BIGINT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    code VARCHAR(100) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    contact_name VARCHAR(100),
    contact_email VARCHAR(200),
    contact_phone VARCHAR(50),
    domain VARCHAR(200),
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tenant_shop (
    id BIGINT PRIMARY KEY,
    tenant_id BIGINT NOT NULL REFERENCES tenant(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    logo VARCHAR(500),
    banner VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'CLOSED',
    sort_order INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tenant_user (
    id BIGINT PRIMARY KEY,
    tenant_id BIGINT NOT NULL REFERENCES tenant(id),
    user_id BIGINT NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'ADMIN',
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_tenant_shop_tenant ON tenant_shop(tenant_id);
CREATE INDEX IF NOT EXISTS idx_tenant_user_tenant ON tenant_user(tenant_id);
CREATE INDEX IF NOT EXISTS idx_tenant_user_user ON tenant_user(user_id);
CREATE INDEX IF NOT EXISTS idx_tenant_code ON tenant(code);
CREATE INDEX IF NOT EXISTS idx_tenant_status ON tenant(status);


