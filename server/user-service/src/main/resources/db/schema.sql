-- User Service Schema (UMS + RBAC)
-- Aligned with mall-admin-web and mall-app-web requirements

-- Base users (unified for now or keeping separation if needed)
CREATE TABLE IF NOT EXISTS tb_user (
    id BIGINT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password VARCHAR(64) NOT NULL,
    nickname VARCHAR(64),
    avatar VARCHAR(255),
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(32) UNIQUE,
    status SMALLINT DEFAULT 1, -- 0->disabled, 1->enabled
    gender INT DEFAULT 0, -- 0->unknown, 1->male, 2->female
    birthday DATE,
    city VARCHAR(100),
    job VARCHAR(100),
    personalized_signature VARCHAR(255),
    integration INT DEFAULT 0,
    growth INT DEFAULT 0,
    member_level_id BIGINT,
    source_type INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Admin RBAC
CREATE TABLE IF NOT EXISTS tb_admin (
    id BIGINT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password VARCHAR(64) NOT NULL,
    icon VARCHAR(255),
    email VARCHAR(100),
    nick_name VARCHAR(64),
    note VARCHAR(500),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    login_time TIMESTAMP,
    status SMALLINT DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tb_role (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    admin_count INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_menu (
    id BIGINT PRIMARY KEY,
    parent_id BIGINT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    title VARCHAR(100),
    level INT DEFAULT 0,
    sort INT DEFAULT 0,
    name VARCHAR(100),
    icon VARCHAR(200),
    hidden SMALLINT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_resource (
    id BIGINT PRIMARY KEY,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    name VARCHAR(200),
    url VARCHAR(200),
    description VARCHAR(500),
    category_id BIGINT
);

CREATE TABLE IF NOT EXISTS tb_admin_role_relation (
    id BIGINT PRIMARY KEY,
    admin_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS tb_role_menu_relation (
    id BIGINT PRIMARY KEY,
    role_id BIGINT NOT NULL,
    menu_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS tb_role_resource_relation (
    id BIGINT PRIMARY KEY,
    role_id BIGINT NOT NULL,
    resource_id BIGINT NOT NULL
);

-- Member management
CREATE TABLE IF NOT EXISTS tb_member_level (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    growth_point INT DEFAULT 0,
    default_status SMALLINT DEFAULT 0,
    free_freight_point DECIMAL(10, 2),
    comment_growth_point INT DEFAULT 0,
    priviledge_free_freight SMALLINT DEFAULT 0,
    priviledge_sign_in SMALLINT DEFAULT 0,
    priviledge_comment SMALLINT DEFAULT 0,
    priviledge_promotion SMALLINT DEFAULT 0,
    priviledge_member_price SMALLINT DEFAULT 0,
    priviledge_birthday SMALLINT DEFAULT 0,
    note VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS tb_member_receive_address (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    name VARCHAR(100),
    phone_number VARCHAR(32),
    default_status SMALLINT DEFAULT 0,
    post_code VARCHAR(32),
    province VARCHAR(64),
    city VARCHAR(64),
    region VARCHAR(64),
    detail_address VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS tb_member_login_log (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip VARCHAR(64),
    city VARCHAR(64),
    login_type INT, -- 0->PC, 1->APP, 2->Web3
    province VARCHAR(64)
);

-- Point / Growth Logic
CREATE TABLE IF NOT EXISTS tb_integration_consume_setting (
    id BIGINT PRIMARY KEY,
    deduction_per_amount INT,
    max_percent_per_order INT,
    use_unit INT,
    coupon_status SMALLINT
);

CREATE TABLE IF NOT EXISTS tb_growth_change_history (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_type INT,
    change_count INT,
    operate_man VARCHAR(100),
    operate_note VARCHAR(255),
    source_type INT
);

CREATE TABLE IF NOT EXISTS tb_integration_change_history (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_type INT,
    change_count INT,
    operate_man VARCHAR(100),
    operate_note VARCHAR(255),
    source_type INT
);

-- Existing Web3 and Auth specific tables from previous project usage
CREATE TABLE IF NOT EXISTS tb_web3_user (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES tb_user(id),
    wallet_address VARCHAR(255) NOT NULL UNIQUE,
    chain_id SMALLINT NOT NULL,
    chain_type VARCHAR(10) NOT NULL DEFAULT 'mainnet',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_author (
    id BIGINT PRIMARY KEY,
    real_name VARCHAR(64) NOT NULL,
    is_enable BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_author_user_mapping (
    id BIGINT PRIMARY KEY,
    author_id BIGINT NOT NULL REFERENCES tb_author(id),
    user_id BIGINT NOT NULL REFERENCES tb_user(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_token_mapping (
    id BIGINT PRIMARY KEY,
    parent_token VARCHAR(512) NOT NULL,
    child_token VARCHAR(512) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL REFERENCES tb_user(id),
    permissions VARCHAR(255),
    expires_at TIMESTAMP NOT NULL,
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Passkey Credential (WebAuthn)
CREATE TABLE IF NOT EXISTS tb_passkey_credential (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES tb_user(id),
    credential_id VARCHAR(255) NOT NULL,
    public_key VARCHAR(4096) NOT NULL,
    rp_id VARCHAR(255) NOT NULL,
    counter BIGINT DEFAULT 0,
    device_type VARCHAR(32) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_passkey_user_id ON tb_passkey_credential(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_passkey_credential_id ON tb_passkey_credential(credential_id);
