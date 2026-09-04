-- Wallet table for blockchain wallet management
CREATE TABLE IF NOT EXISTS tb_wallet (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    chain_type VARCHAR(32) NOT NULL,
    address VARCHAR(256) NOT NULL,
    balance DECIMAL(38, 8) DEFAULT 0,
    status VARCHAR(32) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_wallet_user_id ON tb_wallet (user_id);
CREATE INDEX IF NOT EXISTS idx_wallet_chain_type ON tb_wallet (chain_type);
CREATE INDEX IF NOT EXISTS idx_wallet_address ON tb_wallet (address);
CREATE UNIQUE INDEX IF NOT EXISTS uk_wallet_user_chain ON tb_wallet (user_id, chain_type);

-- Solana off-chain listing metadata
CREATE TABLE IF NOT EXISTS tb_solana_listing (
    id BIGSERIAL PRIMARY KEY,
    listing_address VARCHAR(128) NOT NULL,
    seller_address VARCHAR(128) NOT NULL,
    mint_address VARCHAR(128) NOT NULL,
    payment_mint_address VARCHAR(128) NOT NULL,
    price BIGINT NOT NULL,
    listed_amount BIGINT NOT NULL,
    status SMALLINT DEFAULT 0,
    tx_signature VARCHAR(256) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_listing_address ON tb_solana_listing (listing_address);
CREATE INDEX IF NOT EXISTS idx_seller_address ON tb_solana_listing (seller_address);
CREATE INDEX IF NOT EXISTS idx_mint_address ON tb_solana_listing (mint_address);
CREATE INDEX IF NOT EXISTS idx_listing_status ON tb_solana_listing (status);

-- Solana off-chain activity data
CREATE TABLE IF NOT EXISTS tb_solana_activity (
    id BIGSERIAL PRIMARY KEY,
    activity_address VARCHAR(128) NOT NULL,
    authority_address VARCHAR(128) NOT NULL,
    start_time BIGINT NOT NULL,
    end_time BIGINT NOT NULL,
    entry_fee BIGINT NOT NULL,
    reward_pcts VARCHAR(256) DEFAULT NULL,
    payment_mint VARCHAR(128) NOT NULL,
    total_pool BIGINT DEFAULT 0,
    participant_count INT DEFAULT 0,
    tx_signature VARCHAR(256) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_activity_address ON tb_solana_activity (activity_address);
CREATE INDEX IF NOT EXISTS idx_authority_address ON tb_solana_activity (authority_address);

-- Solana off-chain commission referral tree
CREATE TABLE IF NOT EXISTS tb_solana_commission_relation (
    id BIGSERIAL PRIMARY KEY,
    account_address VARCHAR(128) NOT NULL,
    upline_address VARCHAR(128) NOT NULL,
    level INT DEFAULT 0,
    downline_count INT DEFAULT 0,
    tx_signature VARCHAR(256) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_account_address ON tb_solana_commission_relation (account_address);
CREATE INDEX IF NOT EXISTS idx_upline_address ON tb_solana_commission_relation (upline_address);

-- Solana encrypted keypair store (KMS)
CREATE TABLE IF NOT EXISTS tb_solana_keypair (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(128) NOT NULL,
    encrypted_private_key VARCHAR(256) NOT NULL,
    iv VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uk_keypair_address ON tb_solana_keypair (address);