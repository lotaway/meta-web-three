-- Wallet table for blockchain wallet management
CREATE TABLE IF NOT EXISTS tb_wallet (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    chain_type VARCHAR(32) NOT NULL,
    address VARCHAR(256) NOT NULL,
    balance DECIMAL(38, 8) DEFAULT 0,
    status VARCHAR(32) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_wallet_user_id (user_id),
    INDEX idx_wallet_chain_type (chain_type),
    INDEX idx_wallet_address (address),
    UNIQUE INDEX idx_wallet_user_chain (user_id, chain_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Blockchain wallet table';

-- Solana off-chain listing metadata
CREATE TABLE IF NOT EXISTS tb_solana_listing (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    listing_address VARCHAR(128) NOT NULL COMMENT 'Listing PDA address',
    seller_address VARCHAR(128) NOT NULL COMMENT 'Seller wallet address',
    mint_address VARCHAR(128) NOT NULL COMMENT 'Token mint address',
    payment_mint_address VARCHAR(128) NOT NULL COMMENT 'Payment token mint address',
    price BIGINT NOT NULL COMMENT 'Price in smallest units',
    listed_amount BIGINT NOT NULL COMMENT 'Amount of tokens listed',
    status TINYINT DEFAULT 0 COMMENT '0=Active 1=Sold 2=Cancelled',
    tx_signature VARCHAR(256) NOT NULL COMMENT 'Transaction signature',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_listing_address (listing_address),
    INDEX idx_seller_address (seller_address),
    INDEX idx_mint_address (mint_address),
    INDEX idx_listing_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Solana off-chain listing metadata';

-- Solana off-chain activity data
CREATE TABLE IF NOT EXISTS tb_solana_activity (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    activity_address VARCHAR(128) NOT NULL COMMENT 'Activity PDA address',
    authority_address VARCHAR(128) NOT NULL COMMENT 'Authority wallet address',
    start_time BIGINT NOT NULL COMMENT 'Start timestamp',
    end_time BIGINT NOT NULL COMMENT 'End timestamp',
    entry_fee BIGINT NOT NULL COMMENT 'Entry fee in smallest units',
    reward_pcts VARCHAR(256) DEFAULT NULL COMMENT 'Reward percentages JSON e.g. [5000,3000,2000]',
    payment_mint VARCHAR(128) NOT NULL COMMENT 'Payment token mint address',
    total_pool BIGINT DEFAULT 0 COMMENT 'Total pool amount',
    participant_count INT DEFAULT 0 COMMENT 'Participant count',
    tx_signature VARCHAR(256) NOT NULL COMMENT 'Transaction signature',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_activity_address (activity_address),
    INDEX idx_authority_address (authority_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Solana off-chain activity data';

-- Solana off-chain commission referral tree
CREATE TABLE IF NOT EXISTS tb_solana_commission_relation (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    account_address VARCHAR(128) NOT NULL COMMENT 'Account address',
    upline_address VARCHAR(128) NOT NULL COMMENT 'Upline address',
    level INT DEFAULT 0 COMMENT 'Level in referral tree',
    downline_count INT DEFAULT 0 COMMENT 'Downline count',
    tx_signature VARCHAR(256) NOT NULL COMMENT 'Transaction signature',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_account_address (account_address),
    INDEX idx_upline_address (upline_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Solana off-chain commission referral tree';

-- Solana encrypted keypair store (KMS)
CREATE TABLE IF NOT EXISTS tb_solana_keypair (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    address VARCHAR(128) NOT NULL COMMENT 'Solana wallet address (public key)',
    encrypted_private_key VARCHAR(256) NOT NULL COMMENT 'AES-256-GCM encrypted private key seed (hex)',
    iv VARCHAR(64) NOT NULL COMMENT 'AES-GCM initialization vector (hex)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_keypair_address (address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Solana encrypted keypair store';
