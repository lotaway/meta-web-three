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
