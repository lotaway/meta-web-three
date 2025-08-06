-- 创建payment_service数据库
CREATE DATABASE IF NOT EXISTS payment_service DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE payment_service;

-- 兑换订单表
CREATE TABLE IF NOT EXISTS exchange_orders (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_no VARCHAR(64) NOT NULL UNIQUE COMMENT '订单号',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    order_type VARCHAR(20) NOT NULL COMMENT '订单类型：BUY_CRYPTO, SELL_CRYPTO',
    status VARCHAR(20) NOT NULL COMMENT '订单状态：PENDING, PAID, PROCESSING, COMPLETED, FAILED, EXPIRED, CANCELLED',
    fiat_currency VARCHAR(10) NOT NULL COMMENT '法币币种：USD, CNY, EUR',
    crypto_currency VARCHAR(10) NOT NULL COMMENT '数字币币种：BTC, ETH, USDT, USDC',
    fiat_amount DECIMAL(20,8) NOT NULL COMMENT '法币金额',
    crypto_amount DECIMAL(20,8) NOT NULL COMMENT '数字币数量',
    exchange_rate DECIMAL(20,8) NOT NULL COMMENT '兑换汇率',
    actual_rate DECIMAL(20,8) COMMENT '实际汇率',
    payment_method VARCHAR(20) NOT NULL COMMENT '支付方式：ALIPAY, WECHAT, BANK_TRANSFER, APPLE_PAY, GOOGLE_PAY',
    payment_order_no VARCHAR(64) COMMENT '支付订单号',
    crypto_transaction_hash VARCHAR(128) COMMENT '区块链交易哈希',
    user_wallet_address VARCHAR(128) COMMENT '用户钱包地址',
    failure_reason TEXT COMMENT '失败原因',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    paid_at DATETIME COMMENT '支付时间',
    completed_at DATETIME COMMENT '完成时间',
    expired_at DATETIME COMMENT '过期时间',
    kyc_level VARCHAR(10) COMMENT 'KYC级别',
    kyc_verified BOOLEAN DEFAULT FALSE COMMENT 'KYC是否验证',
    remark TEXT COMMENT '备注',
    INDEX idx_user_id (user_id),
    INDEX idx_order_no (order_no),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_payment_order_no (payment_order_no),
    INDEX idx_crypto_transaction_hash (crypto_transaction_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='兑换订单表';

-- 加密货币价格表
CREATE TABLE IF NOT EXISTS crypto_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL COMMENT '交易对符号：BTC-USD, ETH-USD等',
    base_currency VARCHAR(10) NOT NULL COMMENT '基础币种：BTC, ETH, USDT',
    quote_currency VARCHAR(10) NOT NULL COMMENT '计价币种：USD, CNY, EUR',
    price DECIMAL(20,8) NOT NULL COMMENT '当前价格',
    bid_price DECIMAL(20,8) NOT NULL COMMENT '买价',
    ask_price DECIMAL(20,8) NOT NULL COMMENT '卖价',
    volume_24h DECIMAL(20,8) NOT NULL COMMENT '24小时成交量',
    change_24h DECIMAL(20,8) NOT NULL COMMENT '24小时价格变化',
    change_percent_24h DECIMAL(10,2) NOT NULL COMMENT '24小时价格变化百分比',
    source VARCHAR(20) NOT NULL COMMENT '价格源：binance, coinbase, okx',
    timestamp DATETIME NOT NULL COMMENT '价格时间戳',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_symbol (symbol),
    INDEX idx_base_currency (base_currency),
    INDEX idx_quote_currency (quote_currency),
    INDEX idx_source (source),
    INDEX idx_timestamp (timestamp),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='加密货币价格表';

-- 用户KYC表
CREATE TABLE IF NOT EXISTS user_kyc (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL COMMENT '用户ID',
    level VARCHAR(10) NOT NULL COMMENT 'KYC级别：L0, L1, L2, L3',
    status VARCHAR(20) NOT NULL COMMENT 'KYC状态：PENDING, APPROVED, REJECTED, EXPIRED',
    real_name VARCHAR(100) COMMENT '真实姓名',
    id_number VARCHAR(50) COMMENT '身份证号',
    id_type VARCHAR(20) COMMENT '证件类型：ID_CARD, PASSPORT, DRIVER_LICENSE',
    phone_number VARCHAR(20) COMMENT '手机号',
    email VARCHAR(100) COMMENT '邮箱',
    address TEXT COMMENT '地址',
    country VARCHAR(50) COMMENT '国家',
    nationality VARCHAR(50) COMMENT '国籍',
    date_of_birth VARCHAR(20) COMMENT '出生日期',
    gender VARCHAR(10) COMMENT '性别',
    id_card_front_url VARCHAR(255) COMMENT '身份证正面照片URL',
    id_card_back_url VARCHAR(255) COMMENT '身份证背面照片URL',
    selfie_url VARCHAR(255) COMMENT '自拍照片URL',
    proof_of_address_url VARCHAR(255) COMMENT '地址证明URL',
    bank_account_number VARCHAR(50) COMMENT '银行账号',
    bank_name VARCHAR(100) COMMENT '银行名称',
    bank_branch VARCHAR(100) COMMENT '银行支行',
    tax_id VARCHAR(50) COMMENT '税号',
    occupation VARCHAR(100) COMMENT '职业',
    employer VARCHAR(100) COMMENT '雇主',
    annual_income VARCHAR(50) COMMENT '年收入',
    source_of_funds VARCHAR(100) COMMENT '资金来源',
    purpose_of_transaction VARCHAR(200) COMMENT '交易目的',
    reviewer_id VARCHAR(50) COMMENT '审核员ID',
    review_notes TEXT COMMENT '审核备注',
    submitted_at DATETIME COMMENT '提交时间',
    reviewed_at DATETIME COMMENT '审核时间',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_user_id (user_id),
    INDEX idx_level (level),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_submitted_at (submitted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户KYC表';

-- 插入初始数据
INSERT INTO crypto_prices (symbol, base_currency, quote_currency, price, bid_price, ask_price, volume_24h, change_24h, change_percent_24h, source, timestamp) VALUES
('BTC-USD', 'BTC', 'USD', 45000.00, 44950.00, 45050.00, 1000000.00, 500.00, 1.12, 'binance', NOW()),
('ETH-USD', 'ETH', 'USD', 3000.00, 2995.00, 3005.00, 500000.00, 30.00, 1.01, 'binance', NOW()),
('USDT-USD', 'USDT', 'USD', 1.00, 0.9995, 1.0005, 2000000.00, 0.00, 0.00, 'binance', NOW()),
('USDC-USD', 'USDC', 'USD', 1.00, 0.9995, 1.0005, 1500000.00, 0.00, 0.00, 'binance', NOW()); 