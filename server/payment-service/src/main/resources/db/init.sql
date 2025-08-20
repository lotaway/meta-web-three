CREATE DATABASE payment_service 
WITH ENCODING 'UTF8' 
LC_COLLATE = 'en_US.UTF-8' 
LC_CTYPE = 'en_US.UTF-8';

CREATE TABLE Exchange_Orders (
    id BIGSERIAL PRIMARY KEY,
    order_no VARCHAR(64) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    fiat_currency VARCHAR(10) NOT NULL,
    crypto_currency VARCHAR(10) NOT NULL,
    fiat_amount NUMERIC(20,4) NOT NULL,
    crypto_amount BIGINT NOT NULL,
    crypto_decimals SMALLINT DEFAULT 18,
    fee NUMERIC(20,8) NOT NULL,
    exchange_rate NUMERIC(20,8) NOT NULL,
    settlement_amount NUMERIC(20,4),
    actual_rate NUMERIC(20,8),
    payment_method VARCHAR(20) NOT NULL,
    payment_order_no VARCHAR(64),
    crypto_transaction_hash VARCHAR(128),
    user_wallet_address VARCHAR(128),
    failure_reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    paid_at TIMESTAMP,
    completed_at TIMESTAMP,
    expired_at TIMESTAMP,
    kyc_level VARCHAR(10),
    kyc_verified BOOLEAN DEFAULT FALSE,
    remark TEXT
);

COMMENT ON COLUMN Exchange_Orders.order_type IS 'Order type: BUY_CRYPTO, SELL_CRYPTO';
COMMENT ON COLUMN Exchange_Orders.status IS 'Order status: PENDING, PAID, PROCESSING, COMPLETED, FAILED, EXPIRED, CANCELLED';
COMMENT ON COLUMN Exchange_Orders.fiat_currency IS 'Fiat currency: USD, CNY, EUR';
COMMENT ON COLUMN Exchange_Orders.crypto_currency IS 'Cryptocurrency: BTC, ETH, USDT, USDC';
COMMENT ON COLUMN Exchange_Orders.payment_method IS 'Payment method: ALIPAY, WECHAT, BANK_TRANSFER, APPLE_PAY, GOOGLE_PAY';

CREATE INDEX idx_user_id ON Exchange_Orders (user_id);
CREATE INDEX idx_order_no ON Exchange_Orders (order_no);
CREATE INDEX idx_status ON Exchange_Orders (status);
CREATE INDEX idx_created_at ON Exchange_Orders (created_at);
CREATE INDEX idx_payment_order_no ON Exchange_Orders (payment_order_no);
CREATE INDEX idx_crypto_transaction_hash ON Exchange_Orders (crypto_transaction_hash);

CREATE TABLE Crypto_Prices (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    price NUMERIC(20,8) NOT NULL,
    bid_price NUMERIC(20,8) NOT NULL,
    ask_price NUMERIC(20,8) NOT NULL,
    volume_24h NUMERIC(20,8) NOT NULL,
    change_24h NUMERIC(20,8) NOT NULL,
    change_percent_24h NUMERIC(10,2) NOT NULL,
    source VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON COLUMN Crypto_Prices.symbol IS 'Trading pair: BTC-USD, ETH-USD etc';
COMMENT ON COLUMN Crypto_Prices.base_currency IS 'Base currency: BTC, ETH, USDT';
COMMENT ON COLUMN Crypto_Prices.quote_currency IS 'Quote currency: USD, CNY, EUR';
COMMENT ON COLUMN Crypto_Prices.source IS 'Price source: binance, coinbase, okx';

CREATE TABLE User_Kyc (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    level VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL,
    real_name VARCHAR(100),
    id_number VARCHAR(50),
    id_type VARCHAR(20),
    phone_number VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    country VARCHAR(50),
    nationality VARCHAR(50),
    date_of_birth VARCHAR(20),
    gender VARCHAR(10),
    id_card_front_url VARCHAR(255),
    id_card_back_url VARCHAR(255),
    selfie_url VARCHAR(255),
    proof_of_address_url VARCHAR(255),
    bank_account_number VARCHAR(50),
    bank_name VARCHAR(100),
    bank_branch VARCHAR(100),
    tax_id VARCHAR(50),
    occupation VARCHAR(100),
    employer VARCHAR(100),
    annual_income VARCHAR(50),
    source_of_funds VARCHAR(100),
    purpose_of_transaction VARCHAR(200),
    reviewer_id VARCHAR(50),
    review_notes TEXT,
    submitted_at TIMESTAMP,
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

COMMENT ON COLUMN User_Kyc.level IS 'KYC level: L0, L1, L2, L3';
COMMENT ON COLUMN User_Kyc.status IS 'KYC status: PENDING, APPROVED, REJECTED, EXPIRED';

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_kyc_updated
BEFORE UPDATE ON User_Kyc
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Risk control tables
CREATE TABLE Risk_Decision_Log (
  id BIGSERIAL PRIMARY KEY,
  biz_order_id VARCHAR(64),
  user_id BIGINT NOT NULL,
  device_id VARCHAR(128) NOT NULL,
  scene VARCHAR(32) NOT NULL,
  decision VARCHAR(16) NOT NULL,
  score INT,
  reasons JSON,
  features JSON,
  latency_ms INT,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE Risk_Decision_Log IS 'Risk decision logs';
COMMENT ON COLUMN Risk_Decision_Log.biz_order_id IS 'Business order ID';
COMMENT ON COLUMN Risk_Decision_Log.decision IS 'Decision result: approve, reject, review';
COMMENT ON COLUMN Risk_Decision_Log.score IS 'Risk score';
COMMENT ON COLUMN Risk_Decision_Log.reasons IS 'JSON array of reason codes';
COMMENT ON COLUMN Risk_Decision_Log.features IS 'JSON object of features used in decision';

CREATE INDEX idx_Risk_Decision_Log_user_id ON Risk_Decision_Log (user_id);
CREATE INDEX idx_Risk_Decision_Log_device_id ON Risk_Decision_Log (device_id);
CREATE INDEX idx_Risk_Decision_Log_ts ON Risk_Decision_Log (ts);

CREATE TABLE Risk_Rules (
  id BIGSERIAL PRIMARY KEY,
  scene VARCHAR(32) NOT NULL,
  rule_code VARCHAR(32) NOT NULL,
  expr TEXT NOT NULL,
  priority INT NOT NULL,
  status SMALLINT NOT NULL DEFAULT 1,
  version VARCHAR(16) NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE Risk_Rules IS 'Risk control rules';
COMMENT ON COLUMN Risk_Rules.scene IS 'Business scene: new_credit, transaction, etc';
COMMENT ON COLUMN Risk_Rules.rule_code IS 'Rule code: R101, R203, etc';
COMMENT ON COLUMN Risk_Rules.expr IS 'Rule expression';
COMMENT ON COLUMN Risk_Rules.priority IS 'Execution priority';
COMMENT ON COLUMN Risk_Rules.status IS 'Rule status: 1-active, 0-inactive';

CREATE INDEX idx_risk_rules_scene ON Risk_Rules (scene);
CREATE INDEX idx_risk_rules_status ON Risk_Rules (status);

CREATE TABLE Credit_Profile (
  user_id BIGINT PRIMARY KEY,
  credit_limit INT NOT NULL DEFAULT 0,
  credit_used INT NOT NULL DEFAULT 0,
  risk_level VARCHAR(16) NOT NULL DEFAULT 'C',
  last_score INT,
  last_update TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE Credit_Profile IS 'User credit profiles';
COMMENT ON COLUMN Credit_Profile.risk_level IS 'Risk level: A, B, C, D';
COMMENT ON COLUMN Credit_Profile.last_score IS 'Last risk score';