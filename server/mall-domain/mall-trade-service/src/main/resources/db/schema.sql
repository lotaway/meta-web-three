### ============ order-service ============
-- Order Service Schema (OMS);

CREATE TABLE IF NOT EXISTS tb_order (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    coupon_id BIGINT,
    order_sn VARCHAR(64) NOT NULL UNIQUE,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    member_username VARCHAR(64),
    total_amount DECIMAL(18, 2) DEFAULT 0.00,
    pay_amount DECIMAL(18, 2) DEFAULT 0.00,
    freight_amount DECIMAL(18, 2) DEFAULT 0.00,
    promotion_amount DECIMAL(18, 2) DEFAULT 0.00,
    integration_amount DECIMAL(18, 2) DEFAULT 0.00,
    coupon_amount DECIMAL(18, 2) DEFAULT 0.00,
    discount_amount DECIMAL(18, 2) DEFAULT 0.00,
    pay_type INT,
    source_type INT,
    status INT,
    order_type INT,
    delivery_company VARCHAR(64),
    delivery_sn VARCHAR(64),
    auto_confirm_day INT,
    integration INT,
    growth INT,
    promotion_info VARCHAR(255),
    bill_type INT,
    bill_header VARCHAR(200),
    bill_content VARCHAR(200),
    bill_receiver_phone VARCHAR(32),
    bill_receiver_email VARCHAR(64),
    receiver_name VARCHAR(100) NOT NULL,
    receiver_phone VARCHAR(32) NOT NULL,
    receiver_post_code VARCHAR(32),
    receiver_province VARCHAR(32),
    receiver_city VARCHAR(32),
    receiver_region VARCHAR(32),
    receiver_detail_address VARCHAR(200),
    note VARCHAR(500),
    confirm_status SMALLINT,
    delete_status SMALLINT DEFAULT 0,
    member_receive_address_id BIGINT,
    use_integration INT,
    payment_time TIMESTAMP,
    delivery_time TIMESTAMP,
    receive_time TIMESTAMP,
    comment_time TIMESTAMP,
    modify_time TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tenant_id BIGINT
);

CREATE INDEX IF NOT EXISTS idx_tb_order_tenant ON tb_order(tenant_id);

CREATE TABLE IF NOT EXISTS tb_order_item (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    order_sn VARCHAR(64) NOT NULL,
    product_id BIGINT NOT NULL,
    product_pic VARCHAR(255),
    product_name VARCHAR(200),
    product_brand VARCHAR(200),
    product_sn VARCHAR(64),
    product_price DECIMAL(10, 2),
    product_quantity INT,
    product_sku_id BIGINT,
    product_sku_code VARCHAR(50),
    product_category_id BIGINT,
    promotion_name VARCHAR(255),
    promotion_amount DECIMAL(10, 2),
    coupon_amount DECIMAL(10, 2),
    integration_amount DECIMAL(10, 2),
    real_amount DECIMAL(10, 2),
    gift_integration INT DEFAULT 0,
    gift_growth INT DEFAULT 0,
    product_attr VARCHAR(500),
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_order_item_tenant ON tb_order_item(tenant_id);

CREATE TABLE IF NOT EXISTS tb_order_return_apply (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    company_address_id BIGINT,
    product_id BIGINT,
    order_sn VARCHAR(64) NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    member_username VARCHAR(64),
    return_amount DECIMAL(10, 2),
    return_name VARCHAR(100),
    return_phone VARCHAR(100),
    status INT DEFAULT 0,
    handle_time TIMESTAMP,
    product_pic VARCHAR(500),
    product_name VARCHAR(200),
    product_brand VARCHAR(200),
    product_attr VARCHAR(500),
    product_count INT,
    product_price DECIMAL(10, 2),
    product_real_price DECIMAL(10, 2),
    reason VARCHAR(200),
    description VARCHAR(500),
    proof_pics VARCHAR(1000),
    handle_note VARCHAR(500),
    handle_man VARCHAR(100),
    receive_man VARCHAR(100),
    receive_time TIMESTAMP,
    receive_note VARCHAR(500),
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_order_return_apply_tenant ON tb_order_return_apply(tenant_id);

CREATE TABLE IF NOT EXISTS tb_order_return_reason (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    sort INT DEFAULT 0,
    status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_order_setting (
    id BIGINT PRIMARY KEY,
    flash_order_overtime INT,
    normal_order_overtime INT,
    confirm_overtime INT,
    finish_overtime INT,
    comment_overtime INT
);

CREATE TABLE IF NOT EXISTS tb_company_address (
    id BIGINT PRIMARY KEY,
    address_name VARCHAR(200),
    send_status SMALLINT DEFAULT 0,
    receive_status SMALLINT DEFAULT 0,
    name VARCHAR(64),
    phone VARCHAR(32),
    province VARCHAR(64),
    city VARCHAR(64),
    region VARCHAR(64),
    detail_address VARCHAR(255),
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_company_address_tenant ON tb_company_address(tenant_id);

CREATE TABLE IF NOT EXISTS tb_order_operate_log (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    operate_man VARCHAR(100),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_status INT,
    note VARCHAR(500)
);

-- Saga Transaction Instance Table
CREATE TABLE IF NOT EXISTS tb_saga_instance (
    id BIGSERIAL PRIMARY KEY,
    saga_id VARCHAR(64) NOT NULL,  -- Saga transaction ID,
    biz_id VARCHAR(64),  -- Business ID (e.g., order ID),
    saga_type VARCHAR(32) NOT NULL,  -- Saga type: ORDER_PAYMENT_SAGA,
    status VARCHAR(32) NOT NULL DEFAULT 'RUNNING',  -- Status: RUNNING/COMPLETED/COMPENSATED/FAILED,
    current_step VARCHAR(32),  -- Current executing step,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Start time,
    end_time TIMESTAMP,  -- End time,
    error_message VARCHAR(1024),  -- Error message if failed,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_saga_id UNIQUE (saga_id)
);
CREATE INDEX IF NOT EXISTS idx_biz_id ON tb_saga_instance (biz_id);
CREATE INDEX IF NOT EXISTS idx_status ON tb_saga_instance (status);
CREATE INDEX IF NOT EXISTS idx_saga_type ON tb_saga_instance (saga_type);

-- Saga Step Execution Table
CREATE TABLE IF NOT EXISTS tb_saga_step (
    id BIGSERIAL PRIMARY KEY,
    saga_id VARCHAR(64) NOT NULL,  -- Saga transaction ID,
    step_name VARCHAR(64) NOT NULL,  -- Step name,
    step_order INT NOT NULL,  -- Step execution order,
    service_name VARCHAR(64) NOT NULL,  -- Target service name,
    compensable BOOLEAN DEFAULT TRUE,  -- Whether this step can be compensated,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',  -- Status: PENDING/RUNNING/COMPLETED/COMPENSATED/FAILED,
    request_data TEXT,  -- Request data JSON,
    response_data TEXT,  -- Response data JSON,
    compensation_data TEXT,  -- Compensation data JSON,
    retry_count INT DEFAULT 0,  -- Retry count,
    max_retries INT DEFAULT 3,  -- Max retries,
    error_message VARCHAR(1024),  -- Error message if failed,
    start_time TIMESTAMP,  -- Step start time,
    end_time TIMESTAMP,  -- Step end time,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_saga_id ON tb_saga_step (saga_id);
CREATE INDEX IF NOT EXISTS idx_status ON tb_saga_step (status);
CREATE INDEX IF NOT EXISTS idx_step_order ON tb_saga_step (step_order);
### ============ payment-service ============
-- Payment Service Schema

CREATE TABLE IF NOT EXISTS Exchange_Orders (
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
    remark TEXT,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_exchange_orders_tenant ON Exchange_Orders(tenant_id);

COMMENT ON COLUMN Exchange_Orders.order_type IS 'Order type: BUY_CRYPTO, SELL_CRYPTO';
COMMENT ON COLUMN Exchange_Orders.status IS 'Order status: PENDING, PAID, PROCESSING, COMPLETED, FAILED, EXPIRED, CANCELLED';
COMMENT ON COLUMN Exchange_Orders.fiat_currency IS 'Fiat currency: USD, CNY, EUR';
COMMENT ON COLUMN Exchange_Orders.crypto_currency IS 'Cryptocurrency: BTC, ETH, USDT, USDC';
COMMENT ON COLUMN Exchange_Orders.payment_method IS 'Payment method: ALIPAY, WECHAT, BANK_TRANSFER, APPLE_PAY, GOOGLE_PAY';

CREATE INDEX IF NOT EXISTS idx_user_id ON Exchange_Orders (user_id);
CREATE INDEX IF NOT EXISTS idx_order_no ON Exchange_Orders (order_no);
CREATE INDEX IF NOT EXISTS idx_status ON Exchange_Orders (status);
CREATE INDEX IF NOT EXISTS idx_created_at ON Exchange_Orders (created_at);
CREATE INDEX IF NOT EXISTS idx_payment_order_no ON Exchange_Orders (payment_order_no);
CREATE INDEX IF NOT EXISTS idx_crypto_transaction_hash ON Exchange_Orders (crypto_transaction_hash);

CREATE TABLE IF NOT EXISTS Crypto_Prices (
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

CREATE TABLE IF NOT EXISTS User_Kyc (
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
    id_card_front_url VARCHAR(255) NOT NULL,
    id_card_back_url VARCHAR(255) NOT NULL,
    selfie_url VARCHAR(255) NOT NULL,
    face_compare_score NUMERIC(5,2),
    id_card_verified BOOLEAN DEFAULT FALSE,
    face_verified BOOLEAN DEFAULT FALSE,
    verification_time TIMESTAMP,
    verification_result VARCHAR(20),
    verification_notes TEXT,
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
    updated_at TIMESTAMP,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_user_kyc_tenant ON User_Kyc(tenant_id);

COMMENT ON COLUMN User_Kyc.level IS 'KYC level: L0, L1, L2, L3';
COMMENT ON COLUMN User_Kyc.status IS 'KYC status: PENDING, APPROVED, REJECTED, EXPIRED';

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS user_kyc_updated ON User_Kyc;
CREATE TRIGGER user_kyc_updated
BEFORE UPDATE ON User_Kyc
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Risk control tables
CREATE TABLE IF NOT EXISTS Risk_Decision_Log (
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

CREATE INDEX IF NOT EXISTS idx_Risk_Decision_Log_user_id ON Risk_Decision_Log (user_id);
CREATE INDEX IF NOT EXISTS idx_Risk_Decision_Log_device_id ON Risk_Decision_Log (device_id);
CREATE INDEX IF NOT EXISTS idx_Risk_Decision_Log_ts ON Risk_Decision_Log (ts);

CREATE TABLE IF NOT EXISTS Risk_Rules (
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

CREATE INDEX IF NOT EXISTS idx_risk_rules_scene ON Risk_Rules (scene);
CREATE INDEX IF NOT EXISTS idx_risk_rules_status ON Risk_Rules (status);

CREATE TABLE IF NOT EXISTS Credit_Profile (
  user_id BIGINT PRIMARY KEY,
  base_credit_limit INT NOT NULL DEFAULT 10000,
  current_credit_limit INT NOT NULL DEFAULT 10000,
  credit_used INT NOT NULL DEFAULT 0,
  risk_level VARCHAR(16) NOT NULL DEFAULT 'C',
  last_score INT,
  last_limit_adjustment TIMESTAMP,
  overdue_count_3m INT NOT NULL DEFAULT 0,
  transaction_success_rate NUMERIC(5,2) NOT NULL DEFAULT 100.00,
  credit_utilization_rate NUMERIC(5,2) NOT NULL DEFAULT 0.00,
  last_score_change INT,
  max_adjustment_percentage NUMERIC(5,2) NOT NULL DEFAULT 0.15,
  adjustment_history JSON,
  last_update TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_credit_profile_tenant ON Credit_Profile(tenant_id);

COMMENT ON COLUMN Credit_Profile.base_credit_limit IS 'Base credit limit without adjustments';
COMMENT ON COLUMN Credit_Profile.current_credit_limit IS 'Current adjusted credit limit';
COMMENT ON COLUMN Credit_Profile.last_limit_adjustment IS 'Last time credit limit was adjusted';
COMMENT ON COLUMN Credit_Profile.overdue_count_3m IS 'Overdue count in last 3 months';
COMMENT ON COLUMN Credit_Profile.transaction_success_rate IS 'Transaction success rate percentage';
COMMENT ON COLUMN Credit_Profile.credit_utilization_rate IS 'Credit utilization rate percentage';
COMMENT ON COLUMN Credit_Profile.last_score_change IS 'Last score change compared to previous score';
COMMENT ON COLUMN Credit_Profile.max_adjustment_percentage IS 'Maximum adjustment percentage allowed';
COMMENT ON COLUMN Credit_Profile.adjustment_history IS 'JSON array of adjustment history';

COMMENT ON TABLE Credit_Profile IS 'User credit profiles';
COMMENT ON COLUMN Credit_Profile.risk_level IS 'Risk level: A, B, C, D';
COMMENT ON COLUMN Credit_Profile.last_score IS 'Last risk score';

-- Table: payment_reconciliation_diff (Reconciliation difference record)
CREATE TABLE IF NOT EXISTS payment_reconciliation_diff (
    id BIGSERIAL PRIMARY KEY,
    reconciliation_date DATE NOT NULL,
    diff_type VARCHAR(20) NOT NULL,
    order_no VARCHAR(64),
    internal_amount NUMERIC(20,2),
    external_amount NUMERIC(20,2),
    amount_difference NUMERIC(20,2),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    handle_remark TEXT,
    handled_at TIMESTAMP,
    handled_by VARCHAR(64),
    tenant_id BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_payment_reconciliation_diff_tenant ON payment_reconciliation_diff(tenant_id);
CREATE INDEX IF NOT EXISTS idx_reconciliation_diff_date ON payment_reconciliation_diff (reconciliation_date);
CREATE INDEX IF NOT EXISTS idx_reconciliation_diff_type ON payment_reconciliation_diff (diff_type);
CREATE INDEX IF NOT EXISTS idx_reconciliation_diff_status ON payment_reconciliation_diff (status);
### ============ cart-service ============
-- Cart service schema

CREATE TABLE IF NOT EXISTS oms_cart_item (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_sku_id BIGINT,
    member_id BIGINT NOT NULL,
    quantity INT NOT NULL DEFAULT 1,
    price DECIMAL(10, 2) DEFAULT 0.00,
    product_pic VARCHAR(255),
    product_name VARCHAR(200),
    product_sub_title VARCHAR(500),
    product_sku_code VARCHAR(50),
    member_nickname VARCHAR(100),
    create_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modify_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delete_status INT DEFAULT 0,
    product_category_id BIGINT,
    product_brand VARCHAR(200),
    product_sn VARCHAR(64),
    product_attr VARCHAR(500),
    tenant_id BIGINT
);

CREATE INDEX IF NOT EXISTS idx_cart_member_id ON oms_cart_item (member_id);
CREATE INDEX IF NOT EXISTS idx_cart_product_id ON oms_cart_item (product_id);
CREATE INDEX IF NOT EXISTS idx_cart_product_sku_id ON oms_cart_item (product_sku_id);
CREATE INDEX IF NOT EXISTS idx_oms_cart_item_tenant ON oms_cart_item(tenant_id);
### ============ promotion-service ============
-- Promotion Service Schema (SMS);

CREATE TABLE IF NOT EXISTS tb_coupon (
    id BIGINT PRIMARY KEY,
    type INT DEFAULT 0, -- 0->all, 1->member, 2->buy, 3->register,
    name VARCHAR(100) NOT NULL,
    platform INT DEFAULT 0, -- 0->all, 1->mobile, 2->pc,
    count INT DEFAULT 0,
    amount DECIMAL(10, 2) DEFAULT 0.00,
    per_limit INT DEFAULT 1,
    min_point DECIMAL(10, 2) DEFAULT 0.00,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    use_type INT DEFAULT 0, -- 0->all, 1->category, 2->product,
    note VARCHAR(255),
    publish_count INT DEFAULT 0,
    use_count INT DEFAULT 0,
    receive_count INT DEFAULT 0,
    enable_time TIMESTAMP,
    code VARCHAR(64),
    member_level INT DEFAULT 0,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_coupon_tenant ON tb_coupon(tenant_id);

CREATE TABLE IF NOT EXISTS tb_coupon_history (
    id BIGINT PRIMARY KEY,
    coupon_id BIGINT NOT NULL,
    member_id BIGINT NOT NULL,
    coupon_code VARCHAR(64),
    member_nickname VARCHAR(64),
    get_type INT DEFAULT 0, -- 0->direct, 1->system,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    use_status INT DEFAULT 0, -- 0->unused, 1->used, 2->expired,
    use_time TIMESTAMP,
    order_id BIGINT,
    order_sn VARCHAR(64),
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_coupon_history_tenant ON tb_coupon_history(tenant_id);

CREATE TABLE IF NOT EXISTS tb_coupon_product_relation (
    id BIGINT PRIMARY KEY,
    coupon_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    product_sn VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS tb_coupon_product_category_relation (
    id BIGINT PRIMARY KEY,
    coupon_id BIGINT NOT NULL,
    product_category_id BIGINT NOT NULL,
    product_category_name VARCHAR(200),
    parent_category_name VARCHAR(200)
);

CREATE TABLE IF NOT EXISTS tb_flash_promotion (
    id BIGINT PRIMARY KEY,
    title VARCHAR(200),
    start_date DATE,
    end_date DATE,
    status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_flash_promotion_tenant ON tb_flash_promotion(tenant_id);

CREATE TABLE IF NOT EXISTS tb_flash_promotion_session (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    start_time TIME,
    end_time TIME,
    status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_flash_promotion_session_tenant ON tb_flash_promotion_session(tenant_id);

CREATE TABLE IF NOT EXISTS tb_flash_promotion_product_relation (
    id BIGINT PRIMARY KEY,
    flash_promotion_id BIGINT NOT NULL,
    flash_promotion_session_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    flash_promotion_price DECIMAL(10, 2),
    flash_promotion_count INT,
    flash_promotion_limit INT,
    sort INT DEFAULT 0,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_tb_flash_promotion_product_relation_tenant ON tb_flash_promotion_product_relation(tenant_id);

CREATE TABLE IF NOT EXISTS tb_flash_promotion_log (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    member_phone VARCHAR(32),
    product_name VARCHAR(200),
    subscribe_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    send_time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_home_advertise (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    type INT DEFAULT 0, -- 0->pc, 1->app,
    pic VARCHAR(255),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status SMALLINT DEFAULT 1,
    click_count INT DEFAULT 0,
    order_count INT DEFAULT 0,
    url VARCHAR(500),
    note VARCHAR(255),
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_brand (
    id BIGINT PRIMARY KEY,
    brand_id BIGINT NOT NULL,
    brand_name VARCHAR(100),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_new_product (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_recommend_product (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_recommend_subject (
    id BIGINT PRIMARY KEY,
    subject_id BIGINT NOT NULL,
    subject_name VARCHAR(200),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_cms_subject (
    id BIGINT PRIMARY KEY,
    category_id BIGINT,
    title VARCHAR(200),
    pic VARCHAR(500),
    product_count INT DEFAULT 0,
    recommend_status SMALLINT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    collect_count INT DEFAULT 0,
    read_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    album_pics JSON,
    description TEXT,
    show_status SMALLINT DEFAULT 1,
    forward_count INT DEFAULT 0,
    category_name VARCHAR(100),
    content TEXT
);

CREATE TABLE IF NOT EXISTS tb_cms_subject_category (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    icon VARCHAR(500),
    subject_count INT DEFAULT 0,
    show_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_cms_prefrence_area (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    sub_title VARCHAR(200),
    sort INT DEFAULT 0,
    show_status SMALLINT DEFAULT 1,
    pic BYTEA
);
### ============ after-sale-service ============
-- After-Sale Service Schema

CREATE TABLE IF NOT EXISTS after_sale_order (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL,
    order_no VARCHAR(64),
    user_id BIGINT,
    product_id BIGINT,
    sku_id BIGINT,
    product_name VARCHAR(200),
    product_image VARCHAR(500),
    quantity INT DEFAULT 1,
    refund_amount INT DEFAULT 0,
    after_sale_type INT DEFAULT 0,
    after_sale_status INT DEFAULT 0,
    apply_reason VARCHAR(500),
    reject_reason VARCHAR(500),
    apply_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    process_time TIMESTAMP,
    complete_time TIMESTAMP,
    remark VARCHAR(500),
    tenant_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_after_sale_order_tenant ON after_sale_order(tenant_id);
CREATE INDEX IF NOT EXISTS idx_after_sale_order_user_id ON after_sale_order (user_id);
CREATE INDEX IF NOT EXISTS idx_after_sale_order_status ON after_sale_order (after_sale_status);
