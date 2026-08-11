ALTER TABLE Exchange_Orders ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_exchange_orders_tenant ON Exchange_Orders(tenant_id);

ALTER TABLE User_Kyc ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_user_kyc_tenant ON User_Kyc(tenant_id);

ALTER TABLE Credit_Profile ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_credit_profile_tenant ON Credit_Profile(tenant_id);

ALTER TABLE payment_reconciliation_diff ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_payment_reconciliation_diff_tenant ON payment_reconciliation_diff(tenant_id);