CREATE TABLE IF NOT EXISTS crm_lead (
    id            BIGINT PRIMARY KEY,
    lead_no       VARCHAR(64)  NOT NULL,
    name          VARCHAR(128) NOT NULL,
    company       VARCHAR(256),
    title         VARCHAR(128),
    email         VARCHAR(256),
    phone         VARCHAR(64),
    mobile        VARCHAR(64),
    source        VARCHAR(64),
    status        VARCHAR(64)  NOT NULL DEFAULT 'NEW',
    score         INT          NOT NULL DEFAULT 0,
    industry      VARCHAR(128),
    city          VARCHAR(128),
    province      VARCHAR(128),
    country       VARCHAR(128),
    description   TEXT,
    assigned_to   VARCHAR(128),
    created_by    VARCHAR(128),
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       INT          NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS crm_opportunity (
    id                 BIGINT PRIMARY KEY,
    opportunity_no     VARCHAR(64)  NOT NULL,
    title              VARCHAR(256) NOT NULL,
    lead_id            BIGINT,
    customer_id        BIGINT,
    contact_id         BIGINT,
    pipeline_id        BIGINT,
    stage              VARCHAR(64)  NOT NULL DEFAULT 'PROSPECTING',
    amount             DECIMAL(18,2),
    probability        INT          NOT NULL DEFAULT 0,
    expected_close_date DATE,
    actual_close_date  DATE,
    competitor         VARCHAR(256),
    description        TEXT,
    assigned_to        VARCHAR(128),
    created_by         VARCHAR(128),
    created_at         TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted            INT          NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS crm_cs_ticket (
    id          BIGINT PRIMARY KEY,
    ticket_no   VARCHAR(64)  NOT NULL,
    title       VARCHAR(256) NOT NULL,
    customer_id BIGINT,
    contact_id  BIGINT,
    order_id    BIGINT,
    type        VARCHAR(64),
    priority    VARCHAR(64)  NOT NULL DEFAULT 'MEDIUM',
    status      VARCHAR(64)  NOT NULL DEFAULT 'OPEN',
    assigned_to VARCHAR(128),
    description TEXT,
    resolution  TEXT,
    created_by  VARCHAR(128),
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     INT          NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS crm_campaign (
    id                BIGINT PRIMARY KEY,
    name              VARCHAR(256) NOT NULL,
    description       TEXT,
    type              VARCHAR(64),
    status            VARCHAR(64)  NOT NULL DEFAULT 'DRAFT',
    start_date        DATE,
    end_date          DATE,
    budget            DECIMAL(18,2),
    actual_cost       DECIMAL(18,2),
    expected_revenue  DECIMAL(18,2),
    target_audience   VARCHAR(512),
    leads_generated   INT          NOT NULL DEFAULT 0,
    converted_customers INT        NOT NULL DEFAULT 0,
    roi               DECIMAL(10,4),
    created_by        VARCHAR(128),
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted           INT          NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS crm_contact (
    id          BIGINT PRIMARY KEY,
    first_name  VARCHAR(128),
    last_name   VARCHAR(128),
    email       VARCHAR(256),
    phone       VARCHAR(64),
    mobile      VARCHAR(64),
    position    VARCHAR(128),
    department  VARCHAR(128),
    customer_id BIGINT,
    is_primary  BOOLEAN      NOT NULL DEFAULT FALSE,
    address     VARCHAR(512),
    city        VARCHAR(128),
    province    VARCHAR(128),
    country     VARCHAR(128),
    postal_code VARCHAR(32),
    birthday    DATE,
    notes       TEXT,
    created_by  VARCHAR(128),
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     INT          NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS crm_sales_pipeline (
    id          BIGINT PRIMARY KEY,
    name        VARCHAR(128) NOT NULL,
    description TEXT,
    stages      TEXT,
    is_default  BOOLEAN      NOT NULL DEFAULT FALSE,
    sort_order  INT          NOT NULL DEFAULT 0,
    created_by  VARCHAR(128),
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     INT          NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_crm_lead_status ON crm_lead(status);
CREATE INDEX IF NOT EXISTS idx_crm_lead_source ON crm_lead(source);
CREATE INDEX IF NOT EXISTS idx_crm_lead_assigned_to ON crm_lead(assigned_to);
CREATE INDEX IF NOT EXISTS idx_crm_opportunity_stage ON crm_opportunity(stage);
CREATE INDEX IF NOT EXISTS idx_crm_opportunity_lead_id ON crm_opportunity(lead_id);
CREATE INDEX IF NOT EXISTS idx_crm_opportunity_customer_id ON crm_opportunity(customer_id);
CREATE INDEX IF NOT EXISTS idx_crm_cs_ticket_status ON crm_cs_ticket(status);
CREATE INDEX IF NOT EXISTS idx_crm_cs_ticket_customer_id ON crm_cs_ticket(customer_id);
CREATE INDEX IF NOT EXISTS idx_crm_campaign_status ON crm_campaign(status);
CREATE INDEX IF NOT EXISTS idx_crm_contact_customer_id ON crm_contact(customer_id);
