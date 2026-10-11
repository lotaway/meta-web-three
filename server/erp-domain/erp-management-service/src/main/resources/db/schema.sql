-- ============================================\n-- erp-management-service combined schema\n-- Parts: hrm (hrm_department, hrm_position, hrm_employee), crm (crm_*), project (pm_*)\n-- ============================================\n
-- ---------- Part from hrm-service ----------

-- HRM Department Table
CREATE TABLE IF NOT EXISTS hrm_department (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    parent_id BIGINT DEFAULT 0,
    level INT DEFAULT 1,
    sort_order INT DEFAULT 0,
    leader_id BIGINT,
    status INT DEFAULT 1,
    remark VARCHAR(500),
    tenant_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    UNIQUE (code, tenant_id, deleted)
);
CREATE INDEX IF NOT EXISTS idx_hrm_department_parent_id ON hrm_department (parent_id);
CREATE INDEX IF NOT EXISTS idx_hrm_department_level ON hrm_department (level);

-- HRM Position Table
CREATE TABLE IF NOT EXISTS hrm_position (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    department_id BIGINT NOT NULL,
    level INT DEFAULT 1,
    base_salary DECIMAL(10,2) DEFAULT 0,
    status INT DEFAULT 1,
    description VARCHAR(500),
    requirements VARCHAR(500),
    tenant_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    UNIQUE (code, tenant_id, deleted)
);
CREATE INDEX IF NOT EXISTS idx_hrm_position_department_id ON hrm_position (department_id);

-- HRM Employee Table
CREATE TABLE IF NOT EXISTS hrm_employee (
    id BIGSERIAL PRIMARY KEY,
    employee_no VARCHAR(50) NOT NULL,
    name VARCHAR(50) NOT NULL,
    gender INT DEFAULT 1,
    mobile VARCHAR(20),
    email VARCHAR(100),
    id_card VARCHAR(18),
    birthday DATE,
    native_place VARCHAR(100),
    nation VARCHAR(50),
    marital_status INT,
    political_status VARCHAR(50),
    education INT,
    graduate_school VARCHAR(100),
    major VARCHAR(100),
    hire_date DATE,
    formal_date DATE,
    department_id BIGINT,
    position_id BIGINT,
    work_location VARCHAR(100),
    status INT DEFAULT 0,
    contract_start_date DATE,
    contract_end_date DATE,
    emergency_contact VARCHAR(50),
    emergency_phone VARCHAR(20),
    bank_account VARCHAR(50),
    bank_name VARCHAR(100),
    social_security_no VARCHAR(50),
    housing_fund_no VARCHAR(50),
    photo_url VARCHAR(500),
    remark VARCHAR(500),
    tenant_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    UNIQUE (employee_no, tenant_id, deleted)
);
CREATE INDEX IF NOT EXISTS idx_hrm_employee_department_id ON hrm_employee (department_id);
CREATE INDEX IF NOT EXISTS idx_hrm_employee_position_id ON hrm_employee (position_id);
CREATE INDEX IF NOT EXISTS idx_hrm_employee_status ON hrm_employee (status);
CREATE INDEX IF NOT EXISTS idx_hrm_employee_mobile ON hrm_employee (mobile);
CREATE INDEX IF NOT EXISTS idx_hrm_employee_id_card ON hrm_employee (id_card);

-- ---------- Part from crm-service ----------

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


-- ---------- Part from project-service ----------

-- Project Management Module Database Schema

-- Project Table
CREATE TABLE IF NOT EXISTS pm_project (
    id BIGSERIAL PRIMARY KEY,
    project_code VARCHAR(50) NOT NULL UNIQUE,
    project_name VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',
    department_id BIGINT,
    department_name VARCHAR(100),
    manager_id BIGINT,
    manager_name VARCHAR(50),
    start_date DATE,
    end_date DATE,
    budget_amount DECIMAL(15, 2) DEFAULT 0,
    used_amount DECIMAL(15, 2) DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'CNY',
    progress INT DEFAULT 0,
    created_by BIGINT NOT NULL,
    creator_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    remark VARCHAR(500)
);
CREATE INDEX IF NOT EXISTS idx_pm_project_status ON pm_project (status);
CREATE INDEX IF NOT EXISTS idx_pm_project_department ON pm_project (department_id);
CREATE INDEX IF NOT EXISTS idx_pm_project_manager ON pm_project (manager_id);

-- Task Table
CREATE TABLE IF NOT EXISTS pm_task (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL,
    task_code VARCHAR(50) NOT NULL UNIQUE,
    task_name VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    parent_id BIGINT,
    level INT DEFAULT 1,
    sort INT DEFAULT 0,
    assignee_id BIGINT,
    assignee_name VARCHAR(50),
    planned_start_date TIMESTAMP,
    planned_end_date TIMESTAMP,
    actual_start_date TIMESTAMP,
    actual_end_date TIMESTAMP,
    progress INT DEFAULT 0,
    estimated_hours INT DEFAULT 0,
    actual_hours INT DEFAULT 0,
    created_by BIGINT NOT NULL,
    creator_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    remark VARCHAR(500),
    CONSTRAINT fk_pm_task_project FOREIGN KEY (project_id) REFERENCES pm_project (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pm_task_project ON pm_task (project_id);
CREATE INDEX IF NOT EXISTS idx_pm_task_parent ON pm_task (parent_id);
CREATE INDEX IF NOT EXISTS idx_pm_task_status ON pm_task (status);
CREATE INDEX IF NOT EXISTS idx_pm_task_assignee ON pm_task (assignee_id);

-- Time Entry Table
CREATE TABLE IF NOT EXISTS pm_time_entry (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL,
    project_name VARCHAR(200),
    task_id BIGINT,
    task_name VARCHAR(200),
    employee_id BIGINT NOT NULL,
    employee_name VARCHAR(50) NOT NULL,
    work_date DATE NOT NULL,
    hours DECIMAL(5, 2) NOT NULL,
    work_type VARCHAR(50),
    description VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    approver_id BIGINT,
    approver_name VARCHAR(50),
    approved_at TIMESTAMP,
    created_by BIGINT NOT NULL,
    creator_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    remark VARCHAR(500),
    CONSTRAINT fk_pm_time_entry_project FOREIGN KEY (project_id) REFERENCES pm_project (id) ON DELETE CASCADE,
    CONSTRAINT fk_pm_time_entry_task FOREIGN KEY (task_id) REFERENCES pm_task (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_pm_time_entry_project ON pm_time_entry (project_id);
CREATE INDEX IF NOT EXISTS idx_pm_time_entry_task ON pm_time_entry (task_id);
CREATE INDEX IF NOT EXISTS idx_pm_time_entry_employee ON pm_time_entry (employee_id);
CREATE INDEX IF NOT EXISTS idx_pm_time_entry_work_date ON pm_time_entry (work_date);
CREATE INDEX IF NOT EXISTS idx_pm_time_entry_status ON pm_time_entry (status);

-- Cost Record Table
CREATE TABLE IF NOT EXISTS pm_cost_record (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL,
    project_name VARCHAR(200),
    cost_type VARCHAR(50) NOT NULL,
    cost_code VARCHAR(50),
    cost_name VARCHAR(200) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'CNY',
    cost_date DATE NOT NULL,
    description VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    department_id BIGINT,
    department_name VARCHAR(100),
    created_by BIGINT NOT NULL,
    creator_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    remark VARCHAR(500),
    CONSTRAINT fk_pm_cost_record_project FOREIGN KEY (project_id) REFERENCES pm_project (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pm_cost_record_project ON pm_cost_record (project_id);
CREATE INDEX IF NOT EXISTS idx_pm_cost_record_cost_type ON pm_cost_record (cost_type);
CREATE INDEX IF NOT EXISTS idx_pm_cost_record_cost_date ON pm_cost_record (cost_date);
CREATE INDEX IF NOT EXISTS idx_pm_cost_record_status ON pm_cost_record (status);
CREATE INDEX IF NOT EXISTS idx_pm_cost_record_department ON pm_cost_record (department_id);
