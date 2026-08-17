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