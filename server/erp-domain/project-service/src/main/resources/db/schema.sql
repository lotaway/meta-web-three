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