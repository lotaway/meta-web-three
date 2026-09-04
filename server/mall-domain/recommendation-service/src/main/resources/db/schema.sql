-- =============================================================
-- AI Assisted Shopping (extends recommendation-service)
-- 以图搜图 / 智能匹配 / 文本纠错
-- =============================================================

-- AI provider 运行时配置（管理员可在后台修改，覆盖 application.yml 默认值）
CREATE TABLE IF NOT EXISTS ai_shopping_config (
    config_key VARCHAR(64) PRIMARY KEY,
    config_value TEXT,
    description VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI 搜索日志（观测 / 后台展示）
CREATE TABLE IF NOT EXISTS ai_search_log (
    id BIGINT PRIMARY KEY,
    user_id BIGINT,
    search_type VARCHAR(32),
    query_text TEXT,
    corrected_text TEXT,
    result_count INT,
    response_time_ms BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ai_search_log_created_at ON ai_search_log (created_at);
