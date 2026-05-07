-- Message Service Schema

CREATE TABLE IF NOT EXISTS tb_notification (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    icon VARCHAR(100),
    image_url VARCHAR(500),
    type VARCHAR(32) NOT NULL,
    related_id VARCHAR(64),
    read_status INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_notification_user_id ON tb_notification(user_id);
CREATE INDEX IF NOT EXISTS idx_notification_type ON tb_notification(type);
CREATE INDEX IF NOT EXISTS idx_notification_read_status ON tb_notification(read_status);

COMMENT ON COLUMN tb_notification.type IS 'SYSTEM/ORDER/PROMOTION';
