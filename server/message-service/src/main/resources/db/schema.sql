-- Message Service Schema

CREATE TABLE IF NOT EXISTS tb_notification (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    icon VARCHAR(100),
    image_url VARCHAR(500),
    type VARCHAR(32) NOT NULL COMMENT 'SYSTEM/ORDER/PROMOTION',
    related_id VARCHAR(64),
    read_status INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_type (type),
    INDEX idx_read_status (read_status)
);
