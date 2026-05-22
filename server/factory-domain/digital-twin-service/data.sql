-- Digital Twin Service - Seed Data
-- This script provides reference seed data matching the in-memory DevDataInitializer.
-- Use when switching to a database-backed implementation (e.g., with H2 or PostgreSQL).

-- Workshops
INSERT INTO workshops (workshop_code, workshop_name, description, status, center_x, center_y, width, length)
VALUES ('WS-01', '一号车间', '数字孪生演示车间', 'OPERATING', 0, 0, 40, 40);

-- Production Lines
INSERT INTO production_lines (line_code, line_name, workshop_id, status, capacity, current_output, efficiency)
VALUES ('LINE-01', '演示产线', 'WS-01', 'RUNNING', 100, 85, 85.00);

-- Devices: AGV, ROBOT, PLC, CONVEYOR
INSERT INTO devices (device_code, device_name, device_type, workshop_id, production_line_id, status, position_x, position_y, position_z, rotation_y)
VALUES
('AGV-001',  '搬运机器人A1', 'AGV',     'WS-01', 'LINE-01', 'RUNNING',  2.0,   0.25,  3.0,   0.0),
('AGV-002',  '搬运机器人A2', 'AGV',     'WS-01', 'LINE-01', 'RUNNING',  -3.0,  0.25,  5.0,   0.785),
('ROBOT-001', '机械臂R1',    'ROBOT',   'WS-01', 'LINE-01', 'RUNNING',  5.0,   0.75,  -2.0,  3.14159),
('ROBOT-002', '机械臂R2',    'ROBOT',   'WS-01', 'LINE-01', 'WARNING',  7.0,   0.75,  2.0,   -0.785),
('PLC-001',   'PLC控制器C1', 'PLC',     'WS-01', 'LINE-01', 'ONLINE',   -5.0,  0.3,   -3.0,  0.0),
('PLC-002',   'PLC控制器C2', 'PLC',     'WS-01', 'LINE-01', 'ERROR',    -7.0,  0.3,   4.0,   0.0),
('CONVEYOR-001', '传送带S1', 'CONVEYOR','WS-01', 'LINE-01', 'RUNNING',  0.0,   0.15,  0.0,   0.0),
('AGV-003',  '搬运机器人A3', 'AGV',     'WS-01', 'LINE-01', 'IDLE',     -2.0,  0.25,  -4.0,  1.5708);

-- Alerts
INSERT INTO alerts (alert_code, device_code, workshop_id, level, type, title, description, status, occurred_at)
VALUES
('ALT-001', 'PLC-002',  'WS-01', 'CRITICAL', 'DEVICE_ERROR',     '设备故障',   'PLC控制器C2通信异常',               'TRIGGERED',   CURRENT_TIMESTAMP - INTERVAL '1' HOUR),
('ALT-002', 'ROBOT-002','WS-01', 'WARNING',  'TEMPERATURE_HIGH', '温度告警',   '机械臂R2温度过高',                 'TRIGGERED',   CURRENT_TIMESTAMP - INTERVAL '30' MINUTE),
('ALT-003', 'AGV-001',  'WS-01', 'INFO',     'MAINTENANCE_DUE',  '维护提醒',   '搬运机器人A1即将到期维护',          'ACKNOWLEDGED', CURRENT_TIMESTAMP - INTERVAL '10' MINUTE);
