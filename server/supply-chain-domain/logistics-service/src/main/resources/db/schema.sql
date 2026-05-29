-- logistics-service 数据库初始化脚本

-- 承运商表
CREATE TABLE IF NOT EXISTS logistics_carrier (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    carrier_code VARCHAR(64) NOT NULL UNIQUE COMMENT '承运商编码',
    carrier_name VARCHAR(128) NOT NULL COMMENT '承运商名称',
    carrier_type VARCHAR(32) COMMENT '承运商类型：EXPRESS/FREIGHT/POST',
    contact VARCHAR(64) COMMENT '联系人',
    phone VARCHAR(32) COMMENT '联系电话',
    website VARCHAR(256) COMMENT '官网',
    status VARCHAR(32) DEFAULT 'ACTIVE' COMMENT '状态：ACTIVE/INACTIVE',
    base_freight DECIMAL(10,2) DEFAULT 0 COMMENT '基础运费',
    weight_unit_price DECIMAL(10,4) DEFAULT 0 COMMENT '重量单价(元/公斤)',
    volume_unit_price DECIMAL(10,4) DEFAULT 0 COMMENT '体积单价(元/立方)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_carrier_code (carrier_code),
    INDEX idx_status (status)
) COMMENT '承运商表';

-- 物流订单表
CREATE TABLE IF NOT EXISTS logistics_order (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tracking_no VARCHAR(64) NOT NULL UNIQUE COMMENT '运单号',
    order_no VARCHAR(64) NOT NULL COMMENT '订单号',
    carrier_id BIGINT COMMENT '承运商ID',
    carrier_name VARCHAR(128) COMMENT '承运商名称',
    service_type VARCHAR(32) COMMENT '服务类型',
    sender_name VARCHAR(64) NOT NULL COMMENT '发货人姓名',
    sender_phone VARCHAR(32) NOT NULL COMMENT '发货人电话',
    sender_province VARCHAR(32) COMMENT '发货省',
    sender_city VARCHAR(32) COMMENT '发货市',
    sender_district VARCHAR(32) COMMENT '发货区',
    sender_address VARCHAR(256) COMMENT '发货地址',
    receiver_name VARCHAR(64) NOT NULL COMMENT '收货人姓名',
    receiver_phone VARCHAR(32) NOT NULL COMMENT '收货人电话',
    receiver_province VARCHAR(32) COMMENT '收货省',
    receiver_city VARCHAR(32) COMMENT '收货市',
    receiver_district VARCHAR(32) COMMENT '收货区',
    receiver_address VARCHAR(256) COMMENT '收货地址',
    weight DECIMAL(10,3) COMMENT '重量(kg)',
    volume DECIMAL(10,3) COMMENT '体积(立方)',
    freight DECIMAL(10,2) DEFAULT 0 COMMENT '运费',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT '状态：PENDING/PICKED_UP/IN_TRANSIT/OUT_FOR_DELIVERY/DELIVERED/EXCEPTION',
    picked_up_at TIMESTAMP COMMENT '提货时间',
    in_transit_at TIMESTAMP COMMENT '运输中时间',
    out_for_delivery_at TIMESTAMP COMMENT '派送中时间',
    delivered_at TIMESTAMP COMMENT '签收时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_tracking_no (tracking_no),
    INDEX idx_order_no (order_no),
    INDEX idx_carrier_id (carrier_id),
    INDEX idx_status (status)
) COMMENT '物流订单表';

-- 物流轨迹事件表
CREATE TABLE IF NOT EXISTS logistics_tracking_event (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tracking_no VARCHAR(64) NOT NULL COMMENT '运单号',
    event_type VARCHAR(32) NOT NULL COMMENT '事件类型',
    location VARCHAR(128) COMMENT '位置',
    description VARCHAR(512) COMMENT '描述',
    operator VARCHAR(64) COMMENT '操作人',
    occurred_at TIMESTAMP NOT NULL COMMENT '发生时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_tracking_no (tracking_no),
    INDEX idx_occurred_at (occurred_at)
) COMMENT '物流轨迹事件表';