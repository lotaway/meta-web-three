-- 缺陷代码表
CREATE TABLE IF NOT EXISTS mes_qc_defect_code (
    id BIGSERIAL PRIMARY KEY,
    defect_code VARCHAR(50) NOT NULL UNIQUE COMMENT '缺陷代码',
    defect_name VARCHAR(100) NOT NULL COMMENT '缺陷名称',
    category VARCHAR(30) NOT NULL COMMENT '缺陷分类: DIMENSIONAL, SURFACE, MATERIAL, ASSEMBLY, ELECTRICAL, FUNCTIONAL, PACKAGING, OTHER',
    severity VARCHAR(20) NOT NULL COMMENT '严重等级: CRITICAL, MAJOR, MINOR',
    is_critical BOOLEAN DEFAULT FALSE COMMENT '是否致命',
    description VARCHAR(500) COMMENT '缺陷描述',
    disposition_guide VARCHAR(500) COMMENT '处置指南',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_defect_code_code ON mes_qc_defect_code(defect_code);
CREATE INDEX idx_defect_code_category ON mes_qc_defect_code(category);
CREATE INDEX idx_defect_code_severity ON mes_qc_defect_code(severity);
CREATE INDEX idx_defect_code_enabled ON mes_qc_defect_code(is_enabled);