package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 质检标准实体
 * 定义每个SKU的质检规则
 */
@Data
public class QualityStandard {
    private Long id;
    private String skuCode;
    private String productName;
    private String inspectionType;
    private String inspectionLevel;
    private BigDecimal sampleRate;
    private String checkItems;
    private Integer acceptanceQty;
    private Integer defectQtyThreshold;
    private BigDecimal weightTolerance;
    private String dimensionTolerance;
    private String packagingRequirement;
    private String labelRequirement;
    private Boolean isActive;
    private String remark;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;

    public static final String INSPECTION_TYPE_FULL = "FULL";
    public static final String INSPECTION_TYPE_SAMPLE = "SAMPLE";
    public static final String INSPECTION_TYPE_AUTO = "AUTO";

    public static final String INSPECTION_LEVEL_NORMAL = "NORMAL";
    public static final String INSPECTION_LEVEL_STRICT = "STRICT";

    public boolean isFullInspection() {
        return INSPECTION_TYPE_FULL.equals(this.inspectionType);
    }

    public boolean isSampleInspection() {
        return INSPECTION_TYPE_SAMPLE.equals(this.inspectionType);
    }

    public boolean isActive() {
        return this.isActive == null || this.isActive;
    }
}