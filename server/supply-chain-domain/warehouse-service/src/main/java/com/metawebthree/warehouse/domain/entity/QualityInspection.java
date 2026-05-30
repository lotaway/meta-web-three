package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 质检记录实体
 * 记录每次质检的结果
 */
@Data
public class QualityInspection {
    private Long id;
    private String inspectionNo;
    private Long orderId;
    private String orderNo;
    private String inboundType;
    private Long warehouseId;
    private String supplierCode;
    private String supplierName;
    private String inspectionType;
    private String inspectionStatus;
    private Integer totalQuantity;
    private Integer inspectedQuantity;
    private Integer qualifiedQuantity;
    private Integer unqualifiedQuantity;
    private Integer concessionQuantity;
    private BigDecimal defectRate;
    private String inspector;
    private LocalDateTime inspectionTime;
    private String resultRemark;
    private Boolean isAutoInspection;
    private String sourceSystem;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;

    private List<QualityInspectionItem> items;

    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_IN_PROGRESS = "IN_PROGRESS";
    public static final String STATUS_PASSED = "PASSED";
    public static final String STATUS_FAILED = "FAILED";
    public static final String STATUS_CONCESSION = "CONCESSION";

    public static final String TYPE_FULL = "FULL";
    public static final String TYPE_SAMPLE = "SAMPLE";

    public boolean isPending() {
        return STATUS_PENDING.equals(this.inspectionStatus);
    }

    public boolean isPassed() {
        return STATUS_PASSED.equals(this.inspectionStatus);
    }

    public boolean isFailed() {
        return STATUS_FAILED.equals(this.inspectionStatus);
    }

    public boolean isConcession() {
        return STATUS_CONCESSION.equals(this.inspectionStatus);
    }

    public void calculateDefectRate() {
        if (this.inspectedQuantity != null && this.inspectedQuantity > 0) {
            BigDecimal unqualified = BigDecimal.valueOf(this.unqualifiedQuantity != null ? this.unqualifiedQuantity : 0);
            BigDecimal inspected = BigDecimal.valueOf(this.inspectedQuantity);
            this.defectRate = unqualified.multiply(BigDecimal.valueOf(100)).divide(inspected, 2, BigDecimal.ROUND_HALF_UP);
        }
    }

    public boolean isQualified() {
        return isPassed() || isConcession();
    }
}