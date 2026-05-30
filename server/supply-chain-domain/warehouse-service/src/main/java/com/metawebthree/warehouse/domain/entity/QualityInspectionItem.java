package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * 质检明细实体
 * 记录每个SKU的质检详情
 */
@Data
public class QualityInspectionItem {
    private Long id;
    private Long inspectionId;
    private String skuCode;
    private String productName;
    private String batchNo;
    private String locationCode;
    private Integer planQuantity;
    private Integer actualQuantity;
    private Integer inspectedQuantity;
    private Integer qualifiedQuantity;
    private Integer unqualifiedQuantity;
    private Integer concessionQuantity;
    private Integer sampleQuantity;
    private String defectItems;
    private String checkResult;
    private String remark;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;

    public static final String RESULT_QUALIFIED = "QUALIFIED";
    public static final String RESULT_UNQUALIFIED = "UNQUALIFIED";
    public static final String RESULT_CONCESSION = "CONCESSION";

    public boolean isQualified() {
        return RESULT_QUALIFIED.equals(this.checkResult);
    }

    public boolean isUnqualified() {
        return RESULT_UNQUALIFIED.equals(this.checkResult);
    }

    public boolean isConcession() {
        return RESULT_CONCESSION.equals(this.checkResult);
    }
}