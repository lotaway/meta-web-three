package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class QualityStandardDO {
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
    private Integer isActive;
    private String remark;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Integer deleted;
}