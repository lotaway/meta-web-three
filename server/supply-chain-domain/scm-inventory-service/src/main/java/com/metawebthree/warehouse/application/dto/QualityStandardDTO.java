package com.metawebthree.warehouse.application.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class QualityStandardDTO {
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
}