package com.metawebthree.warehouse.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class QualityInspectionDTO {
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
    private LocalDateTime createTime;
    private List<QualityInspectionItemDTO> items;
}

@Data
class QualityInspectionItemDTO {
    private Long id;
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
}