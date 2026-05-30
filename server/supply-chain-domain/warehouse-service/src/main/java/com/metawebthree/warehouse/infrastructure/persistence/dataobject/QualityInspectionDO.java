package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class QualityInspectionDO {
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
    private Integer isAutoInspection;
    private String sourceSystem;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Integer deleted;
}