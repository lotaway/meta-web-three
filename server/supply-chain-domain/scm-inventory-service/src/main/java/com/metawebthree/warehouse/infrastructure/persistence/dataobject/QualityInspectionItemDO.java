package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class QualityInspectionItemDO {
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
    private Integer deleted;
}