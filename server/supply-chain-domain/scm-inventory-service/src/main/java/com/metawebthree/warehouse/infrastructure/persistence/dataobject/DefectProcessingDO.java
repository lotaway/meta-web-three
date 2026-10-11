package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class DefectProcessingDO {
    private Long id;
    private Long defectId;
    private String processingNo;
    private String processingType;
    private String processingStatus;
    private Integer processingQuantity;
    private BigDecimal processingPrice;
    private String processingReason;
    private String processingRemark;
    private String processor;
    private LocalDateTime processingTime;
    private String relatedDocumentNo;
    private String relatedDocumentType;
    private String approver;
    private LocalDateTime approveTime;
    private String approveRemark;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Integer deleted;
}